"""
VectorStore 自测脚本

覆盖：落盘与重载、faiss / numpy 两种后端结果一致性、upsert 幂等、
按 id 删除、维度校验、向量器指纹警告。

用法:
  python tests/test_vector_store.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from utils import MockEmbedder, VectorStore  # noqa: E402


TEXTS = [
    "向量数据库用于存储和检索高维向量，支持相似度搜索。",
    "文本分块是把长文档切成语义完整的小段，便于向量化。",
    "数据清洗要去除乱码、特殊符号和多余空行。",
    "BERT 是一种预训练语言模型，广泛用于文本理解任务。",
    "余弦相似度衡量两个向量方向的一致性，取值在 -1 到 1 之间。",
]
QUERY = "什么是余弦相似度"


def make_store(tmp: Path, backend: str) -> VectorStore:
    embedder = MockEmbedder(dimension=256)
    store = VectorStore(
        collection_name="test_kb",
        persist_directory=str(tmp),
        backend=backend,
        embedder_signature=embedder.signature,
    )
    embeddings = [r["embedding"] for r in embedder.embed_texts(TEXTS)]
    ids = [f"doc_{i}" for i in range(len(TEXTS))]
    metas = [{"source": f"file_{i}.txt", "chunk_id": i} for i in range(len(TEXTS))]
    result = store.add_documents(TEXTS, embeddings, metas, ids)
    assert result["added"] == len(TEXTS), result
    assert store.count() == len(TEXTS), store.count()
    return store


def check_backend(backend: str) -> None:
    tmp = Path(tempfile.mkdtemp(prefix=f"vec_{backend}_"))
    try:
        embedder = MockEmbedder(dimension=256)
        store = make_store(tmp, backend)
        assert store.backend == backend, store.backend

        # 落盘文件齐全
        for name in ("test_kb.vectors.npy", "test_kb.records.jsonl", "test_kb.manifest.json"):
            assert (tmp / name).exists(), f"缺少持久化文件: {name}"
        # faiss 索引不落盘：IndexFlatIP 的数据就是 vectors.npy，避免不同步脏读
        assert not (tmp / "test_kb.index").exists(), "不应再写入 .index 文件"

        top1_before = store.search(embedder.embed_single(QUERY)["embedding"], 3)
        assert top1_before, "检索为空"

        # 重新打开（模拟进程重启），结果应完全一致
        reopened = VectorStore(
            collection_name="test_kb",
            persist_directory=str(tmp),
            backend=backend,
            embedder_signature=embedder.signature,
        )
        assert reopened.count() == len(TEXTS), reopened.count()
        assert reopened.dimension == 256, reopened.dimension
        top1_after = reopened.search(embedder.embed_single(QUERY)["embedding"], 3)
        assert [h["id"] for h in top1_before] == [h["id"] for h in top1_after], "重载后检索结果不一致"

        # 结果按相似度降序
        scores = [h["score"] for h in top1_after]
        assert scores == sorted(scores, reverse=True), f"结果未按相似度降序: {scores}"

        # 命中语义上最接近的那条（余弦相似度那条）
        best_text = top1_after[0]["text"]
        assert "余弦相似度" in best_text, f"Top1 未命中余弦相似度描述: {best_text}"

        # upsert 幂等：同一批 id 再写一遍，条数不变
        embeddings = [r["embedding"] for r in embedder.embed_texts(TEXTS)]
        ids = [f"doc_{i}" for i in range(len(TEXTS))]
        again = reopened.add_documents(TEXTS, embeddings, None, ids)
        assert again["replaced"] == len(TEXTS), again
        assert reopened.count() == len(TEXTS), f"upsert 后条数异常: {reopened.count()}"

        # 删除
        removed = reopened.delete_documents(["doc_0", "doc_1"])
        assert removed == 2, removed
        assert reopened.count() == len(TEXTS) - 2, reopened.count()
        assert not any(h["id"] in ("doc_0", "doc_1") for h in reopened.get_all_records())

        # 删除后重载仍然一致
        reloaded = VectorStore(
            collection_name="test_kb",
            persist_directory=str(tmp),
            backend=backend,
            embedder_signature=embedder.signature,
        )
        assert reloaded.count() == len(TEXTS) - 2, reloaded.count()

        # 维度不一致时拒绝写入
        bad = reloaded.add_documents(["x"], [[0.1] * 128], None, ["bad_dim"])
        assert "error" in bad and "维度" in bad["error"], bad

        # 查询维度不一致时报错
        try:
            reloaded.search([0.1] * 128, 3)
            raise AssertionError("维度不一致的查询应当报错")
        except ValueError:
            pass

        # 指纹不一致时给出警告
        other = VectorStore(
            collection_name="test_kb",
            persist_directory=str(tmp),
            backend=backend,
            embedder_signature="hashing-v2:999:1-3",
        )
        assert any("不一致" in w for w in other.warnings), other.warnings

        # 历史遗留的 .index 文件应被自动清理，避免被误当作有效索引
        stale = tmp / "test_kb.index"
        stale.write_bytes(b"stale")
        VectorStore("test_kb", str(tmp), backend=backend)
        assert not stale.exists(), "遗留 .index 文件未被清理"

        # reset 后应为空库，且允许改用不同维度重建
        reloaded.reset()
        assert reloaded.count() == 0, reloaded.count()
        assert VectorStore("test_kb", str(tmp), backend=backend).count() == 0, "reset 未持久化"
        rebuilt = reloaded.add_documents(["换了向量器"], [[0.1] * 64], None, ["new_dim"])
        assert rebuilt.get("added") == 1, rebuilt
        assert reloaded.dimension == 64, reloaded.dimension

        print(f"  [OK] {backend}: {len(TEXTS)} 条入库, Top1={best_text[:18]}..., "
              f"分数 {scores[0]:.4f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_cross_backend_same_result() -> None:
    """faiss 与 numpy 在同一份数据上应给出相同排序。"""
    embedder = MockEmbedder(dimension=256)
    query = embedder.embed_single(QUERY)["embedding"]
    embeddings = [r["embedding"] for r in embedder.embed_texts(TEXTS)]
    ids = [f"doc_{i}" for i in range(len(TEXTS))]

    orders = {}
    dirs = []
    try:
        for backend in ("numpy", "faiss"):
            tmp = Path(tempfile.mkdtemp(prefix=f"cross_{backend}_"))
            dirs.append(tmp)
            store = VectorStore("cross_kb", str(tmp), backend=backend)
            store.add_documents(TEXTS, embeddings, None, ids)
            orders[backend] = [h["id"] for h in store.search(query, 5)]

        if "faiss" in orders:
            assert orders["numpy"] == orders["faiss"], f"两种后端排序不同: {orders}"
            print(f"  [OK] faiss 与 numpy 排序一致: {orders['numpy']}")
        else:
            print("  [SKIP] 未安装 faiss，跳过跨后端一致性检查")
    finally:
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)


def check_deterministic_embedder() -> None:
    """哈希向量器必须确定性。"""
    a = MockEmbedder(dimension=128)
    b = MockEmbedder(dimension=128)
    v1 = np.array(a.embed_single("向量数据库检索")["embedding"])
    v2 = np.array(b.embed_single("向量数据库检索")["embedding"])
    assert np.allclose(v1, v2), "同一文本两次向量化结果不同"

    v3 = np.array(a.embed_single("向量数据库检索与相似度")["embedding"])
    assert float(np.linalg.norm(v1)) > 0.99, "向量未归一化"
    assert float(v1 @ v3) > float(v1 @ np.array(a.embed_single("今天天气不错")["embedding"]))
    print(f"  [OK] 哈希向量器确定性 & 相关性: 相似文本 {v1 @ v3:.4f} > 无关文本")


if __name__ == "__main__":
    print("VectorStore 自测")
    check_deterministic_embedder()
    for backend in ("numpy", "faiss"):
        check_backend(backend)
    check_cross_backend_same_result()
    print("\n全部通过 ✅")
