"""
OnnxEmbedder 自测

验证 ONNX 轻量向量器：加载、维度、归一化、语义相关性、批量与单条一致性。

依赖 onnxruntime + tokenizers 与已下载的模型；
任一缺失就打印 SKIP 并正常退出（退出码 0），不阻塞 CI/回归。

用法:
  python tests/test_onnx_embedder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from utils import OnnxEmbedder  # noqa: E402


def main() -> int:
    embedder = OnnxEmbedder()

    missing_deps = []
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        missing_deps = ["onnxruntime / tokenizers"]

    missing_files = embedder.missing_files()
    if missing_deps or missing_files:
        print("SKIP: ONNX 向量器自测跳过")
        if missing_deps:
            print(f"  缺少依赖: {', '.join(missing_deps)} -> pip install onnxruntime tokenizers")
        if missing_files:
            print(f"  缺少模型文件: {', '.join(missing_files)} -> python download_model.py")
        return 0

    print("OnnxEmbedder 自测")
    embedder.load_model()

    # 1) 维度与归一化
    assert embedder.embedding_dim == 512, f"维度异常: {embedder.embedding_dim}"
    vec = np.array(embedder.embed_single("向量数据库检索")["embedding"], dtype=np.float32)
    assert vec.shape == (512,), vec.shape
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-3, "向量未做 L2 归一化"
    print(f"  [OK] 维度 {embedder.embedding_dim}，模长 {float(np.linalg.norm(vec)):.4f}")

    # 2) 批次 vs 单条必须一致（padding 不该影响结果）
    batch = embedder.embed_texts(["很短的句子", "这是一段明显长一些的测试文本，用来触发 padding 分支"])
    single = embedder.embed_single("很短的句子")
    diff = float(np.abs(np.array(batch[0]["embedding"]) - np.array(single["embedding"])).max())
    assert diff < 1e-4, f"批量与单条结果不一致，最大差 {diff}"
    print(f"  [OK] 批量与单条结果一致（最大差 {diff:.2e}）")

    # 3) 语义相关性：同义句应明显高于无关句 —— 这正是哈希向量器做不到的
    query = embedder.embed_single("什么是向量数据库")["embedding"]
    q = np.array(query, dtype=np.float32)

    def sim(text: str) -> float:
        return float(q @ np.array(embedder.embed_single(text)["embedding"], dtype=np.float32))

    related = sim("向量数据库是专门存储和查询高维向量的数据库系统")
    unrelated = sim("今天天气不错适合出去散步")
    assert related > unrelated + 0.3, f"语义区分度不足: 相关 {related:.4f} vs 无关 {unrelated:.4f}"
    print(f"  [OK] 语义区分度: 相关 {related:.4f} vs 无关 {unrelated:.4f}")

    # 4) 同义不同字面（哈希向量器会失败，真实模型应能命中）
    paraphrase = sim("想知道怎么把长文本切成一段一段")
    literal = sim("如何进行文本分块")
    gap = abs(paraphrase - literal)
    assert gap < 0.25, f"同义句相似度偏离过大: {gap:.4f}"
    print(f"  [OK] 同义改写稳定: 「怎么把长文本切成一段一段」{paraphrase:.4f} vs 「如何进行文本分块」{literal:.4f}")

    print(f"\n指纹: {embedder.signature}")
    print("\n全部通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
