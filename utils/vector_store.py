"""
向量存储模块（轻量本地后端）

不再强制依赖 ChromaDB，提供三种可插拔后端：

  - ``faiss``  : faiss-cpu 的 ``IndexIDMap2 + IndexFlatIP``，向量已 L2 归一化，
                 内积即余弦相似度。适合几千到几十万条。
  - ``numpy``  : 纯 numpy 矩阵乘法暴力检索，零额外依赖，小规模数据完全够用。
  - ``chroma`` : 旧的 ChromaDB 后端，仅在显式指定且已安装 chromadb 时可用。

``backend="auto"``（默认）按 faiss -> numpy -> chroma 的顺序自动挑一个可用的。

faiss / numpy 两个后端共享同一套持久化格式，落在 ``vector_db/`` 目录：

  ``<collection>.vectors.npy``    L2 归一化后的 float32 向量矩阵，形状 [N, D]
  ``<collection>.records.jsonl``  每行一条记录：{"int_id","id","text","metadata"}
  ``<collection>.manifest.json``  索引元信息：后端、维度、条数、embedding 指纹、时间戳

faiss 用的是 ``IndexFlatIP`` 暴力索引，索引本体就是上面那份向量矩阵，因此**不额外
落盘** —— 加载时用 numpy 反序列化后直接喂给 faiss 重建即可。这样既省一个文件，也
避免了"索引文件与向量文件不同步"导致的脏读；顺带绕开了 faiss 自带 ``write_index``
在 Windows 上无法处理非 ASCII 路径的问题。

因为正文和向量都落盘，进程重启后知识库依旧可用；``add_documents`` 对已存在的
id 采用 upsert 语义，重复跑同一批文件不会产生重复记录。
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

MANIFEST_VERSION = 2
SUPPORTED_BACKENDS = ("auto", "faiss", "numpy", "chroma")


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化，零向量保持为零向量。"""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


def safe_unlink(path: Path) -> bool:
    """
    尽力删除文件，失败时退化为"截断为空文件"。

    顺序刻意是 ``os.remove`` -> ``Path.unlink`` -> 截断：某些环境（公司策略、受管终端、
    沙箱）会把 ``Path.unlink`` 重定向到回收站 API，在没有桌面会话或路径异常时直接抛
    OSError，而 ``os.remove`` 往往仍然可用。
    """
    if not path.exists():
        return True
    for remove in (os.remove, Path.unlink):
        try:
            remove(path)
            return True
        except OSError:
            continue
    try:
        with open(path, "wb"):
            pass
        return True
    except OSError:
        return False


class VectorStore:
    """本地持久化向量库。"""

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        persist_directory: str = "./vector_db",
        use_mock: bool = False,
        backend: str = "auto",
        embedder_signature: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        参数:
            collection_name: 集合名，同时作为持久化文件名前缀
            persist_directory: 持久化目录
            use_mock: 已废弃，仅保留兼容老调用方；不再影响存储行为
            backend: auto / faiss / numpy / chroma
            embedder_signature: 生成向量的 embedder 指纹，用于检索时校验维度源一致性
            normalize: 是否在入库前强制 L2 归一化（余弦检索建议保持 True）
        """
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"不支持的 backend: {backend!r}，可选: {', '.join(SUPPORTED_BACKENDS)}"
            )

        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.use_mock = use_mock  # 兼容字段，不再参与存储决策
        self.normalize = normalize
        self.embedder_signature = embedder_signature

        self.backend = self._resolve_backend(backend)
        self.dimension: Optional[int] = None
        self.manifest: Dict[str, Any] = {}
        self.warnings: List[str] = []

        # 本地后端（faiss / numpy）的内存镜像
        self._matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._records: List[Dict[str, Any]] = []
        self._rec_by_int: Dict[int, Dict[str, Any]] = {}
        self._int_by_id: Dict[str, int] = {}
        self._next_int_id: int = 1
        self._index = None  # faiss index

        # chroma 后端
        self.client = None
        self.collection = None
        # 兼容旧字段：老代码可能读 self.mock_data
        self.mock_data: List[Dict[str, Any]] = []

        if self.backend == "chroma":
            self._init_chromadb()
        else:
            self._load()

    # ------------------------------------------------------------------ #
    # 后端选择与文件路径
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_backend(backend: str) -> str:
        if backend != "auto":
            if backend == "faiss" and not VectorStore._faiss_available():
                raise ImportError(
                    "backend='faiss' 但未安装 faiss-cpu，请执行 pip install faiss-cpu "
                    "或改用 backend='numpy'"
                )
            if backend == "chroma" and not VectorStore._chroma_available():
                raise ImportError("backend='chroma' 但未安装 chromadb，请执行 pip install chromadb")
            return backend

        if VectorStore._faiss_available():
            return "faiss"
        if VectorStore._chroma_available():
            return "chroma"
        return "numpy"

    @staticmethod
    def _faiss_available() -> bool:
        try:
            import faiss  # noqa: F401

            return True
        except Exception:
            return False

    @staticmethod
    def _chroma_available() -> bool:
        try:
            import chromadb  # noqa: F401

            return True
        except Exception:
            return False

    def _path(self, suffix: str) -> Path:
        return self.persist_directory / f"{self.collection_name}{suffix}"

    @property
    def vectors_path(self) -> Path:
        return self._path(".vectors.npy")

    @property
    def records_path(self) -> Path:
        return self._path(".records.jsonl")

    @property
    def manifest_path(self) -> Path:
        return self._path(".manifest.json")

    @property
    def index_path(self) -> Path:
        """历史版本遗留的 faiss 索引文件路径（现版本不再写入，仅用于清理）。"""
        return self._path(".index")

    # ------------------------------------------------------------------ #
    # 载入 / 保存
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        """从磁盘恢复索引；文件缺失或损坏时退化为空库。"""
        self._matrix = np.empty((0, 0), dtype=np.float32)
        self._records = []
        self._rec_by_int = {}
        self._int_by_id = {}
        self._next_int_id = 1

        if not self.records_path.exists():
            self._index = None
            self._load_manifest_only()
            return

        try:
            records: List[Dict[str, Any]] = []
            with open(self.records_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as exc:
            self.warnings.append(f"读取记录文件失败，已按空库启动: {exc}")
            self._load_manifest_only()
            return

        if not records:
            self._load_manifest_only()
            return

        try:
            matrix = np.load(self.vectors_path).astype(np.float32)
        except Exception as exc:
            self.warnings.append(f"读取向量文件失败，已按空库启动: {exc}")
            return

        if matrix.shape[0] != len(records):
            self.warnings.append(
                f"向量条数({matrix.shape[0]})与记录条数({len(records)})不一致，"
                f"已按空库启动，建议重建索引"
            )
            return

        self._matrix = matrix if matrix.ndim == 2 else matrix.reshape(1, -1)
        self._records = records
        for rec in records:
            rec["text"] = rec.get("text", "")
            rec["metadata"] = rec.get("metadata") or {}
            self._rec_by_int[int(rec["int_id"])] = rec
            self._int_by_id[str(rec["id"])] = int(rec["int_id"])
        self._next_int_id = max(self._rec_by_int) + 1

        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
            except Exception:
                self.manifest = {}

        self.dimension = int(self._matrix.shape[1])
        self.mock_data = [
            {
                "id": r["id"],
                "text": r["text"],
                "embedding": self._matrix[i].tolist(),
                "metadata": r["metadata"],
            }
            for i, r in enumerate(self._records)
        ]

        self._check_manifest_consistency()

        if self.backend == "faiss" and not self._rebuild_faiss():
            self.backend = "numpy"
        # 历史版本可能留下过 .index 文件，它已不再被使用，顺手清理以免被误读
        if self.index_path.exists():
            safe_unlink(self.index_path)

    def _load_manifest_only(self) -> None:
        """空库情况：只读 manifest，保留维度/指纹等元信息用于状态展示。"""
        self._index = None
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
            except Exception:
                self.manifest = {}
        dim = self.manifest.get("dimension")
        self.dimension = int(dim) if dim else None

    def _check_manifest_consistency(self) -> None:
        sig = self.manifest.get("embedder_signature")
        if self.embedder_signature and sig and sig != self.embedder_signature:
            self.warnings.append(
                f"索引由 {sig} 构建，当前使用 {self.embedder_signature}，"
                f"向量空间不一致，检索结果不可信"
            )
        dim = self.manifest.get("dimension")
        if dim and self.dimension and int(dim) != int(self.dimension):
            self.warnings.append(
                f"manifest 记录维度 {dim} 与向量实际维度 {self.dimension} 不一致"
            )

    def _rebuild_faiss(self) -> bool:
        """
        用内存中的向量矩阵重建 faiss 索引。

        ``IndexFlatIP`` 是暴力检索索引，其全部数据就是这份矩阵，所以直接
        ``add_with_ids`` 即可，无需读写任何索引文件（见模块开头说明）。
        """
        try:
            import faiss
        except Exception:
            return False

        dim = self._matrix.shape[1] if self._matrix.ndim == 2 and self._matrix.size else 1
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        if self._matrix.shape[0] > 0:
            int_ids = np.array([int(r["int_id"]) for r in self._records], dtype=np.int64)
            index.add_with_ids(np.ascontiguousarray(self._matrix, dtype=np.float32), int_ids)
        self._index = index
        return True

    def save(self) -> None:
        """把索引写回磁盘（chroma 后端由 chromadb 自己负责）。"""
        if self.backend == "chroma":
            return

        if self._matrix.shape[0] == 0:
            # 空库不删文件，改写为空状态，避免依赖删除操作（见 safe_unlink 说明）
            empty_dim = int(self.dimension or 0)
            np.save(self.vectors_path, np.empty((0, empty_dim), dtype=np.float32))
            with open(self.records_path, "w", encoding="utf-8"):
                pass
            safe_unlink(self.index_path)
            self._write_manifest()
            return

        np.save(self.vectors_path, np.ascontiguousarray(self._matrix, dtype=np.float32))

        tmp_records = self.records_path.with_suffix(".jsonl.tmp")
        with open(tmp_records, "w", encoding="utf-8") as f:
            for rec in self._records:
                f.write(
                    json.dumps(
                        {
                            "int_id": rec["int_id"],
                            "id": rec["id"],
                            "text": rec["text"],
                            "metadata": rec["metadata"],
                        },
                        ensure_ascii=False,
                    )
                )
                f.write("\n")
        os.replace(tmp_records, self.records_path)

        self._write_manifest()

    def _write_manifest(self) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        created = self.manifest.get("created_at") or now
        self.manifest = {
            "manifest_version": MANIFEST_VERSION,
            "collection_name": self.collection_name,
            "backend": self.backend,
            "dimension": int(self.dimension or 0),
            "count": len(self._records),
            "normalized": bool(self.normalize),
            "embedder_signature": self.embedder_signature
            or self.manifest.get("embedder_signature"),
            "created_at": created,
            "updated_at": now,
        }
        tmp = self.manifest_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.manifest_path)

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """批量写入，已存在的 id 会被覆盖（upsert）。"""
        if not texts or not embeddings:
            return {"added": 0, "error": "空数据", "total": self.count()}

        if len(texts) != len(embeddings):
            return {"added": 0, "error": "文本和向量数量不匹配", "total": self.count()}

        if self.backend == "chroma":
            return self._chroma_add(texts, embeddings, metadatas, ids)

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        if self.dimension is None and len(self._records) == 0:
            self.dimension = int(matrix.shape[1])
        elif matrix.shape[1] != self.dimension:
            return {
                "added": 0,
                "error": f"向量维度不匹配：库为 {self.dimension}，输入为 {matrix.shape[1]}",
                "total": self.count(),
            }

        if self.normalize:
            matrix = l2_normalize(matrix)

        if ids is None:
            ids = [f"{self.collection_name}_doc_{self._next_int_id + i}" for i in range(len(texts))]
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]

        ids = [str(i) for i in ids]

        # upsert：先清掉同 id 的旧记录
        existing = [i for i in ids if i in self._int_by_id]
        if existing:
            self.delete_documents(existing, save=False)

        new_records: List[Dict[str, Any]] = []
        for text, metadata, doc_id in zip(texts, metadatas, ids):
            int_id = self._next_int_id
            self._next_int_id += 1
            rec = {
                "int_id": int_id,
                "id": doc_id,
                "text": text,
                "metadata": dict(metadata or {}),
            }
            new_records.append(rec)
            self._records.append(rec)
            self._rec_by_int[int_id] = rec
            self._int_by_id[doc_id] = int_id

        self._matrix = np.vstack([self._matrix, matrix]) if self._matrix.size else matrix.copy()

        if self.backend == "faiss":
            if self._index is None or self._index.d != matrix.shape[1]:
                self._rebuild_faiss()
            else:
                int_ids = np.array([r["int_id"] for r in new_records], dtype=np.int64)
                self._index.add_with_ids(np.ascontiguousarray(matrix, dtype=np.float32), int_ids)

        self.mock_data = [
            {
                "id": r["id"],
                "text": r["text"],
                "embedding": self._matrix[i].tolist(),
                "metadata": r["metadata"],
            }
            for i, r in enumerate(self._records)
        ]

        if save:
            self.save()

        return {
            "added": len(new_records),
            "replaced": len(existing),
            "total": self.count(),
        }

    def delete_documents(self, ids: Iterable[str], save: bool = True) -> int:
        """按 id 删除，返回实际删除条数。"""
        if self.backend == "chroma":
            id_list = [str(i) for i in ids]
            if not id_list:
                return 0
            self.collection.delete(ids=id_list)
            return len(id_list)

        targets = {str(i) for i in ids} & set(self._int_by_id)
        if not targets:
            return 0

        keep_idx = [i for i, r in enumerate(self._records) if r["id"] not in targets]
        self._records = [self._records[i] for i in keep_idx]
        self._matrix = (
            self._matrix[keep_idx] if keep_idx else np.empty((0, self.dimension or 0), np.float32)
        )
        self._rec_by_int = {r["int_id"]: r for r in self._records}
        self._int_by_id = {r["id"]: r["int_id"] for r in self._records}
        self.mock_data = [
            {
                "id": r["id"],
                "text": r["text"],
                "embedding": self._matrix[i].tolist(),
                "metadata": r["metadata"],
            }
            for i, r in enumerate(self._records)
        ]

        if self.backend == "faiss":
            self._rebuild_faiss()

        if save:
            self.save()
        return len(targets)

    def reset(self, remove_index_file: bool = True) -> None:
        """清空整个集合（含磁盘文件）。"""
        if self.backend == "chroma":
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
            return

        self._matrix = np.empty((0, 0), dtype=np.float32)
        self._records = []
        self._rec_by_int = {}
        self._int_by_id = {}
        self._next_int_id = 1
        self.mock_data = []
        self.manifest = {}
        # 维度一并清空：允许清空后换用不同维度的向量器重建
        self.dimension = None
        self._index = None
        if remove_index_file:
            safe_unlink(self.index_path)
        # 直接改写为空状态文件，不依赖删除
        self.save()

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        余弦相似度检索。

        返回列表，每项含 ``id`` / ``text`` / ``metadata`` / ``score``（余弦相似度，
        越大越像）/ ``distance``（1 - score，兼容旧调用方）。

        ``min_score`` 可选，用于过滤掉明显不相关的结果。
        """
        if self.backend == "chroma":
            return self._chroma_search(query_embedding, n_results)

        if not self._records:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if self.dimension and query.shape[0] != self.dimension:
            raise ValueError(
                f"查询向量维度 {query.shape[0]} 与库维度 {self.dimension} 不一致，"
                f"请确认用于检索的 embedder 与建库时一致"
            )

        if self.normalize:
            norm = float(np.linalg.norm(query))
            if norm > 0:
                query = query / norm

        k = min(max(int(n_results), 1), len(self._records))
        hits: List[Dict[str, Any]] = []

        if self.backend == "faiss" and self._index is not None and self._index.ntotal > 0:
            scores, int_ids = self._index.search(query.reshape(1, -1).astype(np.float32), k)
            for score, int_id in zip(scores[0], int_ids[0]):
                if int_id == -1:
                    continue
                rec = self._rec_by_int.get(int(int_id))
                if rec is None:
                    continue
                hits.append(self._format_hit(rec, float(score)))
        else:
            sims = self._matrix @ query
            order = np.argsort(-sims)[:k]
            for pos in order:
                rec = self._records[int(pos)]
                hits.append(self._format_hit(rec, float(sims[int(pos)])))

        if min_score is not None:
            hits = [h for h in hits if h["score"] >= min_score]
        return hits

    @staticmethod
    def _format_hit(rec: Dict[str, Any], score: float) -> Dict[str, Any]:
        return {
            "id": rec["id"],
            "text": rec["text"],
            "metadata": rec["metadata"],
            "score": score,
            "distance": 1.0 - score,
        }

    # ------------------------------------------------------------------ #
    # 统计
    # ------------------------------------------------------------------ #
    def count(self) -> int:
        if self.backend == "chroma":
            return int(self.collection.count())
        return len(self._records)

    def list_sources(self) -> Dict[str, int]:
        """按来源文件统计条数，方便一眼看清库里都有什么。"""
        if self.backend == "chroma":
            data = self.collection.get(include=["metadatas"])
            counts: Dict[str, int] = {}
            for meta in data.get("metadatas") or []:
                key = str((meta or {}).get("source", "unknown"))
                counts[key] = counts.get(key, 0) + 1
            return counts

        counts = {}
        for rec in self._records:
            key = str(rec["metadata"].get("source", "unknown"))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "collection_name": self.collection_name,
            "total_documents": self.count(),
            "persist_directory": str(self.persist_directory),
            "backend": self.backend,
            "dimension": self.dimension or 0,
            "embedder_signature": self.manifest.get("embedder_signature")
            or self.embedder_signature,
            "sources": self.list_sources(),
            "updated_at": self.manifest.get("updated_at"),
        }
        if self.warnings:
            stats["warnings"] = list(self.warnings)
        return stats

    def describe(self) -> str:
        """人类可读的一行状态摘要。"""
        if self.count() == 0:
            return f"[{self.backend}] 知识库为空（{self.persist_directory}）"
        return (
            f"[{self.backend}] {self.count()} 条向量 / {self.dimension} 维 / "
            f"{len(self.list_sources())} 个来源文件"
        )

    # ------------------------------------------------------------------ #
    # ChromaDB 兼容后端
    # ------------------------------------------------------------------ #
    def _init_chromadb(self) -> None:
        import chromadb

        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.dimension = None
        print(f"ChromaDB 初始化成功，集合: {self.collection_name}，文档数: {self.collection.count()}")

    def _chroma_add(self, texts, embeddings, metadatas, ids) -> Dict[str, Any]:
        if ids is None:
            ids = [f"{self.collection_name}_doc_{abs(hash(t)) % 10**9}" for t in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]
        try:
            self.collection.upsert(
                documents=texts,
                embeddings=[list(map(float, e)) for e in embeddings],
                metadatas=metadatas,
                ids=[str(i) for i in ids],
            )
            return {"added": len(texts), "total": int(self.collection.count())}
        except Exception as exc:
            return {"added": 0, "error": str(exc), "total": int(self.collection.count())}

    def _chroma_search(self, query_embedding, n_results: int) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[list(map(float, query_embedding))],
            n_results=n_results,
        )
        hits = []
        if results and results.get("documents"):
            for i in range(len(results["documents"][0])):
                distance = results["distances"][0][i] if results.get("distances") else None
                hits.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": (results["metadatas"][0][i] if results.get("metadatas") else {}) or {},
                        "score": (1.0 - distance) if distance is not None else None,
                        "distance": distance,
                    }
                )
        return hits

    # ------------------------------------------------------------------ #
    # 旧接口兼容
    # ------------------------------------------------------------------ #
    def get_all_records(self) -> List[Dict[str, Any]]:
        """导出全部记录，便于调试或做评测。"""
        if self.backend == "chroma":
            data = self.collection.get(include=["documents", "metadatas"])
            return [
                {"id": i, "text": d, "metadata": m or {}}
                for i, d, m in zip(
                    data.get("ids") or [],
                    data.get("documents") or [],
                    data.get("metadatas") or [],
                )
            ]
        return [
            {"id": r["id"], "text": r["text"], "metadata": r["metadata"]} for r in self._records
        ]

    def export_snapshot(self, target_dir: str) -> str:
        """把当前集合的落盘文件复制一份，便于备份。"""
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        if self.backend == "chroma":
            shutil.copytree(self.persist_directory, target, dirs_exist_ok=True)
        else:
            self.save()
            for path in (self.vectors_path, self.records_path, self.manifest_path, self.index_path):
                if path.exists():
                    shutil.copy2(path, target / path.name)
        return str(target)
