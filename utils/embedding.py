"""
文本向量化模块

提供两种向量化器：

  - :class:`TextEmbedder`  : 基于 sentence-transformers 的真实语义向量（需要联网下载模型）
  - :class:`MockEmbedder`  : 无依赖的**确定性**哈希向量器，用于离线跑通全流程

MockEmbedder 早期版本用 ``np.random`` 生成向量，同一段文本每次结果都不同，
导致入库的向量和查询的向量不在同一个空间里，检索必然是噪声。现在改为基于字符
n-gram 的哈希向量（hashing trick）：

  * 同一段文本永远得到同一个向量 -> 索引可持久化、可复现
  * 共享 n-gram 越多的文本，余弦相似度越高 -> 检索结果"看起来是对的"

它不是语义模型（同义词无法匹配），但足以在没有模型、没有网络时验证
分块、入库、检索这条链路。两种向量器都暴露 ``signature`` 属性，写入索引的
manifest 后，检索时可以校验"建库用的向量器"和"查询用的向量器"是否一致。
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def apply_hf_mirror() -> str:
    """
    国内直连 huggingface.co 基本必超时，这里在下载模型前把 HF_ENDPOINT 指向镜像。

    已显式设置过 HF_ENDPOINT 的用户不会被覆盖；``HF_CONFIG["auto_mirror"] = False``
    可以整体关掉这个行为。
    """
    existing = os.environ.get("HF_ENDPOINT")
    if existing:
        return existing
    try:
        from config import HF_CONFIG
    except Exception:
        return ""
    if not HF_CONFIG.get("auto_mirror"):
        return ""
    endpoint = HF_CONFIG.get("endpoint") or "https://hf-mirror.com"
    os.environ["HF_ENDPOINT"] = endpoint
    return endpoint


def check_embedding_deps() -> str:
    """检查真实语义模型所需的依赖，返回缺失说明（空字符串表示都齐了）。"""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return "缺少 sentence-transformers（真实语义模型依赖它，会连带安装 torch）"
    return ""


class TextEmbedder:
    """基于 sentence-transformers 的语义向量化器。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None

    @property
    def signature(self) -> str:
        """向量器指纹，用于校验索引与查询是否同源。"""
        return f"sentence-transformers:{self.model_name}"

    def load_model(self):
        """加载模型（首次调用会触发下载）。"""
        if self.model is None:
            mirror = apply_hf_mirror()
            if mirror:
                print(f"模型下载源: {mirror}（如需官方源请设置 HF_ENDPOINT 或关闭 auto_mirror）")
            try:
                from sentence_transformers import SentenceTransformer

                print(f"正在加载模型: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                test_embedding = self.model.encode(["测试"])
                self.embedding_dim = len(test_embedding[0])
                print(f"模型加载成功，向量维度: {self.embedding_dim}")
            except ImportError as e:
                raise RuntimeError(
                    f"模型加载失败: {e}\n"
                    f"        当前解释器缺少 sentence-transformers。\n"
                    f"        当前解释器: {sys.executable}\n"
                    f"        安装: \"{sys.executable}\" -m pip install sentence-transformers\n"
                    f"        仅验证链路可改用: --mock"
                )
            except Exception as e:
                raise RuntimeError(f"模型加载失败: {e}")

    def embed_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量向量化文本。"""
        if not texts:
            return []

        self.load_model()

        try:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                normalize_embeddings=True,
            )

            results = []
            for text, embedding in zip(texts, embeddings):
                results.append(
                    {
                        "text": text,
                        "embedding": embedding.tolist(),
                        "dimension": len(embedding),
                    }
                )
            return results

        except Exception as e:
            raise RuntimeError(f"文本向量化失败: {e}")

    def embed_single(self, text: str, is_query: bool = False) -> Dict[str, Any]:
        """向量化单个文本（is_query 仅为接口对齐，本实现不使用）。"""
        results = self.embed_texts([text])
        return results[0] if results else None

    def describe(self) -> str:
        return f"sentence-transformers 向量器（{self.model_name}）"


class MockEmbedder:
    """
    确定性哈希向量器（测试模式专用）。

    参数:
        dimension: 向量维度，必须与建库时保持一致
        ngram_range: 字符 n-gram 范围，默认 (1, 3)
    """

    def __init__(self, dimension: int = 512, ngram_range: Tuple[int, int] = (1, 3)):
        self.dimension = int(dimension)
        self.ngram_range = (int(ngram_range[0]), int(ngram_range[1]))
        self.embedding_dim = self.dimension

    @property
    def signature(self) -> str:
        lo, hi = self.ngram_range
        return f"hashing-v2:{self.dimension}:{lo}-{hi}"

    def load_model(self):
        """哈希向量器无需加载模型，保留该方法以对齐接口。"""
        pass

    # -- 内部实现 ------------------------------------------------------- #
    def _ngrams(self, text: str) -> List[str]:
        """字符 n-gram + ASCII 单词，兼顾中英文。"""
        text = text.lower()
        lo, hi = self.ngram_range
        grams: List[str] = []

        for n in range(lo, hi + 1):
            if len(text) >= n:
                grams.extend(text[i : i + n] for i in range(len(text) - n + 1))

        # 独立的英文/数字词，避免被切碎后丢掉整体语义
        word: List[str] = []
        for ch in text + " ":
            if ch.isalnum() and not ("\u4e00" <= ch <= "\u9fff"):
                word.append(ch)
            else:
                if len(word) > 1:
                    grams.append("w:" + "".join(word))
                word = []
        return grams

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        grams = self._ngrams(text)
        if not grams:
            return vector

        counts: Dict[str, int] = {}
        for gram in grams:
            counts[gram] = counts.get(gram, 0) + 1

        for gram, count in counts.items():
            digest = hashlib.md5(gram.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            # 次线性词频，抑制高频字对相似度的主导
            vector[index] += sign * (1.0 + math.log(count))

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def embed_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量生成确定性向量。"""
        results = []
        for text in texts:
            vector = self._embed_one(text or "")
            results.append(
                {
                    "text": text,
                    "embedding": vector.tolist(),
                    "dimension": self.dimension,
                }
            )
        return results

    def embed_single(self, text: str, is_query: bool = False) -> Dict[str, Any]:
        """单个文本向量化（is_query 仅为接口对齐，本实现不使用）。"""
        return {
            "text": text,
            "embedding": self._embed_one(text or "").tolist(),
            "dimension": self.dimension,
        }

    def describe(self) -> str:
        return f"哈希向量器（{self.dimension} 维，仅字面 n-gram 匹配，不懂同义词）"


class HashingEmbedder(MockEmbedder):
    """``MockEmbedder`` 的语义化别名，推荐在新代码中使用这个名字。"""


def _onnx_deps_ready() -> bool:
    """onnxruntime 与 tokenizers 是否都可导入。"""
    try:
        import onnxruntime  # noqa: F401
        from tokenizers import Tokenizer  # noqa: F401

        return True
    except Exception:
        return False


def _config_defaults() -> Dict[str, Any]:
    """从 config 读取默认值；读不到就退回内置默认，保证模块可独立使用。"""
    try:
        from config import (
            EMBEDDING_CONFIG, EMBEDDING_ENGINE, MOCK_EMBEDDING_DIM, ONNX_CONFIG,
        )

        return {
            "model_name": EMBEDDING_CONFIG.get("model_name", "BAAI/bge-small-zh-v1.5"),
            "dimension": MOCK_EMBEDDING_DIM,
            "engine": EMBEDDING_ENGINE,
            "onnx": dict(ONNX_CONFIG),
        }
    except Exception:
        return {
            "model_name": "BAAI/bge-small-zh-v1.5",
            "dimension": 512,
            "engine": "auto",
            "onnx": {},
        }


def get_embedder(
    use_mock: bool = False,
    model_name: Optional[str] = None,
    dimension: Optional[int] = None,
    engine: str = "auto",
    model_dir: Optional[str] = None,
):
    """
    按引擎创建向量器。

    engine:
        ``mock``  离线哈希向量器（零依赖，同一文本结果稳定，但只做字面 n-gram 匹配）
        ``onnx``  onnxruntime + tokenizers 跑 BGE（轻量，推荐）
        ``st``    sentence-transformers（需要 torch）
        ``auto``  ONNX 依赖与模型就绪就用 ONNX，否则退到 sentence-transformers

    ``use_mock=True`` 等价于 ``engine="mock"``，保留以兼容旧调用。
    """
    from .onnx_embedding import OnnxEmbedder, DEFAULT_MODEL_DIR

    defaults = _config_defaults()
    dimension = dimension or defaults["dimension"]
    model_name = model_name or defaults["model_name"]

    if use_mock:
        engine = "mock"
    elif engine == "auto":
        engine = defaults["engine"] or "auto"

    if engine == "mock":
        return MockEmbedder(dimension=dimension)

    if engine == "st":
        return TextEmbedder(model_name=model_name)

    if engine in ("auto", "onnx"):
        onnx_kwargs = dict(defaults["onnx"]) if defaults["onnx"] else {}
        onnx_kwargs.pop("model_dir", None)
        onnx_embedder = OnnxEmbedder(
            model_dir=model_dir or defaults["onnx"].get("model_dir") or DEFAULT_MODEL_DIR,
            **{k: v for k, v in onnx_kwargs.items() if k in
               ("max_length", "batch_size", "pooling", "query_instruction", "threads")},
        )
        ready = _onnx_deps_ready() and not onnx_embedder.missing_files()
        if engine == "onnx" and not ready:
            raise RuntimeError(
                f"指定了 engine='onnx' 但环境未就绪。\n"
                f"        依赖: {'已装' if _onnx_deps_ready() else '缺 onnxruntime/tokenizers -> pip install onnxruntime tokenizers'}\n"
                f"        模型: {'已就绪' if not onnx_embedder.missing_files() else '缺文件 -> python download_model.py'}\n"
                f"        模型目录: {onnx_embedder.model_dir}\n"
                f"        或者改用 --engine mock 走离线哈希向量器"
            )
        if ready:
            return onnx_embedder
        # auto 模式下 ONNX 没就绪，退到 sentence-transformers
        return TextEmbedder(model_name=model_name)

    raise ValueError(f"未知的 engine: {engine!r}，可选 auto / onnx / st / mock")
