"""
ONNX 向量化模块（轻量路线）

用 ``onnxruntime`` + ``tokenizers`` 跑 BGE 中文向量模型，**不需要 torch，也不需要
transformers**：

  - 权重   : ``onnx/model.onnx``（fp32，约 90MB）或 ``model_quantized.onnx``（int8，约 23MB）
  - 分词器 : ``tokenizer.json``，由 ``tokenizers`` 库直接加载，与 HuggingFace 分词结果一致
  - 池化   : BGE 系列用 [CLS] 位置（句子首 token）的隐状态，再 L2 归一化

对比：
  - ``TextEmbedder``（sentence-transformers）需要 torch，装完约 2.5GB
  - ``OnnxEmbedder``（本模块）依赖约 150MB，向量质量与前者基本一致

模型文件用 ``python download_model.py`` 下载。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# 默认模型目录（相对项目根）
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "bge-small-zh-v1.5-onnx"

MODEL_CANDIDATES = ("model.onnx", "model_quantized.onnx")
REQUIRED_FILES = ("tokenizer.json", "config.json")


class OnnxEmbedder:
    """
    ONNX 版 BGE 向量化器。

    参数:
        model_dir: 模型目录，需含 tokenizer.json / config.json 与 onnx 权重
        max_length: 截断长度，BGE 上限 512
        batch_size: 推理批大小
        pooling: "cls"（BGE 默认）或 "mean"
        query_instruction: 检索时给查询加的前缀。BGE 官方建议短查询加
            "为这个句子生成表示以用于检索相关文章："，但建库与检索必须一致，
            默认不加以避免两边不对称。
        threads: onnxruntime 线程数，None 表示交给运行时决定
    """

    def __init__(
        self,
        model_dir: Optional[str | Path] = None,
        max_length: int = 512,
        batch_size: int = 32,
        pooling: str = "cls",
        query_instruction: str = "",
        threads: Optional[int] = None,
    ):
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.pooling = pooling
        self.query_instruction = query_instruction
        self.threads = threads

        self.session = None
        self.tokenizer = None
        self._input_names: set[str] = set()
        self._output_names: List[str] = []

        # 未加载 ORT 前就能确定维度与指纹：直接读 config.json 里的 hidden_size
        self.model_path = self._locate_model()
        self.embedding_dim = self._read_hidden_size()

    # ------------------------------------------------------------------ #
    # 初始化辅助
    # ------------------------------------------------------------------ #
    def _locate_model(self) -> Optional[Path]:
        for name in MODEL_CANDIDATES:
            candidate = self.model_dir / name
            if candidate.exists():
                return candidate
        return None

    def _read_hidden_size(self) -> Optional[int]:
        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            return None
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return int(json.load(f).get("hidden_size")) or None
        except Exception:
            return None

    @property
    def signature(self) -> str:
        """向量器指纹，写进索引 manifest 用于校验建库/检索是否同源。"""
        variant = self.model_path.name.replace(".onnx", "") if self.model_path else "unknown"
        return f"onnx-bge:{self.model_dir.name}:{variant}:{self.pooling}:{self.embedding_dim or '?'}"

    def missing_files(self) -> List[str]:
        """返回缺失的必需文件（空列表表示齐了）。"""
        missing = [name for name in REQUIRED_FILES if not (self.model_dir / name).exists()]
        if self.model_path is None:
            missing.extend(MODEL_CANDIDATES[0])
        return missing

    @staticmethod
    def _require_deps():
        try:
            import onnxruntime  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "缺少 onnxruntime，请执行: pip install onnxruntime tokenizers"
            ) from exc
        try:
            from tokenizers import Tokenizer  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "缺少 tokenizers，请执行: pip install onnxruntime tokenizers"
            ) from exc

    def load_model(self):
        """加载分词器与 ONNX 会话（幂等）。"""
        if self.session is not None:
            return self

        missing = self.missing_files()
        if missing:
            raise RuntimeError(
                f"模型文件不完整，缺少: {', '.join(missing)}\n"
                f"        模型目录: {self.model_dir}\n"
                f"        请先执行: python download_model.py"
            )

        self._require_deps()

        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        options = ort.SessionOptions()
        if self.threads:
            options.intra_op_num_threads = int(self.threads)
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {i.name for i in self.session.get_inputs()}
        self._output_names = [o.name for o in self.session.get_outputs()]

        if self.embedding_dim is None:
            # config.json 缺失时，用一次真实推理兜底探测维度
            probe = self._encode(["维度探测"])
            self.embedding_dim = int(probe.shape[1])
        return self

    # ------------------------------------------------------------------ #
    # 推理
    # ------------------------------------------------------------------ #
    def _encode(self, texts: List[str]) -> np.ndarray:
        """分词 + 推理 + 池化 + L2 归一化，返回 [N, D] float32。"""
        encodings = self.tokenizer.encode_batch(list(texts))

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

        feed: Dict[str, np.ndarray] = {}
        if "input_ids" in self._input_names:
            feed["input_ids"] = input_ids
        if "attention_mask" in self._input_names:
            feed["attention_mask"] = attention_mask
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = token_type_ids

        outputs = self.session.run(None, feed)

        if "last_hidden_state" in self._output_names:
            hidden = outputs[self._output_names.index("last_hidden_state")]
        else:
            hidden = outputs[0]

        if self.pooling == "mean":
            mask = attention_mask[..., None].astype(np.float32)
            pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
        else:
            pooled = hidden[:, 0, :]  # [CLS]

        pooled = np.asarray(pooled, dtype=np.float32)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return (pooled / norms).astype(np.float32)

    def embed_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量向量化，接口与 TextEmbedder 保持一致。"""
        if not texts:
            return []

        self.load_model()

        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t or "" for t in texts[start : start + self.batch_size]]
            vectors.append(self._encode(batch))
        matrix = np.vstack(vectors)

        return [
            {"text": text, "embedding": row.tolist(), "dimension": int(matrix.shape[1])}
            for text, row in zip(texts, matrix)
        ]

    def embed_single(self, text: str, is_query: bool = False) -> Dict[str, Any]:
        """单个文本向量化；is_query=True 时会按需加上查询前缀。"""
        if is_query and self.query_instruction and not text.startswith(self.query_instruction):
            text = self.query_instruction + text
        results = self.embed_texts([text])
        return results[0] if results else None

    def describe(self) -> str:
        variant = self.model_path.name if self.model_path else "未下载"
        return f"ONNX 向量器（{self.model_dir.name} / {variant} / {self.embedding_dim} 维）"
