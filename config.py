"""
项目配置文件
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据目录配置
DATA_RAW_DIR = PROJECT_ROOT / "data_raw"
DATA_CLEAN_DIR = PROJECT_ROOT / "data_clean"
DATA_CHUNK_DIR = PROJECT_ROOT / "data_chunk"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
OUTPUT_DIR = PROJECT_ROOT / "output"

# 确保目录存在
for dir_path in [DATA_RAW_DIR, DATA_CLEAN_DIR, DATA_CHUNK_DIR, VECTOR_DB_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 文本处理配置
TEXT_CONFIG = {
    "min_chunk_length": 50,      # 最小chunk长度（字符数）
    "max_chunk_length": 1024,    # 最大chunk长度
    "target_chunk_length": 512,  # 目标chunk长度
    "overlap": 50,               # chunk重叠长度
}

# 向量化配置
EMBEDDING_CONFIG = {
    "model_name": "BAAI/bge-small-zh-v1.5",  # 中文Embedding模型
    "batch_size": 32,
    "device": "cpu",  # 或 "cuda" 如果有GPU
}

# HuggingFace 下载配置
# 国内直连 huggingface.co 通常超时（表现为 502 Bad Gateway 或连接挂起），
# 这里默认自动切到国内镜像 hf-mirror.com。若你已有代理或想用官方源，
# 把 auto_mirror 设为 False，或提前自行设置环境变量 HF_ENDPOINT。
HF_CONFIG = {
    "auto_mirror": True,
    "endpoint": "https://hf-mirror.com",
}

# 向量器引擎: "auto" | "onnx" | "st" | "mock"
#   auto —— 有 ONNX 模型就用 ONNX（轻量，不需要 torch），否则退到 sentence-transformers
#   onnx —— 强制用 onnxruntime + tokenizers 跑 BGE（推荐，依赖约 150MB）
#   st   —— 强制用 sentence-transformers（需要 torch，约 2.5GB）
#   mock —— 离线哈希向量器（零依赖，仅验证链路）
EMBEDDING_ENGINE = "auto"

# ONNX 向量器配置（模型用 python download_model.py 下载）
ONNX_CONFIG = {
    "model_dir": str(PROJECT_ROOT / "models" / "bge-small-zh-v1.5-onnx"),
    "max_length": 512,
    "batch_size": 32,
    "pooling": "cls",          # BGE 系列用 CLS 池化
    "query_instruction": "",   # 留空表示查询不加前缀；建库与检索必须一致
    "threads": None,           # None = 交给 onnxruntime 决定
}

# 向量库配置
# backend 可选: "auto" | "faiss" | "numpy" | "chroma"
#   auto  —— 按 faiss -> numpy -> chroma 的顺序自动选择可用后端（推荐）
#   faiss —— faiss-cpu 的 IndexIDMap2 + IndexFlatIP，向量已归一化，内积即余弦
#   numpy —— 纯 numpy 暴力检索，零额外依赖
#   chroma—— 旧的 ChromaDB 后端（需自行安装 chromadb）
VECTOR_DB_CONFIG = {
    "collection_name": "knowledge_base",
    "persist_directory": str(VECTOR_DB_DIR),
    "backend": "auto",
    "normalize": True,       # 入库前 L2 归一化，配合内积做余弦检索
}

# 兼容旧配置名（等价于 VECTOR_DB_CONFIG 的集合名与目录）
CHROMA_CONFIG = {
    "collection_name": VECTOR_DB_CONFIG["collection_name"],
    "persist_directory": VECTOR_DB_CONFIG["persist_directory"],
}

# Mock / 哈希向量器维度，建库与检索必须一致
MOCK_EMBEDDING_DIM = 512

# 清洗配置
CLEAN_CONFIG = {
    "min_paragraph_length": 20,  # 最小段落长度
    "remove_special_chars": True,
    "normalize_whitespace": True,
}
