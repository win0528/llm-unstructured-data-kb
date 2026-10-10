"""
工具模块
包含文档解析、文本清洗、分块、向量化、向量存储、质量统计等功能
"""

from .parser import DocumentParser
from .clean import TextCleaner, TextNormalizer, LanguageDetector, TextQualityScorer
from .chunker import TextChunker
from .embedding import (
    TextEmbedder, MockEmbedder, HashingEmbedder, get_embedder,
    apply_hf_mirror, check_embedding_deps,
)
from .onnx_embedding import OnnxEmbedder
from .vector_store import VectorStore
from .quality import QualityStats
from .elements import (
    ElementType, Element, Title, Heading, Paragraph, 
    ListItem, Table, CodeBlock, SensitiveInfo, ElementList
)

__all__ = [
    'DocumentParser',
    'TextCleaner',
    'TextNormalizer',
    'LanguageDetector',
    'TextQualityScorer',
    'TextChunker',
    'TextEmbedder',
    'MockEmbedder',
    'HashingEmbedder',
    'OnnxEmbedder',
    'get_embedder',
    'apply_hf_mirror',
    'check_embedding_deps',
    'VectorStore',
    'QualityStats',
    # 文档元素模型
    'ElementType',
    'Element',
    'Title',
    'Heading',
    'Paragraph',
    'ListItem',
    'Table',
    'CodeBlock',
    'SensitiveInfo',
    'ElementList',
]
