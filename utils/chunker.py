"""
文本分块模块
基于语义和段落结构进行智能分块
"""
import re
from typing import List, Dict, Any


class TextChunker:
    """文本分块器"""
    
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        min_chunk_length: int = 50
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
    
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """对文本进行智能分块"""
        if not text or not text.strip():
            return []
        
        # 按段落分割
        paragraphs = self._split_paragraphs(text)
        
        # 识别结构
        structured_blocks = self._identify_structure(paragraphs)
        
        # 创建块
        chunks = self._create_chunks(structured_blocks)
        
        # 添加元数据
        for i, chunk in enumerate(chunks):
            chunk['chunk_id'] = i
            chunk['metadata'] = metadata or {}
        
        return chunks
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """按段落分割"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _identify_structure(self, paragraphs: List[str]) -> List[Dict[str, Any]]:
        """识别段落结构"""
        blocks = []
        
        for para in paragraphs:
            is_title = self._is_title(para)
            blocks.append({
                'text': para,
                'is_title': is_title,
                'length': len(para)
            })
        
        return blocks
    
    def _is_title(self, text: str) -> bool:
        """判断是否为标题"""
        text = text.strip()
        
        if len(text) > 100:
            return False
        
        title_patterns = [
            r'^#{1,6}\s+.+$',
            r'^第[一二三四五六七八九十百千]+[章节篇部].*$',
            r'^\d+\.\s+.+$',
            r'^[一二三四五六七八九十]+[、.]\s*.+$',
        ]
        
        for pattern in title_patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def _create_chunks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """创建文本块"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for block in blocks:
            block_text = block['text']
            block_length = block['length']
            
            # 标题且当前块不为空，先保存
            if block['is_title'] and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_length:
                    chunks.append({
                        'text': chunk_text,
                        'start_pos': 0,
                        'end_pos': len(chunk_text)
                    })
                current_chunk = []
                current_length = 0
            
            # 超过最大长度
            if current_length + block_length > self.chunk_size:
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    if len(chunk_text) >= self.min_chunk_length:
                        chunks.append({
                            'text': chunk_text,
                            'start_pos': 0,
                            'end_pos': len(chunk_text)
                        })
                    
                    # 重叠处理
                    if self.chunk_overlap > 0 and len(chunk_text) > self.chunk_overlap:
                        overlap_text = chunk_text[-self.chunk_overlap:]
                        current_chunk = [overlap_text]
                        current_length = len(overlap_text)
                    else:
                        current_chunk = []
                        current_length = 0
                
                # 单个块超长，强制分割
                if block_length > self.chunk_size:
                    sub_chunks = self._split_long_text(block_text)
                    for sub in sub_chunks:
                        if len(sub) >= self.min_chunk_length:
                            chunks.append({
                                'text': sub,
                                'start_pos': 0,
                                'end_pos': len(sub)
                            })
                else:
                    current_chunk.append(block_text)
                    current_length += block_length
            else:
                current_chunk.append(block_text)
                current_length += block_length
        
        # 保存最后一块
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_length:
                chunks.append({
                    'text': chunk_text,
                    'start_pos': 0,
                    'end_pos': len(chunk_text)
                })
        
        return chunks
    
    def _split_long_text(self, text: str) -> List[str]:
        """分割超长文本"""
        chunks = []
        sentences = re.split(r'(?<=[。！？.!?])\s*', text)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                if sentence_length > self.chunk_size:
                    for i in range(0, sentence_length, self.chunk_size):
                        chunks.append(sentence[i:i+self.chunk_size])
                else:
                    current_chunk.append(sentence)
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        return chunks
