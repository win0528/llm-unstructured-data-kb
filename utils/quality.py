"""
数据质量统计模块
统计处理过程中的各项指标
"""
from typing import Dict, Any, List
from datetime import datetime
import json
from pathlib import Path


class QualityStats:
    """数据质量统计器"""
    
    def __init__(self):
        self.stats = {
            'timestamp': datetime.now().isoformat(),
            'files': {
                'total': 0,
                'by_format': {},
                'processed': 0,
                'failed': 0
            },
            'text': {
                'original_total_length': 0,
                'cleaned_total_length': 0,
                'reduction_ratio': 0
            },
            'cleaning': {
                'removed_special_chars': 0,
                'removed_empty_lines': 0,
                'removed_page_numbers': 0,
                'removed_short_paragraphs': 0,
                'desensitized_items': 0
            },
            'chunks': {
                'total': 0,
                'avg_length': 0,
                'min_length': 0,
                'max_length': 0
            },
            'vectors': {
                'total': 0,
                'dimension': 0,
                'success_rate': 0
            }
        }
    
    def add_file_stats(self, file_count: int, formats: Dict[str, int]):
        """添加文件统计"""
        self.stats['files']['total'] = file_count
        self.stats['files']['by_format'] = formats
    
    def add_text_stats(self, original_length: int, cleaned_length: int):
        """添加文本统计"""
        self.stats['text']['original_total_length'] = original_length
        self.stats['text']['cleaned_total_length'] = cleaned_length
        
        if original_length > 0:
            self.stats['text']['reduction_ratio'] = (
                (original_length - cleaned_length) / original_length
            )
    
    def add_cleaning_stats(self, cleaning_stats: Dict[str, Any]):
        """添加清洗统计"""
        for key, value in cleaning_stats.items():
            if key in self.stats['cleaning']:
                self.stats['cleaning'][key] += value
    
    def add_chunk_stats(self, chunks: List[Dict[str, Any]]):
        """添加分块统计"""
        if not chunks:
            return
        
        lengths = [len(chunk['text']) for chunk in chunks]
        
        self.stats['chunks']['total'] = len(chunks)
        self.stats['chunks']['avg_length'] = sum(lengths) / len(lengths)
        self.stats['chunks']['min_length'] = min(lengths)
        self.stats['chunks']['max_length'] = max(lengths)
    
    def add_vector_stats(self, vector_count: int, dimension: int, success_rate: float):
        """添加向量统计"""
        self.stats['vectors']['total'] = vector_count
        self.stats['vectors']['dimension'] = dimension
        self.stats['vectors']['success_rate'] = success_rate
    
    def get_stats(self) -> Dict[str, Any]:
        """获取完整统计信息"""
        return self.stats
    
    def print_report(self):
        """打印统计报告"""
        print("=" * 60)
        print("数据质量统计报告")
        print("=" * 60)
        
        print("\n【文件统计】")
        print(f"  总文件数: {self.stats['files']['total']}")
        print(f"  已处理: {self.stats['files']['processed']}")
        print(f"  失败: {self.stats['files']['failed']}")
        print(f"  格式分布: {self.stats['files']['by_format']}")
        
        print("\n【文本统计】")
        print(f"  原始总长度: {self.stats['text']['original_total_length']:,} 字符")
        print(f"  清洗后长度: {self.stats['text']['cleaned_total_length']:,} 字符")
        print(f"  压缩比例: {self.stats['text']['reduction_ratio']:.2%}")
        
        print("\n【清洗统计】")
        print(f"  去除特殊字符: {self.stats['cleaning']['removed_special_chars']}")
        print(f"  去除空行: {self.stats['cleaning']['removed_empty_lines']}")
        print(f"  去除页码: {self.stats['cleaning']['removed_page_numbers']}")
        print(f"  去除短段落: {self.stats['cleaning']['removed_short_paragraphs']}")
        print(f"  脱敏项目: {self.stats['cleaning']['desensitized_items']}")
        
        print("\n【分块统计】")
        print(f"  总块数: {self.stats['chunks']['total']}")
        print(f"  平均长度: {self.stats['chunks']['avg_length']:.0f} 字符")
        print(f"  最小长度: {self.stats['chunks']['min_length']} 字符")
        print(f"  最大长度: {self.stats['chunks']['max_length']} 字符")
        
        print("\n【向量统计】")
        print(f"  本次入库向量数: {self.stats['vectors']['total']}")
        print(f"  向量维度: {self.stats['vectors']['dimension']}")
        print(f"  入库成功率: {self.stats['vectors']['success_rate']:.2%}")
        
        print("=" * 60)
    
    def save_report(self, output_path: str):
        """保存报告到JSON文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        print(f"报告已保存到: {output_path}")
