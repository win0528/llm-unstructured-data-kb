"""
文档解析模块
支持PDF、Word、TXT、Markdown、HTML、CSV、JSON等格式
"""
import re
import json
import csv
from pathlib import Path
from typing import Dict, Any, List

# pdfplumber / python-docx 采用惰性导入：只用 TXT/MD/JSON 时无需安装这两个重依赖，
# 也让 `from utils import VectorStore` 这类用法不必背负可选依赖。
pdfplumber = None
Document = None


def _require_pdfplumber():
    global pdfplumber
    if pdfplumber is None:
        try:
            import pdfplumber as _pdfplumber
        except ImportError as exc:
            raise ImportError("解析 PDF 需要 pdfplumber，请执行 pip install pdfplumber") from exc
        pdfplumber = _pdfplumber
    return pdfplumber


def _require_docx():
    global Document
    if Document is None:
        try:
            from docx import Document as _Document
        except ImportError as exc:
            raise ImportError("解析 Word 需要 python-docx，请执行 pip install python-docx") from exc
        Document = _Document
    return Document


class DocumentParser:
    """文档解析器"""
    
    SUPPORTED_FORMATS = {'.pdf', '.docx', '.doc', '.txt', '.md', 
                         '.html', '.htm', '.csv', '.json'}
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """解析单个文件"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        ext = file_path.suffix.lower()
        
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"不支持的文件格式: {ext}")
        
        parsers = {
            '.pdf': self._parse_pdf,
            '.docx': self._parse_docx,
            '.doc': self._parse_docx,
            '.txt': self._parse_txt,
            '.md': self._parse_txt,
            '.html': self._parse_html,
            '.htm': self._parse_html,
            '.csv': self._parse_csv,
            '.json': self._parse_json,
        }
        
        return parsers[ext](file_path)
    
    def parse_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """批量解析目录下所有文件"""
        dir_path = Path(dir_path)
        results = []
        
        files = [f for f in dir_path.glob('*') 
                if f.is_file() and f.suffix.lower() in self.SUPPORTED_FORMATS]
        
        for file_path in files:
            try:
                result = self.parse_file(str(file_path))
                results.append(result)
            except Exception as e:
                print(f"警告: 跳过文件 {file_path.name}, 错误: {e}")
        
        return results
    
    def _parse_pdf(self, file_path: Path) -> Dict[str, Any]:
        """解析PDF"""
        text_parts = []
        metadata = {'filename': file_path.name, 'format': 'pdf', 'pages': 0}
        
        try:
            with _require_pdfplumber().open(file_path) as pdf:
                metadata['pages'] = len(pdf.pages)
                
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        lines = text.split('\n')
                        # 去除页眉页脚
                        if lines and len(lines[0]) < 50:
                            lines = lines[1:]
                        if lines and len(lines[-1]) < 50:
                            lines = lines[:-1]
                        text_parts.append('\n'.join(lines))
        
        except Exception as e:
            raise RuntimeError(f"PDF解析失败: {e}")
        
        return {'text': '\n'.join(text_parts), 'metadata': metadata, 'format': 'pdf'}
    
    def _parse_docx(self, file_path: Path) -> Dict[str, Any]:
        """解析Word文档"""
        try:
            doc = _require_docx()(file_path)
            text_parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            
            metadata = {
                'filename': file_path.name,
                'format': 'docx',
                'paragraphs': len(text_parts)
            }
            
            return {'text': '\n'.join(text_parts), 'metadata': metadata, 'format': 'docx'}
        
        except Exception as e:
            raise RuntimeError(f"Word文档解析失败: {e}")
    
    def _parse_txt(self, file_path: Path) -> Dict[str, Any]:
        """解析纯文本/Markdown"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        text = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if text is None:
            raise ValueError(f"无法解码文件: {file_path}")
        
        metadata = {
            'filename': file_path.name,
            'format': file_path.suffix[1:],
            'size': len(text)
        }
        
        return {'text': text, 'metadata': metadata, 'format': file_path.suffix[1:]}
    
    def _parse_html(self, file_path: Path) -> Dict[str, Any]:
        """解析HTML"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html = f.read()
            
            # 移除script和style
            html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
            html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
            
            # 提取title
            title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else ''
            
            # 移除标签
            text = re.sub(r'<[^>]+>', ' ', html)
            text = text.replace('&nbsp;', ' ').replace('&lt;', '<').replace('&gt;', '>')
            text = text.replace('&amp;', '&').replace('&quot;', '"')
            text = re.sub(r'\s+', ' ', text).strip()
            
            metadata = {
                'filename': file_path.name,
                'format': 'html',
                'title': title,
                'size': len(text)
            }
            
            return {'text': text, 'metadata': metadata, 'format': 'html'}
        
        except Exception as e:
            raise RuntimeError(f"HTML解析失败: {e}")
    
    def _parse_csv(self, file_path: Path) -> Dict[str, Any]:
        """解析CSV"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        text = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, newline='') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                
                text_parts = []
                if rows:
                    text_parts.append(' | '.join(rows[0]))
                    text_parts.append('-' * 40)
                    for row in rows[1:]:
                        text_parts.append(' | '.join(row))
                
                text = '\n'.join(text_parts)
                break
            except UnicodeDecodeError:
                continue
        
        if text is None:
            raise ValueError(f"无法解码CSV: {file_path}")
        
        metadata = {
            'filename': file_path.name,
            'format': 'csv',
            'rows': len(rows) if rows else 0,
            'size': len(text)
        }
        
        return {'text': text, 'metadata': metadata, 'format': 'csv'}
    
    def _parse_json(self, file_path: Path) -> Dict[str, Any]:
        """解析JSON"""
        encodings = ['utf-8', 'gbk', 'latin-1']
        data = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    data = json.load(f)
                break
            except UnicodeDecodeError:
                continue
        
        if data is None:
            raise ValueError(f"无法解码JSON: {file_path}")
        
        text = self._json_to_text(data)
        
        metadata = {
            'filename': file_path.name,
            'format': 'json',
            'size': len(text)
        }
        
        return {'text': text, 'metadata': metadata, 'format': 'json'}
    
    def _json_to_text(self, data, prefix="") -> str:
        """JSON转文本"""
        paragraphs = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    paragraphs.append(f"{prefix}{key}：")
                    paragraphs.append(self._json_to_text(value, prefix + "  "))
                else:
                    paragraphs.append(f"{prefix}{key}：{value}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    parts = [f"{k}：{v}" for k, v in item.items() 
                            if not isinstance(v, (dict, list))]
                    paragraphs.append(prefix + "；".join(parts))
                else:
                    paragraphs.append(f"{prefix}- {item}")
        else:
            paragraphs.append(f"{prefix}{data}")
        
        return '\n\n'.join(p for p in paragraphs if p.strip())
