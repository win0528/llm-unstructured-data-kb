"""
文档元素模型
参考 Unstructured 项目的 Element 类型系统
将文档解析为结构化的元素类型，而非简单的纯文本
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json


class ElementType(Enum):
    """文档元素类型"""
    TITLE = "Title"                    # 文档标题
    HEADING = "Heading"                # 章节标题
    PARAGRAPH = "Paragraph"            # 段落
    LIST_ITEM = "ListItem"             # 列表项
    TABLE = "Table"                    # 表格
    FIGURE = "Figure"                  # 图表
    IMAGE = "Image"                    # 图片
    CODE_BLOCK = "CodeBlock"           # 代码块
    FORMULA = "Formula"                # 公式
    FOOTNOTE = "Footnote"              # 脚注
    PAGE_NUMBER = "PageNumber"         # 页码
    HEADER = "Header"                  # 页眉
    FOOTER = "Footer"                  # 页脚
    EMAIL = "Email"                    # 邮箱地址
    PHONE = "Phone"                    # 电话号码
    URL = "URL"                        # 网址
    ADDRESS = "Address"                # 地址
    DATE = "Date"                      # 日期
    MONEY = "Money"                    # 金额
    UNKNOWN = "Unknown"                # 未知类型


@dataclass
class Element:
    """文档元素基类"""
    element_id: str
    element_type: ElementType
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'element_id': self.element_id,
            'element_type': self.element_type.value,
            'text': self.text,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Element':
        """从字典创建"""
        return cls(
            element_id=data['element_id'],
            element_type=ElementType(data['element_type']),
            text=data['text'],
            metadata=data.get('metadata', {})
        )


@dataclass
class Title(Element):
    """文档标题"""
    def __init__(self, element_id: str, text: str, metadata: Dict[str, Any] = None):
        super().__init__(element_id, ElementType.TITLE, text, metadata or {})


@dataclass
class Heading(Element):
    """章节标题"""
    level: int = 1  # 标题层级 1-6
    
    def __init__(self, element_id: str, text: str, level: int = 1, metadata: Dict[str, Any] = None):
        super().__init__(element_id, ElementType.HEADING, text, metadata or {})
        self.level = level
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d['level'] = self.level
        return d


@dataclass
class Paragraph(Element):
    """段落"""
    def __init__(self, element_id: str, text: str, metadata: Dict[str, Any] = None):
        super().__init__(element_id, ElementType.PARAGRAPH, text, metadata or {})


@dataclass
class ListItem(Element):
    """列表项"""
    list_type: str = "bullet"  # bullet, numbered, lettered
    level: int = 0  # 嵌套层级
    
    def __init__(self, element_id: str, text: str, list_type: str = "bullet", 
                 level: int = 0, metadata: Dict[str, Any] = None):
        super().__init__(element_id, ElementType.LIST_ITEM, text, metadata or {})
        self.list_type = list_type
        self.level = level
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d['list_type'] = self.list_type
        d['level'] = self.level
        return d


@dataclass
class Table(Element):
    """表格"""
    rows: List[List[str]] = field(default_factory=list)
    headers: List[str] = field(default_factory=list)
    
    def __init__(self, element_id: str, text: str, rows: List[List[str]] = None,
                 headers: List[str] = None, metadata: Dict[str, Any] = None):
        super().__init__(element_id, ElementType.TABLE, text, metadata or {})
        self.rows = rows or []
        self.headers = headers or []
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d['rows'] = self.rows
        d['headers'] = self.headers
        return d
    
    def to_markdown(self) -> str:
        """转换为Markdown表格"""
        if not self.rows:
            return self.text
        
        lines = []
        # 表头
        if self.headers:
            lines.append("| " + " | ".join(self.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")
        
        # 数据行
        for row in self.rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        
        return "\n".join(lines)


@dataclass
class CodeBlock(Element):
    """代码块"""
    language: str = ""
    
    def __init__(self, element_id: str, text: str, language: str = "", 
                 metadata: Dict[str, Any] = None):
        super().__init__(element_id, ElementType.CODE_BLOCK, text, metadata or {})
        self.language = language
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d['language'] = self.language
        return d


@dataclass
class SensitiveInfo(Element):
    """敏感信息（脱敏标记）"""
    info_type: str = ""  # phone, email, id_card, url, address
    
    def __init__(self, element_id: str, text: str, info_type: str = "",
                 metadata: Dict[str, Any] = None):
        super().__init__(element_id, ElementType.UNKNOWN, text, metadata or {})
        self.info_type = info_type
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d['info_type'] = self.info_type
        return d


class ElementList:
    """元素列表容器"""
    
    def __init__(self, elements: List[Element] = None):
        self.elements = elements or []
        self._id_counter = 0
    
    def add(self, element: Element):
        """添加元素"""
        if not element.element_id:
            element.element_id = f"elem_{self._id_counter}"
            self._id_counter += 1
        self.elements.append(element)
    
    def filter_by_type(self, element_type: ElementType) -> List[Element]:
        """按类型过滤"""
        return [e for e in self.elements if e.element_type == element_type]
    
    def get_text_elements(self) -> List[Element]:
        """获取所有文本类元素（排除敏感信息）"""
        text_types = {
            ElementType.TITLE, ElementType.HEADING, ElementType.PARAGRAPH,
            ElementType.LIST_ITEM, ElementType.TABLE, ElementType.CODE_BLOCK
        }
        return [e for e in self.elements if e.element_type in text_types]
    
    def to_text(self) -> str:
        """转换为纯文本"""
        return "\n\n".join(e.text for e in self.get_text_elements())
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """转换为字典列表"""
        return [e.to_dict() for e in self.elements]
    
    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict_list(), ensure_ascii=False, indent=indent)
    
    @classmethod
    def from_dict_list(cls, data: List[Dict[str, Any]]) -> 'ElementList':
        """从字典列表创建"""
        elements = []
        for item in data:
            elem_type = ElementType(item['element_type'])
            
            if elem_type == ElementType.HEADING:
                elem = Heading(
                    element_id=item['element_id'],
                    text=item['text'],
                    level=item.get('level', 1),
                    metadata=item.get('metadata', {})
                )
            elif elem_type == ElementType.LIST_ITEM:
                elem = ListItem(
                    element_id=item['element_id'],
                    text=item['text'],
                    list_type=item.get('list_type', 'bullet'),
                    level=item.get('level', 0),
                    metadata=item.get('metadata', {})
                )
            elif elem_type == ElementType.TABLE:
                elem = Table(
                    element_id=item['element_id'],
                    text=item['text'],
                    rows=item.get('rows', []),
                    headers=item.get('headers', []),
                    metadata=item.get('metadata', {})
                )
            elif elem_type == ElementType.CODE_BLOCK:
                elem = CodeBlock(
                    element_id=item['element_id'],
                    text=item['text'],
                    language=item.get('language', ''),
                    metadata=item.get('metadata', {})
                )
            elif elem_type == ElementType.TITLE:
                elem = Title(
                    element_id=item['element_id'],
                    text=item['text'],
                    metadata=item.get('metadata', {})
                )
            else:
                elem = Paragraph(
                    element_id=item['element_id'],
                    text=item['text'],
                    metadata=item.get('metadata', {})
                )
            
            elements.append(elem)
        
        return cls(elements)
    
    def __len__(self) -> int:
        return len(self.elements)
    
    def __iter__(self):
        return iter(self.elements)
    
    def __getitem__(self, idx):
        return self.elements[idx]
