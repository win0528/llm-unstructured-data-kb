"""
文本清洗和脱敏模块
去除乱码、特殊符号、多余空行，统一编码，脱敏隐私信息
"""
import re
from typing import Dict, Any, Tuple


class TextCleaner:
    """文本清洗器"""
    
    def __init__(self, min_paragraph_length: int = 20, remove_duplicates: bool = True):
        self.min_paragraph_length = min_paragraph_length
        self.remove_duplicates = remove_duplicates
    
    def clean(self, text: str) -> Dict[str, Any]:
        """执行完整清洗流程"""
        if not text:
            return {'text': '', 'stats': {'original_length': 0, 'cleaned_length': 0}}
        
        original_length = len(text)
        stats = {
            'original_length': original_length,
            'removed_special_chars': 0,
            'removed_empty_lines': 0,
            'removed_short_paragraphs': 0,
            'removed_page_numbers': 0,
            'removed_duplicates': 0,
        }
        
        # 1. 去除不可见字符
        text, count = self._remove_invisible_chars(text)
        stats['removed_special_chars'] += count
        
        # 2. 去除特殊符号
        text, count = self._remove_special_symbols(text)
        stats['removed_special_chars'] += count
        
        # 3. 去除页码
        text, count = self._remove_page_numbers(text)
        stats['removed_page_numbers'] = count
        
        # 4. 规范化空白
        text, count = self._normalize_whitespace(text)
        stats['removed_empty_lines'] = count
        
        # 5. 过滤短段落
        text, count = self._filter_short_paragraphs(text)
        stats['removed_short_paragraphs'] = count
        
        # 6. 去除重复段落
        if self.remove_duplicates:
            text, count = self._remove_duplicates(text)
            stats['removed_duplicates'] = count
        
        stats['cleaned_length'] = len(text)
        
        return {'text': text.strip(), 'stats': stats}
    
    def _remove_invisible_chars(self, text: str) -> Tuple[str, int]:
        """去除不可见字符"""
        pattern = r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]'
        matches = re.findall(pattern, text)
        count = len(matches)
        cleaned = re.sub(pattern, '', text)
        return cleaned, count
    
    def _remove_special_symbols(self, text: str) -> Tuple[str, int]:
        """去除特殊符号"""
        pattern = r'[^\u4e00-\u9fff\u3000-\u303fa-zA-Z0-9\s\.,;:!?，。；：！？、""''（）《》【】\-\(\)\[\]\{\}/\\@#$%^&*+=<>~`]'
        matches = re.findall(pattern, text)
        count = len(matches)
        cleaned = re.sub(pattern, '', text)
        return cleaned, count
    
    def _remove_page_numbers(self, text: str) -> Tuple[str, int]:
        """去除页码"""
        count = 0
        patterns = [
            r'^\s*-\s*\d+\s*-\s*$',
            r'^\s*第\s*\d+\s*页\s*$',
            r'^\s*Page\s+\d+\s*$',
            r'^\s*\d+\s*/\s*\d+\s*$',
        ]
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            is_page_number = False
            for pattern in patterns:
                if re.match(pattern, line.strip()):
                    is_page_number = True
                    count += 1
                    break
            
            if not is_page_number:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines), count
    
    def _normalize_whitespace(self, text: str) -> Tuple[str, int]:
        """规范化空白字符"""
        count = 0
        lines = [line.strip() for line in text.split('\n')]
        
        cleaned_lines = []
        prev_empty = False
        
        for line in lines:
            if not line:
                if not prev_empty:
                    cleaned_lines.append('')
                    prev_empty = True
                    count += 1
                else:
                    count += 1
            else:
                cleaned_lines.append(line)
                prev_empty = False
        
        return '\n'.join(cleaned_lines), count
    
    def _remove_duplicates(self, text: str) -> Tuple[str, int]:
        """去除重复段落"""
        lines = text.split('\n')
        seen = set()
        cleaned_lines = []
        removed_count = 0
        
        for line in lines:
            line_stripped = line.strip()
            
            # 空行直接保留
            if not line_stripped:
                cleaned_lines.append(line)
                continue
            
            # 标题行保留（即使重复）
            if line_stripped.startswith('#') or re.match(r'^\d+\.\s', line_stripped):
                cleaned_lines.append(line)
                continue
            
            # 检查是否重复
            if line_stripped in seen:
                removed_count += 1
                continue
            
            seen.add(line_stripped)
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines), removed_count
    
    def _filter_short_paragraphs(self, text: str) -> Tuple[str, int]:
        """过滤短段落"""
        count = 0
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if not line.strip():
                cleaned_lines.append(line)
                continue
            
            # 保留标题行
            if line.strip().startswith('#') or re.match(r'^\d+\.\s', line.strip()):
                cleaned_lines.append(line)
                continue
            
            if len(line.strip()) < self.min_paragraph_length:
                count += 1
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines), count


class TextNormalizer:
    """文本标准化和脱敏"""
    
    def __init__(self):
        self.punctuation_map = {
            '，': ',', '。': '.', '；': ';', '：': ':',
            '！': '!', '？': '?', '（': '(', '）': ')',
            '【': '[', '】': ']', '《': '<', '》': '>',
        }
    
    def normalize(self, text: str) -> Dict[str, Any]:
        """标准化文本"""
        if not text:
            return {'text': '', 'stats': {'changes': 0}}
        
        stats = {'original_length': len(text), 'punctuation_changes': 0}
        
        # 统一标点
        text, count = self._normalize_punctuation(text)
        stats['punctuation_changes'] = count
        stats['normalized_length'] = len(text)
        
        return {'text': text, 'stats': stats}
    
    def _normalize_punctuation(self, text: str) -> Tuple[str, int]:
        """统一标点符号"""
        count = 0
        for cn_punc, en_punc in self.punctuation_map.items():
            occurrences = text.count(cn_punc)
            if occurrences > 0:
                text = text.replace(cn_punc, en_punc)
                count += occurrences
        return text, count
    
    def desensitize(self, text: str) -> Dict[str, Any]:
        """脱敏隐私信息"""
        if not text:
            return {'text': '', 'stats': {'total_desensitized': 0}}
        
        stats = {
            'phone_numbers': 0, 'id_cards': 0,
            'emails': 0, 'urls': 0, 'total_desensitized': 0
        }
        
        # 先脱敏身份证（避免被手机号误匹配）
        text, count = self._desensitize_id_card(text)
        stats['id_cards'] = count
        stats['total_desensitized'] += count
        
        # 再脱敏手机号
        text, count = self._desensitize_phone(text)
        stats['phone_numbers'] = count
        stats['total_desensitized'] += count
        
        # 脱敏邮箱
        text, count = self._desensitize_email(text)
        stats['emails'] = count
        stats['total_desensitized'] += count
        
        # 脱敏网址
        text, count = self._desensitize_url(text)
        stats['urls'] = count
        stats['total_desensitized'] += count
        
        return {'text': text, 'stats': stats}
    
    def _desensitize_phone(self, text: str) -> Tuple[str, int]:
        """脱敏手机号"""
        pattern = r'1[3-9]\d{9}'
        matches = re.findall(pattern, text)
        count = len(matches)
        text = re.sub(pattern, '[PHONE]', text)
        return text, count
    
    def _desensitize_id_card(self, text: str) -> Tuple[str, int]:
        """脱敏身份证号"""
        pattern = r'(?<!\d)\d{17}[\dXx](?!\d)'
        matches = re.findall(pattern, text)
        count = len(matches)
        text = re.sub(pattern, '[ID_CARD]', text)
        return text, count
    
    def _desensitize_email(self, text: str) -> Tuple[str, int]:
        """脱敏邮箱"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(pattern, text)
        count = len(matches)
        text = re.sub(pattern, '[EMAIL]', text)
        return text, count
    
    def _desensitize_url(self, text: str) -> Tuple[str, int]:
        """脱敏网址"""
        pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        matches = re.findall(pattern, text)
        count = len(matches)
        text = re.sub(pattern, '[URL]', text)
        return text, count
    
    def process(self, text: str) -> Dict[str, Any]:
        """执行完整标准化和脱敏流程"""
        if not text:
            return {'text': '', 'stats': {}}
        
        normalize_result = self.normalize(text)
        text = normalize_result['text']
        
        desensitize_result = self.desensitize(text)
        text = desensitize_result['text']
        
        stats = {
            'normalize': normalize_result['stats'],
            'desensitize': desensitize_result['stats']
        }
        
        return {'text': text, 'stats': stats}


class LanguageDetector:
    """语言检测器"""
    
    def detect(self, text: str) -> Dict[str, Any]:
        """检测文本语言"""
        if not text:
            return {'language': 'unknown', 'confidence': 0.0}
        
        # 统计字符类型
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = chinese_chars + english_chars
        
        if total_chars == 0:
            return {'language': 'unknown', 'confidence': 0.0}
        
        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_chars / total_chars
        
        if chinese_ratio > 0.5:
            return {'language': 'zh', 'confidence': chinese_ratio, 'details': {'chinese': chinese_chars, 'english': english_chars}}
        elif english_ratio > 0.5:
            return {'language': 'en', 'confidence': english_ratio, 'details': {'chinese': chinese_chars, 'english': english_chars}}
        else:
            return {'language': 'mixed', 'confidence': max(chinese_ratio, english_ratio), 'details': {'chinese': chinese_chars, 'english': english_chars}}


class TextQualityScorer:
    """文本质量评分器"""
    
    def score(self, text: str) -> Dict[str, Any]:
        """评估文本质量"""
        if not text:
            return {'score': 0.0, 'issues': ['empty_text']}
        
        issues = []
        score = 100.0
        
        # 1. 长度检查
        if len(text) < 50:
            score -= 30
            issues.append('too_short')
        
        # 2. 特殊字符比例
        special_chars = len(re.findall(r'[^\w\s\u4e00-\u9fff]', text))
        special_ratio = special_chars / len(text) if text else 0
        if special_ratio > 0.3:
            score -= 20
            issues.append('too_many_special_chars')
        
        # 3. 数字比例
        digits = len(re.findall(r'\d', text))
        digit_ratio = digits / len(text) if text else 0
        if digit_ratio > 0.5:
            score -= 15
            issues.append('too_many_digits')
        
        # 4. 重复内容检测
        lines = text.split('\n')
        unique_lines = set(lines)
        if len(lines) > 0 and len(unique_lines) / len(lines) < 0.5:
            score -= 20
            issues.append('high_repetition')
        
        # 5. 句子完整性（检查是否有完整的句子结束符）
        sentence_endings = len(re.findall(r'[.。!！?？]', text))
        if sentence_endings == 0 and len(text) > 100:
            score -= 10
            issues.append('no_sentence_endings')
        
        # 6. 可读性（平均句长）
        sentences = re.split(r'[.。!！?？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_sentence_length = len(text) / len(sentences)
            if avg_sentence_length > 200:
                score -= 10
                issues.append('sentences_too_long')
            elif avg_sentence_length < 5:
                score -= 10
                issues.append('sentences_too_short')
        
        return {
            'score': max(0, score),
            'issues': issues,
            'metrics': {
                'length': len(text),
                'special_char_ratio': special_ratio,
                'digit_ratio': digit_ratio,
                'unique_line_ratio': len(unique_lines) / len(lines) if lines else 0,
                'sentence_count': len(sentences) if sentences else 0,
                'avg_sentence_length': avg_sentence_length if sentences else 0
            }
        }
