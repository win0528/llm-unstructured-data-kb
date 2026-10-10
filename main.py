"""
命令行入口脚本
用法: python main.py [--chunk_size 800] [--overlap 150] [--model_name BAAI/bge-small-zh]
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

from config import (
    DATA_RAW_DIR, DATA_CLEAN_DIR, DATA_CHUNK_DIR, 
    VECTOR_DB_DIR, OUTPUT_DIR, TEXT_CONFIG, EMBEDDING_CONFIG,
    VECTOR_DB_CONFIG, MOCK_EMBEDDING_DIM, HF_CONFIG, EMBEDDING_ENGINE
)
from utils import (
    DocumentParser, TextCleaner, TextNormalizer, TextChunker,
    get_embedder, VectorStore, QualityStats,
    LanguageDetector, TextQualityScorer
)


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="大模型非结构化数据智能预处理与向量知识库构建工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                                    # 使用默认配置
  python main.py --chunk_size 800 --overlap 150     # 自定义分块参数
  python main.py --model_name BAAI/bge-small-zh     # 指定embedding模型
        """
    )
    
    parser.add_argument(
        '--chunk_size', 
        type=int, 
        default=TEXT_CONFIG['max_chunk_length'],
        help=f'分块大小（字符数），默认: {TEXT_CONFIG["max_chunk_length"]}'
    )
    
    parser.add_argument(
        '--overlap', 
        type=int, 
        default=TEXT_CONFIG['overlap'],
        help=f'分块重叠大小，默认: {TEXT_CONFIG["overlap"]}'
    )
    
    parser.add_argument(
        '--model_name', 
        type=str, 
        default=EMBEDDING_CONFIG['model_name'],
        help=f'Embedding模型名称，默认: {EMBEDDING_CONFIG["model_name"]}'
    )
    
    parser.add_argument(
        '--mock', 
        action='store_true',
        help='使用模拟向量化器（等价于 --engine mock）'
    )

    parser.add_argument(
        '--engine',
        type=str,
        default=EMBEDDING_ENGINE,
        choices=['auto', 'onnx', 'st', 'mock'],
        help=f"向量器引擎，默认: {EMBEDDING_ENGINE}"
             "（auto 优先 ONNX 轻量版，其次 sentence-transformers；mock 为离线哈希向量器）"
    )

    parser.add_argument(
        '--backend',
        type=str,
        default=VECTOR_DB_CONFIG['backend'],
        choices=['auto', 'faiss', 'numpy', 'chroma'],
        help=f"向量库后端，默认: {VECTOR_DB_CONFIG['backend']}（auto 依次尝试 faiss/numpy/chroma）"
    )

    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='先清空已有集合再入库（默认是增量 upsert，重复运行不会产生重复记录）'
    )

    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='只走解析/清洗/分块，不写向量库'
    )
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not DATA_RAW_DIR.exists():
        print(f"错误: 输入目录不存在: {DATA_RAW_DIR}")
        print(f"请先创建目录并将待处理文件放入其中")
        sys.exit(1)
    
    # 统计输入文件
    input_files = list(DATA_RAW_DIR.glob('*'))
    supported_exts = {'.pdf', '.docx', '.doc', '.txt', '.md', '.html', '.htm', '.csv', '.json'}
    valid_files = [f for f in input_files if f.suffix.lower() in supported_exts]
    
    if not valid_files:
        print(f"错误: 输入目录中没有支持的文件: {DATA_RAW_DIR}")
        print(f"支持的格式: {', '.join(sorted(supported_exts))}")
        sys.exit(1)
    
    print("=" * 60)
    print("大模型非结构化数据智能预处理与向量知识库构建")
    print("=" * 60)
    print(f"\n输入目录: {DATA_RAW_DIR}")
    print(f"找到 {len(valid_files)} 个文件待处理")
    print(f"\n配置参数:")
    print(f"  分块大小: {args.chunk_size} 字符")
    print(f"  分块重叠: {args.overlap} 字符")
    print(f"  Embedding模型: {args.model_name}")
    print(f"  向量器引擎: {args.engine}{'（--mock 生效，实际用 mock）' if args.mock else ''}")
    print()
    
    # 初始化各模块
    print("初始化处理模块...")
    doc_parser = DocumentParser()
    text_cleaner = TextCleaner()
    text_normalizer = TextNormalizer()
    text_chunker = TextChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        min_chunk_length=TEXT_CONFIG['min_chunk_length']
    )
    language_detector = LanguageDetector()
    quality_scorer = TextQualityScorer()
    
    # 创建并提前预检向量器：否则每个文件都要跑到第 5 步才失败，白跑一遍还生成误导报告
    try:
        text_embedder = get_embedder(
            use_mock=args.mock,
            model_name=args.model_name,
            dimension=MOCK_EMBEDDING_DIM,
            engine=args.engine,
        )
        text_embedder.load_model()
    except (RuntimeError, ImportError, ValueError) as e:
        print(f"\n错误: {e}")
        print("\n排查方向:")
        print(f"  1) 轻量路线（推荐，依赖约 150MB，不需要 torch）:")
        print(f"       python download_model.py && python main.py --engine onnx")
        print(f"  2) 完整路线（sentence-transformers，含 torch 约 2.5GB）:")
        print(f"       \"{sys.executable}\" -m pip install sentence-transformers && python main.py --engine st")
        print(f"  3) 先跑通链路（离线哈希向量器，检索时也要指定 --engine mock）:")
        print(f"       python main.py --engine mock")
        print(f"\n  注: 模型下载已自动走镜像 {HF_CONFIG.get('endpoint')}，若仍失败请检查网络或代理")
        sys.exit(1)

    print(f"向量器: {text_embedder.describe()}")
    print()

    vector_store = VectorStore(
        collection_name=VECTOR_DB_CONFIG['collection_name'],
        persist_directory=str(VECTOR_DB_DIR),
        backend=args.backend,
        embedder_signature=text_embedder.signature,
        normalize=VECTOR_DB_CONFIG['normalize'],
    )

    print(f"向量库后端: {vector_store.backend}")
    if args.dry_run:
        print("dry-run: 不会写入向量库")
    elif args.rebuild:
        vector_store.reset()
        print("已清空已有集合，本次为全量重建")
    else:
        print(f"当前知识库: {vector_store.describe()}（增量 upsert）")
    for warning in vector_store.warnings:
        print(f"  警告: {warning}")
    print()
    quality_stats = QualityStats()
    
    # 处理统计
    total_files = len(valid_files)
    processed_files = 0
    failed_files = 0
    all_chunks = []
    format_counts = {}
    
    original_total_length = 0
    cleaned_total_length = 0
    
    # 收集每个文件的清洗和脱敏统计
    all_clean_stats = []
    all_desensitize_stats = []

    # 本次真正写入向量库的条数（区别于库里原有总量）
    vectors_added_total = 0
    
    # 处理每个文件
    print("\n开始处理文件...")
    print("-" * 60)
    
    for idx, file_path in enumerate(valid_files, 1):
        print(f"\n[{idx}/{total_files}] 处理: {file_path.name}")
        
        try:
            # 1. 解析文档
            print(f"  1/5 解析文档...")
            parse_result = doc_parser.parse_file(str(file_path))
            text = parse_result['text']
            metadata = parse_result['metadata']
            file_format = parse_result['format']
            
            if not text.strip():
                raise ValueError("文档内容为空")
            
            original_length = len(text)
            original_total_length += original_length
            
            # 统计格式
            format_counts[file_format] = format_counts.get(file_format, 0) + 1
            
            # 2. 文本清洗
            print(f"  2/5 文本清洗...")
            clean_result = text_cleaner.clean(text)
            text = clean_result['text']
            clean_stats = clean_result['stats']
            all_clean_stats.append(clean_stats)
            
            # 3. 标准化和脱敏
            print(f"  3/5 标准化和脱敏...")
            normalize_result = text_normalizer.process(text)
            text = normalize_result['text']
            if 'desensitize' in normalize_result.get('stats', {}):
                all_desensitize_stats.append(normalize_result['stats']['desensitize'])
            
            cleaned_length = len(text)
            cleaned_total_length += cleaned_length
            
            # 4. 智能分块
            print(f"  4/5 智能分块...")
            chunks = text_chunker.chunk(text, metadata)
            
            if not chunks:
                raise ValueError("分块后无有效内容")
            
            # 5. 向量化和入库
            if args.dry_run:
                print(f"  5/5 跳过向量化和入库（dry-run）")
            else:
                print(f"  5/5 向量化和入库...")
            chunk_texts = [chunk['text'] for chunk in chunks]
            
            chunk_metadatas = []
            for i, chunk in enumerate(chunks):
                chunk_meta = {
                    'source': file_path.name,
                    'chunk_id': i,
                    'format': file_format,
                    'original_length': original_length,
                    'cleaned_length': cleaned_length
                }
                chunk_metadatas.append(chunk_meta)
            
            chunk_ids = [f"{file_path.stem}_chunk_{i}" for i in range(len(chunks))]

            if not args.dry_run:
                embedding_results = text_embedder.embed_texts(chunk_texts)
                embeddings = [result['embedding'] for result in embedding_results]

                add_result = vector_store.add_documents(
                    texts=chunk_texts,
                    embeddings=embeddings,
                    metadatas=chunk_metadatas,
                    ids=chunk_ids
                )

                if 'error' in add_result:
                    raise RuntimeError(f"入库失败: {add_result['error']}")
            else:
                add_result = {'added': 0, 'replaced': 0}
            
            # 保存清洗后的文本
            clean_output_path = DATA_CLEAN_DIR / f"{file_path.stem}_cleaned.txt"
            with open(clean_output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # 保存分块结果
            chunk_output_path = DATA_CHUNK_DIR / f"{file_path.stem}_chunks.json"
            import json
            with open(chunk_output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            
            # 更新统计
            all_chunks.extend(chunks)
            vectors_added_total += add_result.get('added', 0)
            processed_files += 1
            
            replaced_note = f", 覆盖旧块{add_result.get('replaced', 0)}" if add_result.get('replaced') else ""
            print(f"  ✓ 成功: 原始{original_length}字符 → 清洗后{cleaned_length}字符 → {len(chunks)}个块{replaced_note}")
            
        except Exception as e:
            failed_files += 1
            print(f"  ✗ 失败: {str(e)}")
            continue
    
    # 更新质量统计
    quality_stats.add_file_stats(total_files, format_counts)
    quality_stats.stats['files']['processed'] = processed_files
    quality_stats.stats['files']['failed'] = failed_files
    
    quality_stats.add_text_stats(original_total_length, cleaned_total_length)
    
    # 汇总清洗统计
    quality_stats.add_cleaning_stats({
        'removed_special_chars': sum(clean_stats.get('removed_special_chars', 0) for clean_stats in all_clean_stats),
        'removed_empty_lines': sum(clean_stats.get('removed_empty_lines', 0) for clean_stats in all_clean_stats),
        'removed_page_numbers': sum(clean_stats.get('removed_page_numbers', 0) for clean_stats in all_clean_stats),
        'removed_short_paragraphs': sum(clean_stats.get('removed_short_paragraphs', 0) for clean_stats in all_clean_stats),
        'desensitized_items': sum(desensitize_stats.get('total_desensitized', 0) for desensitize_stats in all_desensitize_stats)
    })
    
    quality_stats.add_chunk_stats(all_chunks)
    
    # 获取向量库统计
    store_stats = vector_store.get_stats()
    # 报告的向量统计反映"本次入库"，库内总量在下面的总结里单独打印，两者不要混淆
    quality_stats.add_vector_stats(
        vectors_added_total,
        text_embedder.embedding_dim or store_stats.get('dimension') or 0,
        (processed_files / total_files) if total_files else 0.0
    )
    
    # 打印报告
    print("\n" + "=" * 60)
    quality_stats.print_report()
    
    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUT_DIR / f"process_report_{timestamp}.json"
    quality_stats.save_report(str(report_path))
    
    # 最终总结
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"\n输出目录:")
    print(f"  清洗后文本: {DATA_CLEAN_DIR}")
    print(f"  分块结果: {DATA_CHUNK_DIR}")
    print(f"  向量数据库: {VECTOR_DB_DIR}")
    print(f"  处理报告: {report_path}")
    print(f"\n处理结果:")
    print(f"  成功: {processed_files}/{total_files}")
    print(f"  失败: {failed_files}/{total_files}")
    print(f"  总分块数: {len(all_chunks)}")
    print(f"  向量库后端: {store_stats['backend']}")
    print(f"  向量器: {text_embedder.signature}")
    print(f"  向量库文档数: {store_stats['total_documents']}")
    if store_stats.get('sources'):
        print(f"  来源文件: {len(store_stats['sources'])} 个")
    engine_note = " --engine mock" if (args.mock or args.engine == "mock") else ""
    print(f"\n检索方式:")
    print(f"  python query.py --query \"你的问题\"{engine_note}")
    print("=" * 60)


if __name__ == '__main__':
    main()
