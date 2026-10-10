"""
Streamlit Web界面
用法: streamlit run app.py
"""
import streamlit as st
from pathlib import Path
import sys
import json
import tempfile
from datetime import datetime
import io
import zipfile

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_RAW_DIR, DATA_CLEAN_DIR, DATA_CHUNK_DIR,
    VECTOR_DB_DIR, OUTPUT_DIR, TEXT_CONFIG, EMBEDDING_CONFIG,
    VECTOR_DB_CONFIG, MOCK_EMBEDDING_DIM
)
from utils import (
    DocumentParser, TextCleaner, TextNormalizer, TextChunker,
    get_embedder, VectorStore, QualityStats
)


def init_session_state():
    """初始化会话状态"""
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'stats' not in st.session_state:
        st.session_state.stats = None
    if 'results' not in st.session_state:
        st.session_state.results = None


def page_upload():
    """文件上传页面"""
    st.header("文件上传")
    st.markdown("支持格式: PDF, Word, TXT, Markdown, HTML, CSV, JSON")

    uploaded_files = st.file_uploader(
        "选择文件",
        type=['pdf', 'docx', 'txt', 'md', 'html', 'htm', 'csv', 'json'],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"已选择 {len(uploaded_files)} 个文件:")
        for f in uploaded_files:
            st.write(f"  - {f.name} ({f.size / 1024:.1f} KB)")

        if st.button("上传文件", type="primary"):
            save_count = 0
            for f in uploaded_files:
                save_path = DATA_RAW_DIR / f.name
                with open(save_path, 'wb') as out:
                    out.write(f.read())
                save_count += 1

            st.success(f"成功保存 {save_count} 个文件到 data_raw 目录")
            st.info("请切换到「数据处理」页面进行一键处理")


def page_process():
    """数据处理页面"""
    st.header("数据处理")

    # 检查输入目录
    files = list(DATA_RAW_DIR.glob('*'))
    supported_exts = {'.pdf', '.docx', '.doc', '.txt', '.md', '.html', '.htm', '.csv', '.json'}
    valid_files = [f for f in files if f.suffix.lower() in supported_exts]

    if not valid_files:
        st.warning("data_raw 目录为空，请先上传文件")
        return

    st.write(f"data_raw 目录中有 {len(valid_files)} 个待处理文件:")
    for f in valid_files:
        st.write(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    # 配置选项
    st.subheader("处理配置")

    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input(
            "分块大小（字符数）",
            value=TEXT_CONFIG['max_chunk_length'],
            min_value=128,
            max_value=4096
        )
    with col2:
        overlap = st.number_input(
            "分块重叠（字符数）",
            value=TEXT_CONFIG['overlap'],
            min_value=0,
            max_value=1024
        )

    model_name = st.selectbox(
        "Embedding模型",
        ["BAAI/bge-small-zh-v1.5", "BAAI/bge-base-zh-v1.5", "BAAI/bge-large-zh-v1.5"]
    )

    backend = st.selectbox(
        "向量库后端",
        ["auto", "faiss", "numpy", "chroma"],
        index=0,
        help="auto 会按 faiss -> numpy -> chroma 的顺序自动选择可用后端；"
             "numpy 为零依赖的纯 Python 实现，faiss 更快",
    )

    engine = st.selectbox(
        "向量器引擎",
        ["auto", "onnx", "st", "mock"],
        index=0,
        help="auto 优先 ONNX 轻量版（模型约 90MB，不需要 torch）；"
             "st 需要 sentence-transformers（含 torch 约 2.5GB）；"
             "mock 为离线哈希向量器，零依赖但只做字面匹配。建库与检索必须用同一个引擎。",
    )

    col_rebuild, col_append = st.columns(2)
    with col_rebuild:
        rebuild = st.checkbox("重建索引（先清空已有集合）", value=False,
                              help="不勾选时为增量 upsert，重复运行不会产生重复记录")
    with col_append:
        dry_run = st.checkbox("干跑（只清洗分块，不写向量库）", value=False)

    if st.button("开始处理", type="primary"):
        with st.spinner("正在处理文件，请稍候..."):
            # 初始化模块
            parser = DocumentParser()
            cleaner = TextCleaner()
            normalizer = TextNormalizer()
            chunker = TextChunker(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                min_chunk_length=TEXT_CONFIG['min_chunk_length']
            )

            # 先建向量器并加载，失败就在动手处理文件前停下（否则会白跑一遍还生成误导报告）
            try:
                embedder = get_embedder(
                    model_name=model_name,
                    dimension=MOCK_EMBEDDING_DIM,
                    engine=engine,
                )
                embedder.load_model()
            except Exception as e:
                st.error(f"向量器初始化失败: {e}")
                st.info(
                    "排查方向：\n"
                    "1) 轻量路线：在终端执行 `python download_model.py` 下载 ONNX 模型，"
                    "并确认已装 `pip install onnxruntime tokenizers`\n"
                    "2) 完整路线：`pip install sentence-transformers`（含 torch 约 2.5GB）\n"
                    "3) 零依赖：把「向量器引擎」改成 `mock`（离线哈希向量器，检索时也要选 mock）"
                )
                return

            store = VectorStore(
                collection_name=VECTOR_DB_CONFIG['collection_name'],
                persist_directory=str(VECTOR_DB_DIR),
                backend=backend,
                embedder_signature=embedder.signature,
                normalize=VECTOR_DB_CONFIG['normalize'],
            )

            if rebuild:
                store.reset()

            stats = QualityStats()
            format_counts = {}
            all_chunks = []
            original_total = 0
            cleaned_total = 0
            results = []

            for idx, file_path in enumerate(valid_files, 1):
                try:
                    # 1. 解析
                    parse_result = parser.parse_file(str(file_path))
                    text = parse_result['text']
                    metadata = parse_result['metadata']
                    file_format = parse_result['format']

                    if not text.strip():
                        results.append({
                            'file': file_path.name,
                            'success': False,
                            'error': '文档内容为空'
                        })
                        continue

                    original_length = len(text)
                    original_total += original_length
                    format_counts[file_format] = format_counts.get(file_format, 0) + 1

                    # 2. 清洗
                    clean_result = cleaner.clean(text)
                    text = clean_result['text']

                    # 3. 标准化
                    normalize_result = normalizer.process(text)
                    text = normalize_result['text']
                    cleaned_length = len(text)
                    cleaned_total += cleaned_length

                    # 4. 分块
                    chunks = chunker.chunk(text, metadata)

                    if not chunks:
                        results.append({
                            'file': file_path.name,
                            'success': False,
                            'error': '分块后无有效内容'
                        })
                        continue

                    # 5. 向量化 + 入库
                    chunk_texts = [c['text'] for c in chunks]

                    chunk_metadatas = [{
                        'source': file_path.name,
                        'chunk_id': i,
                        'format': file_format
                    } for i in range(len(chunks))]

                    chunk_ids = [f"{file_path.stem}_chunk_{i}" for i in range(len(chunks))]

                    if dry_run:
                        add_result = {'added': 0, 'replaced': 0}
                    else:
                        embeddings_result = embedder.embed_texts(chunk_texts)
                        embeddings = [r['embedding'] for r in embeddings_result]

                        add_result = store.add_documents(
                            texts=chunk_texts,
                            embeddings=embeddings,
                            metadatas=chunk_metadatas,
                            ids=chunk_ids
                        )

                        if add_result.get('error'):
                            raise RuntimeError(f"入库失败: {add_result['error']}")

                    # 保存清洗后文本
                    clean_path = DATA_CLEAN_DIR / f"{file_path.stem}_cleaned.txt"
                    with open(clean_path, 'w', encoding='utf-8') as f:
                        f.write(text)

                    # 保存分块结果
                    chunk_path = DATA_CHUNK_DIR / f"{file_path.stem}_chunks.json"
                    with open(chunk_path, 'w', encoding='utf-8') as f:
                        json.dump(chunks, f, ensure_ascii=False, indent=2)

                    all_chunks.extend(chunks)
                    results.append({
                        'file': file_path.name,
                        'success': True,
                        'original_length': original_length,
                        'cleaned_length': cleaned_length,
                        'chunks': len(chunks),
                        'vectors_added': add_result.get('added', 0),
                        'vectors_replaced': add_result.get('replaced', 0)
                    })

                except Exception as e:
                    results.append({
                        'file': file_path.name,
                        'success': False,
                        'error': str(e)
                    })

            # 更新统计
            success_results = [r for r in results if r['success']]
            stats.add_file_stats(len(valid_files), format_counts)
            stats.stats['files']['processed'] = len(success_results)
            stats.stats['files']['failed'] = len(results) - len(success_results)
            stats.add_text_stats(original_total, cleaned_total)
            stats.add_chunk_stats(all_chunks)

            store_stats = store.get_stats()
            stats.add_vector_stats(
                sum(r.get('vectors_added', 0) for r in success_results),
                embedder.embedding_dim or store_stats.get('dimension') or 0,
                (len(success_results) / len(valid_files)) if valid_files else 0.0
            )

            # 保存报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = OUTPUT_DIR / f"process_report_{timestamp}.json"
            stats.save_report(str(report_path))

            st.session_state.processed = True
            st.session_state.stats = stats.get_stats()
            st.session_state.results = results

        st.success("处理完成!")
        if dry_run:
            st.info("干跑模式：本次没有写入向量库。")
        else:
            st.caption(
                f"向量库后端: **{store_stats['backend']}** ｜ "
                f"库内共 **{store_stats['total_documents']}** 条向量"
                f"（{len(store_stats['sources'])} 个来源文件）"
            )

        # 显示结果
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总文件数", len(valid_files))
        with col2:
            st.metric("成功处理", len(success_results))
        with col3:
            st.metric("处理失败", len(results) - len(success_results))

        st.subheader("详细结果")
        for r in results:
            if r['success']:
                with st.expander(f"✅ {r['file']}"):
                    st.write(f"原始长度: {r['original_length']:,} 字符")
                    st.write(f"清洗后长度: {r['cleaned_length']:,} 字符")
                    st.write(f"分块数: {r['chunks']}")
                    if r.get('vectors_replaced'):
                        st.write(f"覆盖同名旧块: {r['vectors_replaced']}")
                    st.write(f"入库向量数: {r['vectors_added']}")
            else:
                with st.expander(f"❌ {r['file']}"):
                    st.error(r['error'])

        # 下载区域
        if success_results:
            st.subheader("下载处理结果")

            col1, col2, col3 = st.columns(3)

            with col1:
                # 下载清洗后文本
                clean_files = list(DATA_CLEAN_DIR.glob('*_cleaned.txt'))
                if clean_files:
                    # 创建ZIP包
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for f in clean_files:
                            zip_file.write(f, f.name)
                    zip_buffer.seek(0)

                    st.download_button(
                        label="下载清洗后文本 (ZIP)",
                        data=zip_buffer,
                        file_name="cleaned_texts.zip",
                        mime="application/zip"
                    )

            with col2:
                # 下载分块结果
                chunk_files = list(DATA_CHUNK_DIR.glob('*_chunks.json'))
                if chunk_files:
                    # 合并所有分块
                    all_chunks_data = []
                    for f in chunk_files:
                        with open(f, 'r', encoding='utf-8') as file:
                            chunks = json.load(file)
                            all_chunks_data.extend(chunks)

                    chunks_json = json.dumps(all_chunks_data, ensure_ascii=False, indent=2)

                    st.download_button(
                        label="下载分块结果 (JSON)",
                        data=chunks_json,
                        file_name="all_chunks.json",
                        mime="application/json"
                    )

            with col3:
                # 下载报告
                report_files = sorted(OUTPUT_DIR.glob('process_report_*.json'), reverse=True)
                if report_files:
                    with open(report_files[0], 'r', encoding='utf-8') as f:
                        report_content = f.read()

                    st.download_button(
                        label="下载报告 (JSON)",
                        data=report_content,
                        file_name="process_report.json",
                        mime="application/json"
                    )


def page_preview():
    """清洗效果预览页面"""
    st.header("清洗效果预览")

    files = list(DATA_RAW_DIR.glob('*'))
    supported_exts = {'.pdf', '.docx', '.doc', '.txt', '.md', '.html', '.htm', '.csv', '.json'}
    valid_files = [f for f in files if f.suffix.lower() in supported_exts]

    if not valid_files:
        st.warning("请先上传文件")
        return

    selected_file = st.selectbox("选择文件预览", [f.name for f in valid_files])

    if selected_file:
        file_path = DATA_RAW_DIR / selected_file

        parser = DocumentParser()
        cleaner = TextCleaner()
        normalizer = TextNormalizer()

        try:
            with st.spinner("处理中..."):
                parse_result = parser.parse_file(str(file_path))
                original_text = parse_result['text']

                clean_result = cleaner.clean(original_text)
                cleaned_text = clean_result['text']

                normalize_result = normalizer.process(cleaned_text)
                final_text = normalize_result['text']

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("原始文本")
                st.text_area("原始文本内容", original_text[:2000], height=300,
                             label_visibility="hidden")
                st.write(f"长度: {len(original_text)} 字符")

            with col2:
                st.subheader("处理后文本")
                st.text_area("处理后文本内容", final_text[:2000], height=300,
                             label_visibility="hidden")
                st.write(f"长度: {len(final_text)} 字符")

            st.subheader("清洗统计")
            stats = clean_result['stats']

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("去除特殊字符", stats.get('removed_special_chars', 0))
            with col2:
                st.metric("去除空行", stats.get('removed_empty_lines', 0))
            with col3:
                st.metric("去除页码", stats.get('removed_page_numbers', 0))
            with col4:
                if stats.get('original_length', 0) > 0:
                    ratio = (stats['original_length'] - stats['cleaned_length']) / stats['original_length']
                    st.metric("压缩比例", f"{ratio:.1%}")

            norm_stats = normalize_result['stats']
            if 'desensitize' in norm_stats:
                st.subheader("脱敏统计")
                d_stats = norm_stats['desensitize']

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("手机号", d_stats.get('phone_numbers', 0))
                with col2:
                    st.metric("身份证", d_stats.get('id_cards', 0))
                with col3:
                    st.metric("邮箱", d_stats.get('emails', 0))
                with col4:
                    st.metric("网址", d_stats.get('urls', 0))

        except Exception as e:
            st.error(f"处理失败: {str(e)}")


def open_store(embedder=None, backend: str = "auto"):
    """打开知识库。embedder 仅用于提供指纹，可为 None。"""
    return VectorStore(
        collection_name=VECTOR_DB_CONFIG['collection_name'],
        persist_directory=str(VECTOR_DB_DIR),
        backend=backend,
        embedder_signature=getattr(embedder, 'signature', None),
        normalize=VECTOR_DB_CONFIG['normalize'],
    )


def page_search():
    """知识库检索页面"""
    st.header("知识库检索")

    col_left, col_right = st.columns([3, 1])
    with col_left:
        engine = st.selectbox(
            "向量器引擎（必须与建库时一致）",
            ["auto", "onnx", "st", "mock"],
            index=0,
            help="用错引擎会导致向量空间不一致、检索结果无意义（下面会给出告警）。"
                 "auto 会在 ONNX 就绪时优先用 ONNX 轻量版。",
        )
    with col_right:
        backend = st.selectbox("向量库后端", ["auto", "faiss", "numpy", "chroma"])

    try:
        embedder = get_embedder(
            model_name=EMBEDDING_CONFIG['model_name'],
            dimension=MOCK_EMBEDDING_DIM,
            engine=engine,
        )
    except Exception as e:
        st.error(f"向量器初始化失败: {e}")
        st.info("请检查 onnxruntime/tokenizers 是否已装、ONNX 模型是否已下载"
                "（`python download_model.py`），或把引擎改成 mock。")
        return

    try:
        store = open_store(embedder, backend)
    except Exception as e:
        st.error(f"知识库打开失败: {e}")
        return

    stats = store.get_stats()

    # ---- 库状态 ----
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("向量条数", stats['total_documents'])
    with c2:
        st.metric("向量维度", stats['dimension'] or "—")
    with c3:
        st.metric("后端", stats['backend'])
    with c4:
        st.metric("来源文件", len(stats['sources']))

    for warning in stats.get('warnings', []):
        st.warning(warning)

    if stats['total_documents'] == 0:
        st.info("知识库是空的。请先到「数据处理」页面点击开始处理，把 data_raw 里的文件入库。")
    else:
        with st.expander("库内来源分布"):
            for name, count in sorted(stats['sources'].items(), key=lambda kv: -kv[1]):
                st.write(f"- {name}: {count} 块")

    stored_sig = stats.get('embedder_signature')
    if stored_sig and stored_sig != embedder.signature:
        st.error(
            f"向量器不匹配：索引由 `{stored_sig}` 构建，当前查询使用 `{embedder.signature}`。"
            f"结果不可信，请切换上面的向量器选项，或重新建库。"
        )

    st.caption(f"当前向量器: {embedder.describe()}")

    st.divider()

    # ---- 检索 ----
    query = st.text_input("输入查询内容", placeholder="例如: 什么是向量数据库？")
    col_a, col_b = st.columns(2)
    with col_a:
        n_results = st.slider("返回结果数", min_value=1, max_value=10, value=5)
    with col_b:
        min_score = st.slider("最低相似度过滤", min_value=0.0, max_value=1.0,
                              value=0.0, step=0.05)

    if st.button("检索", type="primary"):
        if not query:
            st.warning("请输入查询内容")
            return
        if stats['total_documents'] == 0:
            st.warning("知识库为空，无法检索")
            return

        with st.spinner("正在检索..."):
            try:
                query_result = embedder.embed_single(query, is_query=True)
                results = store.search(
                    query_result['embedding'],
                    n_results=n_results,
                    min_score=min_score if min_score > 0 else None
                )

                if results:
                    st.success(f"找到 {len(results)} 条相关结果")
                    for i, res in enumerate(results):
                        title = f"结果 {i+1} ｜ 相似度 {res['score']:.4f}"
                        with st.expander(title, expanded=(i == 0)):
                            st.write(res['text'])
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"来源: {res['metadata'].get('source', 'unknown')}")
                            with col2:
                                st.write(f"块号: {res['metadata'].get('chunk_id', '-')}")
                            with col3:
                                st.write(f"余弦距离: {res['distance']:.4f}")
                else:
                    st.warning("未找到相关结果。可尝试调低最低相似度，或确认向量器配置。")

            except Exception as e:
                st.error(f"检索失败: {str(e)}")

    # ---- 危险操作 ----
    st.divider()
    with st.expander("危险操作：清空知识库"):
        st.markdown(
            f"将删除 `{VECTOR_DB_DIR}` 目录下该集合的索引与记录文件。"
            f"清洗后的文本和分块结果不受影响。"
        )
        confirmed = st.checkbox("我确认要清空当前知识库")
        if st.button("清空知识库", disabled=not confirmed):
            store.reset()
            st.success("已清空知识库，请重新到「数据处理」页面建库。")


def page_stats():
    """统计报告页面"""
    st.header("数据质量统计")

    report_files = sorted(OUTPUT_DIR.glob('process_report_*.json'), reverse=True)

    if not report_files:
        st.warning("暂无处理报告，请先进行数据处理")
        return

    latest_report = report_files[0]

    with open(latest_report, 'r', encoding='utf-8') as f:
        stats = json.load(f)

    st.write(f"报告文件: {latest_report.name}")

    # 文件统计
    st.subheader("文件统计")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总文件数", stats['files']['total'])
    with col2:
        st.metric("成功处理", stats['files']['processed'])
    with col3:
        st.metric("处理失败", stats['files']['failed'])

    # 文本统计
    st.subheader("文本统计")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("原始总长度", f"{stats['text']['original_total_length']:,} 字符")
    with col2:
        st.metric("清洗后长度", f"{stats['text']['cleaned_total_length']:,} 字符")
    with col3:
        st.metric("压缩比例", f"{stats['text']['reduction_ratio']:.1%}")

    # 分块统计
    st.subheader("分块统计")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总块数", stats['chunks']['total'])
    with col2:
        st.metric("平均长度", f"{stats['chunks']['avg_length']:.0f} 字符")
    with col3:
        st.metric("最小长度", f"{stats['chunks']['min_length']} 字符")
    with col4:
        st.metric("最大长度", f"{stats['chunks']['max_length']} 字符")

    # 向量统计
    st.subheader("向量统计")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("本次入库向量数", stats['vectors']['total'])
    with col2:
        st.metric("向量维度", stats['vectors']['dimension'])
    with col3:
        st.metric("入库成功率", f"{stats['vectors']['success_rate']:.1%}")


def main():
    """主函数"""
    st.set_page_config(
        page_title="大模型数据预处理与向量知识库",
        page_icon="📚",
        layout="wide"
    )

    init_session_state()

    # 侧边栏导航
    st.sidebar.title("导航菜单")
    page = st.sidebar.radio(
        "选择功能页面",
        ["文件上传", "数据处理", "清洗效果预览", "知识库检索", "统计报告"]
    )

    # 页面路由
    if page == "文件上传":
        page_upload()
    elif page == "数据处理":
        page_process()
    elif page == "清洗效果预览":
        page_preview()
    elif page == "知识库检索":
        page_search()
    elif page == "统计报告":
        page_stats()

    # 页脚
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 项目信息")
    st.sidebar.markdown("大模型非结构化数据智能预处理与向量知识库构建项目")
    st.sidebar.markdown(f"data_raw 目录: `{DATA_RAW_DIR}`")
    st.sidebar.markdown(f"vector_db 目录: `{VECTOR_DB_DIR}`")

    # 知识库状态（直接读 manifest，无需加载向量器）
    manifest_path = VECTOR_DB_DIR / f"{VECTOR_DB_CONFIG['collection_name']}.manifest.json"
    st.sidebar.markdown("### 知识库状态")
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            st.sidebar.markdown(
                f"- 后端: `{manifest.get('backend', '?')}`\n"
                f"- 条数: `{manifest.get('count', 0)}`\n"
                f"- 维度: `{manifest.get('dimension', '?')}`\n"
                f"- 向量器: `{manifest.get('embedder_signature') or '未记录'}`\n"
                f"- 更新: {manifest.get('updated_at', '—')}"
            )
        except Exception:
            st.sidebar.markdown("manifest 读取失败")
    else:
        st.sidebar.markdown("尚未建立索引，请到「数据处理」页面生成。")


def running_under_streamlit() -> bool:
    """
    判断当前是否由 `streamlit run`（或 AppTest）拉起。

    裸跑 `python app.py` 时为 False —— 那种用法不会启动 web 服务，
    脚本跑完就退出，只会刷一屏 missing ScriptRunContext 警告。
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is not None:
            return True
    except Exception:
        pass
    try:
        from streamlit import runtime

        return bool(runtime.exists())
    except Exception:
        return False


def relaunch_via_streamlit() -> int:
    """裸跑兜底：转交给 `streamlit run`，让 `python app.py` 也能正常启动界面。"""
    from streamlit.web import cli as stcli

    script = str(Path(__file__).resolve())
    print("提示: 这是 Streamlit 应用，不能直接用 python 运行，正在自动改用 streamlit run 启动...")
    print(f'      （等价命令: streamlit run "{Path(__file__).name}"）')
    print("      按 Ctrl+C 可停止服务\n")
    sys.argv = ["streamlit", "run", script, *sys.argv[1:]]
    return int(stcli.main())


if __name__ == '__main__':
    if not running_under_streamlit():
        sys.exit(relaunch_via_streamlit())
    main()
