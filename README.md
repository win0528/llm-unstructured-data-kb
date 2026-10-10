# 大模型非结构化数据智能预处理与向量知识库构建项目

将杂乱的 PDF、Word、TXT、网页文本全自动处理成大模型可用的高质量训练/问答数据，构建本地向量知识库。

## 功能特性

- **多格式文档解析**：支持 PDF、Word、TXT、Markdown、HTML、CSV、JSON 等格式
- **智能文本清洗**：去除乱码、特殊符号、多余空行，统一编码
- **隐私信息脱敏**：自动识别并脱敏手机号、身份证、邮箱、网址
- **语义分块**：基于段落和标题的智能分块，保持语义完整性
- **三档向量器**：ONNX 轻量版（约 150MB，推荐）/ sentence-transformers（含 torch）/ 离线哈希向量器（零依赖）
- **可插拔向量库后端**：faiss / numpy / chroma 三选一，**默认不再依赖 ChromaDB**
- **本地持久化 + 增量更新**：向量、正文、元信息全部落盘，重启即可检索；重复入库自动 upsert，不产生重复记录
- **命令行检索工具**：`query.py` 支持单次查询、交互式提问、JSON 输出
- **质量统计**：自动生成数据质量报告

## 环境要求

- Python 3.9 ~ 3.10
- Windows / macOS / Linux

## 安装步骤

```bash
# 1. 克隆或下载项目
git clone <repository-url>
cd 大模型非结构化数据智能预处理与向量知识库

# 2. 安装依赖（依赖按用途分层，详见 requirements.txt）
pip install -r requirements.txt

# 【推荐】轻量语义路线：约 150MB，不需要 torch
pip install numpy streamlit faiss-cpu onnxruntime tokenizers
python download_model.py            # 下 BGE 的 ONNX 模型（约 90MB）

# 完整路线（要 sentence-transformers）：会连带装 torch，约 2.5GB
pip install sentence-transformers

# 最小化安装：只要 numpy + streamlit，用 numpy 后端 + 哈希向量器即可全程离线跑通
pip install numpy streamlit
```

> **依赖装进哪个解释器？** 装进你**实际用来跑脚本的那个**解释器。
> 如果你建了虚拟环境（`venv\Scripts\python.exe`），它默认是隔离的
> （`include-system-site-packages = false`），**看不到全局环境里已装的包**。
> 请用 venv 自己的 pip 安装，不要用全局的：
>
> ```bash
> # 用 venv 的解释器装
> venv\Scripts\python.exe -m pip install faiss-cpu streamlit numpy
> # 想用真实语义模型（会连带装 torch，约 2.5 GB）
> venv\Scripts\python.exe -m pip install sentence-transformers
>
> # 或者反过来：不装模型依赖，直接用内置哈希向量器离线跑
> venv\Scripts\python.exe main.py --mock
> ```
>
> 各依赖的作用：`numpy`+`streamlit` 是必须的；`faiss-cpu` 让检索更快（没装会自动退化为 numpy 后端）；
> `sentence-transformers` 只有走真实语义模型时才需要；`pdfplumber`/`python-docx` 只在解析 PDF/Word 时需要。

### 模型下载源

国内直连 `huggingface.co` 通常超时（表现为 `502 Bad Gateway` 或长时间挂起）。项目默认在加载模型前
自动把 `HF_ENDPOINT` 指向镜像 `https://hf-mirror.com`，无需手动配置。

若你有代理或想用官方源，二选一：

```bash
# 方式一：自己设环境变量（项目检测到已有 HF_ENDPOINT 就不会再改）
set HF_ENDPOINT=https://huggingface.co

# 方式二：改 config.py
HF_CONFIG = {"auto_mirror": False, "endpoint": ""}
```

## 使用方式

### 方式一：命令行工具（推荐）

```bash
# 1. 将待处理文件放入 data_raw 目录
# 支持格式：PDF、Word、TXT、Markdown、HTML、CSV、JSON

# 2. 运行处理命令（使用默认配置）
python main.py

# 3. 常用组合
python main.py --engine onnx --rebuild   # 用 ONNX 真实语义模型全量重建（推荐）
python main.py --engine mock             # 离线哈希向量器，不下载模型，先跑通链路
python main.py --backend numpy           # 指定零依赖的 numpy 向量库后端
python main.py --dry_run                 # 只做清洗分块，不写向量库
python main.py --chunk_size 800 --overlap 150 --model_name BAAI/bge-small-zh-v1.5

# 4. 检索知识库
python query.py --stats                          # 查看库状态、来源分布
python query.py -q "什么是向量数据库" -k 3         # 单次检索（引擎必须与建库时一致）
python query.py -i                               # 交互式连续提问
python query.py -q "分块策略" --engine mock --json  # JSON 输出，便于管道处理

# 5. 下载 / 更新 ONNX 模型
python download_model.py                # fp32，约 90MB
python download_model.py --quantized    # int8 量化版，约 23MB
python download_model.py --list         # 只看要下什么
```

**命令行参数说明（`main.py`）：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--chunk_size` | 分块大小（字符数） | 1024 |
| `--overlap` | 分块重叠大小 | 50 |
| `--model_name` | Embedding 模型名称 | BAAI/bge-small-zh-v1.5 |
| `--engine` | 向量器引擎：auto/onnx/st/mock | auto |
| `--mock` | 等价于 `--engine mock` | False |
| `--backend` | 向量库后端：auto/faiss/numpy/chroma | auto |
| `--rebuild` | 先清空集合再入库 | False |
| `--dry_run` | 只清洗分块，不写向量库 | False |

**输出目录：**

- `data_clean/` - 清洗后的文本文件
- `data_chunk/` - 分块结果（JSON 格式）
- `vector_db/` - 向量库（持久化文件见下）
- `output/` - 处理报告

### 方式二：Web 界面

```bash
streamlit run app.py
# 或者直接 python app.py —— 脚本会检测到裸跑并自动转交到 streamlit run
python app.py
```

浏览器会自动打开 http://localhost:8501，提供：

- 文件上传界面
- 一键处理按钮（可选向量器引擎、向量库后端，可重建索引、可干跑）
- 清洗效果预览
- 知识库检索测试（含库状态、来源分布、相似度排序、向量器指纹校验、清空知识库）
- 数据质量统计报告

## 项目结构

```
.
├── data_raw/              # 输入：待处理的原始文件
├── data_clean/            # 输出：清洗后的文本
├── data_chunk/            # 输出：分块结果
├── vector_db/             # 输出：向量库持久化文件
├── models/                # ONNX 向量模型（由 download_model.py 下载）
├── utils/                 # 工具模块
│   ├── parser.py          # 文档解析（pdfplumber / python-docx 惰性导入）
│   ├── clean.py           # 文本清洗、脱敏、语种检测、质量打分
│   ├── chunker.py         # 智能分块
│   ├── elements.py        # 文档元素模型（标题/段落/列表/表格/代码块）
│   ├── embedding.py       # 向量器工厂 + 哈希向量器 + sentence-transformers 向量器
│   ├── onnx_embedding.py  # ONNX 轻量向量器（onnxruntime + tokenizers，不需要 torch）
│   ├── vector_store.py    # 向量库（faiss / numpy / chroma 可插拔后端）
│   ├── quality.py         # 质量统计
│   └── logger.py          # 日志
├── tests/
│   ├── test_vector_store.py    # 向量库自测：落盘重载、后端一致性、upsert、删除、清空
│   ├── test_onnx_embedder.py   # ONNX 向量器自测（依赖缺失自动跳过）
│   └── test_app_smoke.py       # 界面冒烟测试：无头渲染全部 5 个页面
├── main.py                # 命令行处理入口
├── query.py               # 命令行检索入口
├── download_model.py      # ONNX 模型下载（hf-mirror + 断点续传 + 自动重试）
├── app.py                 # Streamlit Web 界面
├── config.py              # 配置文件
├── requirements.txt       # 依赖清单
└── README.md              # 使用说明
```

## 向量器引擎说明

`engine` 可选 `auto` / `onnx` / `st` / `mock`，用 `--engine` 或 `config.EMBEDDING_ENGINE` 指定。

| 引擎 | 实现 | 依赖体积 | 效果 |
|------|------|----------|------|
| `onnx` | onnxruntime + tokenizers 跑 BGE ONNX | **约 150MB**（onnxruntime + 模型） | 真实语义向量，推荐 |
| `st` | sentence-transformers | 约 2.5GB（含 torch） | 真实语义向量，与 onnx 基本等价 |
| `mock` | 内置哈希向量器（字符 n-gram） | 0 | 只做字面匹配，**不懂同义词**，仅验证链路 |
| `auto` | 优先 `onnx`，不可用则退到 `st` | — | 默认值 |

> **建库和检索必须用同一个引擎。** 索引的 `manifest.json` 里记了向量器指纹
> （如 `onnx-bge:bge-small-zh-v1.5-onnx:model:cls:512`），用错引擎时程序会告警并提示
> "向量空间不一致，结果不可信"。

**为什么推荐 ONNX 而非 sentence-transformers？** 两者用的是同一个 BGE 模型，向量质量一致，
但 ONNX 路线不需要 PyTorch —— 依赖从 2.5GB 降到约 150MB。模型文件用
`python download_model.py` 下载（fp32 约 90MB，`--quantized` 走 int8 约 23MB）。

**实测效果对比**（同一批 5 条同义提问，命中正确来源文件的比例）：

| 引擎 | 命中率 | 典型相似度 |
|------|--------|-----------|
| `mock` 哈希向量器 | 3/5 | 0.07 ~ 0.20（几乎无区分度） |
| `onnx` 真实语义 | 4/5 | 0.44 ~ 0.63 |

例：问「怎么把长文档切段」——哈希向量器错命中 `vector_database.md`（0.071）；
ONNX 正确命中 `data_preprocessing.txt`（0.573）。

## 向量库后端说明

`backend` 可选 `auto` / `faiss` / `numpy` / `chroma`，默认 `auto` 按 **faiss → numpy → chroma** 的顺序
自动挑选可用的后端。

| 后端 | 依赖 | 适用场景 |
|------|------|----------|
| `faiss` | `faiss-cpu` | 推荐。`IndexIDMap2 + IndexFlatIP`，向量已 L2 归一化，内积即余弦相似度 |
| `numpy` | 无（仅 numpy） | 零额外依赖，小规模数据（上万条以内）完全够用 |
| `chroma` | `chromadb` | 旧的 ChromaDB 后端，仅需兼容历史数据时使用 |

`faiss` 与 `numpy` 共享同一套持久化格式，落盘文件位于 `vector_db/`：

| 文件 | 内容 |
|------|------|
| `<集合名>.vectors.npy` | L2 归一化后的 float32 向量矩阵 `[N, D]` |
| `<集合名>.records.jsonl` | 每行一条记录：`{"int_id","id","text","metadata"}` |
| `<集合名>.manifest.json` | 索引元信息：后端、维度、条数、embedding 指纹、时间戳 |

faiss 用的是 `IndexFlatIP` 暴力索引，索引本体就是 `vectors.npy`，所以**不再额外落盘索引
文件** —— 加载时直接重建，省一个文件，也避免"索引与向量不同步"的脏读。

因为正文和向量都落盘，**进程重启后知识库依旧可用**。两个后端可以互相读取同一份
数据（用 `numpy` 打开 `faiss` 建的库也一样能检索），方便在没有 faiss 的环境里迁移。

## 配置说明

编辑 `config.py` 可修改默认配置：

```python
# 文本处理配置
TEXT_CONFIG = {
    "min_chunk_length": 50,      # 最小分块长度
    "max_chunk_length": 1024,    # 最大分块长度
    "overlap": 50,               # 分块重叠
}

# 向量化配置
EMBEDDING_CONFIG = {
    "model_name": "BAAI/bge-small-zh-v1.5",  # Embedding 模型
}

# 向量库配置
VECTOR_DB_CONFIG = {
    "collection_name": "knowledge_base",
    "persist_directory": str(VECTOR_DB_DIR),
    "backend": "auto",   # auto / faiss / numpy / chroma
    "normalize": True,   # 入库前 L2 归一化，配合内积做余弦检索
}

MOCK_EMBEDDING_DIM = 512  # 离线哈希向量器维度，建库与检索必须一致

# 向量器引擎: auto / onnx / st / mock
EMBEDDING_ENGINE = "auto"

# ONNX 向量器配置（模型用 python download_model.py 下载）
ONNX_CONFIG = {
    "model_dir": str(PROJECT_ROOT / "models" / "bge-small-zh-v1.5-onnx"),
    "max_length": 512,
    "batch_size": 32,
    "pooling": "cls",          # BGE 系列用 CLS 池化
    "query_instruction": "",   # 留空表示查询不加前缀；建库与检索必须一致
    "threads": None,
}

# HuggingFace 下载配置（国内直连 huggingface.co 通常超时）
HF_CONFIG = {
    "auto_mirror": True,
    "endpoint": "https://hf-mirror.com",
}
```

> 关于 `query_instruction`：BGE 官方建议检索时给查询加前缀「为这个句子生成表示以用于检索相关文章：」。
> 本机实测**加了没有提升**（命中率不变、相似度略降），所以默认留空。要试的话改这里即可，
> 查询侧会自动加上，文档侧不受影响。

## 自测

```bash
python tests/test_vector_store.py    # 向量库核心逻辑（不需要模型）
python tests/test_onnx_embedder.py   # ONNX 向量器（依赖/模型缺失会自动跳过）
python tests/test_app_smoke.py       # Web 界面无头冒烟
```

`test_vector_store.py` 覆盖：哈希向量器确定性、两种后端落盘与重载、持久化文件齐全、
重载后检索结果一致、结果按相似度降序、upsert 幂等、按 id 删除、清空后换维度重建、
维度校验、向量器指纹警告、遗留索引文件清理、faiss 与 numpy 排序一致性。

`test_onnx_embedder.py` 覆盖：加载、维度与归一化、**批量与单条结果一致性**（padding 不该影响结果）、
语义区分度（相关句应比无关句高 0.3 以上）、同义改写稳定性。

`test_app_smoke.py` 用 Streamlit 官方 `AppTest` 逐个切换页面并断言无异常。

## 完整使用流程

### 命令行版本

1. 将待处理文档放入 `data_raw/` 目录
2. 执行 `python main.py --mock --rebuild`（离线跑通）或 `python main.py`（真实模型）
3. 等待处理完成
4. `python query.py -i` 开始提问
5. 在输出目录查看结果：
   - `data_clean/` - 清洗后的文本
   - `data_chunk/` - 分块结果
   - `vector_db/` - 向量库
   - `output/` - 处理报告

### Web 版本

1. 执行 `streamlit run app.py`（或 `python app.py`，会自动转交）
2. 浏览器打开界面
3. 上传文件到 `data_raw/`
4. 到「数据处理」页选择向量器引擎与后端，点击"开始处理"
5. 到「知识库检索」页查看库状态并检索

## 已知问题

1. **大文件处理**：超大 PDF 文件可能占用较多内存，建议分批处理
2. **模型下载需要网络**：ONNX 模型约 90MB，首次要联网下载；下载源默认走 hf-mirror，
   中断了直接重跑 `python download_model.py` 会**断点续传**。完全离线时请用
   `--engine mock`（哈希向量器）
3. **Windows 兼容性**：部分库在 Windows 上可能需要额外配置，建议使用 Python 3.10
4. **删除权限受限的环境**：受管终端 / 沙箱有时会把 `Path.unlink` 重定向到回收站 API，
   无桌面会话时直接抛 `OSError`。本项目清空集合时**不依赖删除**（改写为空状态文件），
   必须删除时也走 `os.remove -> Path.unlink -> 截断` 三级降级（见 `utils/vector_store.py`
   的 `safe_unlink`），因此在这种环境里也能正常工作
5. **哈希向量器不懂同义词**：`--engine mock` 只做字符 n-gram 字面匹配，
   问「怎么把长文档切段」可能命中错误文件。它是给"无网络/无依赖时验证链路"用的，
   真实问答请用 `onnx` 引擎

## 常见问题

**Q: 直接 `python app.py` 只刷一屏 `missing ScriptRunContext` 警告就退出了？**

A: 这不是报错，是**启动方式不对**。Streamlit 应用必须由 `streamlit run` 拉起 ——
它需要一个 web 服务来承载会话，直接 `python app.py` 只是把脚本裸跑一遍，
所有 `st.*` 调用都没有运行时上下文，于是刷警告、执行完就退出（退出码 0）。

正确做法二选一：

```bash
streamlit run app.py      # 标准方式
python app.py             # 项目已做兜底：检测到裸跑会自动转交到 streamlit run
```

底层的 `st.*` API（`st.file_uploader` 等）在裸跑时不会生效，所以任何"直接用 python 跑
Streamlit 脚本"的想法都行不通；要拿它做自动化测试请用官方的 `streamlit.testing.v1.AppTest`
（本项目 `tests/test_app_smoke.py` 就是这么做的）。

**Q: 报 `模型加载失败: No module named 'sentence_transformers'` 怎么办？**

A: 说明**当前这个解释器**没装 sentence-transformers（虚拟环境与全局环境是隔离的，全局装了不算数）。
报错信息里会直接指出当前解释器路径和该执行的安装命令。三条路：

1. **推荐**：改用 ONNX 轻量路线（不需要 torch）——
   `pip install onnxruntime tokenizers` + `python download_model.py`，然后加 `--engine onnx`；
2. 装 sentence-transformers 走完整路线：`"<报错里显示的解释器>" -m pip install sentence-transformers`
   （含 torch，约 2.5 GB）；
3. 不装模型，用内置哈希向量器：`python main.py --engine mock`
   （**检索时也要加 `--engine mock`**，两者必须一致）。

**Q: 加了 `--engine onnx` 报"环境未就绪"怎么办？**

A: 报错信息会分别指出**依赖**和**模型文件**哪一边没齐：

```bash
# 依赖缺失
pip install onnxruntime tokenizers
# 模型缺失（会在 models/bge-small-zh-v1.5-onnx/ 下补文件，支持断点续传）
python download_model.py
# 先看看要下什么、从哪个源下
python download_model.py --list
```

装完用 `python tests/test_onnx_embedder.py` 验证，正常应输出维度 512、模长 1.0、
以及"相关 0.88 vs 无关 0.26"这类语义区分度。若下载失败（SSL 超时 / 403），
注意两点：下载源必须是能通的镜像（`HF_ENDPOINT`），以及 hf-mirror 的 CDN
会拦截 Python 默认 User-Agent —— 这两点 `download_model.py` 都已处理，自己写脚本时要注意。

**Q: ONNX 和 sentence-transformers 出来的向量一样吗？**

A: 是同一个 BGE 模型（`Xenova/bge-small-zh-v1.5` 是官方 PyTorch 版的 ONNX 导出），
向量质量一致，但**两者指纹不同**（`onnx-bge:...` vs `sentence-transformers:...`），
索引不能混用 —— 换引擎后要加 `--rebuild` 重建。

**Q: 还需要装 ChromaDB 吗？**

A: 不需要。默认 `backend="auto"` 会优先用 faiss，没有 faiss 就用零依赖的 numpy 后端，两者都不需要
ChromaDB。只有显式指定 `--backend chroma` 时才要求安装 chromadb。

**Q: 怎么换 Embedding 模型？**

A: 修改 `config.py` 中的 `EMBEDDING_CONFIG["model_name"]`，或使用命令行参数 `--model_name`。

**Q: 怎么换 Embedding 模型？**

A: 修改 `config.py` 中的 `EMBEDDING_CONFIG["model_name"]`，或使用命令行参数 `--model_name`。

**Q: 检索不到结果 / 结果无关怎么办？**

A: 按以下顺序排查：

1. `python query.py --stats` 确认库里确实有数据；
2. 确认查询用的向量器与建库时一致 —— **用 `--mock` 建的库就带 `--mock` 查，用真实模型建的库就不要带**。
   索引里记录了向量器指纹（manifest 的 `embedder_signature`），不一致时程序会给出警告；
3. 向量器一致但仍不相关时，调低 `--min_score`，或检查分块是否把语义切碎了；
4. 哈希向量器只做字面 n-gram 匹配，**不理解同义词**，验证链路够用，真实问答请换语义模型。

**Q: 向量数据库在哪里？**

A: `vector_db/` 目录，具体文件见上文「向量库后端说明」。

**Q: 重复运行 `main.py` 会不会把数据写两遍？**

A: 不会。分块 id 由 `文件名_chunk_序号` 决定，重复入库走 upsert 覆盖旧块，条数保持不变。
需要彻底重来时才加 `--rebuild`。

**Q: 如何测试环境是否正常？**

A: 运行 `python tests/test_vector_store.py`（向量库逻辑）和 `python tests/test_app_smoke.py`
（界面渲染），再跑 `python main.py --mock` 和 `python query.py -q "向量" --mock` 走一遍完整链路。

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request。
