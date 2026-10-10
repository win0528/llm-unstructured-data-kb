"""
知识库检索命令行入口

用法:
  python query.py --stats                              # 查看知识库状态
  python query.py --query "什么是向量数据库？"           # 单次检索
  python query.py --query "分块策略" --top_k 3 --mock    # 指定条数与向量器
  python query.py -i                                   # 交互式连续提问
  python query.py --query "..." --json                 # 输出 JSON，便于管道处理

注意：检索用的向量器必须与建库时一致。若索引由 --mock 构建，
查询时也要带 --mock；反之索引由真实模型构建，查询就不要带 --mock。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from config import (
    VECTOR_DB_DIR, VECTOR_DB_CONFIG, MOCK_EMBEDDING_DIM, EMBEDDING_CONFIG, EMBEDDING_ENGINE
)
from utils import MockEmbedder, get_embedder, VectorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="大模型向量知识库检索工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--query", "-q", type=str, default=None, help="查询内容")
    parser.add_argument("--interactive", "-i", action="store_true", help="进入交互式提问模式")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="返回结果数，默认 5")
    parser.add_argument(
        "--min_score",
        type=float,
        default=None,
        help="最低余弦相似度阈值，低于该值的结果会被丢弃",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="使用哈希向量器（等价于 --engine mock，必须与建库时一致）",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=EMBEDDING_ENGINE,
        choices=["auto", "onnx", "st", "mock"],
        help=f"向量器引擎，默认: {EMBEDDING_ENGINE}（必须与建库时一致）",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=EMBEDDING_CONFIG["model_name"],
        help=f"Embedding 模型名称，默认: {EMBEDDING_CONFIG['model_name']}",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "faiss", "numpy", "chroma"],
        help="向量库后端，默认 auto",
    )
    parser.add_argument("--stats", action="store_true", help="只打印知识库状态后退出")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出结果")
    parser.add_argument(
        "--snippet",
        type=int,
        default=240,
        help="结果文本预览的最大字符数，默认 240（0 表示完整输出）",
    )
    return parser


def print_stats(store: VectorStore) -> None:
    stats = store.get_stats()
    print("=" * 60)
    print("知识库状态")
    print("=" * 60)
    print(f"  集合名称   : {stats['collection_name']}")
    print(f"  存储目录   : {stats['persist_directory']}")
    print(f"  后端       : {stats['backend']}")
    print(f"  向量条数   : {stats['total_documents']}")
    print(f"  向量维度   : {stats['dimension']}")
    print(f"  向量器指纹 : {stats['embedder_signature'] or '未记录'}")
    print(f"  最近更新   : {stats['updated_at'] or '—'}")
    if stats["sources"]:
        print(f"  来源文件   : {len(stats['sources'])} 个")
        for name, count in sorted(stats["sources"].items(), key=lambda kv: -kv[1]):
            print(f"      - {name}: {count} 块")
    else:
        print("  来源文件   : 无（知识库为空，请先运行 python main.py）")
    for warning in stats.get("warnings", []):
        print(f"  ! 警告     : {warning}")
    print("=" * 60)


def format_hits(hits, snippet: int) -> None:
    for rank, hit in enumerate(hits, 1):
        text = hit["text"] if not snippet else hit["text"][:snippet]
        suffix = "…" if snippet and len(hit["text"]) > snippet else ""
        meta = hit["metadata"] or {}
        print(f"\n[{rank}] 相似度 {hit['score']:.4f}  来源: {meta.get('source', 'unknown')}"
              f"  块号: {meta.get('chunk_id', '-')}")
        print("-" * 60)
        print(f"{text}{suffix}")


def embed_query(embedder, query: str):
    """向量化查询；模型不可用时给出可操作的提示而不是抛栈。"""
    if not query or not query.strip():
        print("查询内容为空。")
        return None
    try:
        result = embedder.embed_single(query, is_query=True)
    except Exception as exc:
        print(f"向量化失败: {exc}")
        if not isinstance(embedder, MockEmbedder):
            print("提示: 如果只是验证检索链路，可以加 --engine mock 使用离线哈希向量器；"
                  "注意它必须与建库时使用的向量器一致。")
        return None
    return result["embedding"] if result else None


def answer_once(store: VectorStore, embedder, query: str, top_k: int, min_score, snippet: int, as_json: bool):
    embedding = embed_query(embedder, query)
    if embedding is None:
        return []
    hits = store.search(embedding, n_results=top_k, min_score=min_score)

    if as_json:
        print(json.dumps(
            {
                "query": query,
                "backend": store.backend,
                "count": len(hits),
                "results": hits,
            },
            ensure_ascii=False,
            indent=2,
        ))
        return hits

    if not hits:
        print("未找到相关结果。可能原因：知识库为空、min_score 阈值过高，"
              "或查询用的向量器与建库时不一致。")
        return hits

    print(f"\n查询: {query}")
    print(f"命中 {len(hits)} 条（后端 {store.backend}）")
    format_hits(hits, snippet)
    print()
    return hits


def interactive(store: VectorStore, embedder, args) -> None:
    print("进入交互式检索，输入问题回车即可；输入 :q / exit 退出，:k N 调整返回条数。")
    top_k = args.top_k
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query in (":q", "exit", "quit", ":quit"):
            break
        if query.startswith(":k "):
            try:
                top_k = max(1, int(query[3:].strip()))
                print(f"返回条数已设为 {top_k}")
            except ValueError:
                print("用法: :k 3")
            continue
        answer_once(store, embedder, query, top_k, args.min_score, args.snippet, args.json)


def main() -> int:
    args = build_parser().parse_args()

    try:
        embedder = get_embedder(
            use_mock=args.mock,
            model_name=args.model_name,
            dimension=MOCK_EMBEDDING_DIM,
            engine=args.engine,
        )
        # 提前加载一次，失败就给出明确提示而不是每问一次报一次错
        embedder.load_model()
    except Exception as exc:
        print(f"向量器加载失败: {exc}", file=sys.stderr)
        print("排查方向:", file=sys.stderr)
        print("  1) 轻量路线: python download_model.py && python query.py --engine onnx", file=sys.stderr)
        print("  2) 离线链路: python query.py --engine mock（需与建库时一致）", file=sys.stderr)
        return 1

    store = VectorStore(
        collection_name=VECTOR_DB_CONFIG["collection_name"],
        persist_directory=str(VECTOR_DB_DIR),
        backend=args.backend,
        embedder_signature=embedder.signature,
        normalize=VECTOR_DB_CONFIG["normalize"],
    )

    if args.stats:
        print_stats(store)
        return 0

    if store.count() == 0:
        print_stats(store)
        print(f"知识库是空的。先跑 `python main.py --engine {args.engine}` 建库，再回来检索。")
        return 1

    stored_sig = store.manifest.get("embedder_signature")
    if stored_sig and stored_sig != embedder.signature:
        print(
            f"提示: 索引由 {stored_sig} 构建，当前查询使用 {embedder.signature}，\n"
            f"      向量空间不一致，结果可能没有意义。",
            file=sys.stderr,
        )

    if not args.json:
        print(f"向量器: {embedder.describe()}")

    if args.interactive or not args.query:
        interactive(store, embedder, args)
        return 0

    hits = answer_once(
        store, embedder, args.query, args.top_k, args.min_score, args.snippet, args.json
    )
    return 0 if hits else 1


if __name__ == "__main__":
    sys.exit(main())
