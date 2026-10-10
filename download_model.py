"""
ONNX 模型下载脚本

从镜像站拉取 BGE 中文向量模型的 ONNX 版本，存到项目 `models/` 目录，
供 `OnnxEmbedder` 离线加载（不需要 torch，不需要 transformers）。

用法:
  python download_model.py                    # 默认下 fp32 版（约 90MB）
  python download_model.py --quantized        # 下 int8 量化版（约 23MB，更小）
  python download_model.py --list             # 只看要下什么，不下载
  python download_model.py --force            # 已存在也重下

为什么不用官方仓库：BAAI/bge-small-zh-v1.5 只提供 PyTorch 权重，没有 ONNX。
这里用的是社区导出仓库 Xenova/bge-small-zh-v1.5（ONNX 权重与官方 PyTorch 版数值等价）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TARGET = PROJECT_ROOT / "models" / "bge-small-zh-v1.5-onnx"

REPO = "Xenova/bge-small-zh-v1.5"
MIRROR = "https://hf-mirror.com"

# hf-mirror 的 CDN 会拦截 Python 默认的 User-Agent（Python-urllib/x.y）并返回 403，
# 必须伪装成常见客户端才能下载。
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

# 运行时必需的文件（缺任何一个 OnnxEmbedder 都起不来）
COMMON_FILES = ["tokenizer.json", "config.json", "tokenizer_config.json"]
FP32_MODEL = "onnx/model.onnx"
QUANTIZED_MODEL = "onnx/model_quantized.onnx"


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def resolve_endpoint() -> str:
    """优先用用户已设的 HF_ENDPOINT，其次 config.HF_CONFIG，最后默认镜像。"""
    env = os.environ.get("HF_ENDPOINT")
    if env:
        return env.rstrip("/")
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from config import HF_CONFIG

        if not HF_CONFIG.get("auto_mirror") and not HF_CONFIG.get("endpoint"):
            return "https://huggingface.co"
        return (HF_CONFIG.get("endpoint") or MIRROR).rstrip("/")
    except Exception:
        return MIRROR


def build_url(endpoint: str, remote_path: str) -> str:
    return f"{endpoint}/{REPO}/resolve/main/{remote_path}"


def remote_size(url: str) -> int:
    """拿远端文件大小，用于展示与判断是否需要重下。失败返回 -1。"""
    headers = dict(HEADERS)
    headers["Range"] = "bytes=0-0"
    req = urllib.request.Request(url, method="GET", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_range = resp.headers.get("Content-Range") or ""
            if "/" in content_range:
                return int(content_range.rsplit("/", 1)[1])
    except Exception:
        pass
    return -1


def download(url: str, dest: Path, force: bool = False, retries: int = 4) -> bool:
    """
    下载单个文件，已存在且大小一致则跳过。

    支持**断点续传 + 自动重试**：网络抖动（SSL 握手超时、连接重置）在这类镜像站很常见，
    90MB 的权重一次下完的概率并不高。中断后保留 `.part`，下次从断点继续。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    expect = remote_size(url)

    if dest.exists() and not force:
        local = dest.stat().st_size
        if expect > 0 and local == expect:
            print(f"  [跳过] {dest.name}（已存在，{human(local)}）")
            return False
        if expect < 0 and local > 0:
            print(f"  [跳过] {dest.name}（已存在，{human(local)}，远端大小未知）")
            return False
        if expect > 0:
            print(f"  [重下] {dest.name}（本地 {human(local)} != 远端 {human(expect)}）")

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  下载 {dest.name} ... {human(expect) if expect > 0 else '大小未知'}")

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        done = tmp.stat().st_size if tmp.exists() else 0
        headers = dict(HEADERS)
        if done > 0:
            headers["Range"] = f"bytes={done}-"

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                resumed = resp.status == 206 and done > 0
                if not resumed:
                    done = 0
                total = expect if expect > 0 else 0
                if resumed:
                    print(f"    续传自 {human(done)}/{human(total)}")

                with open(tmp, "ab" if resumed else "wb") as out:
                    last_print = done * 100 / total if total else 0
                    while True:
                        chunk = resp.read(256 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = done * 100 / total
                            if pct - last_print >= 10 or done == total:
                                last_print = pct
                                print(f"    {pct:5.1f}%  {human(done)}/{human(total)}")
                        else:
                            print(f"    {human(done)}", end="\r")

            if expect > 0 and tmp.stat().st_size != expect:
                raise RuntimeError(
                    f"大小不符（期望 {human(expect)}，实得 {human(tmp.stat().st_size)}）"
                )
            last_error = None
            break

        except Exception as e:
            last_error = e
            saved = tmp.stat().st_size if tmp.exists() else 0
            if attempt >= retries:
                print(f"    第 {attempt} 次失败，放弃: {e}")
                break
            wait = min(3 * attempt, 10)
            print(f"    第 {attempt} 次中断（已存 {human(saved)}）: {e}")
            print(f"    {wait}s 后续传...")
            time.sleep(wait)

    if last_error is not None:
        saved = human(tmp.stat().st_size) if tmp.exists() else "0B"
        raise RuntimeError(
            f"下载失败（已重试 {retries} 次）: {last_error}\n"
            f"        已保存 {saved}，直接重跑本脚本即可续传"
        )

    os.replace(tmp, dest)
    print(f"  [完成] {dest.name} -> {dest}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 BGE 中文向量模型的 ONNX 版本")
    ap.add_argument("--quantized", action="store_true",
                    help="下载 int8 量化版（约 23MB，默认下 fp32 版约 90MB）")
    ap.add_argument("--target", type=str, default=str(DEFAULT_TARGET),
                    help=f"保存目录，默认: {DEFAULT_TARGET}")
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    ap.add_argument("--list", action="store_true", help="只列出要下载的文件")
    args = ap.parse_args()

    endpoint = resolve_endpoint()
    target = Path(args.target)
    model_remote = QUANTIZED_MODEL if args.quantized else FP32_MODEL
    model_local = "model_quantized.onnx" if args.quantized else "model.onnx"

    plan = [(f, f) for f in COMMON_FILES] + [(model_remote, model_local)]

    print("=" * 62)
    print("ONNX 向量模型下载")
    print("=" * 62)
    print(f"仓库     : {REPO}")
    print(f"下载源   : {endpoint}")
    print(f"版本     : {'int8 量化 (' + human(23 * 1024 * 1024) + ')' if args.quantized else 'fp32 (约 90MB)'}")
    print(f"保存目录 : {target}")
    print()
    print("文件清单:")
    for remote, local in plan:
        size = remote_size(build_url(endpoint, remote))
        size_text = human(size) if size > 0 else "大小未知"
        print(f"  {local:26s} <- {remote:32s} {size_text}")

    if args.list:
        print("\n仅列表模式，未下载。")
        return 0

    print("\n开始下载...")
    try:
        for remote, local in plan:
            download(build_url(endpoint, remote), target / local, force=args.force)
    except RuntimeError as e:
        print(f"\n错误: {e}")
        print("\n排查方向:")
        print(f"  1) 换个下载源: set HF_ENDPOINT=https://hf-mirror.com")
        print(f"  2) 或手动下载后放到 {target}，文件名保持上面清单里的 local 列")
        return 1

    total = sum(p.stat().st_size for p in target.glob("*") if p.is_file())
    print()
    print("=" * 62)
    print(f"完成，共 {human(total)}，位置: {target}")
    print("=" * 62)
    print("\n下一步:")
    print("  python main.py --engine onnx --rebuild      # 用真实语义模型重建知识库")
    print("  python query.py --engine onnx -i            # 用真实模型检索")
    return 0


if __name__ == "__main__":
    sys.exit(main())
