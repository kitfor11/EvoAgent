"""工具集：Agent 能调用的「手」。

每个工具 = 一个普通 Python 函数 + 一段描述（docstring）。
描述很重要：模型就是靠读这段描述，判断「现在该用哪个工具、怎么传参」。

TOOLS 是注册表：名字 -> 函数。Agent 说「我要调 count_words」，
我们就从这张表里按名字查到函数来执行。

安全设计（已按需求放开到任意路径）：
- 绝对路径直接用，不再限制在沙盒内（用户明确要求操作真实磁盘）。
- 相对路径仍以 SANDBOX_ROOT 为基准，方便 Agent 拿文件名直接操作。
- 删除是「软删除」：文件被移到项目下的 .trash 回收站，而不是永久删除。
  这是对不可逆操作留的后路——误删还能找回来。
"""
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict

from app.config import settings

TOOLS: Dict[str, Callable] = {}

# 默认工作目录：相对路径的基准。默认 demo_files，可在 .env 的 SANDBOX_ROOT 里改。
SANDBOX_ROOT = Path(settings.sandbox_root).resolve()

# 回收站：删除时把文件移到这里（软删除，可恢复），放在项目根目录下。
TRASH_DIR = Path(__file__).resolve().parent.parent.parent / ".trash"

# 当前工作目录（运行时可变）：初始 = 默认工作目录。
# 用户在前端「选择文件夹」后切到这里，之后 Agent 的相对路径都以它为基准，
# 相当于给 Agent 换了「当前所在的文件夹」。
_current_workdir = SANDBOX_ROOT


def get_workdir() -> Path:
    """返回当前工作目录（Agent 相对路径的基准）。"""
    return _current_workdir


def set_workdir(path: str) -> Path:
    """切换当前工作目录。path 必须是已存在的目录。"""
    global _current_workdir
    p = Path(path).resolve()
    if not p.is_dir():
        raise ValueError(f"Not a directory: {path}")
    _current_workdir = p
    return p


def _resolve_path(path: str) -> Path:
    """把路径解析成绝对路径（相对路径以当前工作目录为基准）。

    现在放开到「任意路径」：绝对路径直接用，不再做沙盒越界检查。
    相对路径（如 "report.pdf"）以当前工作目录（get_workdir）为基准，
    因为 list_files 只回传文件名，Agent 自然会拿文件名去操作。
    """
    p = Path(path)
    if not p.is_absolute():
        p = get_workdir() / p  # 相对路径默认相对于当前工作目录
    return p.resolve()


def register(func: Callable) -> Callable:
    """装饰器：把函数按名字自动登记进 TOOLS 表。"""
    TOOLS[func.__name__] = func
    return func


@register
def count_words(text: str) -> int:
    """Count how many words are in the text. Input: the text. Output: an integer count."""
    return len(text.split())


@register
def reverse_text(text: str) -> str:
    """Reverse the characters of the text. Input: the text. Output: the reversed text."""
    return text[::-1]


@register
def to_upper(text: str) -> str:
    """Convert the text to UPPERCASE. Input: the text. Output: the uppercase text."""
    return text.upper()


@register
def to_lower(text: str) -> str:
    """Convert the text to lowercase. Input: the text. Output: the lowercase text."""
    return text.lower()


@register
def list_files(path: str) -> str:
    """List all entries directly inside a directory (files and subdirectories).
    Input: path - the directory to inspect. Output: a text list; subdirectories are marked [dir]."""
    dir_path = _resolve_path(path)
    if not dir_path.exists():
        return f"Directory not found: {path}"
    lines = []
    for p in sorted(dir_path.iterdir()):  # iterdir 遍历目录下的每一项
        if p.is_dir():                    # 子目录标 [dir]，方便 Agent 继续深入
            lines.append(f"[dir] {p.name}")
        else:
            size_kb = p.stat().st_size / 1024
            lines.append(f"{p.name}  ({size_kb:.1f} KB)")
    if not lines:
        return f"(empty directory: {path})"
    return "\n".join(lines)


@register
def read_text(path: str) -> str:
    """Read a text file and return its content (truncated to 2000 chars).
    Input: path - the file to read. Output: the file's text content."""
    p = _resolve_path(path)
    if not p.exists():
        return f"File not found: {path}"
    # errors="ignore"：遇到非 UTF-8 的字节就跳过，不崩；[:2000] 防止超长文件撑爆 prompt
    return p.read_text(encoding="utf-8", errors="ignore")[:2000]


@register
def make_dir(path: str) -> str:
    """Create a directory (and any missing parents).
    Input: path - the directory to create. Output: a confirmation message."""
    p = _resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)  # parents=True 递归建目录；exist_ok=True 已存在不报错
    return f"Created directory: {path}"


@register
def move_file(src: str, dst: str) -> str:
    """Move or rename a file from src to dst.
    Input: src - current path, dst - target path. Output: a confirmation message."""
    src_p = _resolve_path(src)
    dst_p = _resolve_path(dst)
    if not src_p.exists():
        return f"File not found: {src}"
    dst_p.parent.mkdir(parents=True, exist_ok=True)  # 先保证目标目录存在
    shutil.move(str(src_p), str(dst_p))
    return f"Moved {src} -> {dst}"


@register
def delete_file(path: str) -> str:
    """Delete a file or directory by moving it to the trash (recoverable, not permanent).
    Input: path - the file or directory to delete. Output: a confirmation message."""
    p = _resolve_path(path)
    if not p.exists():
        return f"Not found: {path}"
    TRASH_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 时间戳前缀，避免重名互相覆盖
    dest = TRASH_DIR / f"{stamp}_{p.name}"
    shutil.move(str(p), str(dest))
    return f"Deleted {path} -> moved to trash: {dest}"


@register
def read_image_text(path: str) -> str:
    """Read the text inside an image file using OCR (看图识字).
    Input: path - the image file (.png/.jpg/.jpeg/.bmp). Output: the recognized text."""
    p = _resolve_path(path)
    if not p.exists():
        return f"File not found: {path}"
    if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
        return f"Not an image file: {path} (only .png/.jpg/.jpeg/.bmp)"
    # 懒 import：只有真正调用这个工具时才加载 OCR 引擎（paddleocr 很重）
    from app.multimodal.ocr import get_ocr
    try:
        return get_ocr().read_text(str(p))
    except Exception as e:  # OCR 失败不崩，把错误作为观察返回给 Agent
        return f"OCR failed: {e}"


@register
def search_docs(query: str, k: int = 3) -> str:
    """Search the document knowledge base (indexed from workdir files) for chunks related to a question.
    Input: query - the question to search for; k - how many chunks to return.
    Output: the top related chunks, each prefixed with its source file."""
    # 懒 import：只有真正调用这个工具时才加载向量库（嵌入模型很重）
    from app.rag.indexer import ensure_indexed
    from app.rag.store import get_store

    store = get_store()  # 单例，进程内只建一次
    ensure_indexed(store)  # 空库或工作目录变了就（重）建索引

    try:  # 模型可能把 k 写成字符串（如 "3"），兜底转 int
        k = int(k)
    except (TypeError, ValueError):
        k = 3

    hits = store.hybrid_search(query, k=k)
    if not hits:
        return "(知识库为空或没有检索到相关内容)"
    return "\n".join(f"[{h['source']}] {h['text']}" for h in hits)


def call_tool(name: str, args: Dict[str, Any]) -> Any:
    """按名字执行工具。找不到工具时报错（这样 Agent 乱编工具名时能被发现）。"""
    if name not in TOOLS:
        raise ValueError(f"Unknown tool: {name}")
    return TOOLS[name](**args)
