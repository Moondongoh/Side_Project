import os
import fnmatch

# ROOT_PATH = r"D:\gachon\manifest-1654812109500\Duke-Breast-Cancer-MRI"
ROOT_PATH = r"경로지정"
OUT_PATH = r"경로지정"
MAX_DEPTH = 4  # 폴더 안으로 몇단계 까지 들어갈건지 지정 변수
EXCLUDES = [".git", "__pycache__"]
FOLLOW_SYMLINKS = False

LINE_MID = "├── "
LINE_END = "└── "
VERT_BAR = "│   "
EMPTY_PAD = "    "


def iter_entries(path):
    try:
        with os.scandir(path) as it:
            entries = [e for e in it]
    except Exception:
        return []

    entries.sort(key=lambda e: (not e.is_dir(), e.name.lower()))
    for e in entries:
        if any(fnmatch.fnmatch(e.name, pat) for pat in EXCLUDES):
            continue
        yield e


def build_tree(path, prefix="", depth=1, lines=None):
    if lines is None:
        lines = []
        base = os.path.basename(path) or path
        lines.append(base)

    if MAX_DEPTH is not None and depth > MAX_DEPTH:
        return lines

    children = list(iter_entries(path))
    total = len(children)

    for i, entry in enumerate(children):
        is_last = i == total - 1
        connector = LINE_END if is_last else LINE_MID
        name = entry.name + ("/" if entry.is_dir() else "")
        lines.append(prefix + connector + name)

        if entry.is_dir(follow_symlinks=FOLLOW_SYMLINKS):
            sub_prefix = prefix + (EMPTY_PAD if is_last else VERT_BAR)
            build_tree(entry.path, sub_prefix, depth + 1, lines)

    return lines


def main():
    lines = build_tree(ROOT_PATH)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
