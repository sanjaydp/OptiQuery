import difflib

def generate_diff(original, optimized):
    diff = difflib.unified_diff(
        original.strip().splitlines(),
        optimized.strip().splitlines(),
        lineterm="",
        fromfile="Original",
        tofile="Optimized"
    )
    return "\n".join(diff)
