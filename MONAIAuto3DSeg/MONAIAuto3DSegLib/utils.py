import sys


def get_safe_path(path):
    """Convert a path to a short (8.3) Windows path if it contains non-ASCII characters.

    On Windows, paths with accented or non-ASCII characters can cause failures in subprocess
    calls and in libraries like PyTorch, ITK, and pynrrd. Converting to the short (8.3) path
    format produces an ASCII-only equivalent that bypasses these issues.

    On non-Windows platforms, the path is returned unchanged.

    :param path: A file or directory path (str or Path object). The path must exist on disk.
    :return: The short path (str) on Windows if non-ASCII characters are present,
             or the original path as a string otherwise.
    """
    path_str = str(path)
    if sys.platform != "win32":
        return path_str
    try:
        path_str.encode("ascii")
        return path_str
    except UnicodeEncodeError:
        pass
    import ctypes
    from ctypes import wintypes
    GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
    GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    GetShortPathNameW.restype = wintypes.DWORD
    buf_size = GetShortPathNameW(path_str, None, 0)
    if buf_size == 0:
        import logging
        logging.warning(
            f"get_safe_path: GetShortPathNameW failed for '{path_str}'. "
            f"Short (8.3) names may be disabled on this volume."
        )
        return path_str
    buf = ctypes.create_unicode_buffer(buf_size)
    GetShortPathNameW(path_str, buf, buf_size)
    return buf.value


def humanReadableTimeFromSec(seconds):
  import math
  if not seconds:
    return "N/A"
  if seconds < 55:
    # if less than a minute, round up to the nearest 5 seconds
    return f"{math.ceil(seconds / 5) * 5} sec"
  elif seconds < 60 * 60:
    # if less than 1 hour, round up to the nearest minute
    return f"{math.ceil(seconds / 60)} min"
  # Otherwise round up to the nearest 0.1 hour
  return f"{seconds / 3600:.1f} h"
