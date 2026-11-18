"""Work around stale Visual C++ runtime DLLs loaded ahead of PyTorch on Windows."""

from __future__ import annotations

import ctypes
import os
import sys


def preload_windows_cpp_runtime() -> None:
    """Load the Windows system C++ runtime before Anaconda's older copies."""
    if sys.platform != "win32":
        return

    system_directory = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    for dll_name in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"):
        dll_path = os.path.join(system_directory, dll_name)
        if os.path.isfile(dll_path):
            ctypes.WinDLL(dll_path)


preload_windows_cpp_runtime()
