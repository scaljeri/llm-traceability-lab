"""Use installed Chrome on macOS or Playwright's bundled Chromium elsewhere."""

import os
from pathlib import Path


def browser_executable():
    if override := os.environ.get("BROWSER_EXECUTABLE"):
        return override
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    return str(chrome) if chrome.is_file() else None
