import os
from pathlib import Path

import pytesseract
from slack_bolt.adapter.socket_mode import SocketModeHandler


def _configure_tesseract_cmd() -> None:
    """Configure pytesseract executable path in a portable way."""
    env_cmd = (os.getenv("TESSERACT_CMD") or "").strip()
    if env_cmd and Path(env_cmd).exists():
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return

    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"D:\Tesseract-OCR\tesseract.exe",
    ]
    for cmd in candidates:
        if Path(cmd).exists():
            pytesseract.pytesseract.tesseract_cmd = cmd
            return
    # If no explicit path is found, keep the default so pytesseract can resolve via PATH.


_configure_tesseract_cmd()

from zh_cosearch_agent_app import app, SLACK_APP_TOKEN


if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
