import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from slack_bolt.adapter.socket_mode import SocketModeHandler

from zh_cosearch_agent_app import app, SLACK_APP_TOKEN


if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
