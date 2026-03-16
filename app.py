from slack_bolt.adapter.socket_mode import SocketModeHandler

from zh_cosearch_agent_app import app, SLACK_APP_TOKEN


if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
