from slack_sdk import WebClient
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import config.settings as settings

APP_TOKEN = settings.SLACK_APP_TOKEN

slack_app = App(
    token=settings.SLACK_BOT_TOKEN,
    signing_secret=settings.SLACK_SECRET_KEY,
)

@slack_app.event("message")
def event_test(say):
    say("Hi")

@slack_app.event("app_mention")
def handle_app_mention(event, say, logger):
    logger.info(event)
    say(f"Hi")

if __name__ == "__main__":
    SocketModeHandler(slack_app, APP_TOKEN).start()
    # client = WebClient(token=BOT_TOKEN)

    # resp = client.chat_postMessage(
    #     channel="C0AU08WDKBN",
    #     text="안녕하세요. 슬랙봇 테스트 메시지입니다."
    # )

    # print(resp["ok"], resp["ts"])

