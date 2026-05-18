import re
from slack_bolt.async_app import AsyncApp
from apiserver.service import request_chat
import config.settings as settings

APP_TOKEN = settings.SLACK_APP_TOKEN


def get_slack():
    slack_app = AsyncApp(
        token=settings.SLACK_BOT_TOKEN,
        signing_secret=settings.SLACK_SECRET_KEY,
    )
    return slack_app

def register_slack_app(slack_app: AsyncApp):

    @slack_app.event("message")
    async def event_test(say):
        await say("Hi")

    @slack_app.event("app_mention")
    async def handle_app_mention(event, say,  client, logger):
        logger.info(event)
        text = event.get("text")
        thread_ts = event.get("ts")
        channel = event.get("channel")
        user_id = event.get("user")
        
        loading = await say(
            text="처리 중입니다... 잠시만 기다려 주세요.",
            thread_ts=thread_ts
        )

        text = re.sub(rf"<@[A-Z0-9]+>", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        result = request_chat(text, user_id = user_id)
        print(user_id)

        await client.chat_update(
                channel=channel,
                ts=loading["ts"],
                text="처리가 완료되었습니다."
            )

        await say(f"<@{user_id}> {result}")
