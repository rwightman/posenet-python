# https://github.com/slackapi/python-slack-sdk
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

def slack_init()
    global client 
    global slackBotToken
    global redis
    slackBotToken = ""
    client = WebClient(token=slackBotToken)
    redis_init()

def stalemate_notification(id)
    global client
    try:
        content = {'text': "{}は手が止まっています".format(id)}
        response = client.chat_postMessage(channel = '#student_state', text = content)
        assert response["message"]["text"] == content
    except SlackApiError as e:
        # You will get a SlackApiError if "ok" is False
        assert e.response["ok"] is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
        print(f"Got an error: {e.response['error']}")