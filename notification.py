import requests
import json

def stalemate_notification(id, webhook):
    content = {'text': "{}は手が止まっています".format(id)}
    requests.post(webhook, data=json.dumps(content))