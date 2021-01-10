import sentry_sdk
from flask import Flask, request, jsonify
from sentry_sdk.integrations.flask import FlaskIntegration
from transformers import AutoModelForCausalLM
import torch
import math
import os
import requests

MILLION = 1000 * 1000

env = os.environ.get('PRODUCT_ENV')

if env == "production":
    sentry_sdk.init(
        dsn=os.environ.get('SENTRY_DSN'),
        integrations=[FlaskIntegration()],
        traces_sample_rate=1.0
    )

MODEL_NAME = os.environ["MODEL_NAME"]
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, return_dict=True)
model.to(device="cuda")

app = Flask(__name__)


def send_message_to_slack(text):
    url = os.environ.get('SLACK_INCOMING_WEBHOOKS_URL')
    payload = {
        "pretext": "*GPT2-EVALUTE SERVER ERROR OCCURED!*",
        "text" : f"*ERROR*: {text}",
        "color": "danger",
    }
    requests.post(url, json=payload)

    if env == "production":
        sentry_sdk.capture_message(text, "fatal")


def check(encoded_text):
    if not isinstance(encoded_text, list):
        return False
    for l in encoded_text:
        if l < 0 or 50257 <= l:
            return False

    return True


def score(encoded_text):
    if not check(encoded_text):
        return -1
    if len(encoded_text) == 1:
        return 100.0 * MILLION
    encoded_text = torch.cuda.LongTensor(encoded_text)
    output = model(input_ids=encoded_text, labels=encoded_text)
    print(output.loss, type(output.loss))
    print(output.loss.cpu().detach())
    return math.exp(output.loss.cpu().detach())


@app.route("/evaluate", methods=["POST"])
def evaluate():
    try:
        if request.is_json:
            content = request.get_json()
            print(content, type(content))
            return jsonify({"score": score(content)}), 200
        return jsonify({"score": -1}), 400
    except Exception as e:
        if request.is_json:
            send_message_to_slack(f'requested json: *{request.get_json()}*. \n *{e}*')
        else:
            send_message_to_slack(f'requested data: *{request.data}*. \n *{e}*')
        return jsonify({"score": -1}), 500


if __name__ == "__main__":
    app.run(debug=False, port=8001, host="0.0.0.0")
