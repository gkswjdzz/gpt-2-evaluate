from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM
import torch
import math
import os

MILLION = 1000 * 1000

MODEL_NAME = os.environ["MODEL_NAME"]
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, return_dict=True)
model.to(device="cuda")

app = Flask(__name__)


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
    if request.is_json:
        content = request.get_json()
        print(content, type(content))
        return jsonify({"score": score(content)}), 200
    return jsonify({"score": -1}), 400


if __name__ == "__main__":
    app.run(debug=False, port=8001, host="0.0.0.0")
