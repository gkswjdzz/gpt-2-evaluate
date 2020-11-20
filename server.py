from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM
import torch
import math
import os


MODEL_NAME=os.environ['MODEL_NAME']
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, return_dict=True)
model.to(device='cuda')

app = Flask(__name__)


def score(encoded_text):
    encoded_text = torch.cuda.LongTensor(encoded_text)
    output = model(input_ids=encoded_text, labels=encoded_text)
    return math.exp(output.loss)


@app.route('/evaluate', methods=['POST'])
def evaluate():
    if request.is_json:
        content = request.get_json()
        print(content, type(content))
        return jsonify({"score": score(content)}), 200 
    return jsonify({"score": -1}), 400


if __name__ == "__main__":
    app.run(debug=False, port=8001, host='0.0.0.0')
