from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import os


MODEL_NAME=os.environ['MODEL_NAME']
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, return_dict=True)

app = Flask(__name__)


def score(raw_text):
    encoded_text = tokenizer(raw_text, return_tensors='pt')
    output = model(**encoded_text, labels=encoded_text['input_ids'])
    return math.exp(output.loss)


@app.route('/evaluate', methods=['POST'])
def torch():
    if request.is_json:
        content = request.get_json()
        return jsonify({ "score": score(content['context'])}), 200 
    return jsonify({ "score": -1 }), 400


if __name__ == "__main__":
    app.run(debug=False, port=8001, host='0.0.0.0')
