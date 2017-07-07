import numpy as np
from flask import Flask, jsonify, render_template, request

from model.NeuralNet import NeuralNet

neural_nets = [
    (NeuralNet('model/checkpoints/no-reg', 0.), 256.),
    (NeuralNet('model/checkpoints/0.02', 0.02), 1.),
    (NeuralNet('model/checkpoints/lamb29', 0.02), 1.),
]
# neural_net = NeuralNet('model/checkpoints/augmented_data', 0.)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def convolutional(input, ensemble):
    if np.sum(input) == 0:
        return [0] * 26, [' '] * len(ensemble)

    probabilities = []
    predictions = []
    for i in range(len(ensemble)):
        if ensemble[i]:
            pred = softmax(neural_nets[i][0].predict(input * neural_nets[i][1])[0])
            probabilities.append(pred)
            predictions.append(chr(ord('A') + np.argmax(pred)))
        else:
            predictions.append(' ')

    if probabilities:
        probability = np.around(sum(probabilities) / len(probabilities), 3).tolist()
    else:
        probability = [0] * 26
    return probability, predictions



# webapp
app = Flask(__name__)


@app.route('/api/query', methods=['POST'])
def query():

    req = request.json
    inp = ((255 - np.array(req['inputs'], dtype=np.uint8)) / 255.0).reshape(1, 784)
    probability, predictions = convolutional(inp, req['ensemble'])
    print predictions
    return jsonify(probability=probability, predictions=predictions)


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
