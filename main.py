import numpy as np
from flask import Flask, jsonify, render_template, request

from model.NeuralNet import NeuralNet

neural_net = NeuralNet('augmented_data', 0.)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def convolutional(input):
    prediction = softmax(neural_net.predict(input*256)[0])
    print prediction
    return np.around(prediction, 3).tolist()



# webapp
app = Flask(__name__)


@app.route('/api/mnist', methods=['POST'])
def mnist():
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    output = convolutional(input)
    return jsonify(results=[output])


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
