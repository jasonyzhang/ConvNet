import numpy as np
from flask import Flask, jsonify, render_template, request

from model.NeuralNet import NeuralNet

neural_net = NeuralNet('no-reg', 0.)

def regression(input):
    return [0] * 10


def convolutional(input):
    print repr(input)
    print np.argmax(neural_net.predict(input*256), 1)
    return [0] * 10


# webapp
app = Flask(__name__)


@app.route('/api/mnist', methods=['POST'])
def mnist():
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    output1 = regression(input)
    output2 = convolutional(input)
    return jsonify(results=[output1, output2])


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()