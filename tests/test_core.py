import numpy as np

from nnlib.datasets import mnist
from nnlib.core import Model
from nnlib.layers.core import Input, FullyConnected
from nnlib.optimizers import SGD


model = Model(
    Input((784,)),
    FullyConnected(512, activation="relu"),
    FullyConnected(10 , activation="linear"),
    loss="softmax_crossentropy",
    optimizer=SGD()
)


mock_input, mock_output = mnist.training_data(flat=True)
mock_input = np.array(mock_input[:100], dtype="float32")
mock_output = np.array(mock_output[:100], dtype="float32")


num_grad = model.numeric_grads(mock_input, mock_output, 1e-4)
grad = model._train_on_batch(mock_input, mock_output)

for layer in model.layers:
    print("Layer: ", layer.uuid)
    print("==================================")
    print("Weights: ")
    print("Backprop:")
    print(grad[layer.uuid][0][:5, :5])
    print("Numeric:")
    print(num_grad[layer.uuid][0])
    print("Biases: ")
    print("Backprop:")
    print(grad[layer.uuid][1][:5])
    print("Numeric:")
    print(num_grad[layer.uuid][1])
    print("===========================================")


def check_grad(mock_input, mock_output):
    grad = model._train_on_batch(mock_input, mock_output)
    """
    for gr, vals in grad.items():
        print("Grad w.r.t.", gr)
        print("=======================")
        print("w.r.t weights:")
        print(vals[0].shape)
        print("w.r.t biases")
        print(vals[1].shape)
        print("============================================")
    """
    #num_grads = numeric_grads(mock_input, mock_output)



def numeric_grads():
    pass


if __name__ == '__main__':
    #model.summary()
    #check_grad(mock_input, mock_output)
    pass
