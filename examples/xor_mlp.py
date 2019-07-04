import numpy as np
from nnlib.core import Model
from nnlib.layers.core import FullyConnected
from nnlib.optimizers import SGD

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype="float32")
y = np.array([0, 1, 1, 0], dtype="float32").reshape(4, 1)

model = Model(
    FullyConnected(2, activation="sigmoid", input_shape=(2,), weight_initializer="xavier_uniform"),
    FullyConnected(1, activation="sigmoid", weight_initializer="xavier_uniform"),
    loss="binary_crossentropy",
    optimizer=SGD(lr=1, decay=1e-4)
)

model.train(x, y, epochs=200)
model.test(x, y, logging_level=1)
