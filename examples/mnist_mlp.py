from nnlib.datasets import mnist
from nnlib.core import Model
from nnlib.layers.core import FullyConnected
from nnlib.optimizers import SGD

model = Model(
    FullyConnected(256, activation="relu", input_shape=(784,), weight_initializer="he_normal"),
    FullyConnected(128, activation="relu", weight_initializer="he_normal"),
    FullyConnected(10 , weight_initializer="xavier_normal"),
    loss="softmax_crossentropy",
    optimizer=SGD(lr=4e-2, decay=2e-4)
)

model.summary()
model.train(*mnist.training_data(flat=True), batch_size=64, epochs=10)
model.test(*mnist.test_data(flat=True), batch_size=64, logging_level=1)
