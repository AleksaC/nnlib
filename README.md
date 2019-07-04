# NNLib [![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/AleksaC/dldidact/blob/master/LICENSE)

You just found nnlib - a minimalistic deep learning library built for
educational purposes.

## About

### Motivation
Richard Feynman once said: *'What I cannot create, I do not understand'*

And in general I agree with that statement especially when talking about
programming.

So while I was learning deep learning I decided to implement some of the 
algorithms from scratch in numpy. Since there were a lot of common things
that could be shared between implementations of various algorithms I decided to
use some of the code I've written to create a small deep learning library with
an interface similar to Keras.

### Why call it nnlib?
It's a library for building neural nets, hence nnlib... Unfortunately, naming things isn't
something I'm good at.

**Note:** The library is in early alpha version and is undergoing major changes.
I've been working on it for a long time with huge breaks so coding style and
documentation are inconsistent and I've only started writing tests recently.
I'm doing my best to clean it up.

## Installation

The developement of the library was done using Python 3.6 but it should work 
for any version after 3.4. The only dependency of nnlib is `numpy`.

#### Using PyPI
The library hasn't been published to PyPI yet so you need to install it from
source.

#### From source
* `git clone https://github.com/AleksaC/dldidact.git`
* `cd dldidact`
* `python -m pip install .`

## Getting started

To get a taste of nnlib take a look at this simple MLP that achieves 98.25%
test accuracy on MNIST classification in around a minute of training on a
machine with a decent CPU:

```python
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
model.test(*mnist.test_data(flat=True), logging_level=1)
```

For more examples check out the [examples folder](https://github.com/AleksaC/dldidact/tree/master/examples)
of this repo.

## Contact
If there's anything I can help you with you can find me at my personal [website](https://www.aleksac.me)
where you can contact me directly or via social media linked there. If you
liked this project you can follow me on twitter to stay up to date with my
latest projects.
<a target="_blank" href="http://twitter.com/aleksa_c_"><img alt='Twitter followers' src="https://img.shields.io/twitter/follow/aleksa_c_.svg?style=social"></a>
