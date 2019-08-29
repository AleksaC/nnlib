# nnlib 
[![license](https://img.shields.io/github/license/AleksaC/nnlib.svg?maxAge=2592000)](https://github.com/AleksaC/nnlib/blob/master/LICENSE)
[![Build Status](https://travis-ci.com/AleksaC/nnlib.svg?branch=master)](https://travis-ci.com/AleksaC/nnlib)
[![Coverage Status](https://coveralls.io/repos/github/AleksaC/nnlib/badge.svg?branch=master)](https://coveralls.io/github/AleksaC/nnlib?branch=master)

You just found nnlib - a minimalistic deep learning library built for
educational purposes.

## About

### Motivation 🤔
Richard Feynman once said: *'What I cannot create, I do not understand'* and in
general I agree with that statement especially when talking about programming.

So while I was learning deep learning I decided to implement some of the 
algorithms from scratch in numpy. Since there were a lot of common things
that could be shared between implementations of various algorithms I decided to
use some of the code I've written to create a small deep learning library with
an interface similar to Keras.

### Why call it nnlib? 🤷‍♂️ 
It's a library for building neural nets, hence nnlib... Unfortunately, naming things isn't
something I'm good at.

**Note:** The library is in early alpha version and is undergoing major changes.
I've been working on it for a long time with huge breaks so coding style and
documentation are inconsistent and I've only started writing tests recently.
I'm doing my best to clean it up.

## Installation

The development of the library was done using Python 3.6 but it should work 
for python 3.4 or higher. It is strongly recommended to use some sort of 
virtual environment, I prefer plain old virtualenv but you can use pipenv, 
conda environments or anything else you like. To install the dependencies run
`python -m pip install -r requirements.txt` or simply 
`python -m pip install numpy` as numpy is the only dependency of nnlib. 

### Using PyPI
The library hasn't been published to PyPI yet so you need to install it from
source.

### From source
```commandline
git clone https://github.com/AleksaC/nnlib.git
cd nnlib
python -m pip install .
```
If you want to make changes to the library it is best to use
[editable](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs)
install by running 
```commandline
python -m pip install -e .
```
That way all the changes to the code will immediately be reflected in the 
installed package.

## Running tests 👩‍🔬
To run test you need to have `pytest` installed. To run tests with coverage you
should install `pytest-cov` and run `pytest --cov=nnlib tests/`. To run tests
without coverage simpy run `pytest` while being in the root directory of the
project.

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

For more examples check out the [examples folder](https://github.com/AleksaC/nnlib/tree/master/examples)
of this repo.

## Contact 🙋‍♂️
If there's anything I can help you with you can find me at my personal [website](https://aleksac.me)
where you can contact me directly or via social media linked there. If you
liked this project you can follow me on twitter to stay up to date with my
latest projects.
<a target="_blank" href="http://twitter.com/aleksa_c_"><img alt='Twitter followers' src="https://img.shields.io/twitter/follow/aleksa_c_.svg?style=social"></a>
