# Example Neural Network with Rust
This is an example Neural Network using the [TensorFlow bindings for Rust](https://github.com/tensorflow/rust). I wrote this as a learning lesson for myself while reading through the book [Neural Networks: A Visual Introduction For Beginners by Michael Taylor](https://www.amazon.com/Machine-Learning-Neural-Networks-depth-ebook/dp/B075882XCP/ref=sr_1_1?crid=80UXT3HUU7HZ&dchild=1&keywords=neural+networks%2C+a+visual+introduction+for+beginners&qid=1591340282&s=digital-text&sprefix=neural+networks+visual+%2Cdigital-text%2C237&sr=1-1), so it's basically a one-to-one translation from the Python code in that book into Rust. This example is part of [my blog posts on how to build a Neural Network with Rust](https://blog.robban.eu/tags/tensorflow/).

## Data set
The example expects MNIST data in CSV format, one for training and one for testing, where the first 784 values are the pixel values (0-255) and the last one is the label (0-9). I used [this one from from Kaggle](https://www.kaggle.com/oddrationale/mnist-in-csv). However, actually reading in and parsing the data is the only part of the example that is extracted into a separate function, so you can easily substitute this with your own.

## Some notes about the Rust code
* The code is intentionally verbose and has not been refactored or simplified. Expect a lot of repetition. The idea is to make it easier to follow along in a top-to-bottom fashion.
* This is a simple feed forward Neural Network with pretty much default settings - the example is not intended to build the highest accuracy network for this task (a CNN would be more accurate) but rather showcase how to use the TensorFlow bindings with Rust in a clear way.
* I am a machine learning hobbyist - the example above is tested and works but may contain bugs or logic errors.

_Thanks to Adam Crume and the contributors of the TensorFlow Rust crate, working on this in their spare time and always helpful in responding to questions and helping out!_
