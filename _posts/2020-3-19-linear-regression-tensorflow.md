---
type: post
title: Linear Regression in TensorFlow 2.x
category: Tensorflow 2.x
tags: [machine learning, tensorflow]
---

### Workflow of TensorFlow Models

TensorFlow 2.x came out a couple of years ago with earer execution workflow like PyTorch. Most of the development takes place during the summer by GSoC students. Some documentation pages are a bit blant. In this article, I have provided a common workflow for TensorFlow 2.x users and show how to implement a Linear Regression Model using the autodiff tool tf-2.x provides.

Following are the steps you may want to follow to train your models:

1. Getting data and using the ``tf.data`` containers.
2. Creating your model.
3. Creating a loss function.
4. Training your model.

### 1. Getting data and using the tf.data container

Just like TensorFlow 1.x, TensorFlow 2.x provides a data contaianer in the ``tf.data`` module tha can be used to efficiently load and work with data from within the workflow. Often, large datasets contain millions of samples, all of which can't be allocated in the main memory together. You need to load a batch of data in the main memory, work with them, store the results in secondary memory, load the next batch and so on. You can easily do that with ``tf.data.Dataset`` container. Below is a example that shows how to use ``tf.Data.Dataset`` to generate batches of data.

```python
import numpy as np
import tensorflow as tf

# A dummy dataset with 1000 samples and 2 features.
data = np.random.randn(1000, 2)

# Let's create a Dataset instance
dataset = tf.data.Dataset.from_tensor_slices(data)

# Load the data in the form of batches
batch_size = 10
batch_generator = dataset.batch(batch_size).__iter__()

# You can get the next batch of data using .get_next() method
batch_1 = batch_generator.get_next()
batch_2 = batch_generator.get_next()
...
batch_n = batch_generator.get_next()
# Once the generator iterates through all the data points,
# .get_next() method throws a ``OutOfRangeError: End of sequence``
# error message.

# Operate on the batch of data.
...

# Store the results back in your secondary memory.
...
```

As shown in the example, you can generate a batch using the ``.get_next()`` method. Once the generator iterates through all the data points, ``.get_next()`` method throws a ``OutOfRangeError: End of sequence`` error message. You can also use the ``batch_generator`` in a for loop to iterate through the whole dataset without worring about the ``OutOfRangeError`` error.

#### Vectorization

Batches of data give one more advantage of vectorization or, in common words, parallelization. This is achieved through the ``vectorized_map`` function introduced in TensorFlow 2.x. Citing from the tensorflow 2.x documentation pages:

> This method works similar to tf.map_fn but is optimized to run much faster, possibly with a much larger memory footprint. The speedups are obtained by vectorization (see [https://arxiv.org/pdf/1903.04243.pdf](https://arxiv.org/pdf/1903.04243.pdf)). The idea behind vectorization is to semantically launch all the invocations of fn in parallel and fuse corresponding operations across all these invocations. This fusion is done statically at graph generation time and the generated code is often similar in performance to a manually fused version.

Below example shows how to use ``vectorized_map`` to find the sum of a dataset in parallel.

```python
import numpy as np
import tensorflow as tf

# A dummy dataset with 1000 samples and 2 features.
data = np.random.randn(1000, 2)

batched_data = data.reshape(10, -1, 2)

# Sum all the elements in the tensor in parallel
result = tf.vectorized_map(lambda t: tf.reduce_sum(t), batched_data)

print(result)
```
