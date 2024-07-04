# Part 2: Differentiable Communication Systems

This tutorial will guide you through Sionna, from its basic principles to the implementation of a point-to-point link with a 5G NR compliant code and a 3GPP channel model. You will also learn how to write custom trainable layers by implementing a state of the art neural receiver, and how to train and evaluate end-to-end communication systems.

The tutorial is structured in four notebooks:

- Part I: Getting started with Sionna
- **Part II: Differentiable Communication Systems**
- Part III: Advanced Link-level Simulations
- Part IV: Toward Learned Receivers


The [official documentation](https://nvlabs.github.io/sionna) provides key material on how to use Sionna and how its components are implemented.

## Imports


```python
import os
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Import Sionna
try:
    import sionna as sn
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna as sn
# Import TensorFlow and NumPy
import tensorflow as tf
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
import numpy as np
# For plotting
%matplotlib inline
import matplotlib.pyplot as plt
# For saving complex Python data structures efficiently
import pickle
# For the implementation of the neural receiver
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer
```

## Gradient Computation Through End-to-end Systems

Lets start by setting up a simple communication system that transmit bits modulated as QAM symbols over an AWGN channel.

However, compared to what we have previously done, we now make the constellation *trainable*. With Sionna, achieving this requires only setting a boolean parameter to `True` when instantiating the `Constellation` object.


```python
# Binary source to generate uniform i.i.d. bits
binary_source = sn.utils.BinarySource()
# 256-QAM constellation
NUM_BITS_PER_SYMBOL = 6
constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL, trainable=True) # The constellation is set to be trainable
# Mapper and demapper
mapper = sn.mapping.Mapper(constellation=constellation)
demapper = sn.mapping.Demapper("app", constellation=constellation)
# AWGN channel
awgn_channel = sn.channel.AWGN()
```


As we have already seen, we can now easily simulate forward passes through the system we have just setup


```python
BATCH_SIZE = 128 # How many examples are processed by Sionna in parallel
EBN0_DB = 17.0 # Eb/N0 in dB
no = sn.utils.ebnodb2no(ebno_db=EBN0_DB,
                        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                        coderate=1.0) # Coderate set to 1 as we do uncoded transmission here
bits = binary_source([BATCH_SIZE,
                        1200]) # Blocklength
x = mapper(bits)
y = awgn_channel([x, no])
llr = demapper([y,no])
```


Just for fun, lets visualize the channel inputs and outputs


```python
plt.figure(figsize=(8,8))
plt.axes().set_aspect(1.0)
plt.grid(True)
plt.scatter(tf.math.real(y), tf.math.imag(y), label='Output')
plt.scatter(tf.math.real(x), tf.math.imag(x), label='Input')
plt.legend(fontsize=20);
```


Lets now *optimize* the constellation through *stochastic gradient descent* (SGD). As we will see, this is made very easy by Sionna.

We need to define a *loss function* that we will aim to minimize.

We can see the task of the receiver as jointly solving, for each received symbol, `NUM_BITS_PER_SYMBOL` binary classification problems in order to reconstruct the transmitted bits. Therefore, a natural choice for the loss function is the *binary cross-entropy* (BCE) applied to each bit and to each received symbol.

*Remark:* The LLRs computed by the demapper are *logits* on the transmitted bits, and can therefore be used as-is to compute the BCE without any additional processing. *Remark 2:* The BCE is closely related to an achieveable information rate for bit-interleaved coded modulation systems [1,2]

[1] Georg Bcherer, Principles of Coded Modulation, [available online](http://www.georg-boecherer.de/bocherer2018principles.pdf)

[2] F. Ait Aoudia and J. Hoydis, End-to-End Learning for OFDM: From Neural Receivers to Pilotless Communication, in IEEE Transactions on Wireless Communications, vol.21, no. 2, pp.1049-1063, Feb.2022, doi: 10.1109/TWC.2021.3101364.


```python
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
print(f"BCE: {bce(bits, llr)}")
```


```python
BCE: 0.0001015052548609674
```


One iteration of SGD consists in three steps: 1. Perform a forward pass through the end-to-end system and compute the loss function 2. Compute the gradient of the loss function with respect to the trainable weights 3. Apply the gradient to the weights

To enable gradient computation, we need to perform the forward pass (step 1) within a `GradientTape`


```python
with tf.GradientTape() as tape:
    bits = binary_source([BATCH_SIZE,
                            1200]) # Blocklength
    x = mapper(bits)
    y = awgn_channel([x, no])
    llr = demapper([y,no])
    loss = bce(bits, llr)
```


Using the `GradientTape`, computing the gradient is done as follows


```python
gradient = tape.gradient(loss, tape.watched_variables())
```


`gradient` is a list of tensor, each tensor corresponding to a trainable variable of our model.

For this model, we only have a single trainable tensor: The constellation of shape [`2`, `2^NUM_BITS_PER_SYMBOL`], the first dimension corresponding to the real and imaginary components of the constellation points.

*Remark:* It is important to notice that the gradient computation was performed *through the demapper and channel*, which are conventional non-trainable algorithms implemented as *differentiable* Keras layers. This key feature of Sionna enables the training of end-to-end communication systems that combine both trainable and conventional and/or non-trainable signal processing algorithms.


```python
for g in gradient:
    print(g.shape)
```


```python
(2, 64)
```


Applying the gradient (third step) is performed using an *optimizer*. [Many optimizers are available as part of TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers), and we use in this notebook `Adam`.


```python
optimizer = tf.keras.optimizers.Adam(1e-2)
```


Using the optimizer, the gradients can be applied to the trainable weights to update them


```python
optimizer.apply_gradients(zip(gradient, tape.watched_variables()));
```


Let compare the constellation before and after the gradient application


```python
fig = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL).show()
fig.axes[0].scatter(tf.math.real(constellation.points), tf.math.imag(constellation.points), label='After SGD')
fig.axes[0].legend();
```


The SGD step has led to slight change in the position of the constellation points. Training of a communication system using SGD consists in looping over such SGD steps until a stop criterion is met.

## Creating Custom Layers

Custom trainable (or not trainable) algorithms should be implemented as [Keras layers](https://keras.io/api/layers/). All Sionna components, such as the mapper, demapper, channel are implemented as Keras layers.

To illustrate how this can be done, the next cell implements a simple neural network-based demapper which consists of three dense layers.


```python
class NeuralDemapper(Layer): # Inherits from Keras Layer
    def __init__(self):
        super().__init__()
        # The three dense layers that form the custom trainable neural network-based demapper
        self.dense_1 = Dense(64, 'relu')
        self.dense_2 = Dense(64, 'relu')
        self.dense_3 = Dense(NUM_BITS_PER_SYMBOL, None) # The last layer has no activation and therefore outputs logits, i.e., LLRs
    def call(self, y):
        # y : complex-valued with shape [batch size, block length]
        # y is first mapped to a real-valued tensor with shape
        #  [batch size, block length, 2]
        # where the last dimension consists of the real and imaginary components
        # The dense layers operate on the last dimension, and treat the inner dimensions as batch dimensions, i.e.,
        # all the received symbols are independently processed.
        nn_input = tf.stack([tf.math.real(y), tf.math.imag(y)], axis=-1)
        z = self.dense_1(nn_input)
        z = self.dense_2(z)
        z = self.dense_3(z) # [batch size, number of symbols per block, number of bits per symbol]
        llr = tf.reshape(z, [tf.shape(y)[0], -1]) # [batch size, number of bits per block]
        return llr
```


A custom Keras layer is used as any other Sionna layer, and therefore integration to a Sionna-based communication is straightforward.

The following model uses the neural demapper instead of the conventional demapper. It takes at initialization a parameter that indicates if the model is intantiated to be trained or evaluated. When instantiated to be trained, the loss function is returned. Otherwise, the transmitted bits and LLRs are returned.


```python
class End2EndSystem(Model): # Inherits from Keras Model
    def __init__(self, training):
        super().__init__() # Must call the Keras model initializer
        self.constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL, trainable=True) # Constellation is trainable
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = NeuralDemapper() # Intantiate the NeuralDemapper custom layer as any other
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True) # Loss function
        self.training = training
    @tf.function(jit_compile=True) # Enable graph execution to speed things up
    def __call__(self, batch_size, ebno_db):
        # no channel coding used; we set coderate=1.0
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                coderate=1.0)
        bits = self.binary_source([batch_size, 1200]) # Blocklength set to 1200 bits
        x = self.mapper(bits)
        y = self.awgn_channel([x, no])
        llr = self.demapper(y)  # Call the NeuralDemapper custom layer as any other
        if self.training:
            loss = self.bce(bits, llr)
            return loss
        else:
            return bits, llr
```


When a model that includes a neural network is created, the neural network weights are randomly initialized typically leading to very poor performance.

To see this, the following cell benchmarks the previously defined untrained model against a conventional baseline.


```python
EBN0_DB_MIN = 10.0
EBN0_DB_MAX = 20.0

###############################
# Baseline
###############################
class Baseline(Model): # Inherits from Keras Model
    def __init__(self):
        super().__init__() # Must call the Keras model initializer
        self.constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()
    @tf.function # Enable graph execution to speed things up
    def __call__(self, batch_size, ebno_db):
        # no channel coding used; we set coderate=1.0
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                coderate=1.0)
        bits = self.binary_source([batch_size, 1200]) # Blocklength set to 1200 bits
        x = self.mapper(bits)
        y = self.awgn_channel([x, no])
        llr = self.demapper([y,no])
        return bits, llr
###############################
# Benchmarking
###############################
baseline = Baseline()
model = End2EndSystem(False)
ber_plots = sn.utils.PlotBER("Neural Demapper")
ber_plots.simulate(baseline,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Baseline",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);
ber_plots.simulate(model,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Untrained model",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=True);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     10.0 | 2.6927e-02 | 1.0000e+00 |        4136 |      153600 |          128 |         128 |         0.7 |reached target block errors
   10.526 | 2.1426e-02 | 1.0000e+00 |        3291 |      153600 |          128 |         128 |         0.0 |reached target block errors
   11.053 | 1.6100e-02 | 1.0000e+00 |        2473 |      153600 |          128 |         128 |         0.0 |reached target block errors
   11.579 | 1.2051e-02 | 1.0000e+00 |        1851 |      153600 |          128 |         128 |         0.0 |reached target block errors
   12.105 | 9.1927e-03 | 1.0000e+00 |        1412 |      153600 |          128 |         128 |         0.0 |reached target block errors
   12.632 | 6.5234e-03 | 1.0000e+00 |        1002 |      153600 |          128 |         128 |         0.0 |reached target block errors
   13.158 | 4.4792e-03 | 9.8438e-01 |         688 |      153600 |          126 |         128 |         0.0 |reached target block errors
   13.684 | 2.7474e-03 | 9.6875e-01 |         422 |      153600 |          124 |         128 |         0.0 |reached target block errors
   14.211 | 1.6146e-03 | 8.8281e-01 |         248 |      153600 |          113 |         128 |         0.0 |reached target block errors
   14.737 | 9.9609e-04 | 7.0312e-01 |         306 |      307200 |          180 |         256 |         0.0 |reached target block errors
   15.263 | 5.2083e-04 | 4.7266e-01 |         160 |      307200 |          121 |         256 |         0.0 |reached target block errors
   15.789 | 3.4071e-04 | 3.3333e-01 |         157 |      460800 |          128 |         384 |         0.0 |reached target block errors
   16.316 | 1.4193e-04 | 1.5781e-01 |         109 |      768000 |          101 |         640 |         0.0 |reached target block errors
   16.842 | 6.0961e-05 | 7.1023e-02 |         103 |     1689600 |          100 |        1408 |         0.1 |reached target block errors
   17.368 | 2.4113e-05 | 2.8935e-02 |         100 |     4147200 |          100 |        3456 |         0.2 |reached target block errors
   17.895 | 7.6593e-06 | 9.1912e-03 |         100 |    13056000 |          100 |       10880 |         0.5 |reached target block errors
   18.421 | 2.7995e-06 | 3.3594e-03 |          43 |    15360000 |           43 |       12800 |         0.6 |reached max iter
   18.947 | 6.5104e-07 | 7.8125e-04 |          10 |    15360000 |           10 |       12800 |         0.6 |reached max iter
   19.474 | 6.5104e-08 | 7.8125e-05 |           1 |    15360000 |            1 |       12800 |         0.5 |reached max iter
     20.0 | 0.0000e+00 | 0.0000e+00 |           0 |    15360000 |            0 |       12800 |         0.5 |reached max iter
Simulation stopped as no error occurred @ EbNo = 20.0 dB.
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     10.0 | 4.7460e-01 | 1.0000e+00 |       72899 |      153600 |          128 |         128 |         1.3 |reached target block errors
   10.526 | 4.7907e-01 | 1.0000e+00 |       73585 |      153600 |          128 |         128 |         0.0 |reached target block errors
   11.053 | 4.7525e-01 | 1.0000e+00 |       72999 |      153600 |          128 |         128 |         0.0 |reached target block errors
   11.579 | 4.7865e-01 | 1.0000e+00 |       73521 |      153600 |          128 |         128 |         0.0 |reached target block errors
   12.105 | 4.7684e-01 | 1.0000e+00 |       73242 |      153600 |          128 |         128 |         0.0 |reached target block errors
   12.632 | 4.7469e-01 | 1.0000e+00 |       72913 |      153600 |          128 |         128 |         0.0 |reached target block errors
   13.158 | 4.7614e-01 | 1.0000e+00 |       73135 |      153600 |          128 |         128 |         0.0 |reached target block errors
   13.684 | 4.7701e-01 | 1.0000e+00 |       73268 |      153600 |          128 |         128 |         0.0 |reached target block errors
   14.211 | 4.7544e-01 | 1.0000e+00 |       73027 |      153600 |          128 |         128 |         0.0 |reached target block errors
   14.737 | 4.7319e-01 | 1.0000e+00 |       72682 |      153600 |          128 |         128 |         0.0 |reached target block errors
   15.263 | 4.7740e-01 | 1.0000e+00 |       73329 |      153600 |          128 |         128 |         0.0 |reached target block errors
   15.789 | 4.7385e-01 | 1.0000e+00 |       72783 |      153600 |          128 |         128 |         0.0 |reached target block errors
   16.316 | 4.7344e-01 | 1.0000e+00 |       72721 |      153600 |          128 |         128 |         0.0 |reached target block errors
   16.842 | 4.7303e-01 | 1.0000e+00 |       72658 |      153600 |          128 |         128 |         0.0 |reached target block errors
   17.368 | 4.7378e-01 | 1.0000e+00 |       72773 |      153600 |          128 |         128 |         0.0 |reached target block errors
   17.895 | 4.7257e-01 | 1.0000e+00 |       72586 |      153600 |          128 |         128 |         0.0 |reached target block errors
   18.421 | 4.7377e-01 | 1.0000e+00 |       72771 |      153600 |          128 |         128 |         0.0 |reached target block errors
   18.947 | 4.7315e-01 | 1.0000e+00 |       72676 |      153600 |          128 |         128 |         0.0 |reached target block errors
   19.474 | 4.7217e-01 | 1.0000e+00 |       72525 |      153600 |          128 |         128 |         0.0 |reached target block errors
     20.0 | 4.7120e-01 | 1.0000e+00 |       72376 |      153600 |          128 |         128 |         0.0 |reached target block errors
```

## Setting up Training Loops

Training of end-to-end communication systems consists in iterating over SGD steps.

The next cell implements a training loop of `NUM_TRAINING_ITERATIONS` iterations. The training SNR is set to $E_b/N_0 = 15$ dB.

At each iteration: - A forward pass through the end-to-end system is performed within a gradient tape - The gradients are computed using the gradient tape, and applied using the Adam optimizer - The estimated loss is periodically printed to follow the progress of training


```python
# Number of iterations used for training
NUM_TRAINING_ITERATIONS = 30000
# Set a seed for reproducibility
tf.random.set_seed(1)
# Instantiating the end-to-end model for training
model_train = End2EndSystem(training=True)
# Adam optimizer (SGD variant)
optimizer = tf.keras.optimizers.Adam()
# Training loop
for i in range(NUM_TRAINING_ITERATIONS):
    # Forward pass
    with tf.GradientTape() as tape:
        loss = model_train(BATCH_SIZE, 15.0) # The model is assumed to return the BMD rate
    # Computing and applying gradients
    grads = tape.gradient(loss, model_train.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_train.trainable_weights))
    # Print progress
    if i % 100 == 0:
        print(f"{i}/{NUM_TRAINING_ITERATIONS}  Loss: {loss:.2E}", end="\r")
```


```python
29900/30000  Loss: 2.02E-03
```


The weights of the trained model are saved using [pickle](https://docs.python.org/3/library/pickle.html).


```python
# Save the weightsin a file
weights = model_train.get_weights()
with open('weights-neural-demapper', 'wb') as f:
    pickle.dump(weights, f)
```


Finally, we evaluate the trained model and benchmark it against the previously introduced baseline.

We first instantiate the model for evaluation and load the saved weights.


```python
# Instantiating the end-to-end model for evaluation
model = End2EndSystem(training=False)
# Run one inference to build the layers and loading the weights
model(tf.constant(1, tf.int32), tf.constant(10.0, tf.float32))
with open('weights-neural-demapper', 'rb') as f:
    weights = pickle.load(f)
    model.set_weights(weights)
```


The trained model is then evaluated.


```python
# Computing and plotting BER
ber_plots.simulate(model,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100,
                  legend="Trained model",
                  soft_estimates=True,
                  max_mc_iter=100,
                  show_fig=True);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     10.0 | 2.6094e-02 | 1.0000e+00 |        4008 |      153600 |          128 |         128 |         0.4 |reached target block errors
   10.526 | 2.0768e-02 | 1.0000e+00 |        3190 |      153600 |          128 |         128 |         0.0 |reached target block errors
   11.053 | 1.5729e-02 | 1.0000e+00 |        2416 |      153600 |          128 |         128 |         0.0 |reached target block errors
   11.579 | 1.1667e-02 | 1.0000e+00 |        1792 |      153600 |          128 |         128 |         0.0 |reached target block errors
   12.105 | 8.3789e-03 | 1.0000e+00 |        1287 |      153600 |          128 |         128 |         0.0 |reached target block errors
   12.632 | 6.1458e-03 | 1.0000e+00 |         944 |      153600 |          128 |         128 |         0.0 |reached target block errors
   13.158 | 3.8411e-03 | 9.7656e-01 |         590 |      153600 |          125 |         128 |         0.0 |reached target block errors
   13.684 | 2.8971e-03 | 9.7656e-01 |         445 |      153600 |          125 |         128 |         0.0 |reached target block errors
   14.211 | 1.6602e-03 | 8.4375e-01 |         255 |      153600 |          108 |         128 |         0.0 |reached target block errors
   14.737 | 9.8958e-04 | 6.7578e-01 |         304 |      307200 |          173 |         256 |         0.0 |reached target block errors
   15.263 | 5.0130e-04 | 4.7656e-01 |         154 |      307200 |          122 |         256 |         0.0 |reached target block errors
   15.789 | 2.5228e-04 | 2.6367e-01 |         155 |      614400 |          135 |         512 |         0.0 |reached target block errors
   16.316 | 1.4453e-04 | 1.6250e-01 |         111 |      768000 |          104 |         640 |         0.0 |reached target block errors
   16.842 | 5.2548e-05 | 5.8594e-02 |         113 |     2150400 |          105 |        1792 |         0.1 |reached target block errors
   17.368 | 2.7083e-05 | 3.1875e-02 |         104 |     3840000 |          102 |        3200 |         0.1 |reached target block errors
   17.895 | 8.6520e-06 | 1.0382e-02 |         101 |    11673600 |          101 |        9728 |         0.3 |reached target block errors
   18.421 | 2.7344e-06 | 3.2812e-03 |          42 |    15360000 |           42 |       12800 |         0.4 |reached max iter
   18.947 | 8.4635e-07 | 1.0156e-03 |          13 |    15360000 |           13 |       12800 |         0.4 |reached max iter
   19.474 | 1.3021e-07 | 1.5625e-04 |           2 |    15360000 |            2 |       12800 |         0.4 |reached max iter
     20.0 | 0.0000e+00 | 0.0000e+00 |           0 |    15360000 |            0 |       12800 |         0.4 |reached max iter
Simulation stopped as no error occurred @ EbNo = 20.0 dB.

```



