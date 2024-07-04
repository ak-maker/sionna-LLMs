# Channel Models from Datasets

In this notebook, you will learn how to create a channel model from a [generator](https://wiki.python.org/moin/Generators). This can be used, e.g., to import datasets of channel impulse responses.
## GPU Configuration and Imports


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
import numpy as np
import h5py
```

```python
# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
```

## Simulation Parameters


```python
num_rx = 2
num_rx_ant = 2
num_tx = 1
num_tx_ant = 8
num_time_steps = 100
num_paths = 10
```

## Creating a Simple Dataset

To illustrate how to load dataset, we will first create one.

The next cell creates a very small HDF5 file storing Gaussian distributed i.i.d. path coefficients and uniformly distributed i.i.d. path delays.


```python
# Number of examples in the dataset
dataset_size = 1000
# Random path coefficients
a_shape = [dataset_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
a = (np.random.normal(size=a_shape) + 1j*np.random.normal(size=a_shape))/np.sqrt(2)
# Random path delays
tau = np.random.uniform(size=[dataset_size, num_rx, num_tx, num_paths])
```

```python
filename = 'my_dataset.h5'
hf = h5py.File(filename, 'w')
hf.create_dataset('a', data=a)
hf.create_dataset('tau', data=tau)
hf.close()
```

## Generators

The first step to load a dataset is to create a [generator](https://wiki.python.org/moin/Generators). A generator is a callable object, i.e., a function or a class that implements the `__call__()` method, and that behaves like an iterator.

The next cell shows how to create a generator that parses an HDF5 file storing path coefficients and delays. Note that how the HDF5 file is parsed depends on its structure. The following generator is specific to the dataset previously created.

If you have another dataset, you will need to change the way it is parsed in the generator. The generator can also carry out any type of desired pre-processing of your data, e.g., normalization.


```python
class HD5CIRGen:
    def __init__(self, filename):
        self.filename = filename
    def __call__(self):
        with h5py.File(self.filename, 'r') as hf:
            for im in zip(hf["a"], hf["tau"]):
                a = im[0]
                tau = im[1]
                # One could do some preprocessing on the dataset here
                # ...
                yield im
```

```python
generator = HD5CIRGen(filename)
```


We can use the generator to sample the first 5 items of the dataset:


```python
i = 0
for (a,tau) in generator():
    print(a.shape)
    print(tau.shape)
    i = i + 1
    if i == 5:
        break
```


```python
(2, 2, 1, 8, 10, 100)
(2, 1, 10)
(2, 2, 1, 8, 10, 100)
(2, 1, 10)
(2, 2, 1, 8, 10, 100)
(2, 1, 10)
(2, 2, 1, 8, 10, 100)
(2, 1, 10)
(2, 2, 1, 8, 10, 100)
(2, 1, 10)
```


Let us create a channel model from this dataset:


```python
batch_size = 64 # The batch_size cannot be changed after the creation of the channel model
channel_model = sn.channel.CIRDataset(generator,
                                      batch_size,
                                      num_rx,
                                      num_rx_ant,
                                      num_tx,
                                      num_tx_ant,
                                      num_paths,
                                      num_time_steps)
```


We can now sample from this dataset in the same way as we would from a stochastic channel model:


```python
# Note that the arguments batch_size, num_time_steps, and smapling_frequency
# of the __call__ function are ignored as they are already specified by the dataset.
a, tau = channel_model()
```

```python
print(a.shape)
print(a.dtype)
print(tau.shape)
print(tau.dtype)
```


```python
(64, 2, 2, 1, 8, 10, 100)
<dtype: 'complex64'>
(64, 2, 1, 10)
<dtype: 'float32'>
```
## Use the Channel Model for OFDM Transmissions

The following code demonstrates how you can use the channel model to generate channel frequency responses that can be used for the simulation of communication system based on OFDM.


```python
# Create an OFDM resource grid
# Each time step is assumed to correspond to one OFDM symbol over which it is constant.
resource_grid = sn.ofdm.ResourceGrid(num_ofdm_symbols=num_time_steps,
                                     fft_size=76,
                                     subcarrier_spacing=15e3,
                                     num_tx=num_tx,
                                     num_streams_per_tx=num_tx_ant)
```

```python
ofdm_channel = sn.channel.GenerateOFDMChannel(channel_model, resource_grid)
```

```python
# Generate a batch of frequency responses
# Shape: [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
h_freq = ofdm_channel()
print(h_freq.shape)
```


```python
(64, 2, 2, 1, 8, 100, 76)
```


