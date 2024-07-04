# Hello, world!

Import Sionna:


```python
import os
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Import Sionna
try:
    import sionna
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna
# IPython "magic function" for inline plots
%matplotlib inline
import matplotlib.pyplot as plt
```


Let us first create a [BinarySource](https://nvlabs.github.io/sionna/api/utils.html?highlight=binarysource#binarysource) to generate a random batch of bit vectors that we can map to constellation symbols:


```python
batch_size = 1000 # Number of symbols we want to generate
num_bits_per_symbol = 4 # 16-QAM has four bits per symbol
binary_source = sionna.utils.BinarySource()
b = binary_source([batch_size, num_bits_per_symbol])
b
```
```python
<tf.Tensor: shape=(1000, 4), dtype=float32, numpy=
array([[1., 0., 1., 0.],
       [0., 1., 1., 1.],
       [0., 1., 0., 0.],
       ...,
       [1., 0., 1., 0.],
       [1., 1., 0., 0.],
       [0., 1., 0., 1.]], dtype=float32)>
```


Next, let us create a [Constellation](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation) and visualize it:


```python
constellation = sionna.mapping.Constellation("qam", num_bits_per_symbol)
constellation.show();
```


We now need a [Mapper](https://nvlabs.github.io/sionna/api/mapping.html#mapper) that maps each row of b to the constellation symbols according to the bit labeling shown above.


```python
mapper = sionna.mapping.Mapper(constellation=constellation)
x = mapper(b)
x[:10]
```
```python
<tf.Tensor: shape=(10, 1), dtype=complex64, numpy=
array([[-0.9486833+0.3162278j],
       [ 0.9486833-0.9486833j],
       [ 0.3162278-0.3162278j],
       [-0.3162278-0.3162278j],
       [ 0.9486833-0.3162278j],
       [-0.3162278+0.3162278j],
       [ 0.3162278-0.3162278j],
       [-0.9486833-0.9486833j],
       [ 0.9486833+0.3162278j],
       [ 0.9486833+0.9486833j]], dtype=complex64)>
```


Let us now make things a bit more interesting a send our symbols over and [AWGN channel](https://nvlabs.github.io/sionna/api/channel.html#sionna.channel.AWGN):


```python
awgn = sionna.channel.AWGN()
ebno_db = 15 # Desired Eb/No in dB
no = sionna.utils.ebnodb2no(ebno_db, num_bits_per_symbol, coderate=1)
y = awgn([x, no])
# Visualize the received signal
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
plt.scatter(np.real(y), np.imag(y));
ax.set_aspect("equal", adjustable="box")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.grid(True, which="both", axis="both")
plt.title("Received Symbols");
```



