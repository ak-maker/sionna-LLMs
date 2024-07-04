<table>
        <td style="padding: 0px 0px;">
            <a href=" https://colab.research.google.com/github/NVlabs/sionna/blob/main/examples/Hello_World.ipynb" style="vertical-align:text-bottom">
                <img alt="Colab logo" src="https://nvlabs.github.io/sionna/_static/colab_logo.svg" style="width: 40px; min-width: 40px">
            </a>
        </td>
        <td style="padding: 4px 0px;">
            <a href=" https://colab.research.google.com/github/nvlabs/sionna/blob/main/examples/Hello_World.ipynb" style="vertical-align:text-bottom">
                Run in Google Colab
            </a>
        </td>
        <td style="padding: 0px 15px;">
        </td>
        <td class="wy-breadcrumbs-aside" style="padding: 0 30px;">
            <a href="https://github.com/nvlabs/sionna/blob/main/examples/Hello_World.ipynb" style="vertical-align:text-bottom">
                <i class="fa fa-github" style="font-size:24px;"></i>
                View on GitHub
            </a>
        </td>
        <td class="wy-breadcrumbs-aside" style="padding: 0 35px;">
            <a href="../examples/Hello_World.ipynb" download target="_blank" style="vertical-align:text-bottom">
                <i class="fa fa-download" style="font-size:24px;"></i>
                Download notebook
            </a>
        </td>
    </table>

# “Hello, world!”<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Hello_World.html#“Hello,-world!”" title="Permalink to this headline"></a>
    
Import Sionna:

```python
[1]:
```

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

    
Let us first create a <a class="reference external" href="https://nvlabs.github.io/sionna/api/utils.html?highlight=binarysource#binarysource">BinarySource</a> to generate a random batch of bit vectors that we can map to constellation symbols:

```python
[2]:
```

```python
batch_size = 1000 # Number of symbols we want to generate
num_bits_per_symbol = 4 # 16-QAM has four bits per symbol
binary_source = sionna.utils.BinarySource()
b = binary_source([batch_size, num_bits_per_symbol])
b
```
```python
[2]:
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

    
Next, let us create a <a class="reference external" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation">Constellation</a> and visualize it:

```python
[3]:
```

```python
constellation = sionna.mapping.Constellation("qam", num_bits_per_symbol)
constellation.show();
```

<img alt="../_images/examples_Hello_World_6_0.png" src="https://nvlabs.github.io/sionna/_images/examples_Hello_World_6_0.png" />

    
We now need a <a class="reference external" href="https://nvlabs.github.io/sionna/api/mapping.html#mapper">Mapper</a> that maps each row of b to the constellation symbols according to the bit labeling shown above.

```python
[4]:
```

```python
mapper = sionna.mapping.Mapper(constellation=constellation)
x = mapper(b)
x[:10]
```
```python
[4]:
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

    
Let us now make things a bit more interesting a send our symbols over and <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html#sionna.channel.AWGN">AWGN channel</a>:

```python
[5]:
```

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

<img alt="../_images/examples_Hello_World_10_0.png" src="https://nvlabs.github.io/sionna/_images/examples_Hello_World_10_0.png" />

<script type="application/vnd.jupyter.widget-state+json">
{"state": {}, "version_major": 2, "version_minor": 0}
</script>