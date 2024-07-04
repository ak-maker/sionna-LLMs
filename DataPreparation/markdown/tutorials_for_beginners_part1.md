<table>
        <td style="padding: 0px 0px;">
            <a href=" https://colab.research.google.com/github/NVlabs/sionna/blob/main/examples/Sionna_tutorial_part1.ipynb" style="vertical-align:text-bottom">
                <img alt="Colab logo" src="https://nvlabs.github.io/sionna/_static/colab_logo.svg" style="width: 40px; min-width: 40px">
            </a>
        </td>
        <td style="padding: 4px 0px;">
            <a href=" https://colab.research.google.com/github/nvlabs/sionna/blob/main/examples/Sionna_tutorial_part1.ipynb" style="vertical-align:text-bottom">
                Run in Google Colab
            </a>
        </td>
        <td style="padding: 0px 15px;">
        </td>
        <td class="wy-breadcrumbs-aside" style="padding: 0 30px;">
            <a href="https://github.com/nvlabs/sionna/blob/main/examples/Sionna_tutorial_part1.ipynb" style="vertical-align:text-bottom">
                <i class="fa fa-github" style="font-size:24px;"></i>
                View on GitHub
            </a>
        </td>
        <td class="wy-breadcrumbs-aside" style="padding: 0 35px;">
            <a href="../examples/Sionna_tutorial_part1.ipynb" download target="_blank" style="vertical-align:text-bottom">
                <i class="fa fa-download" style="font-size:24px;"></i>
                Download notebook
            </a>
        </td>
    </table>

# Part 1: Getting Started with Sionna<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html#Part-1:-Getting-Started-with-Sionna" title="Permalink to this headline"></a>
    
This tutorial will guide you through Sionna, from its basic principles to the implementation of a point-to-point link with a 5G NR compliant code and a 3GPP channel model. You will also learn how to write custom trainable layers by implementing a state of the art neural receiver, and how to train and evaluate end-to-end communication systems.
    
The tutorial is structured in four notebooks:
 
- **Part I: Getting started with Sionna**
- Part II: Differentiable Communication Systems
- Part III: Advanced Link-level Simulations
- Part IV: Toward Learned Receivers

    
The <a class="reference external" href="https://nvlabs.github.io/sionna">official documentation</a> provides key material on how to use Sionna and how its components are implemented.

## Table of Contents
- <a class="reference external" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html#Imports-&-Basics">Imports & Basics</a>
- <a class="reference external" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html#Sionna-Data-flow-and-Design-Paradigms">Sionna Data-flow and Design Paradigms</a>
- <a class="reference external" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html#Hello,-Sionna!">Hello, Sionna!</a>
- <a class="reference external" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html#Communication-Systems-as-Keras-Models">Communication Systems as Keras Models</a>
- <a class="reference external" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html#Forward-Error-Correction-(FEC)">Forward Error Correction (FEC)</a>
- <a class="reference external" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html#Eager-vs-Graph-Mode">Eager vs Graph Mode</a>
- <a class="reference external" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html#Exercise">Exercise</a>
## Imports & Basics<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html#Imports-&-Basics" title="Permalink to this headline"></a>

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
# also try %matplotlib widget
import matplotlib.pyplot as plt
# for performance measurements
import time
# For the implementation of the Keras models
from tensorflow.keras import Model
```

    
We can now access Sionna functions within the `sn` namespace.
    
**Hint**: In Jupyter notebooks, you can run bash commands with `!`.

```python
[2]:
```

```python
!nvidia-smi
```


```python
Tue Mar 15 14:47:45 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.103.01   Driver Version: 470.103.01   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
| 30%   51C    P8    23W / 350W |     53MiB / 24265MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce ...  Off  | 00000000:4C:00.0 Off |                  N/A |
|  0%   33C    P8    24W / 350W |      8MiB / 24268MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```
## Sionna Data-flow and Design Paradigms<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html#Sionna-Data-flow-and-Design-Paradigms" title="Permalink to this headline"></a>
    
Sionna inherently parallelizes simulations via <em>batching</em>, i.e., each element in the batch dimension is simulated independently.
    
This means the first tensor dimension is always used for <em>inter-frame</em> parallelization similar to an outer <em>for-loop</em> in Matlab/NumPy simulations, but operations can be operated in parallel.
    
To keep the dataflow efficient, Sionna follows a few simple design principles:
 
- Signal-processing components are implemented as an individual <a class="reference external" href="https://keras.io/api/layers/">Keras layer</a>.
- `tf.float32` is used as preferred datatype and `tf.complex64` for complex-valued datatypes, respectively.
This allows simpler re-use of components (e.g., the same scrambling layer can be used for binary inputs and LLR-values).
- `tf.float64`/`tf.complex128` are available when high precision is needed.
- Models can be developed in <em>eager mode</em> allowing simple (and fast) modification of system parameters.
- Number crunching simulations can be executed in the faster <em>graph mode</em> or even <em>XLA</em> acceleration (experimental) is available for most components.
- Whenever possible, components are automatically differentiable via <a class="reference external" href="https://www.tensorflow.org/guide/autodiff">auto-grad</a> to simplify the deep learning design-flow.
- Code is structured into sub-packages for different tasks such as channel coding, mapping,… (see <a class="reference external" href="http://nvlabs.github.io/sionna/api/sionna.html">API documentation</a> for details).

    
These paradigms simplify the re-useability and reliability of our components for a wide range of communications related applications.

## Hello, Sionna!<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html#Hello,-Sionna!" title="Permalink to this headline"></a>
    
Let’s start with a very simple simulation: Transmitting QAM symbols over an AWGN channel. We will implement the system shown in the figure below.
    
<img alt="QAM AWGN" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAa8AAAC/CAYAAABJylMuAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAaGVYSWZNTQAqAAAACAAEAQYAAwAAAAEAAgAAARIAAwAAAAEAAQAAASgAAwAAAAEAAgAAh2kABAAAAAEAAAA+AAAAAAADoAEAAwAAAAEAAQAAoAIABAAAAAEAAAGvoAMABAAAAAEAAAC/AAAAAO5zGdUAAALkaVRYdFhNTDpjb20uYWRvYmUueG1wAAAAAAA8eDp4bXBtZXRhIHhtbG5zOng9ImFkb2JlOm5zOm1ldGEvIiB4OnhtcHRrPSJYTVAgQ29yZSA2LjAuMCI+CiAgIDxyZGY6UkRGIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI+CiAgICAgIDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiCiAgICAgICAgICAgIHhtbG5zOnRpZmY9Imh0dHA6Ly9ucy5hZG9iZS5jb20vdGlmZi8xLjAvIgogICAgICAgICAgICB4bWxuczpleGlmPSJodHRwOi8vbnMuYWRvYmUuY29tL2V4aWYvMS4wLyI+CiAgICAgICAgIDx0aWZmOkNvbXByZXNzaW9uPjE8L3RpZmY6Q29tcHJlc3Npb24+CiAgICAgICAgIDx0aWZmOlJlc29sdXRpb25Vbml0PjI8L3RpZmY6UmVzb2x1dGlvblVuaXQ+CiAgICAgICAgIDx0aWZmOk9yaWVudGF0aW9uPjE8L3RpZmY6T3JpZW50YXRpb24+CiAgICAgICAgIDx0aWZmOlBob3RvbWV0cmljSW50ZXJwcmV0YXRpb24+MjwvdGlmZjpQaG90b21ldHJpY0ludGVycHJldGF0aW9uPgogICAgICAgICA8ZXhpZjpQaXhlbFhEaW1lbnNpb24+NDMxPC9leGlmOlBpeGVsWERpbWVuc2lvbj4KICAgICAgICAgPGV4aWY6Q29sb3JTcGFjZT4xPC9leGlmOkNvbG9yU3BhY2U+CiAgICAgICAgIDxleGlmOlBpeGVsWURpbWVuc2lvbj4xOTE8L2V4aWY6UGl4ZWxZRGltZW5zaW9uPgogICAgICA8L3JkZjpEZXNjcmlwdGlvbj4KICAgPC9yZGY6UkRGPgo8L3g6eG1wbWV0YT4K3524XQAANrxJREFUeAHtnQe4VNX19tct9N6LghQrWLCjYkDFGn2woYio2DUxsURiHo0t1vjPJxJrYk80GjV2E6MRQVExNuwNRZAiSu/lcu93fpvs6zDM3Dsz90w5M+/mGc7cM+fss8+7y7vW2muvXVYTJFMSAkJACAgBIRAhBMojVFYVVQgIASEgBISAQ0DkpYYgBISAEBACkUNA5BW5KlOBhYAQEAJCQOSlNiAEhIAQEAKRQ0DkFbkqU4GFgBAQAkJA5KU2IASEgBAQApFDQOQVuSpTgYWAEBACQkDkpTYgBISAEBACkUNA5BW5KlOBhYAQEAJCQOSlNiAEhIAQEAKRQ0DkFbkqU4GFgBAQAkJA5KU2IASEgBAQApFDQOQVuSpTgYWAEBACQkDkpTYgBISAEBACkUNA5BW5KlOBhYAQEAJCQOSlNiAEhIAQEAKRQ0DkFbkqU4GFgBAQAkJA5KU2IASEgBAQApFDQOQVuSpTgYWAEBACQkDkpTYgBISAEBACkUNA5BW5KlOBhYAQEAJCQOSlNiAEhIAQEAKRQ0DkFbkqU4GFgBAQAkJA5KU2IASEgBAQApFDQOQVuSpTgYWAEBACQkDkpTYgBISAEBACkUNA5BW5KlOBhYAQEAJCQOSlNiAEhIAQEAKRQ0DkFbkqU4GFgBAQAkJA5KU2IASEgBAQApFDQOQVuSpTgYWAEBACQkDkpTYgBISAEBACkUNA5BW5KlOBhYAQEAJCQOSlNiAEhIAQEAKRQ0DkFbkqU4GFgBAQAkKgUhAIASGQHgKTJ082/+FOvislR+CUU06x1q1b24477mgDBw60Jk2aJL9YvwiBFBEoqwlSitfqMiFQ8giMGDHCVq1aZQMGDHCDMYDwXSk5Ao899pgtXbrUHnzwQRs1apSdf/75jsyS36FfhED9CIi86sdIVwgBp11BXMcdd5ydd955Tnto2rSpQ8YfBVNiBJYsWWLV1dXG8dFHH7V7773XrrvuOjvggAOkhSWGTGdTQEDklQJIukQIdO3a1caNG2cHHnigtW3bVoBkiAAE9vLLL9vFF19szz//vPXo0SPDnHRbqSMg8ir1FqD3rxeBm266yaZPn+60BWlZ9cJV7wWrV6+266+/3r788ksbO3asderUqd57dIEQiEdA3obxiOhvIRCDAMSFhnD55ZebiCsGmAZ8xWGDea81a9bYO++8Y5CZkhBIFwGRV7qI6fqSQmDRokXOMUOmwnCrHe/DnXfe2d577z3nABNu7sqtFBAQeZVCLesdM0YAN/jBgwdnfL9uTI4AbvMir+T46Je6ERB51Y2Pfi1xBN588023NqnEYcjK67Pu65NPPnHmw6w8QJkWNQIir6KuXr1cQxHAbCiTYUNRTHw/psNly5Y5N/rEV+isEEiOgMgrOTb6RQgIASEgBAoUAZFXgVaMiiUEhIAQEALJERB5JcdGvwgBISAEhECBIiDyKtCKUbGEgBAQAkIgOQIir+TY6BchUHAIsKB38eLFCcu1bt06LfhNiIxOFiMCIq9irFW9U9EhQDSKG2+80Q499FD3Of744110CgLe+vTaa6+5oLdc69Ptt99ukyZNMoiNdM8999iECRNq//bX6SgEooaAyCtqNabyliQCDz30kLG1CKR12mmnORfzCy+80ObPn1+LB1uO3HnnnbZy5cracy+99JI9+eSTtnbtWnfukUcesVdffdWqqqpqr9EXIRBFBEReUaw1lbnkELjjjjtsp512smHDhtnw4cNdVPYPPvjAXnnlFUdEs2bNcqQ0d+5ce/fdd51mxVZ9H330kSMv9iDjmvfff9/efvttaV4l14KK74VFXsVXp3qjIkOAOa4vvvjC9t57b2vVqpU1b97cEdkmm2ziIlRgEnzqqadsyy23tH333deRFabDOXPm2IIFC5zWNWXKFBs/frxbcA2BSfMqskZSgq8j8irBStcrRwsByAmyadasWW3BGzVqZJWVlY6YMAliMjzssMPsyCOPdFHwFy5caJ9++ql16dLFDj74YHvxxRftmWeesZEjR7pdjSE2baJeC6e+RBABkVcEK01FLi0ECKNEiCpMgN7xYt68ec4M2K1bN3vjjTecBvb55587s+DMmTOdUwbmw6222sqZGZ977jl33RFHHGF9+/at1dhKC0m9bTEhIPIqptrUuxQlAmhYaFVPP/20TZ061diN+K677jJ2dx46dKjhhEGQW3Yl5hzmRZw7IK+tt97amRjZQ2vzzTe3zTbbzLbbbjtjvswTYVGCppcqegQqi/4N9YJCoAgQOPfcc+23v/2tnX766VZeXm7Tpk2z3/3ud4b5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" />
    
We will use upper case for naming simulation parameters that are used throughout this notebook
    
Every layer needs to be initialized once before it can be used.
    
**Tip**: Use the <a class="reference external" href="http://nvlabs.github.io/sionna/api/sionna.html">API documentation</a> to find an overview of all existing components. You can directly access the signature and the docstring within jupyter via `Shift+TAB`.
    
<em>Remark</em>: Most layers are defined to be complex-valued.
    
We first need to create a QAM constellation.

```python
[3]:
```

```python
NUM_BITS_PER_SYMBOL = 2 # QPSK
constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
constellation.show();
```

<img alt="../_images/examples_Sionna_tutorial_part1_13_0.png" src="https://nvlabs.github.io/sionna/_images/examples_Sionna_tutorial_part1_13_0.png" />

    
**Task:** Try to change the modulation order, e.g., to 16-QAM.
    
We then need to setup a mapper to map bits into constellation points. The mapper takes as parameter the constellation.
    
We also need to setup a corresponding demapper to compute log-likelihood ratios (LLRs) from received noisy samples.

```python
[4]:
```

```python
mapper = sn.mapping.Mapper(constellation=constellation)
# The demapper uses the same constellation object as the mapper
demapper = sn.mapping.Demapper("app", constellation=constellation)
```

    
**Tip**: You can access the signature+docstring via `?` command and print the complete class definition via `??` operator.
    
Obviously, you can also access the source code via <a class="reference external" href="https://github.com/nvlabs/sionna/">https://github.com/nvlabs/sionna/</a>.

```python
[5]:
```

```python
# print class definition of the Constellation class
sn.mapping.Mapper??
```


```python
Init signature: sn.mapping.Mapper(*args, **kwargs)
Source:
class Mapper(Layer):
    # pylint: disable=line-too-long
    r&#34;&#34;&#34;
    Mapper(constellation_type=None, num_bits_per_symbol=None, constellation=None, dtype=tf.complex64, **kwargs)
    Maps binary tensors to points of a constellation.
    This class defines a layer that maps a tensor of binary values
    to a tensor of points from a provided constellation.
    Parameters
    ----------
    constellation_type : One of [&#34;qam&#34;, &#34;pam&#34;, &#34;custom&#34;], str
        For &#34;custom&#34;, an instance of :class:`~sionna.mapping.Constellation`
        must be provided.
    num_bits_per_symbol : int
        The number of bits per constellation symbol, e.g., 4 for QAM16.
        Only required for ``constellation_type`` in [&#34;qam&#34;, &#34;pam&#34;].
    constellation :  Constellation
        An instance of :class:`~sionna.mapping.Constellation` or
        `None`. In the latter case, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.
    dtype : One of [tf.complex64, tf.complex128], tf.DType
        The output dtype. Defaults to tf.complex64.
    Input
    -----
    : [..., n], tf.float or tf.int
        Tensor with with binary entries.
    Output
    ------
    : [...,n/Constellation.num_bits_per_symbol], tf.complex
        The mapped constellation symbols.
    Note
    ----
    The last input dimension must be an integer multiple of the
    number of bits per constellation symbol.
    &#34;&#34;&#34;
    def __init__(self,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 dtype=tf.complex64,
                 **kwargs
                ):
        super().__init__(dtype=dtype, **kwargs)
        assert dtype in [tf.complex64, tf.complex128],\
            &#34;dtype must be tf.complex64 or tf.complex128&#34;
        #self._dtype = dtype
        #print(dtype, self._dtype)
        if constellation is not None:
            assert constellation_type in [None, &#34;custom&#34;], \
                &#34;&#34;&#34;`constellation_type` must be &#34;custom&#34;.&#34;&#34;&#34;
            assert num_bits_per_symbol in \
                     [None, constellation.num_bits_per_symbol], \
                &#34;&#34;&#34;`Wrong value of `num_bits_per_symbol.`&#34;&#34;&#34;
            self._constellation = constellation
        else:
            assert constellation_type in [&#34;qam&#34;, &#34;pam&#34;], \
                &#34;Wrong constellation type.&#34;
            assert num_bits_per_symbol is not None, \
                &#34;`num_bits_per_symbol` must be provided.&#34;
            self._constellation = Constellation(constellation_type,
                                                num_bits_per_symbol,
                                                dtype=self._dtype)
        self._binary_base = 2**tf.constant(
                        range(self.constellation.num_bits_per_symbol-1,-1,-1))
    @property
    def constellation(self):
        &#34;&#34;&#34;The Constellation used by the Mapper.&#34;&#34;&#34;
        return self._constellation
    def call(self, inputs):
        tf.debugging.assert_greater_equal(tf.rank(inputs), 2,
            message=&#34;The input must have at least rank 2&#34;)
        # Reshape inputs to the desired format
        new_shape = [-1] + inputs.shape[1:-1].as_list() + \
           [int(inputs.shape[-1] / self.constellation.num_bits_per_symbol),
            self.constellation.num_bits_per_symbol]
        inputs_reshaped = tf.cast(tf.reshape(inputs, new_shape), tf.int32)
        # Convert the last dimension to an integer
        int_rep = tf.reduce_sum(inputs_reshaped * self._binary_base, axis=-1)
        # Map integers to constellation symbols
        x = tf.gather(self.constellation.points, int_rep, axis=0)
        return x
File:           ~/.local/lib/python3.8/site-packages/sionna/mapping.py
Type:           type
Subclasses:
```

    
As can be seen, the `Mapper` class inherits from `Layer`, i.e., implements a Keras layer.
    
This allows to simply built complex systems by using the <a class="reference external" href="https://keras.io/guides/functional_api/">Keras functional API</a> to stack layers.
    
Sionna provides as utility a binary source to sample uniform i.i.d. bits.

```python
[6]:
```

```python
binary_source = sn.utils.BinarySource()
```

    
Finally, we need the AWGN channel.

```python
[7]:
```

```python
awgn_channel = sn.channel.AWGN()
```

    
Sionna provides a utility function to compute the noise power spectral density ratio $N_0$ from the energy per bit to noise power spectral density ratio $E_b/N_0$ in dB and a variety of parameters such as the coderate and the nunber of bits per symbol.

```python
[8]:
```

```python
no = sn.utils.ebnodb2no(ebno_db=10.0,
                        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                        coderate=1.0) # Coderate set to 1 as we do uncoded transmission here
```

    
We now have all the components we need to transmit QAM symbols over an AWGN channel.
    
Sionna natively supports multi-dimensional tensors.
    
Most layers operate at the last dimension and can have arbitrary input shapes (preserved at output).

```python
[9]:
```

```python
BATCH_SIZE = 64 # How many examples are processed by Sionna in parallel
bits = binary_source([BATCH_SIZE,
                      1024]) # Blocklength
print("Shape of bits: ", bits.shape)
x = mapper(bits)
print("Shape of x: ", x.shape)
y = awgn_channel([x, no])
print("Shape of y: ", y.shape)
llr = demapper([y, no])
print("Shape of llr: ", llr.shape)
```


```python
Shape of bits:  (64, 1024)
Shape of x:  (64, 512)
Shape of y:  (64, 512)
Shape of llr:  (64, 1024)
```

    
In <em>Eager</em> mode, we can directly access the values of each tensor. This simplify debugging.

```python
[10]:
```

```python
num_samples = 8 # how many samples shall be printed
num_symbols = int(num_samples/NUM_BITS_PER_SYMBOL)
print(f"First {num_samples} transmitted bits: {bits[0,:num_samples]}")
print(f"First {num_symbols} transmitted symbols: {np.round(x[0,:num_symbols], 2)}")
print(f"First {num_symbols} received symbols: {np.round(y[0,:num_symbols], 2)}")
print(f"First {num_samples} demapped llrs: {np.round(llr[0,:num_samples], 2)}")
```


```python
First 8 transmitted bits: [0. 1. 1. 1. 1. 0. 1. 0.]
First 4 transmitted symbols: [ 0.71-0.71j -0.71-0.71j -0.71+0.71j -0.71+0.71j]
First 4 received symbols: [ 0.65-0.61j -0.62-0.69j -0.6 +0.72j -0.86+0.63j]
First 8 demapped llrs: [-36.65  34.36  35.04  38.83  33.96 -40.5   48.47 -35.65]
```

    
Let’s visualize the received noisy samples.

```python
[11]:
```

```python
plt.figure(figsize=(8,8))
plt.axes().set_aspect(1)
plt.grid(True)
plt.title('Channel output')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.scatter(tf.math.real(y), tf.math.imag(y))
plt.tight_layout()
```

<img alt="../_images/examples_Sionna_tutorial_part1_31_0.png" src="https://nvlabs.github.io/sionna/_images/examples_Sionna_tutorial_part1_31_0.png" />

    
**Task:** One can play with the SNR to visualize the impact on the received samples.
    
**Advanced Task:** Compare the LLR distribution for “app” demapping with “maxlog” demapping. The <a class="reference external" href="https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html">Bit-Interleaved Coded Modulation</a> example notebook can be helpful for this task.

## Communication Systems as Keras Models<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html#Communication-Systems-as-Keras-Models" title="Permalink to this headline"></a>
    
It is typically more convenient to wrap a Sionna-based communication system into a <a class="reference external" href="https://keras.io/api/models/model/">Keras models</a>.
    
These models can be simply built by using the <a class="reference external" href="https://keras.io/guides/functional_api/">Keras functional API</a> to stack layers.
    
The following cell implements the previous system as a Keras model.
    
The key functions that need to be defined are `__init__()`, which instantiates the required components, and `__call()__`, which performs forward pass through the end-to-end system.

```python
[12]:
```

```python
class UncodedSystemAWGN(Model): # Inherits from Keras Model
    def __init__(self, num_bits_per_symbol, block_length):
        """
        A keras model of an uncoded transmission over the AWGN channel.
        Parameters
        ----------
        num_bits_per_symbol: int
            The number of bits per constellation symbol, e.g., 4 for QAM16.
        block_length: int
            The number of bits per transmitted message block (will be the codeword length later).
        Input
        -----
        batch_size: int
            The batch_size of the Monte-Carlo simulation.
        ebno_db: float
            The `Eb/No` value (=rate-adjusted SNR) in dB.
        Output
        ------
        (bits, llr):
            Tuple:
        bits: tf.float32
            A tensor of shape `[batch_size, block_length] of 0s and 1s
            containing the transmitted information bits.
        llr: tf.float32
            A tensor of shape `[batch_size, block_length] containing the
            received log-likelihood-ratio (LLR) values.
        """
        super().__init__() # Must call the Keras model initializer
        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = block_length
        self.constellation = sn.mapping.Constellation("qam", self.num_bits_per_symbol)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()
    # @tf.function # Enable graph execution to speed things up
    def __call__(self, batch_size, ebno_db):
        # no channel coding used; we set coderate=1.0
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)
        bits = self.binary_source([batch_size, self.block_length]) # Blocklength set to 1024 bits
        x = self.mapper(bits)
        y = self.awgn_channel([x, no])
        llr = self.demapper([y,no])
        return bits, llr
```

    
We need first to instantiate the model.

```python
[13]:
```

```python
model_uncoded_awgn = UncodedSystemAWGN(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=1024)
```

    
Sionna provides a utility to easily compute and plot the bit error rate (BER).

```python
[14]:
```

```python
EBN0_DB_MIN = -3.0 # Minimum value of Eb/N0 [dB] for simulations
EBN0_DB_MAX = 5.0 # Maximum value of Eb/N0 [dB] for simulations
BATCH_SIZE = 2000 # How many examples are processed by Sionna in parallel
ber_plots = sn.utils.PlotBER("AWGN")
ber_plots.simulate(model_uncoded_awgn,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Uncoded",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=True);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -3.0 | 1.5825e-01 | 1.0000e+00 |      324099 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
   -2.579 | 1.4687e-01 | 1.0000e+00 |      300799 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
   -2.158 | 1.3528e-01 | 1.0000e+00 |      277061 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
   -1.737 | 1.2323e-01 | 1.0000e+00 |      252373 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
   -1.316 | 1.1246e-01 | 1.0000e+00 |      230320 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
   -0.895 | 1.0107e-01 | 1.0000e+00 |      206992 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
   -0.474 | 9.0021e-02 | 1.0000e+00 |      184362 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
   -0.053 | 8.0165e-02 | 1.0000e+00 |      164177 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
    0.368 | 6.9933e-02 | 1.0000e+00 |      143222 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
    0.789 | 6.0897e-02 | 1.0000e+00 |      124717 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
    1.211 | 5.2020e-02 | 1.0000e+00 |      106537 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
    1.632 | 4.3859e-02 | 1.0000e+00 |       89823 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
    2.053 | 3.6686e-02 | 1.0000e+00 |       75132 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
    2.474 | 3.0071e-02 | 1.0000e+00 |       61586 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
    2.895 | 2.4304e-02 | 1.0000e+00 |       49775 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
    3.316 | 1.9330e-02 | 1.0000e+00 |       39588 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
    3.737 | 1.4924e-02 | 1.0000e+00 |       30565 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
    4.158 | 1.1227e-02 | 1.0000e+00 |       22992 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
    4.579 | 8.2632e-03 | 1.0000e+00 |       16923 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
      5.0 | 5.9722e-03 | 9.9850e-01 |       12231 |     2048000 |         1997 |        2000 |         0.0 |reached target block errors
```

<img alt="../_images/examples_Sionna_tutorial_part1_39_1.png" src="https://nvlabs.github.io/sionna/_images/examples_Sionna_tutorial_part1_39_1.png" />

    
The `sn.utils.PlotBER` object stores the results and allows to add additional simulations to the previous curves.
    
<em>Remark</em>: In Sionna, a block error is defined to happen if for two tensors at least one position in the last dimension differs (i.e., at least one bit wrongly received per codeword). The bit error rate the total number of erroneous positions divided by the total number of transmitted bits.

## Forward Error Correction (FEC)<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html#Forward-Error-Correction-(FEC)" title="Permalink to this headline"></a>
    
We now add channel coding to our transceiver to make it more robust against transmission errors. For this, we will use <a class="reference external" href="https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3214">5G compliant low-density parity-check (LDPC) codes and Polar codes</a>. You can find more detailed information in the notebooks <a class="reference external" href="https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html">Bit-Interleaved Coded Modulation (BICM)</a> and <a class="reference external" href="https://nvlabs.github.io/sionna/examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html">5G Channel Coding
and Rate-Matching: Polar vs. LDPC Codes</a>.

```python
[15]:
```

```python
k = 12
n = 20
encoder = sn.fec.ldpc.LDPC5GEncoder(k, n)
decoder = sn.fec.ldpc.LDPC5GDecoder(encoder, hard_out=True)
```

    
Let us encode some random input bits.

```python
[16]:
```

```python
BATCH_SIZE = 1 # one codeword in parallel
u = binary_source([BATCH_SIZE, k])
print("Input bits are: \n", u.numpy())
c = encoder(u)
print("Encoded bits are: \n", c.numpy())
```


```python
Input bits are:
 [[1. 1. 0. 0. 1. 1. 0. 0. 0. 0. 1. 1.]]
Encoded bits are:
 [[1. 1. 0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 1. 1.]]
```

    
One of the fundamental paradigms of Sionna is batch-processing. Thus, the example above could be executed with for arbitrary batch-sizes to simulate `batch_size` codewords in parallel.
    
However, Sionna can do more - it supports <em>N</em>-dimensional input tensors and, thereby, allows the processing of multiple samples of multiple users and several antennas in a single command line. Let’s say we want to encoded `batch_size` codewords of length `n` for each of the `num_users` connected to each of the `num_basestations`. This means in total we transmit `batch_size` * `n` * `num_users` * `num_basestations` bits.

```python
[17]:
```

```python
BATCH_SIZE = 10 # samples per scenario
num_basestations = 4
num_users = 5 # users per basestation
n = 1000 # codeword length per transmitted codeword
coderate = 0.5 # coderate
k = int(coderate * n) # number of info bits per codeword
# instantiate a new encoder for codewords of length n
encoder = sn.fec.ldpc.LDPC5GEncoder(k, n)
# the decoder must be linked to the encoder (to know the exact code parameters used for encoding)
decoder = sn.fec.ldpc.LDPC5GDecoder(encoder,
                                    hard_out=True, # binary output or provide soft-estimates
                                    return_infobits=True, # or also return (decoded) parity bits
                                    num_iter=20, # number of decoding iterations
                                    cn_type="boxplus-phi") # also try "minsum" decoding
# draw random bits to encode
u = binary_source([BATCH_SIZE, num_basestations, num_users, k])
print("Shape of u: ", u.shape)
# We can immediately encode u for all users, basetation and samples
# This all happens with a single line of code
c = encoder(u)
print("Shape of c: ", c.shape)
print("Total number of processed bits: ", np.prod(c.shape))
```


```python
Shape of u:  (10, 4, 5, 500)
Shape of c:  (10, 4, 5, 1000)
Total number of processed bits:  200000
```

    
This works for arbitrary dimensions and allows a simple extension of the designed system to multi-user or multi-antenna scenarios.
    
Let us now replace the LDPC code by a Polar code. The API remains similar.

```python
[18]:
```

```python
k = 64
n = 128
encoder = sn.fec.polar.Polar5GEncoder(k, n)
decoder = sn.fec.polar.Polar5GDecoder(encoder,
                                      dec_type="SCL") # you can also use "SCL"
```

    
<em>Advanced Remark:</em> The 5G Polar encoder/decoder class directly applies rate-matching and the additional CRC concatenation. This is all done internally and transparent to the user.
    
In case you want to access low-level features of the Polar codes, please use `sionna.fec.polar.PolarEncoder` and the desired decoder (`sionna.fec.polar.PolarSCDecoder`, `sionna.fec.polar.PolarSCLDecoder` or `sionna.fec.polar.PolarBPDecoder`).
    
Further details can be found in the tutorial notebook on <a class="reference external" href="https://nvlabs.github.io/sionna/examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html">5G Channel Coding and Rate-Matching: Polar vs. LDPC Codes</a>.
    
<img alt="QAM FEC AWGN" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmkAAACtCAYAAADrl+hZAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAaGVYSWZNTQAqAAAACAAEAQYAAwAAAAEAAgAAARIAAwAAAAEAAQAAASgAAwAAAAEAAgAAh2kABAAAAAEAAAA+AAAAAAADoAEAAwAAAAEAAQAAoAIABAAAAAEAAAJpoAMABAAAAAEAAACtAAAAAF4rGUEAAALkaVRYdFhNTDpjb20uYWRvYmUueG1wAAAAAAA8eDp4bXBtZXRhIHhtbG5zOng9ImFkb2JlOm5zOm1ldGEvIiB4OnhtcHRrPSJYTVAgQ29yZSA2LjAuMCI+CiAgIDxyZGY6UkRGIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI+CiAgICAgIDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiCiAgICAgICAgICAgIHhtbG5zOnRpZmY9Imh0dHA6Ly9ucy5hZG9iZS5jb20vdGlmZi8xLjAvIgogICAgICAgICAgICB4bWxuczpleGlmPSJodHRwOi8vbnMuYWRvYmUuY29tL2V4aWYvMS4wLyI+CiAgICAgICAgIDx0aWZmOkNvbXByZXNzaW9uPjE8L3RpZmY6Q29tcHJlc3Npb24+CiAgICAgICAgIDx0aWZmOlJlc29sdXRpb25Vbml0PjI8L3RpZmY6UmVzb2x1dGlvblVuaXQ+CiAgICAgICAgIDx0aWZmOk9yaWVudGF0aW9uPjE8L3RpZmY6T3JpZW50YXRpb24+CiAgICAgICAgIDx0aWZmOlBob3RvbWV0cmljSW50ZXJwcmV0YXRpb24+MjwvdGlmZjpQaG90b21ldHJpY0ludGVycHJldGF0aW9uPgogICAgICAgICA8ZXhpZjpQaXhlbFhEaW1lbnNpb24+NjE3PC9leGlmOlBpeGVsWERpbWVuc2lvbj4KICAgICAgICAgPGV4aWY6Q29sb3JTcGFjZT4xPC9leGlmOkNvbG9yU3BhY2U+CiAgICAgICAgIDxleGlmOlBpeGVsWURpbWVuc2lvbj4xNzM8L2V4aWY6UGl4ZWxZRGltZW5z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" />

```python
[19]:
```

```python
class CodedSystemAWGN(Model): # Inherits from Keras Model
    def __init__(self, num_bits_per_symbol, n, coderate):
        super().__init__() # Must call the Keras model initializer
        self.num_bits_per_symbol = num_bits_per_symbol
        self.n = n
        self.k = int(n*coderate)
        self.coderate = coderate
        self.constellation = sn.mapping.Constellation("qam", self.num_bits_per_symbol)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()
        self.encoder = sn.fec.ldpc.LDPC5GEncoder(self.k, self.n)
        self.decoder = sn.fec.ldpc.LDPC5GDecoder(self.encoder, hard_out=True)
    #@tf.function # activate graph execution to speed things up
    def __call__(self, batch_size, ebno_db):
        no = sn.utils.ebnodb2no(ebno_db, num_bits_per_symbol=self.num_bits_per_symbol, coderate=self.coderate)
        bits = self.binary_source([batch_size, self.k])
        codewords = self.encoder(bits)
        x = self.mapper(codewords)
        y = self.awgn_channel([x, no])
        llr = self.demapper([y,no])
        bits_hat = self.decoder(llr)
        return bits, bits_hat
```
```python
[20]:
```

```python
CODERATE = 0.5
BATCH_SIZE = 2000
model_coded_awgn = CodedSystemAWGN(num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                   n=2048,
                                   coderate=CODERATE)
ber_plots.simulate(model_coded_awgn,
                   ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 15),
                   batch_size=BATCH_SIZE,
                   num_target_block_errors=500,
                   legend="Coded",
                   soft_estimates=False,
                   max_mc_iter=15,
                   show_fig=True,
                   forward_keyboard_interrupt=False);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -3.0 | 2.7896e-01 | 1.0000e+00 |      571312 |     2048000 |         2000 |        2000 |         1.8 |reached target block errors
   -2.429 | 2.6327e-01 | 1.0000e+00 |      539174 |     2048000 |         2000 |        2000 |         1.7 |reached target block errors
   -1.857 | 2.4783e-01 | 1.0000e+00 |      507546 |     2048000 |         2000 |        2000 |         1.7 |reached target block errors
   -1.286 | 2.2687e-01 | 1.0000e+00 |      464636 |     2048000 |         2000 |        2000 |         1.7 |reached target block errors
   -0.714 | 2.0312e-01 | 1.0000e+00 |      415988 |     2048000 |         2000 |        2000 |         1.7 |reached target block errors
   -0.143 | 1.7154e-01 | 1.0000e+00 |      351316 |     2048000 |         2000 |        2000 |         1.7 |reached target block errors
    0.429 | 1.1296e-01 | 9.9050e-01 |      231337 |     2048000 |         1981 |        2000 |         1.7 |reached target block errors
      1.0 | 2.0114e-02 | 4.7150e-01 |       41193 |     2048000 |          943 |        2000 |         1.7 |reached target block errors
    1.571 | 1.7617e-04 | 1.1633e-02 |        5412 |    30720000 |          349 |       30000 |        25.2 |reached max iter
    2.143 | 0.0000e+00 | 0.0000e+00 |           0 |    30720000 |            0 |       30000 |        25.3 |reached max iter
Simulation stopped as no error occurred @ EbNo = 2.1 dB.

```

<img alt="../_images/examples_Sionna_tutorial_part1_52_1.png" src="https://nvlabs.github.io/sionna/_images/examples_Sionna_tutorial_part1_52_1.png" />

    
As can be seen, the `BerPlot` class uses multiple stopping conditions and stops the simulation after no error occured at a specifc SNR point.
    
**Task**: Replace the coding scheme by a Polar encoder/decoder or a convolutional code with Viterbi decoding.

## Eager vs Graph Mode<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html#Eager-vs-Graph-Mode" title="Permalink to this headline"></a>
    
So far, we have executed the example in <em>eager</em> mode. This allows to run TensorFlow ops as if it was written NumPy and simplifies development and debugging.
    
However, to unleash Sionna’s full performance, we need to activate <em>graph</em> mode which can be enabled with the function decorator <em>@tf.function()</em>.
    
We refer to <a class="reference external" href="https://www.tensorflow.org/guide/function">TensorFlow Functions</a> for further details.

```python
[21]:
```

```python
@tf.function() # enables graph-mode of the following function
def run_graph(batch_size, ebno_db):
    # all code inside this function will be executed in graph mode, also calls of other functions
    print(f"Tracing run_graph for values batch_size={batch_size} and ebno_db={ebno_db}.") # print whenever this function is traced
    return model_coded_awgn(batch_size, ebno_db)
```
```python
[22]:
```

```python
batch_size = 10 # try also different batch sizes
ebno_db = 1.5
# run twice - how does the output change?
run_graph(batch_size, ebno_db)
```


```python
Tracing run_graph for values batch_size=10 and ebno_db=1.5.
```
```python
[22]:
```
```python
(<tf.Tensor: shape=(10, 1024), dtype=float32, numpy=
 array([[1., 1., 0., ..., 0., 1., 0.],
        [1., 1., 1., ..., 0., 1., 1.],
        [1., 1., 0., ..., 0., 1., 1.],
        ...,
        [0., 1., 0., ..., 0., 0., 0.],
        [0., 1., 0., ..., 0., 1., 0.],
        [0., 1., 1., ..., 0., 1., 1.]], dtype=float32)>,
 <tf.Tensor: shape=(10, 1024), dtype=float32, numpy=
 array([[1., 1., 0., ..., 0., 1., 0.],
        [1., 1., 1., ..., 0., 1., 1.],
        [1., 1., 0., ..., 0., 1., 1.],
        ...,
        [0., 1., 0., ..., 0., 0., 0.],
        [0., 1., 0., ..., 0., 1., 0.],
        [0., 1., 1., ..., 0., 1., 1.]], dtype=float32)>)
```

    
In graph mode, Python code (i.e., <em>non-TensorFlow code</em>) is only executed whenever the function is <em>traced</em>. This happens whenever the input signature changes.
    
As can be seen above, the print statement was executed, i.e., the graph was traced again.
    
To avoid this re-tracing for different inputs, we now input tensors. You can see that the function is now traced once for input tensors of same dtype.
    
See <a class="reference external" href="https://www.tensorflow.org/guide/function#rules_of_tracing">TensorFlow Rules of Tracing</a> for details.
    
**Task:** change the code above such that tensors are used as input and execute the code with different input values. Understand when re-tracing happens.
    
<em>Remark</em>: if the input to a function is a tensor its signature must change and not <em>just</em> its value. For example the input could have a different size or datatype. For efficient code execution, we usually want to avoid re-tracing of the code if not required.

```python
[23]:
```

```python
# You can print the cached signatures with
print(run_graph.pretty_printed_concrete_signatures())
```


```python
run_graph(batch_size=10, ebno_db=1.5)
  Returns:
    (<1>, <2>)
      <1>: float32 Tensor, shape=(10, 1024)
      <2>: float32 Tensor, shape=(10, 1024)
```

    
We now compare the throughput of the different modes.

```python
[24]:
```

```python
repetitions = 4 # average over multiple runs
batch_size = BATCH_SIZE # try also different batch sizes
ebno_db = 1.5
# --- eager mode ---
t_start = time.perf_counter()
for _ in range(repetitions):
    bits, bits_hat = model_coded_awgn(tf.constant(batch_size, tf.int32),
                                tf.constant(ebno_db, tf. float32))
t_stop = time.perf_counter()
# throughput in bit/s
throughput_eager = np.size(bits.numpy())*repetitions / (t_stop - t_start) / 1e6
print(f"Throughput in Eager mode: {throughput_eager :.3f} Mbit/s")
# --- graph mode ---
# run once to trace graph (ignored for throughput)
run_graph(tf.constant(batch_size, tf.int32),
          tf.constant(ebno_db, tf. float32))
t_start = time.perf_counter()
for _ in range(repetitions):
    bits, bits_hat = run_graph(tf.constant(batch_size, tf.int32),
                                tf.constant(ebno_db, tf. float32))
t_stop = time.perf_counter()
# throughput in bit/s
throughput_graph = np.size(bits.numpy())*repetitions / (t_stop - t_start) / 1e6
print(f"Throughput in graph mode: {throughput_graph :.3f} Mbit/s")

```


```python
Throughput in Eager mode: 1.212 Mbit/s
Tracing run_graph for values batch_size=Tensor(&#34;batch_size:0&#34;, shape=(), dtype=int32) and ebno_db=Tensor(&#34;ebno_db:0&#34;, shape=(), dtype=float32).
Throughput in graph mode: 8.623 Mbit/s
```

    
Let’s run the same simulation as above in graph mode.

```python
[25]:
```

```python
ber_plots.simulate(run_graph,
                   ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 12),
                   batch_size=BATCH_SIZE,
                   num_target_block_errors=500,
                   legend="Coded (Graph mode)",
                   soft_estimates=True,
                   max_mc_iter=100,
                   show_fig=True,
                   forward_keyboard_interrupt=False);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -3.0 | 2.7922e-01 | 1.0000e+00 |      571845 |     2048000 |         2000 |        2000 |         0.2 |reached target block errors
   -2.273 | 2.5948e-01 | 1.0000e+00 |      531422 |     2048000 |         2000 |        2000 |         0.2 |reached target block errors
   -1.545 | 2.3550e-01 | 1.0000e+00 |      482301 |     2048000 |         2000 |        2000 |         0.2 |reached target block errors
   -0.818 | 2.0768e-01 | 1.0000e+00 |      425335 |     2048000 |         2000 |        2000 |         0.2 |reached target block errors
   -0.091 | 1.6918e-01 | 1.0000e+00 |      346477 |     2048000 |         2000 |        2000 |         0.2 |reached target block errors
    0.636 | 7.6115e-02 | 9.1650e-01 |      155883 |     2048000 |         1833 |        2000 |         0.2 |reached target block errors
    1.364 | 1.7544e-03 | 7.2125e-02 |       14372 |     8192000 |          577 |        8000 |         1.0 |reached target block errors
    2.091 | 7.8125e-08 | 2.0000e-05 |          16 |   204800000 |            4 |      200000 |        24.3 |reached max iter
    2.818 | 0.0000e+00 | 0.0000e+00 |           0 |   204800000 |            0 |      200000 |        24.4 |reached max iter
Simulation stopped as no error occurred @ EbNo = 2.8 dB.

```

<img alt="../_images/examples_Sionna_tutorial_part1_63_1.png" src="https://nvlabs.github.io/sionna/_images/examples_Sionna_tutorial_part1_63_1.png" />

    
**Task:** TensorFlow allows to <em>compile</em> graphs with <a class="reference external" href="https://www.tensorflow.org/xla">XLA</a>. Try to further accelerate the code with XLA (`@tf.function(jit_compile=True)`).
    
<em>Remark</em>: XLA is still an experimental feature and not all TensorFlow (and, thus, Sionna) functions support XLA.
    
**Task 2:** Check the GPU load with `!nvidia-smi`. Find the best tradeoff between batch-size and throughput for your specific GPU architecture.

## Exercise<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html#Exercise" title="Permalink to this headline"></a>
    
Simulate the coded bit error rate (BER) for a Polar coded and 64-QAM modulation. Assume a codeword length of n = 200 and coderate = 0.5.
    
**Hint**: For Polar codes, successive cancellation list decoding (SCL) gives the best BER performance. However, successive cancellation (SC) decoding (without a list) is less complex.

```python
[26]:
```

```python
n = 200
coderate = 0.5
# *You can implement your code here*

```

<script type="application/vnd.jupyter.widget-state+json">
{"state": {}, "version_major": 2, "version_minor": 0}
</script>