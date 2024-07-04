# Using the DeepMIMO Dataset with Sionna

In this example, you will learn how to use the ray-tracing based DeepMIMO dataset.

[DeepMIMO](https://deepmimo.net/) is a generic dataset that enables a wide range of machine/deep learning applications for MIMO systems. It takes as input a set of parameters (such as antenna array configurations and time-domain/OFDM parameters) and generates MIMO channel realizations, corresponding locations, angles of arrival/departure, etc., based on these parameters and on a ray-tracing scenario selected [from those available in DeepMIMO](https://deepmimo.net/scenarios/).

## GPU Configuration and Imports


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

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import os
# Load the required Sionna components
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.ofdm import OFDMModulator, OFDMDemodulator, ZFPrecoder, RemoveNulledSubcarriers
from sionna.channel.tr38901 import AntennaArray, CDL, Antenna
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, time_lag_discrete_time_channel
from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel, TimeChannel
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper
from sionna.utils import BinarySource, ebnodb2no, sim_ber
from sionna.utils.metrics import compute_ber
```

## Configuration of DeepMIMO

DeepMIMO provides multiple [scenarios](https://deepmimo.net/scenarios/) that one can select from. In this example, we use the O1 scenario with the carrier frequency set to 60 GHz (O1_60). To run this example, please download the O1_60 data files [from this page](https://deepmimo.net/scenarios/o1-scenario/). The downloaded zip file should be extracted into a folder, and the parameter `DeepMIMO_params['dataset_folder']` should be set to point to this folder, as done below.

To use DeepMIMO with Sionna, the DeepMIMO dataset first needs to be generated. The generated DeepMIMO dataset contains channels for different locations of the users and basestations. The layout of the O1 scenario is shown in the figure below.


In this example, we generate a dataset that consists of channels for the links from the basestation 6 to the users located on the rows 400 to 450. Each of these rows consists of 181 user locations, resulting in $51 \times 181 = 9231$ basestation-user channels.

The antenna arrays in the DeepMIMO dataset are defined through the x-y-z axes. In the following example, a single-user MISO downlink is considered. The basestation is equipped with a uniform linear array of 16 elements spread along the x-axis. The users are each equipped with a single antenna. These parameters can be configured using the code below (for more information about the DeepMIMO parameters, please check [the DeepMIMO configurations](https://deepmimo.net/versions/v2-python/)).


```python
# Import DeepMIMO
try:
    import DeepMIMO
except ImportError as e:
    # Install DeepMIMO if package is not already installed
    import os
    os.system("pip install DeepMIMO")
    import DeepMIMO
# Channel generation
DeepMIMO_params = DeepMIMO.default_params() # Load the default parameters
DeepMIMO_params['dataset_folder'] = r'./scenarios' # Path to the downloaded scenarios
DeepMIMO_params['scenario'] = 'O1_60' # DeepMIMO scenario
DeepMIMO_params['num_paths'] = 10 # Maximum number of paths
DeepMIMO_params['active_BS'] = np.array([6]) # Basestation indices to be included in the dataset
# Selected rows of users, whose channels are to be generated.
DeepMIMO_params['user_row_first'] = 400 # First user row to be included in the dataset
DeepMIMO_params['user_row_last'] = 450 # Last user row to be included in the dataset
# Configuration of the antenna arrays
DeepMIMO_params['bs_antenna']['shape'] = np.array([16, 1, 1]) # BS antenna shape through [x, y, z] axes
DeepMIMO_params['ue_antenna']['shape'] = np.array([1, 1, 1]) # UE antenna shape through [x, y, z] axes
# The OFDM_channels parameter allows choosing between the generation of channel impulse
# responses (if set to 0) or frequency domain channels (if set to 1).
# It is set to 0 for this simulation, as the channel responses in frequency domain
# will be generated using Sionna.
DeepMIMO_params['OFDM_channels'] = 0
# Generates a DeepMIMO dataset
DeepMIMO_dataset = DeepMIMO.generate_data(DeepMIMO_params)
```


```python

Basestation 6
UE-BS Channels
```

```python
Reading ray-tracing: 100%|| 81450/81450 [00:00<00:00, 129737.29it/s]
Generating channels: 100%|| 9231/9231 [00:00<00:00, 17426.44it/s]
```

```python

BS-BS Channels
```

```python
Reading ray-tracing: 100%|| 6/6 [00:00<00:00, 33509.75it/s]
Generating channels: 100%|| 1/1 [00:00<00:00, 2589.08it/s]
```
### Visualization of the dataset

To provide a better understanding of the user and basestation locations, we next visualize the locations of the users, highlighting the first active row of users (row 400), and basestation 6.


```python
plt.figure(figsize=(12,8))
## User locations
active_bs_idx = 0 # Select the first active basestation in the dataset
plt.scatter(DeepMIMO_dataset[active_bs_idx]['user']['location'][:, 1], # y-axis location of the users
         DeepMIMO_dataset[active_bs_idx]['user']['location'][:, 0], # x-axis location of the users
         s=1, marker='x', c='C0', label='The users located on the rows %i to %i (R%i to R%i)'%
           (DeepMIMO_params['user_row_first'], DeepMIMO_params['user_row_last'],
           DeepMIMO_params['user_row_first'], DeepMIMO_params['user_row_last']))
# First 181 users correspond to the first row
plt.scatter(DeepMIMO_dataset[active_bs_idx]['user']['location'][0:181, 1],
         DeepMIMO_dataset[active_bs_idx]['user']['location'][0:181, 0],
         s=1, marker='x', c='C1', label='First row of users (R%i)'% (DeepMIMO_params['user_row_first']))
## Basestation location
plt.scatter(DeepMIMO_dataset[active_bs_idx]['location'][1],
         DeepMIMO_dataset[active_bs_idx]['location'][0],
         s=50.0, marker='o', c='C2', label='Basestation')
plt.gca().invert_xaxis() # Invert the x-axis to align the figure with the figure above
plt.ylabel('x-axis')
plt.xlabel('y-axis')
plt.grid()
plt.legend();
```


## Using DeepMIMO with Sionna

The DeepMIMO Python package provides [a Sionna-compliant channel impulse response generator](https://nvlabs.github.io/sionna/examples/CIR_Dataset.html#Generators) that adapts the structure of the DeepMIMO dataset to be consistent with Sionna.

An adapter is instantiated for a given DeepMIMO dataset. In addition to the dataset, the adapter takes the indices of the basestations and users, to generate the channels between these basestations and users:

`DeepMIMOSionnaAdapter(DeepMIMO_dataset,` `bs_idx,` `ue_idx)`

**Note:** `bs_idx` and `ue_idx` set the links from which the channels are drawn. For instance, if `bs_idx` `=` `[0,` `1]` and `ue_idx` `=` `[2,` `3]`, the adapter then outputs the 4 channels formed by the combination of the first and second basestations with the third and fourth users.

The default behavior for `bs_idx` and `ue_idx` are defined as follows: - If value for `bs_idx` is not given, it will be set to `[0]` (i.e., the first basestation in the `DeepMIMO_dataset`). - If value for `ue_idx` is not given, then channels are provided for the links between the `bs_idx` and all users (i.e., `ue_idx=range(len(DeepMIMO_dataset[0]['user']['channel']))`. - If the both `bs_idx` and `ue_idx` are not given, the channels between the first basestation and all the
users are provided by the adapter. For this example, `DeepMIMOSionnaAdapter(DeepMIMO_dataset)` returns the channels from the basestation 6 and the 9231 available user locations.

**Note:** The adapter assumes basestations are transmitters and users are receivers. Uplink channels can be obtained using (transpose) reciprocity.

### Random Sampling of Multi-User Channels

When considering multiple basestations, `bs_idx` can be set to a 2D numpy matrix of shape $($ # of samples $\times$ # of basestations per sample $)$. In this case, for each sample of basestations, the `DeepMIMOSionnaAdapter` returns a set of $($ # of basestations per sample $\times$ # of users $)$ channels, which can be provided as a multi-transmitter sample for the Sionna model. For example, `bs_idx` `=` `np.array([[0,` `1],` `[2,` `3],` `[4,` `5]])` provides three
sets of $($ 2 basestations $\times$ # of users $)$ channels. These three channel sets are from the basestation sets `[0,` `1]`, `[2,` `3]`, and `[4,` `5]`, respectively, to the users.

To use the adapter for multi-user channels, `ue_idx` can be set to a 2D numpy matrix of shape $($ # of samples $\times$ # of users per sample $)$. In this case, for each sample of users, the `DeepMIMOSionnaAdapter` returns a set of $($ # of basestations $\times$ # of users per sample $)$ channels, which can be provided as a multi-receiver sample for the Sionna model. For example, `ue_idx` `=` `np.array([[0,` `1` `,2],` `[4,` `5,` `6]])` provides two sets of $($
# of basestations $\times$ 3 users $)$ channels. These two channel sets are from the basestations to the user sets `[0,` `1,` `2]` and `[4,` `5,` `6]`, respectively.

In order to randomly sample channels from all the available user locations considering `num_rx` users, one may set `ue_idx` as in the following cell. In this example, the channels will be randomly chosen from the links between the basestation 6 and the 9231 available user locations.


```python
from DeepMIMO import DeepMIMOSionnaAdapter
# Number of receivers for the Sionna model.
# MISO is considered here.
num_rx = 1
# The number of UE locations in the generated DeepMIMO dataset
num_ue_locations = len(DeepMIMO_dataset[0]['user']['channel']) # 9231
# Pick the largest possible number of user locations that is a multiple of ``num_rx``
ue_idx = np.arange(num_rx*(num_ue_locations//num_rx))
# Optionally shuffle the dataset to not select only users that are near each others
np.random.shuffle(ue_idx)
# Reshape to fit the requested number of users
ue_idx = np.reshape(ue_idx, [-1, num_rx]) # In the shape of (floor(9231/num_rx) x num_rx)
DeepMIMO_Sionna_adapter = DeepMIMOSionnaAdapter(DeepMIMO_dataset, ue_idx=ue_idx)
```

## Link-level Simulations using Sionna and DeepMIMO

In the following cell, we define a Sionna model implementing the end-to-end link.

**Note:** The Sionna CIRDataset object shuffles the DeepMIMO channels provided by the adapter. Therefore, channel samples are passed through the model in a random order.


```python
class LinkModel(tf.keras.Model):
    def __init__(self,
                 DeepMIMO_Sionna_adapter,
                 carrier_frequency,
                 cyclic_prefix_length,
                 pilot_ofdm_symbol_indices,
                 subcarrier_spacing = 60e3,
                 batch_size = 64
                ):
        super().__init__()
        self._batch_size = batch_size
        self._cyclic_prefix_length = cyclic_prefix_length
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices
        # CIRDataset to parse the dataset
        self._CIR = sionna.channel.CIRDataset(DeepMIMO_Sionna_adapter,
                                              self._batch_size,
                                              DeepMIMO_Sionna_adapter.num_rx,
                                              DeepMIMO_Sionna_adapter.num_rx_ant,
                                              DeepMIMO_Sionna_adapter.num_tx,
                                              DeepMIMO_Sionna_adapter.num_tx_ant,
                                              DeepMIMO_Sionna_adapter.num_paths,
                                              DeepMIMO_Sionna_adapter.num_time_steps)
        # System parameters
        self._carrier_frequency = carrier_frequency
        self._subcarrier_spacing = subcarrier_spacing
        self._fft_size = 76
        self._num_ofdm_symbols = 14
        self._num_streams_per_tx = DeepMIMO_Sionna_adapter.num_rx
        self._dc_null = False
        self._num_guard_carriers = [0, 0]
        self._pilot_pattern = "kronecker"
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices
        self._num_bits_per_symbol = 4
        self._coderate = 0.5
        # Setup the OFDM resource grid and stream management
        self._sm = StreamManagement(np.ones([DeepMIMO_Sionna_adapter.num_rx, 1], int), self._num_streams_per_tx)
        self._rg = ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                fft_size=self._fft_size,
                                subcarrier_spacing = self._subcarrier_spacing,
                                num_tx=DeepMIMO_Sionna_adapter.num_tx,
                                num_streams_per_tx=self._num_streams_per_tx,
                                cyclic_prefix_length=self._cyclic_prefix_length,
                                num_guard_carriers=self._num_guard_carriers,
                                dc_null=self._dc_null,
                                pilot_pattern=self._pilot_pattern,
                                pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)
        # Components forming the link
        # Codeword length
        self._n = int(self._rg.num_data_symbols * self._num_bits_per_symbol)
        # Number of information bits per codeword
        self._k = int(self._n * self._coderate)
        # OFDM channel
        self._frequencies = subcarrier_frequencies(self._rg.fft_size, self._rg.subcarrier_spacing)
        self._ofdm_channel = sionna.channel.GenerateOFDMChannel(self._CIR, self._rg, normalize_channel=True)
        self._channel_freq = ApplyOFDMChannel(add_awgn=True)
        # Transmitter
        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(self._k, self._n)
        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)
        self._zf_precoder = ZFPrecoder(self._rg, self._sm, return_effective_channel=True)
        # Receiver
        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="lin_time_avg")
        self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
        self._demapper = Demapper("app", "qam", self._num_bits_per_symbol)
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._rg)
    @tf.function
    def call(self, batch_size, ebno_db):
        # Transmitter
        b = self._binary_source([self._batch_size, 1, self._num_streams_per_tx, self._k])
        c = self._encoder(b)
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)
        # Generate the OFDM channel
        h_freq = self._ofdm_channel()
        # Precoding
        x_rg, g = self._zf_precoder([x_rg, h_freq])
        # Apply OFDM channel
        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        y = self._channel_freq([x_rg, h_freq, no])
        # Receiver
        h_hat, err_var = self._ls_est ([y, no])
        x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no])
        llr = self._demapper([x_hat, no_eff])
        b_hat = self._decoder(llr)
        return b, b_hat
```


We next evaluate the setup with different $E_b/N_0$ values to obtain BLER curves.


```python
sim_params = {
              "ebno_db": np.linspace(-7, -5.25, 10),
              "cyclic_prefix_length" : 0,
              "pilot_ofdm_symbol_indices" : [2, 11],
              }
batch_size = 64
model = LinkModel(DeepMIMO_Sionna_adapter=DeepMIMO_Sionna_adapter,
                  carrier_frequency=DeepMIMO_params['scenario_params']['carrier_freq'],
                  cyclic_prefix_length=sim_params["cyclic_prefix_length"],
                  pilot_ofdm_symbol_indices=sim_params["pilot_ofdm_symbol_indices"])
ber, bler = sim_ber(model,
                    sim_params["ebno_db"],
                    batch_size=batch_size,
                    max_mc_iter=100,
                    num_target_block_errors=100)
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -7.0 | 1.2077e-01 | 1.0000e+00 |       28196 |      233472 |          128 |         128 |         5.3 |reached target block errors
   -6.806 | 9.1514e-02 | 1.0000e+00 |       21366 |      233472 |          128 |         128 |         0.2 |reached target block errors
   -6.611 | 5.5651e-02 | 9.6094e-01 |       12993 |      233472 |          123 |         128 |         0.2 |reached target block errors
   -6.417 | 2.3723e-02 | 7.2396e-01 |        8308 |      350208 |          139 |         192 |         0.3 |reached target block errors
   -6.222 | 5.3968e-03 | 3.5938e-01 |        3150 |      583680 |          115 |         320 |         0.4 |reached target block errors
   -6.028 | 8.9899e-04 | 9.2014e-02 |        1889 |     2101248 |          106 |        1152 |         1.6 |reached target block errors
   -5.833 | 9.4144e-05 | 1.3437e-02 |        1099 |    11673600 |           86 |        6400 |         8.6 |reached max iter
   -5.639 | 4.2832e-07 | 3.1250e-04 |           5 |    11673600 |            2 |        6400 |         8.6 |reached max iter
   -5.444 | 1.1993e-06 | 1.5625e-04 |          14 |    11673600 |            1 |        6400 |         8.5 |reached max iter
    -5.25 | 0.0000e+00 | 0.0000e+00 |           0 |    11673600 |            0 |        6400 |         8.5 |reached max iter
Simulation stopped as no error occurred @ EbNo = -5.2 dB.

```

```python
plt.figure(figsize=(12,8))
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.semilogy(sim_params["ebno_db"], bler)
```
```python
[<matplotlib.lines.Line2D at 0x7fe0b00c13a0>]
```


## DeepMIMO License and Citation
<ol class="upperalpha simple">
- Alkhateeb, [DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications](https://arxiv.org/pdf/1902.06435.pdf), in Proc. of Information Theory and Applications Workshop (ITA), San Diego, CA, Feb.2019.
</ol>

To use the DeepMIMO dataset, please check the license information [here](https://deepmimo.net/license/).To use the DeepMIMO dataset, please check the license information [here](https://deepmimo.net/license/).