# Realistic Multiuser MIMO OFDM Simulations

In this notebook, you will learn how to setup realistic simulations of multiuser MIMO uplink transmissions. Multiple user terminals (UTs) are randomly distributed in a cell sector and communicate with a multi-antenna base station.


The block-diagramm of the system model looks as follows:


It includes the following components:

- 5G LDPC FEC
- QAM modulation
- OFDM resource grid with configurable pilot pattern
- Multiple single-antenna transmitters and a multi-antenna receiver
- 3GPP 38.901 UMi, UMa, and RMa channel models and antenna patterns
- LS Channel estimation with nearest-neighbor interpolation as well as perfect CSI
- LMMSE MIMO equalization


You will learn how to setup the topologies required to simulate such scenarios and investigate

- the performance over different models, and
- the impact of imperfect CSI.


We will first walk through the configuration of all components of the system model, before simulating some simple uplink transmissions in the frequency domain. We will then simulate CDFs of the channel condition number and look into frequency-selectivity of the different channel models to understand the reasons for the observed performance differences.

It is recommended that you familiarize yourself with the [API documentation](https://nvlabs.github.io/sionna/api/channel.html) of the `Channel` module and, in particular, the 3GPP 38,901 models that require a substantial amount of configuration. The last set of simulations in this notebook take some time, especially when you have no GPU available. For this reason, we provide the simulation results directly in the cells generating the figures. Simply uncomment the corresponding lines to show
this results.

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
import time
import pickle
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.ofdm import OFDMModulator, OFDMDemodulator, ZFPrecoder, RemoveNulledSubcarriers
from sionna.channel.tr38901 import Antenna, AntennaArray, CDL, UMi, UMa, RMa
from sionna.channel import gen_single_sector_topology as gen_topology
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel
from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper
from sionna.utils import BinarySource, ebnodb2no, sim_ber, QAMSource
from sionna.utils.metrics import compute_ber
```

## System Setup

We will now configure all components of the system model step-by-step.


```python
scenario = "umi"
carrier_frequency = 3.5e9
direction = "uplink"
num_ut = 4
batch_size = 32
```

```python
tf.random.set_seed(1)
# Define the UT antenna array
ut_array = Antenna(polarization="single",
                   polarization_type="V",
                   antenna_pattern="omni",
                   carrier_frequency=carrier_frequency)
# Define the BS antenna array
bs_array = AntennaArray(num_rows=1,
                        num_cols=4,
                        polarization="dual",
                        polarization_type="VH",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)
# Create channel model
channel_model = UMi(carrier_frequency=carrier_frequency,
                    o2i_model="low",
                    ut_array=ut_array,
                    bs_array=bs_array,
                    direction=direction,
                    enable_pathloss=False,
                    enable_shadow_fading=False)
# Generate the topology
topology = gen_topology(batch_size, num_ut, scenario)
# Set the topology
channel_model.set_topology(*topology)
# Visualize the topology
channel_model.show_topology()
```


```python
# The number of transmitted streams is equal to the number of UT antennas
num_streams_per_tx = 1
# Create an RX-TX association matrix
# rx_tx_association[i,j]=1 means that receiver i gets at least one stream
# from transmitter j. Depending on the transmission direction (uplink or downlink),
# the role of UT and BS can change. However, as we have only a single
# transmitter and receiver, this does not matter:
rx_tx_association = np.zeros([1, num_ut])
rx_tx_association[0, :] = 1
# Instantiate a StreamManagement object
# This determines which data streams are determined for which receiver.
# In this simple setup, this is fairly simple. However, it can get complicated
# for simulations with many transmitters and receivers.
sm = StreamManagement(rx_tx_association, num_streams_per_tx)
```

```python
rg = ResourceGrid(num_ofdm_symbols=14,
                  fft_size=128,
                  subcarrier_spacing=30e3,
                  num_tx=num_ut,
                  num_streams_per_tx=num_streams_per_tx,
                  cyclic_prefix_length=20,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=[2,11])
rg.show();
```


```python
num_bits_per_symbol = 2 # QPSK modulation
coderate = 0.5 # The code rate
n = int(rg.num_data_symbols*num_bits_per_symbol) # Number of coded bits
k = int(n*coderate) # Number of information bits
# The binary source will create batches of information bits
binary_source = BinarySource()
qam_source = QAMSource(num_bits_per_symbol)
# The encoder maps information bits to coded bits
encoder = LDPC5GEncoder(k, n)
# The mapper maps blocks of information bits to constellation symbols
mapper = Mapper("qam", num_bits_per_symbol)
# The resource grid mapper maps symbols onto an OFDM resource grid
rg_mapper = ResourceGridMapper(rg)
# This function removes nulled subcarriers from any tensor having the shape of a resource grid
remove_nulled_scs = RemoveNulledSubcarriers(rg)
# The LS channel estimator will provide channel estimates and error variances
ls_est = LSChannelEstimator(rg, interpolation_type="nn")
# The LMMSE equalizer will provide soft symbols together with noise variance estimates
lmmse_equ = LMMSEEqualizer(rg, sm)
# The demapper produces LLR for all coded bits
demapper = Demapper("app", "qam", num_bits_per_symbol)
# The decoder provides hard-decisions on the information bits
decoder = LDPC5GDecoder(encoder, hard_out=True)
# OFDM CHannel
ofdm_channel = OFDMChannel(channel_model, rg, add_awgn=True, normalize_channel=False, return_channel=True)
channel_freq = ApplyOFDMChannel(add_awgn=True)
frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
```

## Uplink Transmissions in the Frequency Domain

We now simulate a batch of uplink transmissions. We keep references to the estimated and actual channel frequency responses.


```python
ebno_db = 10
no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)
b = binary_source([batch_size, num_ut, rg.num_streams_per_tx, encoder.k])
c = encoder(b)
x = mapper(c)
x_rg = rg_mapper(x)
a, tau = channel_model(num_time_samples=rg.num_ofdm_symbols, sampling_frequency=1/rg.ofdm_symbol_duration)
h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
y = channel_freq([x_rg, h_freq, no])
h_hat, err_var = ls_est ([y, no])
x_hat, no_eff = lmmse_equ([y, h_hat, err_var, no])
llr = demapper([x_hat, no_eff])
b_hat = decoder(llr)
print("BER: {}".format(compute_ber(b, b_hat).numpy()))
```


```python
BER: 6.103515625e-05
```
### Compare Estimated and Actual Frequency Responses

We can now compare the estimated frequency responses and ground truth:


```python
# In the example above, we assumed perfect CSI, i.e.,
# h_hat correpsond to the exact ideal channel frequency response.
h_perf = remove_nulled_scs(h_freq)[0,0,0,0,0,0]
# We now compute the LS channel estimate from the pilots.
h_est = h_hat[0,0,0,0,0,0]
plt.figure()
plt.plot(np.real(h_perf))
plt.plot(np.imag(h_perf))
plt.plot(np.real(h_est), "--")
plt.plot(np.imag(h_est), "--")
plt.xlabel("Subcarrier index")
plt.ylabel("Channel frequency response")
plt.legend(["Ideal (real part)", "Ideal (imaginary part)", "Estimated (real part)", "Estimated (imaginary part)"]);
plt.title("Comparison of channel frequency responses");
```


### Understand the Difference Between the Channel Models

Before we proceed with more advanced simulations, it is important to understand the differences between the UMi, UMa, and RMa models. In the following code snippet, we compute the empirical cummulative distribution function (CDF) of the condition number of the channel frequency response matrix between all receiver and transmit antennas.


```python
def cond_hist(scenario):
    """Generates a histogram of the channel condition numbers"""
    # Setup a CIR generator
    if scenario == "umi":
        channel_model = UMi(carrier_frequency=carrier_frequency,
                                      o2i_model="low",
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
    elif scenario == "uma":
        channel_model = UMa(carrier_frequency=carrier_frequency,
                                      o2i_model="low",
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
    elif scenario == "rma":
        channel_model = RMa(carrier_frequency=carrier_frequency,
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
    topology = gen_topology(1024, num_ut, scenario)
    # Set the topology
    channel_model.set_topology(*topology)
    # Generate random CIR realizations
    # As we nned only a single sample in time, the sampling_frequency
    # does not matter.
    cir = channel_model(1, 1)
    # Compute the frequency response
    h = cir_to_ofdm_channel(frequencies, *cir, normalize=True)
    h = tf.squeeze(h)
    h = tf.transpose(h, [0,3,1,2])
    # Compute condition number
    c = np.reshape(np.linalg.cond(h), [-1])
    # Compute normalized histogram
    hist, bins = np.histogram(c, 100, (1, 100))
    hist = hist/np.sum(hist)
    return bins[:-1], hist
plt.figure()
for cdl_model in ["umi", "uma", "rma"]:
    bins, hist = cond_hist(cdl_model)
    plt.plot(bins, np.cumsum(hist))
plt.xlim([0,40])
plt.legend(["UMi", "UMa", "RMa"]);
plt.xlabel("Channel Condition Number")
plt.ylabel("CDF")
plt.title("CDF of the channel condition number");
```


From the figure above, you can observe that the UMi and UMa models are substantially better conditioned than the RMa models. This makes them more suitable for MIMO transmissions as we will observe in the next section.

It is also interesting to look at the channel frequency responses of these different models, as done in the next cell:


```python
def freq_response(scenario):
    """Generates an example frequency response"""
    tf.random.set_seed(2)
    # Setup a CIR generator
    if scenario == "umi":
        channel_model = UMi(carrier_frequency=carrier_frequency,
                                      o2i_model="low",
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
    elif scenario == "uma":
        channel_model = UMa(carrier_frequency=carrier_frequency,
                                      o2i_model="low",
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
    elif scenario == "rma":
        channel_model = RMa(carrier_frequency=carrier_frequency,
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
    topology = gen_topology(1, num_ut, scenario)
    # Set the topology
    channel_model.set_topology(*topology)
    # Generate random CIR realizations
    # As we nned only a single sample in time, the sampling_frequency
    # does not matter.
    cir = channel_model(1, 1)
    # Compute the frequency response
    h = cir_to_ofdm_channel(frequencies, *cir, normalize=True)
    h = tf.squeeze(h)
    return h[0,0]
plt.figure()
for cdl_model in ["umi", "uma", "rma"]:
    h = freq_response(cdl_model)
    plt.plot(np.real(h))
plt.legend(["UMi", "UMa", "RMa"]);
plt.xlabel("Subcarrier Index")
plt.ylabel(r"$\Re(h)$")
plt.title("Channel frequency response");
```


The RMa model has significantly less frequency selectivity than the other models which makes channel estimation easier.

### Setup a Keras Model for BER simulations


```python
# We need to enable sionna.config.xla_compat before we can use
# tf.function with jit_compile=True.
# See https://nvlabs.github.io/sionna/api/config.html#sionna.Config.xla_compat
sionna.config.xla_compat=True
class Model(tf.keras.Model):
    """Simulate OFDM MIMO transmissions over a 3GPP 38.901 model.
    """
    def __init__(self, scenario, perfect_csi):
        super().__init__()
        self._scenario = scenario
        self._perfect_csi = perfect_csi
        # Internally set parameters
        self._carrier_frequency = 3.5e9
        self._fft_size = 128
        self._subcarrier_spacing = 30e3
        self._num_ofdm_symbols = 14
        self._cyclic_prefix_length = 20
        self._pilot_ofdm_symbol_indices = [2, 11]
        self._num_bs_ant = 8
        self._num_ut = 4
        self._num_ut_ant = 1
        self._num_bits_per_symbol = 2
        self._coderate = 0.5
        # Create an RX-TX association matrix
        # rx_tx_association[i,j]=1 means that receiver i gets at least one stream
        # from transmitter j. Depending on the transmission direction (uplink or downlink),
        # the role of UT and BS can change.
        bs_ut_association = np.zeros([1, self._num_ut])
        bs_ut_association[0, :] = 1
        self._rx_tx_association = bs_ut_association
        self._num_tx = self._num_ut
        self._num_streams_per_tx = self._num_ut_ant

        # Setup an OFDM Resource Grid
        self._rg = ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                fft_size=self._fft_size,
                                subcarrier_spacing=self._subcarrier_spacing,
                                num_tx=self._num_tx,
                                num_streams_per_tx=self._num_streams_per_tx,
                                cyclic_prefix_length=self._cyclic_prefix_length,
                                pilot_pattern="kronecker",
                                pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)
        # Setup StreamManagement
        self._sm = StreamManagement(self._rx_tx_association, self._num_streams_per_tx)
        # Configure antenna arrays
        self._ut_array = AntennaArray(
                                 num_rows=1,
                                 num_cols=1,
                                 polarization="single",
                                 polarization_type="V",
                                 antenna_pattern="omni",
                                 carrier_frequency=self._carrier_frequency)
        self._bs_array = AntennaArray(num_rows=1,
                                      num_cols=int(self._num_bs_ant/2),
                                      polarization="dual",
                                      polarization_type="cross",
                                      antenna_pattern="38.901",
                                      carrier_frequency=self._carrier_frequency)
        # Configure the channel model
        if self._scenario == "umi":
            self._channel_model = UMi(carrier_frequency=self._carrier_frequency,
                                      o2i_model="low",
                                      ut_array=self._ut_array,
                                      bs_array=self._bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
        elif self._scenario == "uma":
            self._channel_model = UMa(carrier_frequency=self._carrier_frequency,
                                      o2i_model="low",
                                      ut_array=self._ut_array,
                                      bs_array=self._bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
        elif self._scenario == "rma":
            self._channel_model = RMa(carrier_frequency=self._carrier_frequency,
                                      ut_array=self._ut_array,
                                      bs_array=self._bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
        # Instantiate other building blocks
        self._binary_source = BinarySource()
        self._qam_source = QAMSource(self._num_bits_per_symbol)
        self._n = int(self._rg.num_data_symbols*self._num_bits_per_symbol) # Number of coded bits
        self._k = int(self._n*self._coderate)                              # Number of information bits
        self._encoder = LDPC5GEncoder(self._k, self._n)
        self._decoder = LDPC5GDecoder(self._encoder)
        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)
        self._ofdm_channel = OFDMChannel(self._channel_model, self._rg, add_awgn=True,
                                         normalize_channel=True, return_channel=True)
        self._remove_nulled_subcarriers = RemoveNulledSubcarriers(self._rg)
        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
        self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
        self._demapper = Demapper("app", "qam", self._num_bits_per_symbol)
    def new_topology(self, batch_size):
        """Set new topology"""
        topology = gen_topology(batch_size,
                                self._num_ut,
                                self._scenario,
                                min_ut_velocity=0.0,
                                max_ut_velocity=0.0)
        self._channel_model.set_topology(*topology)
    @tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db):
        self.new_topology(batch_size)
        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        b = self._binary_source([batch_size, self._num_tx, self._num_streams_per_tx, self._k])
        c = self._encoder(b)
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)
        y, h = self._ofdm_channel([x_rg, no])
        if self._perfect_csi:
            h_hat = self._remove_nulled_subcarriers(h)
            err_var = 0.0
        else:
            h_hat, err_var = self._ls_est ([y, no])
        x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no])
        llr = self._demapper([x_hat, no_eff])
        b_hat = self._decoder(llr)
        return b, b_hat
```


If you do not want to run the simulations (which can take quite some time) yourself, you can skip the next cell and simply visualize the results in the next cell. The reason why we simulate for a rather large number of `max_mc_iter`, is that for every batch, a new topology (i.e., user drop with new large-scale parameters) is generated. Our goal is here to average over many such topologies.


```python
SIMS = {
    "ebno_db" : list(np.arange(-5, 16, 1.0)),
    "scenario" : ["umi", "uma", "rma"],
    "perfect_csi" : [True, False],
    "ber" : [],
    "bler" : [],
    "duration" : None
}
start = time.time()
for scenario in SIMS["scenario"]:
    for perfect_csi in SIMS["perfect_csi"]:
        model = Model(scenario=scenario,
                      perfect_csi=perfect_csi)
        ber, bler = sim_ber(model,
                            SIMS["ebno_db"],
                            batch_size=128,
                            max_mc_iter=1000,
                            num_target_block_errors=1000)
        SIMS["ber"].append(list(ber.numpy()))
        SIMS["bler"].append(list(bler.numpy()))
SIMS["duration"] = time.time() -  start
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -5.0 | 7.1096e-02 | 5.0537e-01 |      223649 |     3145728 |         1035 |        2048 |        27.9 |reached target block errors
     -4.0 | 3.9041e-02 | 2.6855e-01 |      245625 |     6291456 |         1100 |        4096 |         0.4 |reached target block errors
     -3.0 | 2.1455e-02 | 1.3979e-01 |      236218 |    11010048 |         1002 |        7168 |         0.7 |reached target block errors
     -2.0 | 1.1953e-02 | 7.8281e-02 |      235015 |    19660800 |         1002 |       12800 |         1.3 |reached target block errors
     -1.0 | 6.6370e-03 | 4.1260e-02 |      250538 |    37748736 |         1014 |       24576 |         2.5 |reached target block errors
      0.0 | 3.9849e-03 | 2.3321e-02 |      263243 |    66060288 |         1003 |       43008 |         4.4 |reached target block errors
      1.0 | 2.4968e-03 | 1.5171e-02 |      253302 |   101449728 |         1002 |       66048 |         6.7 |reached target block errors
      2.0 | 1.5606e-03 | 9.0965e-03 |      265100 |   169869312 |         1006 |      110592 |        11.3 |reached target block errors
      3.0 | 9.8858e-04 | 5.6669e-03 |      268220 |   271319040 |         1001 |      176640 |        18.1 |reached target block errors
      4.0 | 5.9893e-04 | 3.5849e-03 |      257647 |   430178304 |         1004 |      280064 |        28.7 |reached target block errors
      5.0 | 3.7780e-04 | 2.2014e-03 |      264134 |   699138048 |         1002 |      455168 |        46.8 |reached target block errors
      6.0 | 2.2261e-04 | 1.2910e-03 |      175069 |   786432000 |          661 |      512000 |        52.8 |reached max iter
      7.0 | 1.4676e-04 | 8.3203e-04 |      115414 |   786432000 |          426 |      512000 |        52.8 |reached max iter
      8.0 | 8.2901e-05 | 4.9609e-04 |       65196 |   786432000 |          254 |      512000 |        52.8 |reached max iter
      9.0 | 6.4590e-05 | 3.8477e-04 |       50796 |   786432000 |          197 |      512000 |        52.6 |reached max iter
     10.0 | 3.7591e-05 | 2.5000e-04 |       29563 |   786432000 |          128 |      512000 |        52.7 |reached max iter
     11.0 | 2.3387e-05 | 1.5820e-04 |       18392 |   786432000 |           81 |      512000 |        52.6 |reached max iter
     12.0 | 1.9283e-05 | 1.2305e-04 |       15165 |   786432000 |           63 |      512000 |        52.6 |reached max iter
     13.0 | 8.9823e-06 | 7.0312e-05 |        7064 |   786432000 |           36 |      512000 |        52.6 |reached max iter
     14.0 | 5.1371e-06 | 5.2734e-05 |        4040 |   786432000 |           27 |      512000 |        52.6 |reached max iter
     15.0 | 2.2189e-06 | 2.5391e-05 |        1745 |   786432000 |           13 |      512000 |        52.6 |reached max iter
WARNING:tensorflow:From /usr/local/lib/python3.8/dist-packages/tensorflow/python/util/dispatch.py:1082: calling gather (from tensorflow.python.ops.array_ops) with validate_indices is deprecated and will be removed in a future version.
Instructions for updating:
The `validate_indices` argument has no effect. Indices are always validated on CPU and never validated on GPU.
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -5.0 | 2.9643e-01 | 1.0000e+00 |      466249 |     1572864 |         1024 |        1024 |        21.3 |reached target block errors
     -4.0 | 2.6703e-01 | 1.0000e+00 |      420009 |     1572864 |         1024 |        1024 |         0.1 |reached target block errors
     -3.0 | 2.2791e-01 | 9.9609e-01 |      358464 |     1572864 |         1020 |        1024 |         0.1 |reached target block errors
     -2.0 | 1.6929e-01 | 8.5156e-01 |      399413 |     2359296 |         1308 |        1536 |         0.2 |reached target block errors
     -1.0 | 1.1276e-01 | 5.9424e-01 |      354712 |     3145728 |         1217 |        2048 |         0.2 |reached target block errors
      0.0 | 7.0166e-02 | 3.7630e-01 |      331083 |     4718592 |         1156 |        3072 |         0.3 |reached target block errors
      1.0 | 3.9429e-02 | 2.1723e-01 |      279074 |     7077888 |         1001 |        4608 |         0.5 |reached target block errors
      2.0 | 2.5046e-02 | 1.4089e-01 |      295455 |    11796480 |         1082 |        7680 |         0.8 |reached target block errors
      3.0 | 1.6036e-02 | 8.6362e-02 |      290063 |    18087936 |         1017 |       11776 |         1.2 |reached target block errors
      4.0 | 1.0529e-02 | 5.5971e-02 |      289820 |    27525120 |         1003 |       17920 |         1.9 |reached target block errors
      5.0 | 7.5292e-03 | 3.8274e-02 |      307901 |    40894464 |         1019 |       26624 |         2.8 |reached target block errors
      6.0 | 5.1773e-03 | 2.5930e-02 |      309443 |    59768832 |         1009 |       38912 |         4.0 |reached target block errors
      7.0 | 3.6257e-03 | 1.8008e-02 |      310801 |    85721088 |         1005 |       55808 |         5.8 |reached target block errors
      8.0 | 2.6579e-03 | 1.3781e-02 |      298907 |   112459776 |         1009 |       73216 |         7.6 |reached target block errors
      9.0 | 2.2009e-03 | 1.1557e-02 |      292515 |   132907008 |         1000 |       86528 |         9.0 |reached target block errors
     10.0 | 1.6736e-03 | 9.7851e-03 |      264557 |   158072832 |         1007 |      102912 |        10.7 |reached target block errors
     11.0 | 1.3004e-03 | 8.8067e-03 |      227037 |   174587904 |         1001 |      113664 |        11.8 |reached target block errors
     12.0 | 1.2042e-03 | 9.6689e-03 |      191298 |   158859264 |         1000 |      103424 |        10.8 |reached target block errors
     13.0 | 1.1146e-03 | 1.0589e-02 |      162159 |   145489920 |         1003 |       94720 |         9.8 |reached target block errors
     14.0 | 1.0679e-03 | 1.2168e-02 |      135214 |   126615552 |         1003 |       82432 |         8.6 |reached target block errors
     15.0 | 1.1502e-03 | 1.4826e-02 |      119396 |   103809024 |         1002 |       67584 |         7.0 |reached target block errors
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -5.0 | 7.2186e-02 | 5.1758e-01 |      227077 |     3145728 |         1060 |        2048 |        21.7 |reached target block errors
     -4.0 | 3.4300e-02 | 2.4609e-01 |      242769 |     7077888 |         1134 |        4608 |         0.5 |reached target block errors
     -3.0 | 1.6501e-02 | 1.2017e-01 |      220608 |    13369344 |         1046 |        8704 |         0.9 |reached target block errors
     -2.0 | 8.1806e-03 | 5.9837e-02 |      212304 |    25952256 |         1011 |       16896 |         1.7 |reached target block errors
     -1.0 | 4.3676e-03 | 3.0048e-02 |      223264 |    51118080 |         1000 |       33280 |         3.4 |reached target block errors
      0.0 | 2.6359e-03 | 1.7006e-02 |      240466 |    91226112 |         1010 |       59392 |         6.1 |reached target block errors
      1.0 | 1.3907e-03 | 8.9862e-03 |      238431 |   171442176 |         1003 |      111616 |        11.5 |reached target block errors
      2.0 | 8.0677e-04 | 5.0052e-03 |      248078 |   307494912 |         1002 |      200192 |        20.7 |reached target block errors
      3.0 | 4.0438e-04 | 2.5365e-03 |      244874 |   605552640 |         1000 |      394240 |        40.9 |reached target block errors
      4.0 | 2.1496e-04 | 1.3184e-03 |      169054 |   786432000 |          675 |      512000 |        53.1 |reached max iter
      5.0 | 1.2614e-04 | 8.0664e-04 |       99201 |   786432000 |          413 |      512000 |        53.0 |reached max iter
      6.0 | 6.7013e-05 | 4.1016e-04 |       52701 |   786432000 |          210 |      512000 |        53.0 |reached max iter
      7.0 | 3.5152e-05 | 2.0508e-04 |       27645 |   786432000 |          105 |      512000 |        52.9 |reached max iter
      8.0 | 2.0339e-05 | 1.3477e-04 |       15995 |   786432000 |           69 |      512000 |        52.9 |reached max iter
      9.0 | 1.2319e-05 | 6.4453e-05 |        9688 |   786432000 |           33 |      512000 |        52.9 |reached max iter
     10.0 | 7.5582e-06 | 5.2734e-05 |        5944 |   786432000 |           27 |      512000 |        53.0 |reached max iter
     11.0 | 3.7626e-06 | 2.9297e-05 |        2959 |   786432000 |           15 |      512000 |        53.0 |reached max iter
     12.0 | 1.7904e-06 | 2.1484e-05 |        1408 |   786432000 |           11 |      512000 |        53.0 |reached max iter
     13.0 | 1.1584e-06 | 1.1719e-05 |         911 |   786432000 |            6 |      512000 |        53.0 |reached max iter
     14.0 | 7.2988e-07 | 3.9063e-06 |         574 |   786432000 |            2 |      512000 |        53.0 |reached max iter
     15.0 | 1.0351e-06 | 5.8594e-06 |         814 |   786432000 |            3 |      512000 |        53.0 |reached max iter
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -5.0 | 2.9996e-01 | 1.0000e+00 |      471791 |     1572864 |         1024 |        1024 |        21.1 |reached target block errors
     -4.0 | 2.7072e-01 | 1.0000e+00 |      425811 |     1572864 |         1024 |        1024 |         0.1 |reached target block errors
     -3.0 | 2.3427e-01 | 1.0000e+00 |      368467 |     1572864 |         1024 |        1024 |         0.1 |reached target block errors
     -2.0 | 1.7587e-01 | 8.7174e-01 |      414940 |     2359296 |         1339 |        1536 |         0.2 |reached target block errors
     -1.0 | 1.2033e-01 | 6.3867e-01 |      378535 |     3145728 |         1308 |        2048 |         0.2 |reached target block errors
      0.0 | 6.9556e-02 | 3.9141e-01 |      273507 |     3932160 |         1002 |        2560 |         0.3 |reached target block errors
      1.0 | 4.4011e-02 | 2.4132e-01 |      311505 |     7077888 |         1112 |        4608 |         0.5 |reached target block errors
      2.0 | 2.8742e-02 | 1.5415e-01 |      293844 |    10223616 |         1026 |        6656 |         0.7 |reached target block errors
      3.0 | 1.7114e-02 | 9.0376e-02 |      296091 |    17301504 |         1018 |       11264 |         1.1 |reached target block errors
      4.0 | 1.0157e-02 | 5.1282e-02 |      311521 |    30670848 |         1024 |       19968 |         2.0 |reached target block errors
      5.0 | 8.0138e-03 | 3.9414e-02 |      315114 |    39321600 |         1009 |       25600 |         2.6 |reached target block errors
      6.0 | 5.7654e-03 | 2.7103e-02 |      330990 |    57409536 |         1013 |       37376 |         3.8 |reached target block errors
      7.0 | 4.3783e-03 | 2.0406e-02 |      330547 |    75497472 |         1003 |       49152 |         5.0 |reached target block errors
      8.0 | 3.5508e-03 | 1.6734e-02 |      329507 |    92798976 |         1011 |       60416 |         6.1 |reached target block errors
      9.0 | 3.0306e-03 | 1.5171e-02 |      307449 |   101449728 |         1002 |       66048 |         6.7 |reached target block errors
     10.0 | 2.6278e-03 | 1.4482e-02 |      278990 |   106168320 |         1001 |       69120 |         7.0 |reached target block errors
     11.0 | 2.3524e-03 | 1.5159e-02 |      240503 |   102236160 |         1009 |       66560 |         6.7 |reached target block errors
     12.0 | 2.5297e-03 | 1.8139e-02 |      214857 |    84934656 |         1003 |       55296 |         5.6 |reached target block errors
     13.0 | 2.3190e-03 | 2.1918e-02 |      164137 |    70778880 |         1010 |       46080 |         4.7 |reached target block errors
     14.0 | 2.2989e-03 | 2.5619e-02 |      139211 |    60555264 |         1010 |       39424 |         4.0 |reached target block errors
     15.0 | 2.2612e-03 | 2.8674e-02 |      122701 |    54263808 |         1013 |       35328 |         3.6 |reached target block errors
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -5.0 | 1.1218e-01 | 6.0400e-01 |      352900 |     3145728 |         1237 |        2048 |        20.2 |reached target block errors
     -4.0 | 7.6053e-02 | 4.0469e-01 |      299052 |     3932160 |         1036 |        2560 |         0.3 |reached target block errors
     -3.0 | 6.6383e-02 | 3.2878e-01 |      313232 |     4718592 |         1010 |        3072 |         0.3 |reached target block errors
     -2.0 | 5.0129e-02 | 2.5195e-01 |      315385 |     6291456 |         1032 |        4096 |         0.4 |reached target block errors
     -1.0 | 3.3127e-02 | 1.7057e-01 |      312623 |     9437184 |         1048 |        6144 |         0.6 |reached target block errors
      0.0 | 2.7713e-02 | 1.3984e-01 |      326920 |    11796480 |         1074 |        7680 |         0.8 |reached target block errors
      1.0 | 1.9857e-02 | 9.7656e-02 |      312320 |    15728640 |         1000 |       10240 |         1.0 |reached target block errors
      2.0 | 1.6031e-02 | 8.0078e-02 |      315191 |    19660800 |         1025 |       12800 |         1.3 |reached target block errors
      3.0 | 1.1741e-02 | 5.7904e-02 |      313944 |    26738688 |         1008 |       17408 |         1.7 |reached target block errors
      4.0 | 9.4707e-03 | 4.6596e-02 |      312820 |    33030144 |         1002 |       21504 |         2.1 |reached target block errors
      5.0 | 7.3535e-03 | 3.6350e-02 |      312285 |    42467328 |         1005 |       27648 |         2.7 |reached target block errors
      6.0 | 5.7442e-03 | 2.8292e-02 |      316219 |    55050240 |         1014 |       35840 |         3.6 |reached target block errors
      7.0 | 4.3868e-03 | 2.2978e-02 |      293242 |    66846720 |         1000 |       43520 |         4.3 |reached target block errors
      8.0 | 3.4328e-03 | 1.7235e-02 |      307761 |    89653248 |         1006 |       58368 |         5.8 |reached target block errors
      9.0 | 2.7156e-03 | 1.3672e-02 |      305395 |   112459776 |         1001 |       73216 |         7.3 |reached target block errors
     10.0 | 2.0763e-03 | 1.1172e-02 |      285749 |   137625600 |         1001 |       89600 |         8.9 |reached target block errors
     11.0 | 1.5832e-03 | 8.3993e-03 |      290097 |   183238656 |         1002 |      119296 |        11.9 |reached target block errors
     12.0 | 1.1819e-03 | 6.4524e-03 |      281639 |   238288896 |         1001 |      155136 |        15.4 |reached target block errors
     13.0 | 9.3162e-04 | 4.9874e-03 |      287202 |   308281344 |         1001 |      200704 |        20.0 |reached target block errors
     14.0 | 7.0943e-04 | 3.9981e-03 |      272823 |   384565248 |         1001 |      250368 |        24.9 |reached target block errors
     15.0 | 5.7079e-04 | 3.2230e-03 |      272025 |   476577792 |         1000 |      310272 |        30.9 |reached target block errors
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -5.0 | 2.9893e-01 | 1.0000e+00 |      470184 |     1572864 |         1024 |        1024 |        21.0 |reached target block errors
     -4.0 | 2.7396e-01 | 1.0000e+00 |      430909 |     1572864 |         1024 |        1024 |         0.1 |reached target block errors
     -3.0 | 2.2892e-01 | 9.8145e-01 |      360059 |     1572864 |         1005 |        1024 |         0.1 |reached target block errors
     -2.0 | 1.7590e-01 | 7.8841e-01 |      414991 |     2359296 |         1211 |        1536 |         0.2 |reached target block errors
     -1.0 | 1.2533e-01 | 5.8057e-01 |      394240 |     3145728 |         1189 |        2048 |         0.2 |reached target block errors
      0.0 | 1.0202e-01 | 4.6133e-01 |      401178 |     3932160 |         1181 |        2560 |         0.3 |reached target block errors
      1.0 | 7.9942e-02 | 3.6165e-01 |      377212 |     4718592 |         1111 |        3072 |         0.3 |reached target block errors
      2.0 | 6.4012e-02 | 2.9492e-01 |      352387 |     5505024 |         1057 |        3584 |         0.4 |reached target block errors
      3.0 | 5.1941e-02 | 2.3698e-01 |      367634 |     7077888 |         1092 |        4608 |         0.5 |reached target block errors
      4.0 | 3.8278e-02 | 1.7464e-01 |      361239 |     9437184 |         1073 |        6144 |         0.6 |reached target block errors
      5.0 | 2.9897e-02 | 1.3503e-01 |      352682 |    11796480 |         1037 |        7680 |         0.8 |reached target block errors
      6.0 | 2.4803e-02 | 1.1306e-01 |      351106 |    14155776 |         1042 |        9216 |         0.9 |reached target block errors
      7.0 | 1.8750e-02 | 8.5003e-02 |      339141 |    18087936 |         1001 |       11776 |         1.2 |reached target block errors
      8.0 | 1.4703e-02 | 6.7619e-02 |      335327 |    22806528 |         1004 |       14848 |         1.5 |reached target block errors
      9.0 | 1.1462e-02 | 5.3262e-02 |      333508 |    29097984 |         1009 |       18944 |         1.9 |reached target block errors
     10.0 | 8.3913e-03 | 3.9570e-02 |      329960 |    39321600 |         1013 |       25600 |         2.5 |reached target block errors
     11.0 | 7.3333e-03 | 3.3766e-02 |      340262 |    46399488 |         1020 |       30208 |         3.0 |reached target block errors
     12.0 | 5.8008e-03 | 2.7127e-02 |      328462 |    56623104 |         1000 |       36864 |         3.7 |reached target block errors
     13.0 | 4.6824e-03 | 2.1832e-02 |      331412 |    70778880 |         1006 |       46080 |         4.6 |reached target block errors
     14.0 | 3.4007e-03 | 1.6544e-02 |      318254 |    93585408 |         1008 |       60928 |         6.1 |reached target block errors
     15.0 | 2.5160e-03 | 1.2143e-02 |      318566 |   126615552 |         1001 |       82432 |         8.2 |reached target block errors
```

```python
# Load results (uncomment to show saved results from the cell above)
#SIMS = eval("{'ebno_db': [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], 'scenario': ['umi', 'uma', 'rma'], 'perfect_csi': [True, False], 'ber': [[0.07905292510986328, 0.03808736801147461, 0.017681360244750977, 0.009239894087596606, 0.0050665537516276045, 0.0027629886053304755, 0.0016827532040175571, 0.0008500541736877042, 0.0004983015045593167, 0.00031632105509440104, 0.00018594996134440104, 0.00010455576578776041, 6.090927124023438e-05, 3.9520263671875e-05, 2.5684356689453124e-05, 1.4940897623697916e-05, 7.539113362630208e-06, 4.683176676432292e-06, 3.59344482421875e-06, 1.5436808268229167e-06, 7.578531901041666e-07], [0.29719797770182294, 0.26843706766764325, 0.2296581268310547, 0.17483605278862846, 0.10778331756591797, 0.07155566745334202, 0.04232830471462674, 0.022064606348673504, 0.015027618408203125, 0.008189432548754143, 0.005684130119554924, 0.00370962642928929, 0.002616008663393743, 0.0019156403011745876, 0.0015677095994417248, 0.0013081868489583333, 0.0010344430083840252, 0.0010432377567997685, 0.0009155009900555915, 0.0009102860117346291, 0.0008864811488560267], [0.06746260325113933, 0.0329127311706543, 0.014757650869864004, 0.007593437477394387, 0.003813561333550347, 0.0018911941496331237, 0.001028917273696588, 0.000513251788663255, 0.0002842496236165365, 0.00016032155354817708, 8.596547444661459e-05, 4.677454630533854e-05, 2.0359039306640624e-05, 1.1446634928385416e-05, 1.0133107503255209e-05, 3.92913818359375e-07, 1.682281494140625e-06, 6.421407063802083e-07, 1.3987223307291666e-08, 4.895528157552083e-07, 1.2715657552083333e-09], [0.2960662841796875, 0.2712268829345703, 0.2315998077392578, 0.17950481838650173, 0.11626561482747395, 0.0681504143608941, 0.04071949146412037, 0.02562223161969866, 0.014265790397738233, 0.009787991515591614, 0.006755871242947049, 0.004930473776424632, 0.003845776165569867, 0.003375189644949777, 0.0026965757616523173, 0.002434003298685431, 0.002402254330214634, 0.0021742226420969203, 0.0020746665425819925, 0.0021730139552350023, 0.0022606077648344492], [0.09145228068033855, 0.06702995300292969, 0.05034939448038737, 0.03308937766335227, 0.024936834971110027, 0.018199747258966618, 0.014243278974368249, 0.010329062478584155, 0.00815982288784451, 0.006009458884214744, 0.004231770833333333, 0.0033478243597622574, 0.0025576324126052015, 0.0019759736530521373, 0.0014438384618514623, 0.001123528113731971, 0.0008716690492438504, 0.0006736387408842243, 0.0004936694871486797, 0.00040878831294544955, 0.0002836583455403646], [0.2979132334391276, 0.2661628723144531, 0.22218640645345053, 0.1630952623155382, 0.11636797587076823, 0.08418807983398438, 0.06609598795572917, 0.047115184642650465, 0.035385449727376304, 0.026970704396565754, 0.02249379743609512, 0.016286409818209134, 0.011899021693638392, 0.008838085418051861, 0.007266274813948007, 0.005744590415610923, 0.0042660244551720895, 0.003129789240790991, 0.002527833716269651, 0.002067384265718006, 0.0014839694274598686]], 'bler': [[0.53173828125, 0.26806640625, 0.1279296875, 0.0637600806451613, 0.03380926724137931, 0.017422566371681415, 0.010500672043010752, 0.00547640931372549, 0.0030405569595645414, 0.001916015625, 0.0010703125, 0.00063671875, 0.00039453125, 0.000236328125, 0.000154296875, 9.1796875e-05, 6.25e-05, 3.515625e-05, 2.5390625e-05, 1.5625e-05, 1.171875e-05], [1.0, 1.0, 0.994140625, 0.86328125, 0.59619140625, 0.392578125, 0.2348090277777778, 0.127197265625, 0.081484375, 0.04469992897727273, 0.03009588068181818, 0.019152002427184466, 0.01331313775510204, 0.01019287109375, 0.008390893240343348, 0.00784375, 0.007462130248091603, 0.008697916666666666, 0.01016029792746114, 0.011945026676829269, 0.014048549107142858], [0.49609375, 0.25244140625, 0.10894097222222222, 0.05495876736111111, 0.026328125, 0.01235750786163522, 0.006649925595238095, 0.0034094621080139375, 0.001748046875, 0.0010078125, 0.000537109375, 0.000291015625, 0.000130859375, 9.1796875e-05, 6.0546875e-05, 9.765625e-06, 1.953125e-05, 1.3671875e-05, 5.859375e-06, 3.90625e-06, 1.953125e-06], [1.0, 1.0, 0.9951171875, 0.8912760416666666, 0.62353515625, 0.3834635416666667, 0.2265625, 0.14020647321428573, 0.07458043981481481, 0.050380608974358976, 0.03264973958333333, 0.023115808823529413, 0.017648507882882882, 0.015516493055555556, 0.013366284013605442, 0.013377568493150685, 0.015814012096774195, 0.017085597826086957, 0.0193958849009901, 0.02383753765060241, 0.02837611607142857], [0.5185546875, 0.3645833333333333, 0.267822265625, 0.17844460227272727, 0.128662109375, 0.09259588068181818, 0.07241030092592593, 0.05180921052631579, 0.041056315104166664, 0.030048076923076924, 0.021399456521739132, 0.016904633620689655, 0.013012210264900662, 0.01011981865284974, 0.007512019230769231, 0.0060276442307692305, 0.004715737951807229, 0.0035807291666666665, 0.0027969644134477824, 0.0022843567251461987, 0.001625], [1.0, 1.0, 0.9765625, 0.7649739583333334, 0.55517578125, 0.40625, 0.30747767857142855, 0.22200520833333334, 0.169921875, 0.1259765625, 0.10546875, 0.0751953125, 0.05613839285714286, 0.041555851063829786, 0.034078663793103446, 0.026551942567567568, 0.02025612113402062, 0.014954079198473283, 0.012090203220858896, 0.009994818239795918, 0.007302355410447761]], 'duration': 14960.869339227676}")
plt.figure()
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
i=0
legend = []
for scenario in SIMS["scenario"]:
    for perfect_csi in SIMS["perfect_csi"]:
        if scenario=="umi":
            r = "r"
            t = "UMi"
        elif scenario=="uma":
            r = "b"
            t = "UMa"
        else:
            r = "g"
            t = "RMa"
        if perfect_csi:
            r += "-"
        else:
            r += "--"
        plt.semilogy(SIMS["ebno_db"], SIMS["bler"][i], r);
        s = "{} - {} CSI".format(t,"perf." if perfect_csi else "imperf.")
        legend.append(s)
        i += 1
plt.legend(legend)
plt.ylim([1e-3, 1])
plt.title("Multiuser 4x8 MIMO Uplink over Different 3GPP 38.901 Models");
```


Due to the worse channel conditioning, the RMa model achieves the worst performance with perfect CSI. However, as a result of the smaller frequency selectivity, imperfect channel estimation only leads to a constant 5dB performace loss. For the UMI and UMa models, the used channel estimator with nearest-neighbor interpolation is not accurate enough so that the BER curves saturate at high SNR. This could, for example, be circumvented with another interpolation method (e.g., <a class="reference external" href="https://nvlabs.github.io/sionna/api/ofdm.html#linearinterpolator">linear interpolation
with time averaging</a>) or a different pilot pattern.
with time averaging</a>) or a different pilot pattern.
