# OFDM MIMO Channel Estimation and Detection

In this notebook, we will evaluate some of the OFDM channel estimation and MIMO detection algorithms available in Sionna.

We will start by evaluating the mean square error (MSE) preformance of various channel estimation and interpolation methods.

Then, we will compare some of the MIMO detection algorithms under both perfect and imperfect channel state information (CSI) in terms of uncoded symbol error rate (SER) and coded bit error rate (BER).

The developed end-to-end Keras models in this notebook are a great tool for benchmarking of MIMO receivers under realistic conditions. They can be easily extended to new channel estimation methods or MIMO detection algorithms.

For MSE evaluations, the block diagram of the system looks as follows:


where the channel estimation module is highlighted as it is the focus of this evaluation. The channel covariance matrices are required for linear minimum mean square error (LMMSE) channel interpolation.

For uncoded SER evaluations, the block diagram of the system looks as follows:


where the channel estimation and detection modules are highlighted as they are the focus of this evaluation.

Finally, for coded BER evaluations, the block diagram of the system looks as follows:


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
import pickle
from tensorflow.keras import Model
from sionna.mimo import StreamManagement
from sionna.utils import QAMSource, compute_ser, BinarySource, sim_ber, ebnodb2no, QAMSource
from sionna.mapping import Mapper
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEInterpolator, LinearDetector, KBestDetector, EPDetector, MMSEPICDetector
from sionna.channel import GenerateOFDMChannel, OFDMChannel, gen_single_sector_topology
from sionna.channel.tr38901 import UMi, Antenna, PanelArray
from sionna.fec.ldpc import LDPC5GEncoder
from sionna.fec.ldpc import LDPC5GDecoder
```

## Simulation parameters

The next cell defines the simulation parameters used throughout this notebook.

This includes the OFDM waveform parameters, [antennas geometries and patterns](https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray), and the [3GPP UMi channel model](https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi).


```python
NUM_OFDM_SYMBOLS = 14
FFT_SIZE = 12*4 # 4 PRBs
SUBCARRIER_SPACING = 30e3 # Hz
CARRIER_FREQUENCY = 3.5e9 # Hz
SPEED = 3. # m/s
# The user terminals (UTs) are equipped with a single antenna
# with vertial polarization.
UT_ANTENNA = Antenna(polarization='single',
                     polarization_type='V',
                     antenna_pattern='omni', # Omnidirectional antenna pattern
                     carrier_frequency=CARRIER_FREQUENCY)
# The base station is equipped with an antenna
# array of 8 cross-polarized antennas,
# resulting in a total of 16 antenna elements.
NUM_RX_ANT = 16
BS_ARRAY = PanelArray(num_rows_per_panel=4,
                      num_cols_per_panel=2,
                      polarization='dual',
                      polarization_type='cross',
                      antenna_pattern='38.901', # 3GPP 38.901 antenna pattern
                      carrier_frequency=CARRIER_FREQUENCY)
# 3GPP UMi channel model is considered
CHANNEL_MODEL = UMi(carrier_frequency=CARRIER_FREQUENCY,
                    o2i_model='low',
                    ut_array=UT_ANTENNA,
                    bs_array=BS_ARRAY,
                    direction='uplink',
                    enable_shadow_fading=False,
                    enable_pathloss=False)
```

## Estimation of the channel time, frequency, and spatial covariance matrices

The linear minimum mean square (LMMSE) interpolation method requires knowledge of the time (i.e., across OFDM symbols), frequency (i.e., across sub-carriers), and spatial (i.e., across receive antennas) covariance matrices of the channel frequency response.

These are estimated in this section using Monte Carlo sampling.

We explain below how this is achieved for the frequency covariance matrix. The same approach is used for the time and spatial covariance matrices.

Let $N$ be the number of sub-carriers. The first step for estimating the frequency covariance matrix is to sample the channel model in order to build a set of frequency-domain channel realizations $\left\{ \mathbf{h}_k \right\}, 1 \leq k \leq K$, where $K$ is the number of samples and $\mathbf{h}_k \in \mathbb{C}^{N}$ are complex-valued samples of the channel frequency response.

The frequency covariance matrix $\mathbf{R}^{(f)} \in \mathbb{C}^{N \times N}$ is then estimated by

\begin{equation}
\mathbf{R}^{(f)} \approx \frac{1}{K} \sum_{k = 1}^K \mathbf{h}_k \mathbf{h}_k^{\mathrm{H}}
\end{equation}

where we assume that the frequency-domain channel response has zero mean.

The following cells implement this process for all three dimensions (frequency, time, and space).

The next cell defines a [resource grid](https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid) and an [OFDM channel generator](https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateOFDMChannel) for sampling the channel in the frequency domain.


```python
rg = ResourceGrid(num_ofdm_symbols=NUM_OFDM_SYMBOLS,
                  fft_size=FFT_SIZE,
                  subcarrier_spacing=SUBCARRIER_SPACING)
channel_sampler = GenerateOFDMChannel(CHANNEL_MODEL, rg)
```


Then, a function that samples the channel is defined. It randomly samples a network topology for every batch and for every batch example using the [appropriate utility function](https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.gen_single_sector_topology).


```python
def sample_channel(batch_size):
    # Sample random topologies
    topology = gen_single_sector_topology(batch_size, 1, 'umi', min_ut_velocity=SPEED, max_ut_velocity=SPEED)
    CHANNEL_MODEL.set_topology(*topology)
    # Sample channel frequency responses
    # [batch size, 1, num_rx_ant, 1, 1, num_ofdm_symbols, fft_size]
    h_freq = channel_sampler(batch_size)
    # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
    h_freq = h_freq[:,0,:,0,0]
    return h_freq
```


We now define a function that estimates the frequency, time, and spatial covariance matrcies using Monte Carlo sampling.


```python
@tf.function(jit_compile=True) # Use XLA for speed-up
def estimate_covariance_matrices(num_it, batch_size):
    freq_cov_mat = tf.zeros([FFT_SIZE, FFT_SIZE], tf.complex64)
    time_cov_mat = tf.zeros([NUM_OFDM_SYMBOLS, NUM_OFDM_SYMBOLS], tf.complex64)
    space_cov_mat = tf.zeros([NUM_RX_ANT, NUM_RX_ANT], tf.complex64)
    for _ in tf.range(num_it):
        # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
        h_samples = sample_channel(batch_size)
        #################################
        # Estimate frequency covariance
        #################################
        # [batch size, num_rx_ant, fft_size, num_ofdm_symbols]
        h_samples_ = tf.transpose(h_samples, [0,1,3,2])
        # [batch size, num_rx_ant, fft_size, fft_size]
        freq_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [fft_size, fft_size]
        freq_cov_mat_ = tf.reduce_mean(freq_cov_mat_, axis=(0,1))
        # [fft_size, fft_size]
        freq_cov_mat += freq_cov_mat_
        ################################
        # Estimate time covariance
        ################################
        # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
        time_cov_mat_ = tf.matmul(h_samples, h_samples, adjoint_b=True)
        # [num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat_ = tf.reduce_mean(time_cov_mat_, axis=(0,1))
        # [num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat += time_cov_mat_
        ###############################
        #Estimate spatial covariance
        ###############################
        # [batch size, num_ofdm_symbols, num_rx_ant, fft_size]
        h_samples_ = tf.transpose(h_samples, [0,2,1,3])
        # [batch size, num_ofdm_symbols, num_rx_ant, num_rx_ant]
        space_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [num_rx_ant, num_rx_ant]
        space_cov_mat_ = tf.reduce_mean(space_cov_mat_, axis=(0,1))
        # [num_rx_ant, num_rx_ant]
        space_cov_mat += space_cov_mat_
    freq_cov_mat /= tf.complex(tf.cast(NUM_OFDM_SYMBOLS*num_it, tf.float32), 0.0)
    time_cov_mat /= tf.complex(tf.cast(FFT_SIZE*num_it, tf.float32), 0.0)
    space_cov_mat /= tf.complex(tf.cast(FFT_SIZE*num_it, tf.float32), 0.0)
    return freq_cov_mat, time_cov_mat, space_cov_mat
```


We then compute the estimates by executing the function defined in the previous cell.

The batch size and number of iterations determine the total number of samples, i.e.,
```python
number of samples = batch_size x num_iterations
```


and hence control the tradeoff between the accuracy of the estimates and the time needed for their computation.


```python
batch_size = 1000
num_iterations = 100
sionna.Config.xla_compat = True # Enable Sionna's support of XLA
FREQ_COV_MAT, TIME_COV_MAT, SPACE_COV_MAT = estimate_covariance_matrices(batch_size, num_iterations)
sionna.Config.xla_compat = False # Disable Sionna's support of XLA
```


Finally, the estimated matrices are saved (as numpy arrays) for future use.


```python
# FREQ_COV_MAT : [fft_size, fft_size]
# TIME_COV_MAT : [num_ofdm_symbols, num_ofdm_symbols]
# SPACE_COV_MAT : [num_rx_ant, num_rx_ant]
np.save('freq_cov_mat', FREQ_COV_MAT.numpy())
np.save('time_cov_mat', TIME_COV_MAT.numpy())
np.save('space_cov_mat', SPACE_COV_MAT.numpy())
```

## Loading the channel covariance matrices

The next cell loads saved estimates of the time, frequency, and space covariance matrices.


```python
FREQ_COV_MAT = np.load('freq_cov_mat.npy')
TIME_COV_MAT = np.load('time_cov_mat.npy')
SPACE_COV_MAT = np.load('space_cov_mat.npy')
```


We then visualize the loaded matrices.

As one can see, the frequency correlation slowly decays with increasing spectral distance.

The time-correlation is much stronger as the mobility low. The covariance matrix is hence very badly conditioned with rank almost equal to one.

The spatial covariance matrix has a regular structure which is determined by the array geometry and polarization of its elements.


```python
fig, ax = plt.subplots(3,2, figsize=(10,12))
fig.suptitle("Time and frequency channel covariance matrices")
ax[0,0].set_title("Freq. cov. Real")
im = ax[0,0].imshow(FREQ_COV_MAT.real, vmin=-0.3, vmax=1.8)
ax[0,1].set_title("Freq. cov. Imag")
im = ax[0,1].imshow(FREQ_COV_MAT.imag, vmin=-0.3, vmax=1.8)
ax[1,0].set_title("Time cov. Real")
im = ax[1,0].imshow(TIME_COV_MAT.real, vmin=-0.3, vmax=1.8)
ax[1,1].set_title("Time cov. Imag")
im = ax[1,1].imshow(TIME_COV_MAT.imag, vmin=-0.3, vmax=1.8)
ax[2,0].set_title("Space cov. Real")
im = ax[2,0].imshow(SPACE_COV_MAT.real, vmin=-0.3, vmax=1.8)
ax[2,1].set_title("Space cov. Imag")
im = ax[2,1].imshow(SPACE_COV_MAT.imag, vmin=-0.3, vmax=1.8)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax);
```
```python
<matplotlib.colorbar.Colorbar at 0x7effa863f0d0>
```


## Comparison of OFDM estimators

This section focuses on comparing the available OFDM channel estimators in Sionna for the considered setup.

OFDM channel estimation consists of two steps:
<ol class="arabic simple">
- Channel estimation at pilot-carrying resource elements using [least-squares (LS)](https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LSChannelEstimator).
- Interpolation for data-carrying resource elements, for which three methods are available in Sionna:
</ol>

- [Nearest-neighbor](https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.NearestNeighborInterpolator), which uses the channel estimate of the nearest pilot
- [Linear](https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LinearInterpolator), with optional averaging over the OFDM symbols (time dimension) for low mobility scenarios
- [LMMSE](https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LMMSEInterpolator), which requires knowledge of the time and frequency covariance matrices


The LMMSE interpolator also features optional spatial smoothin, which requires the spatial covarance matrix. The [API documentation](https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LMMSEInterpolator) explains in more detail how this interpolator operates.

### End-to-end model

In the next cell, we will create a Keras model which uses the interpolation method specified at initialization.

It computes the mean square error (MSE) for a specified batch size and signal-to-noise ratio (SNR) (in dB).

The following interpolation methods are available (set through the `int_method` parameter):

- `"nn"` : Nearest-neighbor interpolation
- `"lin"` : Linear interpolation
- `"lmmse"` : LMMSE interpolation


When LMMSE interpolation is used, it is required to specified the order in which interpolation and optional spatial smoothing is performed. This is achieved using the `lmmse_order` parameter. For example, setting this parameter to `"f-t"` leads to frequency interpolation being performed first followed by time interpolation, and no spatial smoothing. Setting it to `"t-f-s"` leads to time interpolation being performed first, followed by frequency interpolation, and finally spatial smoothing.


```python
class MIMOOFDMLink(Model):
    def __init__(self, int_method, lmmse_order=None, **kwargs):
        super().__init__(kwargs)
        assert int_method in ('nn', 'lin', 'lmmse')

        # Configure the resource grid
        rg = ResourceGrid(num_ofdm_symbols=NUM_OFDM_SYMBOLS,
                          fft_size=FFT_SIZE,
                          subcarrier_spacing=SUBCARRIER_SPACING,
                          num_tx=1,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2,11])
        self.rg = rg
        # Stream management
        # Only a sinlge UT is considered for channel estimation
        sm = StreamManagement([[1]], 1)
        ##################################
        # Transmitter
        ##################################
        self.qam_source = QAMSource(num_bits_per_symbol=2) # Modulation order does not impact the channel estimation. Set to QPSK
        self.rg_mapper = ResourceGridMapper(rg)
        ##################################
        # Channel
        ##################################
        self.channel = OFDMChannel(CHANNEL_MODEL, rg, return_channel=True)
        ###################################
        # Receiver
        ###################################
        # Channel estimation
        freq_cov_mat = tf.constant(FREQ_COV_MAT, tf.complex64)
        time_cov_mat = tf.constant(TIME_COV_MAT, tf.complex64)
        space_cov_mat = tf.constant(SPACE_COV_MAT, tf.complex64)
        if int_method == 'nn':
            self.channel_estimator = LSChannelEstimator(rg, interpolation_type='nn')
        elif int_method == 'lin':
            self.channel_estimator = LSChannelEstimator(rg, interpolation_type='lin')
        elif int_method == 'lmmse':
            lmmse_int_freq_first = LMMSEInterpolator(rg.pilot_pattern, time_cov_mat, freq_cov_mat, space_cov_mat, order=lmmse_order)
            self.channel_estimator = LSChannelEstimator(rg, interpolator=lmmse_int_freq_first)
    @tf.function
    def call(self, batch_size, snr_db):

        ##################################
        # Transmitter
        ##################################
        x = self.qam_source([batch_size, 1, 1, self.rg.num_data_symbols])
        x_rg = self.rg_mapper(x)
        ##################################
        # Channel
        ##################################
        no = tf.pow(10.0, -snr_db/10.0)
        topology = gen_single_sector_topology(batch_size, 1, 'umi', min_ut_velocity=SPEED, max_ut_velocity=SPEED)
        CHANNEL_MODEL.set_topology(*topology)
        y_rg, h_freq = self.channel((x_rg, no))
        ###################################
        # Channel estimation
        ###################################
        h_hat,_ = self.channel_estimator((y_rg,no))
        ###################################
        # MSE
        ###################################
        mse = tf.reduce_mean(tf.square(tf.abs(h_freq-h_hat)))
        return mse
```


The next cell defines a function for evaluating the mean square error (MSE) of a `model` over a range of SNRs (`snr_dbs`).

The `batch_size` and `num_it` parameters control the number of samples used to compute the MSE for each SNR value.


```python
def evaluate_mse(model, snr_dbs, batch_size, num_it):
    # Casting model inputs to TensorFlow types to avoid
    # re-building of the graph
    snr_dbs = tf.cast(snr_dbs, tf.float32)
    batch_size = tf.cast(batch_size, tf.int32)
    mses = []
    for snr_db in snr_dbs:
        mse_ = 0.0
        for _ in range(num_it):
            mse_ += model(batch_size, snr_db).numpy()
        # Averaging over the number of iterations
        mse_ /= float(num_it)
        mses.append(mse_)
    return mses
```


The next cell defines the evaluation parameters.


```python
# Range of SNR (in dB)
SNR_DBs = np.linspace(-10.0, 20.0, 20)
# Number of iterations and batch size.
# These parameters control the number of samples used to compute each SNR value.
# The higher the number of samples is, the more accurate the MSE estimation is, at
# the cost of longer compute time.
BATCH_SIZE = 512
NUM_IT = 10
# Interpolation/filtering order for the LMMSE interpolator.
# All valid configurations are listed.
# Some are commented to speed-up simulations.
# Uncomment configurations to evaluate them!
ORDERS = ['s-t-f', # Space - time - frequency
          #'s-f-t', # Space - frequency - time
          #'t-s-f', # Time - space - frequency
          't-f-s', # Time - frequency - space
          #'f-t-s', # Frequency - time - space
          #'f-s-t', # Frequency - space- time
          #'f-t',   # Frequency - time (no spatial smoothing)
          't-f'   # Time - frequency (no spatial smoothing)
          ]
```


The next cell evaluates the nearest-neighbor, linear, and LMMSE interpolator. For the LMMSE interpolator, we loop through the configuration listed in `ORDERS`.


```python
MSES = {}
# Nearest-neighbor interpolation
e2e = MIMOOFDMLink("nn")
MSES['nn'] = evaluate_mse(e2e, SNR_DBs, BATCH_SIZE, NUM_IT)
# Linear interpolation
e2e = MIMOOFDMLink("lin")
MSES['lin'] = evaluate_mse(e2e, SNR_DBs, BATCH_SIZE, NUM_IT)
# LMMSE
for order in ORDERS:
    e2e = MIMOOFDMLink("lmmse", order)
    MSES[f"lmmse: {order}"] = evaluate_mse(e2e, SNR_DBs, BATCH_SIZE, NUM_IT)

```


```python
WARNING:tensorflow:From /home/faycal/.local/lib/python3.8/site-packages/tensorflow/python/util/dispatch.py:1176: calling gather (from tensorflow.python.ops.array_ops) with validate_indices is deprecated and will be removed in a future version.
Instructions for updating:
The `validate_indices` argument has no effect. Indices are always validated on CPU and never validated on GPU.
```


Finally, we plot the MSE.


```python
plt.figure(figsize=(8,6))
for est_label in MSES:
    plt.semilogy(SNR_DBs, MSES[est_label], label=est_label)
plt.xlabel(r"SNR (dB)")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
```


Unsurprisingly, the LMMSE interpolator leads to more accurate estimates compared to the two other methods, as it leverages knowledge of the the channel statistics. Moreover, the order in which the LMMSE interpolation steps are performed strongly impacts the accuracy of the estimator. This is because the LMMSE interpolation operates in one dimension at a time which is not equivalent to full-blown LMMSE estimation across all dimensions at one.

Also note that the order that leads to the best accuracy depends on the channel statistics. As a rule of thumb, it might be good to start with the dimension that is most strongly correlated (i.e., time in our example).
## Comparison of MIMO detectors

An OFDM MIMO receiver consists of two stages: **OFDM channel estimation** and **MIMO detection**.

While the previous section focused on OFDM channel estimation, this section focuses now on MIMO detection.

The following MIMO detection algorithms, all available out-of-the-box in Sionna, are considered:

- [LMMSE equalization followed by APP demapping](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.LinearDetector)
- [K-Best detection](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.KBestDetector)
- [EP detection](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.EPDetector)
- [MMSE-PIC detection](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.MMSEPICDetector)


Both perfect and imperfect channel state information is considered in the simulations. LS estimation combined with LMMSE interpolation is used, with time-frequency-space smoothing (in this order, i.e., `order='t-f-s'`).

### End-to-end model

A Keras model is created in the next cell, which uses the detection method specified at initialization.

It computes either the coded bit error rate (BER) or the uncoded symbol error rate (SER), for a specified batch size, $E_b/N_0$ (in dB), and QAM modulation with a specified modulation order. When computing the BER, a 5G LDPC code is used with the specified coderate.

The following MIMO detection methods are considered (set through the `det_param` parameter):

- `"lmmse"` : No parameter needed
- `"k-best"` : List size `k`, defaults to 64
- `"ep"` : Number of iterations `l`, defaults to 10
- `"mmse-pic"` : Number of self-iterations `num_it`, defaults to 4


The `det_param` parameter corresponds to either `k`, `l`, or `num_it`, for K-Best, EP, or MMSE-PIC, respectively. If set to `None`, a default value is used according to the selected detector.

The `perf_csi` parameter controls whether perfect CSI is assumed or not. If set to `False`, then LS combined with LMMSE interpolation is used to estimate the channel.

You can easily add your own MIMO detector and channel estimator to this model for a fair and realistic benchmark.


```python
class MIMOOFDMLink(Model):
    def __init__(self, output, det_method, perf_csi, num_tx, num_bits_per_symbol, det_param=None, coderate=0.5, **kwargs):
        super().__init__(kwargs)
        assert det_method in ('lmmse', 'k-best', 'ep', 'mmse-pic'), "Unknown detection method"
        self._output = output
        self.num_tx = num_tx
        self.num_bits_per_symbol = num_bits_per_symbol
        self.coderate = coderate
        self.det_method = det_method
        self.perf_csi = perf_csi
        # Configure the resource grid
        rg = ResourceGrid(num_ofdm_symbols=NUM_OFDM_SYMBOLS,
                          fft_size=FFT_SIZE,
                          subcarrier_spacing=SUBCARRIER_SPACING,
                          num_tx=num_tx,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2,11])
        self.rg = rg
        # Stream management
        sm = StreamManagement(np.ones([1,num_tx], int), 1)
        # Codeword length and number of information bits per codeword
        n = int(rg.num_data_symbols*num_bits_per_symbol)
        k = int(coderate*n)
        self.n = n
        self.k = k
        # If output is symbol, then no FEC is used and hard decision are output
        hard_out = (output == "symbol")
        coded = (output == "bit")
        self.hard_out = hard_out
        self.coded = coded
        ##################################
        # Transmitter
        ##################################
        self.binary_source = BinarySource()
        self.mapper = Mapper(constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, return_indices=True)
        self.rg_mapper = ResourceGridMapper(rg)
        if coded:
            self.encoder = LDPC5GEncoder(k, n, num_bits_per_symbol=num_bits_per_symbol)
        ##################################
        # Channel
        ##################################
        self.channel = OFDMChannel(CHANNEL_MODEL, rg, return_channel=True)
        ###################################
        # Receiver
        ###################################
        # Channel estimation
        if not self.perf_csi:
            freq_cov_mat = tf.constant(FREQ_COV_MAT, tf.complex64)
            time_cov_mat = tf.constant(TIME_COV_MAT, tf.complex64)
            space_cov_mat = tf.constant(SPACE_COV_MAT, tf.complex64)
            lmmse_int_time_first = LMMSEInterpolator(rg.pilot_pattern, time_cov_mat, freq_cov_mat, space_cov_mat, order='t-f-s')
            self.channel_estimator = LSChannelEstimator(rg, interpolator=lmmse_int_time_first)
        # Detection
        if det_method == "lmmse":
            self.detector = LinearDetector("lmmse", output, "app", rg, sm, constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, hard_out=hard_out)
        elif det_method == 'k-best':
            if det_param is None:
                k = 64
            else:
                k = det_param
            self.detector = KBestDetector(output, num_tx, k, rg, sm, constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, hard_out=hard_out)
        elif det_method == "ep":
            if det_param is None:
                l = 10
            else:
                l = det_param
            self.detector = EPDetector(output, rg, sm, num_bits_per_symbol, l=l, hard_out=hard_out)
        elif det_method == 'mmse-pic':
            if det_param is None:
                l = 4
            else:
                l = det_param
            self.detector = MMSEPICDetector(output, rg, sm, 'app', num_iter=l, constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, hard_out=hard_out)
        if coded:
            self.decoder = LDPC5GDecoder(self.encoder, hard_out=False)
    @tf.function
    def call(self, batch_size, ebno_db):

        ##################################
        # Transmitter
        ##################################
        if self.coded:
            b = self.binary_source([batch_size, self.num_tx, 1, self.k])
            c = self.encoder(b)
        else:
            c = self.binary_source([batch_size, self.num_tx, 1, self.n])
        bits_shape = tf.shape(c)
        x,x_ind = self.mapper(c)
        x_rg = self.rg_mapper(x)
        ##################################
        # Channel
        ##################################
        no = ebnodb2no(ebno_db, self.num_bits_per_symbol, self.coderate, resource_grid=self.rg)
        topology = gen_single_sector_topology(batch_size, self.num_tx, 'umi', min_ut_velocity=SPEED, max_ut_velocity=SPEED)
        CHANNEL_MODEL.set_topology(*topology)
        y_rg, h_freq = self.channel((x_rg, no))
        ###################################
        # Receiver
        ###################################
        # Channel estimation
        if self.perf_csi:
            h_hat = h_freq
            err_var = 0.0
        else:
            h_hat,err_var = self.channel_estimator((y_rg,no))
        # Detection
        if self.det_method == "mmse-pic":
            if self._output == "bit":
                prior_shape = bits_shape
            elif self._output == "symbol":
                prior_shape = tf.concat([tf.shape(x), [self.num_bits_per_symbol]], axis=0)
            prior = tf.zeros(prior_shape)
            det_out = self.detector((y_rg,h_hat,prior,err_var,no))
        else:
            det_out = self.detector((y_rg,h_hat,err_var,no))
        # (Decoding) and output
        if self._output == "bit":
            llr = tf.reshape(det_out, bits_shape)
            b_hat = self.decoder(llr)
            return b, b_hat
        elif self._output == "symbol":
            x_hat = tf.reshape(det_out, tf.shape(x_ind))
            return x_ind, x_hat
```


The following function is used to evaluate all of the considered detectors for a given setup: It instantiates the end-to-end systems, runs the simulations, and returns the BER or SER.


```python
def run_sim(num_tx, num_bits_per_symbol, output, ebno_dbs, perf_csi, det_param=None):
    lmmse = MIMOOFDMLink(output, "lmmse", perf_csi, num_tx, num_bits_per_symbol, det_param)
    k_best = MIMOOFDMLink(output, "k-best", perf_csi, num_tx, num_bits_per_symbol, det_param)
    ep = MIMOOFDMLink(output, "ep", perf_csi, num_tx, num_bits_per_symbol, det_param)
    mmse_pic = MIMOOFDMLink(output, "mmse-pic", perf_csi, num_tx, num_bits_per_symbol, det_param)
    if output == "symbol":
        soft_estimates = False
        ylabel = "Uncoded SER"
    else:
        soft_estimates = True
        ylabel = "Coded BER"
    er_lmmse,_ = sim_ber(lmmse,
        ebno_dbs,
        batch_size=64,
        max_mc_iter=200,
        num_target_block_errors=200,
        soft_estimates=soft_estimates);
    er_ep,_ = sim_ber(ep,
        ebno_dbs,
        batch_size=64,
        max_mc_iter=200,
        num_target_block_errors=200,
        soft_estimates=soft_estimates);
    er_kbest,_ = sim_ber(k_best,
       ebno_dbs,
       batch_size=64,
       max_mc_iter=200,
       num_target_block_errors=200,
       soft_estimates=soft_estimates);
    er_mmse_pic,_ = sim_ber(mmse_pic,
       ebno_dbs,
       batch_size=64,
       max_mc_iter=200,
       num_target_block_errors=200,
       soft_estimates=soft_estimates);
    return er_lmmse, er_ep, er_kbest, er_mmse_pic
```


The next cell defines the simulation parameters.


```python
# Range of SNR (dB)
EBN0_DBs = np.linspace(-10., 20.0, 10)
# Number of transmitters
NUM_TX = 4
# Modulation order (number of bits per symbol)
NUM_BITS_PER_SYMBOL = 4 # 16-QAM
```


We start by evaluating the uncoded SER. The next cell runs the simulations with perfect CSI and channel estimation. Results are stored in the `SER` dictionnary.


```python
SER = {} # Store the results
# Perfect CSI
ser_lmmse, ser_ep, ser_kbest, ser_mmse_pic = run_sim(NUM_TX, NUM_BITS_PER_SYMBOL, "symbol", EBN0_DBs, True)
SER['Perf. CSI / LMMSE'] = ser_lmmse
SER['Perf. CSI / EP'] = ser_ep
SER['Perf. CSI / K-Best'] = ser_kbest
SER['Perf. CSI / MMSE-PIC'] = ser_mmse_pic
# Imperfect CSI
ser_lmmse, ser_ep, ser_kbest, ser_mmse_pic = run_sim(NUM_TX, NUM_BITS_PER_SYMBOL, "symbol", EBN0_DBs, False)
SER['Ch. Est. / LMMSE'] = ser_lmmse
SER['Ch. Est. / EP'] = ser_ep
SER['Ch. Est. / K-Best'] = ser_kbest
SER['Ch. Est. / MMSE-PIC'] = ser_mmse_pic
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 6.3274e-01 | 1.0000e+00 |       93302 |      147456 |          256 |         256 |         4.7 |reached target block errors
   -6.667 | 5.0724e-01 | 1.0000e+00 |       74796 |      147456 |          256 |         256 |         0.1 |reached target block errors
   -3.333 | 3.7246e-01 | 9.9609e-01 |       54922 |      147456 |          255 |         256 |         0.1 |reached target block errors
      0.0 | 2.3949e-01 | 9.7656e-01 |       35314 |      147456 |          250 |         256 |         0.1 |reached target block errors
    3.333 | 1.2375e-01 | 8.4766e-01 |       18247 |      147456 |          217 |         256 |         0.1 |reached target block errors
    6.667 | 5.7034e-02 | 6.6211e-01 |       16820 |      294912 |          339 |         512 |         0.1 |reached target block errors
     10.0 | 2.5584e-02 | 4.6680e-01 |        7545 |      294912 |          239 |         512 |         0.1 |reached target block errors
   13.333 | 6.7546e-03 | 2.6302e-01 |        2988 |      442368 |          202 |         768 |         0.2 |reached target block errors
   16.667 | 2.0913e-03 | 1.0840e-01 |        2467 |     1179648 |          222 |        2048 |         0.5 |reached target block errors
     20.0 | 5.6708e-04 | 3.9621e-02 |        1756 |     3096576 |          213 |        5376 |         1.4 |reached target block errors
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 6.1009e-01 | 1.0000e+00 |       89961 |      147456 |          256 |         256 |         5.3 |reached target block errors
   -6.667 | 4.8094e-01 | 1.0000e+00 |       70918 |      147456 |          256 |         256 |         0.1 |reached target block errors
   -3.333 | 2.9869e-01 | 9.9609e-01 |       44044 |      147456 |          255 |         256 |         0.1 |reached target block errors
      0.0 | 1.4774e-01 | 9.7656e-01 |       21785 |      147456 |          250 |         256 |         0.1 |reached target block errors
    3.333 | 6.1442e-02 | 7.9688e-01 |        9060 |      147456 |          204 |         256 |         0.1 |reached target block errors
    6.667 | 2.0511e-02 | 4.3750e-01 |        6049 |      294912 |          224 |         512 |         0.2 |reached target block errors
     10.0 | 4.6556e-03 | 1.4453e-01 |        4119 |      884736 |          222 |        1536 |         0.5 |reached target block errors
   13.333 | 8.7167e-04 | 5.3385e-02 |        1928 |     2211840 |          205 |        3840 |         1.2 |reached target block errors
   16.667 | 1.0502e-04 | 1.1217e-02 |        1084 |    10321920 |          201 |       17920 |         5.6 |reached target block errors
     20.0 | 2.3600e-05 | 2.9688e-03 |         696 |    29491200 |          152 |       51200 |        15.8 |reached max iter
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 6.1452e-01 | 1.0000e+00 |       90615 |      147456 |          256 |         256 |         5.6 |reached target block errors
   -6.667 | 4.8480e-01 | 1.0000e+00 |       71487 |      147456 |          256 |         256 |         0.5 |reached target block errors
   -3.333 | 3.0013e-01 | 9.8828e-01 |       44256 |      147456 |          253 |         256 |         0.5 |reached target block errors
      0.0 | 1.2075e-01 | 9.4141e-01 |       17806 |      147456 |          241 |         256 |         0.5 |reached target block errors
    3.333 | 4.2379e-02 | 7.3242e-01 |       12498 |      294912 |          375 |         512 |         0.9 |reached target block errors
    6.667 | 1.5837e-02 | 3.4635e-01 |        7006 |      442368 |          266 |         768 |         1.4 |reached target block errors
     10.0 | 4.0855e-03 | 1.1775e-01 |        4217 |     1032192 |          211 |        1792 |         3.3 |reached target block errors
   13.333 | 7.5164e-04 | 3.3040e-02 |        2660 |     3538944 |          203 |        6144 |        11.1 |reached target block errors
   16.667 | 9.1727e-05 | 1.0116e-02 |        1055 |    11501568 |          202 |       19968 |        36.2 |reached target block errors
     20.0 | 2.4482e-05 | 2.5000e-03 |         722 |    29491200 |          128 |       51200 |        92.6 |reached max iter
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 6.0616e-01 | 1.0000e+00 |       89382 |      147456 |          256 |         256 |         5.0 |reached target block errors
   -6.667 | 4.9781e-01 | 1.0000e+00 |       73405 |      147456 |          256 |         256 |         0.1 |reached target block errors
   -3.333 | 2.9688e-01 | 1.0000e+00 |       43777 |      147456 |          256 |         256 |         0.1 |reached target block errors
      0.0 | 1.3372e-01 | 9.5703e-01 |       19718 |      147456 |          245 |         256 |         0.1 |reached target block errors
    3.333 | 4.9093e-02 | 8.1250e-01 |        7239 |      147456 |          208 |         256 |         0.1 |reached target block errors
    6.667 | 1.7320e-02 | 4.4531e-01 |        5108 |      294912 |          228 |         512 |         0.2 |reached target block errors
     10.0 | 4.3996e-03 | 2.3438e-01 |        2595 |      589824 |          240 |        1024 |         0.3 |reached target block errors
   13.333 | 7.8729e-04 | 7.3509e-02 |        1277 |     1622016 |          207 |        2816 |         0.9 |reached target block errors
   16.667 | 1.5014e-04 | 1.8714e-02 |         952 |     6340608 |          206 |       11008 |         3.6 |reached target block errors
     20.0 | 2.7364e-05 | 3.7695e-03 |         807 |    29491200 |          193 |       51200 |        16.9 |reached max iter
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 6.5757e-01 | 1.0000e+00 |       96962 |      147456 |          256 |         256 |         4.5 |reached target block errors
   -6.667 | 5.3936e-01 | 1.0000e+00 |       79532 |      147456 |          256 |         256 |         0.2 |reached target block errors
   -3.333 | 4.2834e-01 | 1.0000e+00 |       63161 |      147456 |          256 |         256 |         0.2 |reached target block errors
      0.0 | 3.2353e-01 | 9.8828e-01 |       47706 |      147456 |          253 |         256 |         0.2 |reached target block errors
    3.333 | 1.8555e-01 | 9.4141e-01 |       27360 |      147456 |          241 |         256 |         0.2 |reached target block errors
    6.667 | 1.0126e-01 | 7.9297e-01 |       14931 |      147456 |          203 |         256 |         0.2 |reached target block errors
     10.0 | 3.7248e-02 | 5.4492e-01 |       10985 |      294912 |          279 |         512 |         0.4 |reached target block errors
   13.333 | 2.3170e-02 | 4.2773e-01 |        6833 |      294912 |          219 |         512 |         0.4 |reached target block errors
   16.667 | 6.8410e-03 | 2.1777e-01 |        4035 |      589824 |          223 |        1024 |         0.9 |reached target block errors
     20.0 | 4.8977e-03 | 1.7188e-01 |        3611 |      737280 |          220 |        1280 |         1.1 |reached target block errors
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 6.5626e-01 | 1.0000e+00 |       96770 |      147456 |          256 |         256 |         5.3 |reached target block errors
   -6.667 | 5.3429e-01 | 1.0000e+00 |       78785 |      147456 |          256 |         256 |         0.2 |reached target block errors
   -3.333 | 3.4984e-01 | 1.0000e+00 |       51586 |      147456 |          256 |         256 |         0.2 |reached target block errors
      0.0 | 2.3107e-01 | 9.8828e-01 |       34072 |      147456 |          253 |         256 |         0.2 |reached target block errors
    3.333 | 9.8416e-02 | 8.3203e-01 |       14512 |      147456 |          213 |         256 |         0.2 |reached target block errors
    6.667 | 3.5495e-02 | 6.2305e-01 |       10468 |      294912 |          319 |         512 |         0.5 |reached target block errors
     10.0 | 1.1027e-02 | 3.7370e-01 |        4878 |      442368 |          287 |         768 |         0.7 |reached target block errors
   13.333 | 4.2103e-03 | 1.7057e-01 |        3725 |      884736 |          262 |        1536 |         1.4 |reached target block errors
   16.667 | 1.5082e-03 | 7.8125e-02 |        2224 |     1474560 |          200 |        2560 |         2.4 |reached target block errors
     20.0 | 1.9312e-03 | 6.3101e-02 |        3702 |     1916928 |          210 |        3328 |         3.0 |reached target block errors
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 6.5530e-01 | 1.0000e+00 |       96628 |      147456 |          256 |         256 |         5.9 |reached target block errors
   -6.667 | 5.4480e-01 | 1.0000e+00 |       80334 |      147456 |          256 |         256 |         0.6 |reached target block errors
   -3.333 | 3.8673e-01 | 9.9219e-01 |       57026 |      147456 |          254 |         256 |         0.6 |reached target block errors
      0.0 | 2.1018e-01 | 9.8438e-01 |       30993 |      147456 |          252 |         256 |         0.6 |reached target block errors
    3.333 | 8.1733e-02 | 8.1250e-01 |       12052 |      147456 |          208 |         256 |         0.6 |reached target block errors
    6.667 | 3.1857e-02 | 5.5859e-01 |        9395 |      294912 |          286 |         512 |         1.2 |reached target block errors
     10.0 | 9.6594e-03 | 2.7995e-01 |        4273 |      442368 |          215 |         768 |         1.9 |reached target block errors
   13.333 | 3.6594e-03 | 1.5937e-01 |        2698 |      737280 |          204 |        1280 |         3.1 |reached target block errors
   16.667 | 2.2942e-03 | 8.4375e-02 |        3383 |     1474560 |          216 |        2560 |         6.2 |reached target block errors
     20.0 | 1.4678e-03 | 5.6920e-02 |        3030 |     2064384 |          204 |        3584 |         8.7 |reached target block errors
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 6.5718e-01 | 1.0000e+00 |       96905 |      147456 |          256 |         256 |         5.3 |reached target block errors
   -6.667 | 5.3097e-01 | 1.0000e+00 |       78294 |      147456 |          256 |         256 |         0.2 |reached target block errors
   -3.333 | 3.6694e-01 | 1.0000e+00 |       54108 |      147456 |          256 |         256 |         0.2 |reached target block errors
      0.0 | 2.3520e-01 | 9.7656e-01 |       34682 |      147456 |          250 |         256 |         0.2 |reached target block errors
    3.333 | 9.2502e-02 | 8.5156e-01 |       13640 |      147456 |          218 |         256 |         0.2 |reached target block errors
    6.667 | 3.4912e-02 | 6.6211e-01 |       10296 |      294912 |          339 |         512 |         0.5 |reached target block errors
     10.0 | 1.3431e-02 | 4.2383e-01 |        3961 |      294912 |          217 |         512 |         0.5 |reached target block errors
   13.333 | 5.7865e-03 | 2.5098e-01 |        3413 |      589824 |          257 |        1024 |         0.9 |reached target block errors
   16.667 | 2.7466e-03 | 1.1279e-01 |        3240 |     1179648 |          231 |        2048 |         1.9 |reached target block errors
     20.0 | 1.2919e-03 | 6.6732e-02 |        2286 |     1769472 |          205 |        3072 |         2.9 |reached target block errors
```


Next, we evaluate the coded BER. The cell below runs the simulations with perfect CSI and channel estimation. Results are stored in the `BER` dictionnary.


```python
BER = {} # Store the results
# Perfect CSI
ber_lmmse, ber_ep, ber_kbest, ber_mmse_pic = run_sim(NUM_TX, NUM_BITS_PER_SYMBOL, "bit", EBN0_DBs, True)
BER['Perf. CSI / LMMSE'] = ber_lmmse
BER['Perf. CSI / EP'] = ber_ep
BER['Perf. CSI / K-Best'] = ber_kbest
BER['Perf. CSI / MMSE-PIC'] = ber_mmse_pic
# Imperfect CSI
ber_lmmse, ber_ep, ber_kbest, ber_mmse_pic = run_sim(NUM_TX, NUM_BITS_PER_SYMBOL, "bit", EBN0_DBs, False)
BER['Ch. Est. / LMMSE'] = ber_lmmse
BER['Ch. Est. / EP'] = ber_ep
BER['Ch. Est. / K-Best'] = ber_kbest
BER['Ch. Est. / MMSE-PIC'] = ber_mmse_pic
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 1.8888e-01 | 8.5547e-01 |       55703 |      294912 |          219 |         256 |         5.8 |reached target block errors
   -6.667 | 1.1261e-01 | 5.8984e-01 |       66421 |      589824 |          302 |         512 |         0.2 |reached target block errors
   -3.333 | 5.7696e-02 | 3.1641e-01 |       51046 |      884736 |          243 |         768 |         0.3 |reached target block errors
      0.0 | 2.5274e-02 | 1.5039e-01 |       44721 |     1769472 |          231 |        1536 |         0.7 |reached target block errors
    3.333 | 1.0029e-02 | 6.6732e-02 |       35491 |     3538944 |          205 |        3072 |         1.4 |reached target block errors
    6.667 | 2.6471e-03 | 1.9627e-02 |       32007 |    12091392 |          206 |       10496 |         4.6 |reached target block errors
     10.0 | 5.2647e-04 | 4.3645e-03 |       27792 |    52789248 |          200 |       45824 |        20.0 |reached target block errors
   13.333 | 8.6721e-05 | 5.6641e-04 |        5115 |    58982400 |           29 |       51200 |        22.2 |reached max iter
   16.667 | 1.6174e-05 | 9.7656e-05 |         954 |    58982400 |            5 |       51200 |        22.2 |reached max iter
     20.0 | 1.5428e-06 | 1.9531e-05 |          91 |    58982400 |            1 |       51200 |        22.2 |reached max iter
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 1.6863e-01 | 7.8125e-01 |       49731 |      294912 |          200 |         256 |         5.6 |reached target block errors
   -6.667 | 9.9314e-02 | 5.1172e-01 |       58578 |      589824 |          262 |         512 |         0.3 |reached target block errors
   -3.333 | 4.7239e-02 | 2.7474e-01 |       41794 |      884736 |          211 |         768 |         0.4 |reached target block errors
      0.0 | 1.4008e-02 | 8.8108e-02 |       37181 |     2654208 |          203 |        2304 |         1.2 |reached target block errors
    3.333 | 2.2594e-03 | 1.6276e-02 |       31983 |    14155776 |          200 |       12288 |         6.1 |reached target block errors
    6.667 | 3.9112e-04 | 2.9883e-03 |       23069 |    58982400 |          153 |       51200 |        25.2 |reached max iter
     10.0 | 2.0972e-05 | 2.7344e-04 |        1237 |    58982400 |           14 |       51200 |        25.1 |reached max iter
   13.333 | 0.0000e+00 | 0.0000e+00 |           0 |    58982400 |            0 |       51200 |        25.1 |reached max iter
Simulation stopped as no error occurred @ EbNo = 13.3 dB.
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 2.1076e-01 | 9.2969e-01 |       62155 |      294912 |          238 |         256 |         6.5 |reached target block errors
   -6.667 | 1.0710e-01 | 6.1914e-01 |       63171 |      589824 |          317 |         512 |         1.1 |reached target block errors
   -3.333 | 3.8923e-02 | 2.4023e-01 |       45916 |     1179648 |          246 |        1024 |         2.2 |reached target block errors
      0.0 | 1.1103e-02 | 7.1378e-02 |       36018 |     3244032 |          201 |        2816 |         6.2 |reached target block errors
    3.333 | 2.2757e-03 | 1.6927e-02 |       32215 |    14155776 |          208 |       12288 |        27.0 |reached target block errors
    6.667 | 2.9185e-04 | 2.1875e-03 |       17214 |    58982400 |          112 |       51200 |       112.1 |reached max iter
     10.0 | 3.9978e-05 | 2.9297e-04 |        2358 |    58982400 |           15 |       51200 |       112.1 |reached max iter
   13.333 | 0.0000e+00 | 0.0000e+00 |           0 |    58982400 |            0 |       51200 |       112.1 |reached max iter
Simulation stopped as no error occurred @ EbNo = 13.3 dB.
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 1.8315e-01 | 8.5547e-01 |       54013 |      294912 |          219 |         256 |         5.2 |reached target block errors
   -6.667 | 1.1446e-01 | 5.9180e-01 |       67512 |      589824 |          303 |         512 |         0.3 |reached target block errors
   -3.333 | 5.0348e-02 | 2.9297e-01 |       44545 |      884736 |          225 |         768 |         0.4 |reached target block errors
      0.0 | 1.6928e-02 | 1.0596e-01 |       39937 |     2359296 |          217 |        2048 |         1.1 |reached target block errors
    3.333 | 2.9010e-03 | 2.4148e-02 |       28233 |     9732096 |          204 |        8448 |         4.4 |reached target block errors
    6.667 | 5.5365e-04 | 4.4611e-03 |       28737 |    51904512 |          201 |       45056 |        23.3 |reached target block errors
     10.0 | 6.0560e-05 | 7.4219e-04 |        3572 |    58982400 |           38 |       51200 |        26.5 |reached max iter
   13.333 | 2.7466e-06 | 3.9063e-05 |         162 |    58982400 |            2 |       51200 |        26.5 |reached max iter
   16.667 | 1.6954e-08 | 1.9531e-05 |           1 |    58982400 |            1 |       51200 |        26.6 |reached max iter
     20.0 | 0.0000e+00 | 0.0000e+00 |           0 |    58982400 |            0 |       51200 |        26.4 |reached max iter
Simulation stopped as no error occurred @ EbNo = 20.0 dB.
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 2.0073e-01 | 8.4375e-01 |       59199 |      294912 |          216 |         256 |         5.9 |reached target block errors
   -6.667 | 1.5016e-01 | 6.8750e-01 |       88566 |      589824 |          352 |         512 |         0.5 |reached target block errors
   -3.333 | 7.8442e-02 | 4.0430e-01 |       46267 |      589824 |          207 |         512 |         0.5 |reached target block errors
      0.0 | 3.9502e-02 | 2.1777e-01 |       46598 |     1179648 |          223 |        1024 |         1.1 |reached target block errors
    3.333 | 1.7726e-02 | 1.0791e-01 |       41822 |     2359296 |          221 |        2048 |         2.1 |reached target block errors
    6.667 | 6.3252e-03 | 3.7946e-02 |       39173 |     6193152 |          204 |        5376 |         5.6 |reached target block errors
     10.0 | 2.4057e-03 | 1.5855e-02 |       36183 |    15040512 |          207 |       13056 |        13.6 |reached target block errors
   13.333 | 9.3448e-04 | 5.5962e-03 |       38858 |    41582592 |          202 |       36096 |        37.5 |reached target block errors
   16.667 | 2.7039e-04 | 2.0117e-03 |       15948 |    58982400 |          103 |       51200 |        53.3 |reached max iter
     20.0 | 2.7354e-04 | 1.8555e-03 |       16134 |    58982400 |           95 |       51200 |        53.3 |reached max iter
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 1.9646e-01 | 8.9453e-01 |       57939 |      294912 |          229 |         256 |         6.6 |reached target block errors
   -6.667 | 1.3310e-01 | 6.5625e-01 |       78508 |      589824 |          336 |         512 |         0.6 |reached target block errors
   -3.333 | 6.3611e-02 | 3.4505e-01 |       56279 |      884736 |          265 |         768 |         0.8 |reached target block errors
      0.0 | 2.7651e-02 | 1.6562e-01 |       40773 |     1474560 |          212 |        1280 |         1.4 |reached target block errors
    3.333 | 8.5775e-03 | 5.2083e-02 |       37944 |     4423680 |          200 |        3840 |         4.2 |reached target block errors
    6.667 | 2.0052e-03 | 1.3470e-02 |       34298 |    17104896 |          200 |       14848 |        16.4 |reached target block errors
     10.0 | 7.6427e-04 | 5.2083e-03 |       33809 |    44236800 |          200 |       38400 |        42.3 |reached target block errors
   13.333 | 4.1326e-04 | 2.8516e-03 |       24375 |    58982400 |          146 |       51200 |        56.1 |reached max iter
   16.667 | 2.0630e-04 | 1.6602e-03 |       12168 |    58982400 |           85 |       51200 |        56.0 |reached max iter
     20.0 | 1.7263e-04 | 1.6211e-03 |       10182 |    58982400 |           83 |       51200 |        56.3 |reached max iter
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 2.4428e-01 | 9.7266e-01 |       72041 |      294912 |          249 |         256 |         7.3 |reached target block errors
   -6.667 | 1.5539e-01 | 7.4414e-01 |       91650 |      589824 |          381 |         512 |         1.4 |reached target block errors
   -3.333 | 6.4181e-02 | 3.5286e-01 |       56783 |      884736 |          271 |         768 |         2.1 |reached target block errors
      0.0 | 2.6555e-02 | 1.4844e-01 |       46988 |     1769472 |          228 |        1536 |         4.3 |reached target block errors
    3.333 | 5.6042e-03 | 3.4307e-02 |       38013 |     6782976 |          202 |        5888 |        16.4 |reached target block errors
    6.667 | 1.4845e-03 | 9.5538e-03 |       36337 |    24477696 |          203 |       21248 |        59.0 |reached target block errors
     10.0 | 5.6710e-04 | 3.6719e-03 |       33449 |    58982400 |          188 |       51200 |       142.6 |reached max iter
   13.333 | 3.0056e-04 | 1.9727e-03 |       17728 |    58982400 |          101 |       51200 |       142.7 |reached max iter
   16.667 | 2.2124e-04 | 1.5625e-03 |       13049 |    58982400 |           80 |       51200 |       142.7 |reached max iter
     20.0 | 1.3379e-04 | 1.0156e-03 |        7891 |    58982400 |           52 |       51200 |       142.5 |reached max iter
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 2.1431e-01 | 9.0234e-01 |       63203 |      294912 |          231 |         256 |         6.1 |reached target block errors
   -6.667 | 1.3881e-01 | 6.6016e-01 |       81876 |      589824 |          338 |         512 |         0.6 |reached target block errors
   -3.333 | 8.4296e-02 | 4.5117e-01 |       49720 |      589824 |          231 |         512 |         0.6 |reached target block errors
      0.0 | 3.1447e-02 | 1.9062e-01 |       46370 |     1474560 |          244 |        1280 |         1.4 |reached target block errors
    3.333 | 9.9915e-03 | 6.5505e-02 |       38306 |     3833856 |          218 |        3328 |         3.7 |reached target block errors
    6.667 | 2.2112e-03 | 1.6342e-02 |       31954 |    14450688 |          205 |       12544 |        14.0 |reached target block errors
     10.0 | 8.0055e-04 | 6.0562e-03 |       30456 |    38043648 |          200 |       33024 |        36.9 |reached target block errors
   13.333 | 5.1027e-04 | 3.6719e-03 |       30097 |    58982400 |          188 |       51200 |        57.3 |reached max iter
   16.667 | 2.7083e-04 | 2.4609e-03 |       15974 |    58982400 |          126 |       51200 |        57.2 |reached max iter
     20.0 | 2.3241e-04 | 1.7383e-03 |       13708 |    58982400 |           89 |       51200 |        57.2 |reached max iter
```


Finally, we plot the results.


```python
fig, ax = plt.subplots(1,2, figsize=(16,7))
fig.suptitle(f"{NUM_TX}x{NUM_RX_ANT} UMi | {2**NUM_BITS_PER_SYMBOL}-QAM")
## SER
ax[0].set_title("Symbol error rate")
# Perfect CSI
ax[0].semilogy(EBN0_DBs, SER['Perf. CSI / LMMSE'], 'x-', label='Perf. CSI / LMMSE', c='C0')
ax[0].semilogy(EBN0_DBs, SER['Perf. CSI / EP'], 'o--', label='Perf. CSI / EP', c='C0')
ax[0].semilogy(EBN0_DBs, SER['Perf. CSI / K-Best'], 's-.', label='Perf. CSI / K-Best', c='C0')
ax[0].semilogy(EBN0_DBs, SER['Perf. CSI / MMSE-PIC'], 'd:', label='Perf. CSI / MMSE-PIC', c='C0')
# Imperfect CSI
ax[0].semilogy(EBN0_DBs, SER['Ch. Est. / LMMSE'], 'x-', label='Ch. Est. / LMMSE', c='C1')
ax[0].semilogy(EBN0_DBs, SER['Ch. Est. / EP'], 'o--', label='Ch. Est. / EP', c='C1')
ax[0].semilogy(EBN0_DBs, SER['Ch. Est. / K-Best'], 's-.', label='Ch. Est. / K-Best', c='C1')
ax[0].semilogy(EBN0_DBs, SER['Ch. Est. / MMSE-PIC'], 'd:', label='Ch. Est. / MMSE-PIC', c='C1')
ax[0].set_xlabel(r"$E_b/N0$")
ax[0].set_ylabel("SER")
ax[0].set_ylim((1e-4, 1.0))
ax[0].legend()
ax[0].grid(True)
## SER
ax[1].set_title("Bit error rate")
# Perfect CSI
ax[1].semilogy(EBN0_DBs, BER['Perf. CSI / LMMSE'], 'x-', label='Perf. CSI / LMMSE', c='C0')
ax[1].semilogy(EBN0_DBs, BER['Perf. CSI / EP'], 'o--', label='Perf. CSI / EP', c='C0')
ax[1].semilogy(EBN0_DBs, BER['Perf. CSI / K-Best'], 's-.', label='Perf. CSI / K-Best', c='C0')
ax[1].semilogy(EBN0_DBs, BER['Perf. CSI / MMSE-PIC'], 'd:', label='Perf. CSI / MMSE-PIC', c='C0')
# Imperfect CSI
ax[1].semilogy(EBN0_DBs, BER['Ch. Est. / LMMSE'], 'x-', label='Ch. Est. / LMMSE', c='C1')
ax[1].semilogy(EBN0_DBs, BER['Ch. Est. / EP'], 'o--', label='Ch. Est. / EP', c='C1')
ax[1].semilogy(EBN0_DBs, BER['Ch. Est. / K-Best'], 's-.', label='Ch. Est. / K-Best', c='C1')
ax[1].semilogy(EBN0_DBs, BER['Ch. Est. / MMSE-PIC'], 'd:', label='Ch. Est. / MMSE-PIC', c='C1')
ax[1].set_xlabel(r"$E_b/N0$")
ax[1].set_ylabel("BER")
ax[1].set_ylim((1e-4, 1.0))
ax[1].legend()
ax[1].grid(True)
```


For this setup, the non-linear detection algorithms K-Best, EP, and MMSE-PIC, outperform the linear MMSE detection method. It is remarkable that K-Best and EP with imperfect CSI achieve lower BER than LMMSE detection with perfect CSI.

However, one should keep in mind that:

- EP is prone to numerical imprecision and could therefore achieve better BER/SER with double precision (`dtype=tf.complex128`). The number of iterations `l` as well as the update smoothing parameter `beta` impact performance.
- For K-Best, there is not a unique way to compute soft information and better performance could be achieved with improved methods for computing soft information from a list of candidates (see [list2llr](https://nvlabs.github.io/sionna/api/mimo.html#list2llr)). Increasing the list size `k` results in improved accuracy at the cost of higher complexity.
- MMSE-PIC can be easily combined with a decoder to implement iterative detection and decoding, as it takes as input soft prior information on the bits/symbols.


