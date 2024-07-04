# Weighted Belief Propagation Decoding

This notebooks implements the *Weighted Belief Propagation* (BP) algorithm as proposed by Nachmani *et al.* in [1]. The main idea is to leverage BP decoding by additional trainable weights that scale each outgoing variable node (VN) and check node (CN) message. These weights provide additional degrees of freedom and can be trained by stochastic gradient descent (SGD) to improve the BP performance for the given code. If all weights are initialized with *1*, the algorithm equals the *classical* BP
algorithm and, thus, the concept can be seen as a generalized BP decoder.

Our main focus is to show how Sionna can lower the barrier-to-entry for state-of-the-art research. For this, you will investigate:

- How to implement the multi-loss BP decoding with Sionna
- How a single scaling factor can lead to similar results
- What happens for training of the 5G LDPC code


The setup includes the following components:

- LDPC BP Decoder
- Gaussian LLR source


Please note that we implement a simplified version of the original algorithm consisting of two major simplifications:
<ol class="arabic simple">
- ) Only outgoing variable node (VN) messages are weighted. This is possible as the VN operation is linear and it would only increase the memory complexity without increasing the *expressive* power of the neural network.
- ) We use the same shared weights for all iterations. This can potentially influence the final performance, however, simplifies the implementation and allows to run the decoder with different number of iterations.
</ol>

**Note**: If you are not familiar with all-zero codeword-based simulations please have a look into the [Bit-Interleaved Coded Modulation](https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html) example notebook first.

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
# Import required Sionna components
from sionna.fec.ldpc import LDPCBPDecoder, LDPC5GEncoder, LDPC5GDecoder
from sionna.utils.metrics import BitwiseMutualInformation
from sionna.fec.utils import GaussianPriorSource, load_parity_check_examples
from sionna.utils import ebnodb2no, hard_decisions
from sionna.utils.metrics import compute_ber
from sionna.utils.plotting import PlotBER
from tensorflow.keras.losses import BinaryCrossentropy
```

```python
import tensorflow as tf
# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
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
```

## Weighted BP for BCH Codes

First, we define the trainable model consisting of:

- LDPC BP decoder
- Gaussian LLR source


The idea of the multi-loss function in [1]is to average the loss overall iterations, i.e., not just the final estimate is evaluated. This requires to call the BP decoder *iteration-wise* by setting `num_iter=1` and `stateful=True` such that the decoder will perform a single iteration and returns its current estimate while also providing the internal messages for the next iteration.

A few comments:

- We assume the transmission of the all-zero codeword. This allows to train and analyze the decoder without the need of an encoder. Remark: The final decoder can be used for arbitrary codewords.
- We directly generate the channel LLRs with `GaussianPriorSource`. The equivalent LLR distribution could be achieved by transmitting the all-zero codeword over an AWGN channel with BPSK modulation.
- For the proposed *multi-loss* [1] (i.e., the loss is averaged over all iterations), we need to access the decoders intermediate output after each iteration. This is done by calling the decoding function multiple times while setting `stateful` to True, i.e., the decoder continuous the decoding process at the last message state.

```python
class WeightedBP(tf.keras.Model):
    """System model for BER simulations of weighted BP decoding.
    This model uses `GaussianPriorSource` to mimic the LLRs after demapping of
    QPSK symbols transmitted over an AWGN channel.
    Parameters
    ----------
        pcm: ndarray
            The parity-check matrix of the code under investigation.
        num_iter: int
            Number of BP decoding iterations.

    Input
    -----
        batch_size: int or tf.int
            The batch_size used for the simulation.
        ebno_db: float or tf.float
            A float defining the simulation SNR.
    Output
    ------
        (u, u_hat, loss):
            Tuple:
        u: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the transmitted information bits.
        u_hat: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the estimated information bits.
        loss: tf.float32
            Binary cross-entropy loss between `u` and `u_hat`.
    """
    def __init__(self, pcm, num_iter=5):
        super().__init__()
        # init components
        self.decoder = LDPCBPDecoder(pcm,
                                     num_iter=1, # iterations are done via outer loop (to access intermediate results for multi-loss)
                                     stateful=True, # decoder stores internal messages after call
                                     hard_out=False, # we need to access soft-information
                                     cn_type="boxplus",
                                     trainable=True) # the decoder must be trainable, otherwise no weights are generated
        # used to generate llrs during training (see example notebook on all-zero codeword trick)
        self.llr_source = GaussianPriorSource()
        self._num_iter = num_iter
        self._bce = BinaryCrossentropy(from_logits=True)
    def call(self, batch_size, ebno_db):
        noise_var = ebnodb2no(ebno_db,
                              num_bits_per_symbol=2, # QPSK
                              coderate=coderate)
        # all-zero CW to calculate loss / BER
        c = tf.zeros([batch_size, n])
        # Gaussian LLR source
        llr = self.llr_source([[batch_size, n], noise_var])
        # --- implement multi-loss as proposed by Nachmani et al. [1]---
        loss = 0
        msg_vn = None # internal state of decoder
        for i in range(self._num_iter):
            c_hat, msg_vn = self.decoder((llr, msg_vn)) # perform one decoding iteration; decoder returns soft-values
            loss += self._bce(c, c_hat)  # add loss after each iteration
        loss /= self._num_iter # scale loss by number of iterations
        return c, c_hat, loss
```


Load a parity-check matrix used for the experiment. We use the same BCH(63,45) code as in [1]. The code can be replaced by any parity-check matrix of your choice.


```python
pcm_id = 1 # (63,45) BCH code parity check matrix
pcm, k , n, coderate = load_parity_check_examples(pcm_id=pcm_id, verbose=True)
num_iter = 10 # set number of decoding iterations
# and initialize the model
model = WeightedBP(pcm=pcm, num_iter=num_iter)
```


```python

n: 63, k: 45, coderate: 0.714
```


**Note**: weighted BP tends to work better for small number of iterations. The effective gains (compared to the baseline with same number of iterations) vanish with more iterations.

### Weights before Training and Simulation of BER

Let us plot the weights after initialization of the decoder to verify that everything is properly initialized. This is equivalent the *classical* BP decoder.


```python
# count number of weights/edges
print("Total number of weights: ", np.size(model.decoder.get_weights()))
# and show the weight distribution
model.decoder.show_weights()
```


```python
Total number of weights:  432
```


We first simulate (and store) the BER performance *before* training. For this, we use the `PlotBER` class, which provides a convenient way to store the results for later comparison.


```python
# SNR to simulate the results
ebno_dbs = np.array(np.arange(1, 7, 0.5))
mc_iters = 100 # number of Monte Carlo iterations
# we generate a new PlotBER() object to simulate, store and plot the BER results
ber_plot = PlotBER("Weighted BP")
# simulate and plot the BER curve of the untrained decoder
ber_plot.simulate(model,
                  ebno_dbs=ebno_dbs,
                  batch_size=1000,
                  num_target_bit_errors=2000, # stop sim after 2000 bit errors
                  legend="Untrained",
                  soft_estimates=True,
                  max_mc_iter=mc_iters,
                  forward_keyboard_interrupt=False);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      1.0 | 8.9492e-02 | 9.7600e-01 |        5638 |       63000 |          976 |        1000 |         0.2 |reached target bit errors
      1.5 | 7.4079e-02 | 9.0800e-01 |        4667 |       63000 |          908 |        1000 |         0.2 |reached target bit errors
      2.0 | 5.9444e-02 | 8.1300e-01 |        3745 |       63000 |          813 |        1000 |         0.2 |reached target bit errors
      2.5 | 4.4667e-02 | 6.6400e-01 |        2814 |       63000 |          664 |        1000 |         0.2 |reached target bit errors
      3.0 | 3.4365e-02 | 5.1700e-01 |        2165 |       63000 |          517 |        1000 |         0.2 |reached target bit errors
      3.5 | 2.1563e-02 | 3.4950e-01 |        2717 |      126000 |          699 |        2000 |         0.3 |reached target bit errors
      4.0 | 1.3460e-02 | 2.3200e-01 |        2544 |      189000 |          696 |        3000 |         0.5 |reached target bit errors
      4.5 | 7.1778e-03 | 1.2880e-01 |        2261 |      315000 |          644 |        5000 |         0.8 |reached target bit errors
      5.0 | 3.9877e-03 | 7.5889e-02 |        2261 |      567000 |          683 |        9000 |         1.4 |reached target bit errors
      5.5 | 2.1240e-03 | 3.9188e-02 |        2141 |     1008000 |          627 |       16000 |         2.5 |reached target bit errors
      6.0 | 1.0169e-03 | 2.0406e-02 |        2050 |     2016000 |          653 |       32000 |         4.9 |reached target bit errors
      6.5 | 4.4312e-04 | 8.5417e-03 |        2010 |     4536000 |          615 |       72000 |        11.1 |reached target bit errors
```


### Training

We now train the model for a fixed number of SGD training iterations.

**Note**: this is a very basic implementation of the training loop. You can also try more sophisticated training loops with early stopping, different hyper-parameters or optimizers etc.


```python
# training parameters
batch_size = 1000
train_iter = 200
ebno_db = 4.0
clip_value_grad = 10 # gradient clipping for stable training convergence
# bmi is used as metric to evaluate the intermediate results
bmi = BitwiseMutualInformation()
# try also different optimizers or different hyperparameters
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
for it in range(0, train_iter):
    with tf.GradientTape() as tape:
        b, llr, loss = model(batch_size, ebno_db)
    grads = tape.gradient(loss, model.trainable_variables)
    grads = tf.clip_by_value(grads, -clip_value_grad, clip_value_grad, name=None)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    # calculate and print intermediate metrics
    # only for information
    # this has no impact on the training
    if it%10==0: # evaluate every 10 iterations
        # calculate ber from received LLRs
        b_hat = hard_decisions(llr) # hard decided LLRs first
        ber = compute_ber(b, b_hat)
        # and print results
        mi = bmi(b, llr).numpy() # calculate bit-wise mutual information
        l = loss.numpy() # copy loss to numpy for printing
        print(f"Current loss: {l:3f} ber: {ber:.4f} bmi: {mi:.3f}".format())
        bmi.reset_states() # reset the BMI metric
```


```python
Current loss: 0.048708 ber: 0.0120 bmi: 0.934
Current loss: 0.058506 ber: 0.0139 bmi: 0.923
Current loss: 0.052293 ber: 0.0125 bmi: 0.934
Current loss: 0.054314 ber: 0.0134 bmi: 0.928
Current loss: 0.051650 ber: 0.0125 bmi: 0.924
Current loss: 0.047477 ber: 0.0133 bmi: 0.931
Current loss: 0.045135 ber: 0.0122 bmi: 0.935
Current loss: 0.050638 ber: 0.0125 bmi: 0.938
Current loss: 0.045256 ber: 0.0119 bmi: 0.949
Current loss: 0.041335 ber: 0.0124 bmi: 0.952
Current loss: 0.040905 ber: 0.0107 bmi: 0.937
Current loss: 0.043627 ber: 0.0125 bmi: 0.949
Current loss: 0.044397 ber: 0.0126 bmi: 0.942
Current loss: 0.043392 ber: 0.0126 bmi: 0.938
Current loss: 0.043059 ber: 0.0133 bmi: 0.947
Current loss: 0.047521 ber: 0.0130 bmi: 0.937
Current loss: 0.040529 ber: 0.0116 bmi: 0.944
Current loss: 0.041838 ber: 0.0128 bmi: 0.942
Current loss: 0.041801 ber: 0.0130 bmi: 0.940
Current loss: 0.042754 ber: 0.0142 bmi: 0.946
```
### Results

After training, the weights of the decoder have changed. In average, the weights are smaller after training.


```python
model.decoder.show_weights() # show weights AFTER training
```


And let us compare the new BER performance. For this, we can simply call the ber_plot.simulate() function again as it internally stores all previous results (if `add_results` is True).


```python
ebno_dbs = np.array(np.arange(1, 7, 0.5))
batch_size = 10000
mc_ites = 100
ber_plot.simulate(model,
                  ebno_dbs=ebno_dbs,
                  batch_size=1000,
                  num_target_bit_errors=2000, # stop sim after 2000 bit errors
                  legend="Trained",
                  max_mc_iter=mc_iters,
                  soft_estimates=True);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      1.0 | 9.0730e-02 | 9.9600e-01 |        5716 |       63000 |          996 |        1000 |         0.2 |reached target bit errors
      1.5 | 7.8889e-02 | 9.8400e-01 |        4970 |       63000 |          984 |        1000 |         0.1 |reached target bit errors
      2.0 | 6.6365e-02 | 9.2500e-01 |        4181 |       63000 |          925 |        1000 |         0.1 |reached target bit errors
      2.5 | 4.9825e-02 | 8.2000e-01 |        3139 |       63000 |          820 |        1000 |         0.1 |reached target bit errors
      3.0 | 3.6603e-02 | 6.4400e-01 |        2306 |       63000 |          644 |        1000 |         0.1 |reached target bit errors
      3.5 | 2.2302e-02 | 4.2000e-01 |        2810 |      126000 |          840 |        2000 |         0.3 |reached target bit errors
      4.0 | 1.2577e-02 | 2.4400e-01 |        2377 |      189000 |          732 |        3000 |         0.5 |reached target bit errors
      4.5 | 6.5778e-03 | 1.3460e-01 |        2072 |      315000 |          673 |        5000 |         0.7 |reached target bit errors
      5.0 | 2.9769e-03 | 6.2818e-02 |        2063 |      693000 |          691 |       11000 |         1.7 |reached target bit errors
      5.5 | 1.3287e-03 | 2.9667e-02 |        2009 |     1512000 |          712 |       24000 |         3.6 |reached target bit errors
      6.0 | 5.2511e-04 | 1.2967e-02 |        2018 |     3843000 |          791 |       61000 |         9.2 |reached target bit errors
      6.5 | 2.0333e-04 | 5.6000e-03 |        1281 |     6300000 |          560 |      100000 |        15.0 |reached max iter
```


## Further Experiments

You will now see that the memory footprint can be drastically reduced by using the same weight for all messages. In the second part we will apply the concept to the 5G LDPC codes.

### Damped BP

It is well-known that scaling of LLRs / messages can help to improve the performance of BP decoding in some scenarios [3,4]. In particular, this works well for very short codes such as the code we are currently analyzing.

We now follow the basic idea of [2] and scale all weights with the same scalar.


```python
# get weights of trained model
weights_bp = model.decoder.get_weights()
# calc mean value of weights
damping_factor = tf.reduce_mean(weights_bp)
# set all weights to the SAME constant scaling
weights_damped = tf.ones_like(weights_bp) * damping_factor
# and apply the new weights
model.decoder.set_weights(weights_damped)
# let us have look at the new weights again
model.decoder.show_weights()
# and simulate the BER again
leg_str = f"Damped BP (scaling factor {damping_factor.numpy():.3f})"
ber_plot.simulate(model,
                  ebno_dbs=ebno_dbs,
                  batch_size=1000,
                  num_target_bit_errors=2000, # stop sim after 2000 bit errors
                  legend=leg_str,
                  max_mc_iter=mc_iters,
                  soft_estimates=True);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      1.0 | 9.0333e-02 | 9.9500e-01 |        5691 |       63000 |          995 |        1000 |         0.2 |reached target bit errors
      1.5 | 7.6413e-02 | 9.7800e-01 |        4814 |       63000 |          978 |        1000 |         0.2 |reached target bit errors
      2.0 | 6.1556e-02 | 8.9800e-01 |        3878 |       63000 |          898 |        1000 |         0.1 |reached target bit errors
      2.5 | 4.8746e-02 | 7.9700e-01 |        3071 |       63000 |          797 |        1000 |         0.2 |reached target bit errors
      3.0 | 3.5746e-02 | 6.0800e-01 |        2252 |       63000 |          608 |        1000 |         0.1 |reached target bit errors
      3.5 | 2.0857e-02 | 3.7950e-01 |        2628 |      126000 |          759 |        2000 |         0.3 |reached target bit errors
      4.0 | 1.2222e-02 | 2.3433e-01 |        2310 |      189000 |          703 |        3000 |         0.5 |reached target bit errors
      4.5 | 6.4524e-03 | 1.2967e-01 |        2439 |      378000 |          778 |        6000 |         0.9 |reached target bit errors
      5.0 | 2.7712e-03 | 5.8667e-02 |        2095 |      756000 |          704 |       12000 |         1.8 |reached target bit errors
      5.5 | 1.2844e-03 | 2.8960e-02 |        2023 |     1575000 |          724 |       25000 |         3.7 |reached target bit errors
      6.0 | 5.0743e-04 | 1.2032e-02 |        2014 |     3969000 |          758 |       63000 |         9.5 |reached target bit errors
      6.5 | 2.1730e-04 | 5.5200e-03 |        1369 |     6300000 |          552 |      100000 |        15.2 |reached max iter
```


When looking at the results, we observe almost the same performance although we only scale by a single scalar. This implies that the number of weights of our model is by far too large and the memory footprint could be reduced significantly. However, isnt it fascinating to see that this simple concept of weighted BP leads to the same results as the concept of *damped BP*?

**Note**: for more iterations it could be beneficial to implement an individual damping per iteration.

### Learning the 5G LDPC Code

In this Section, you will experience what happens if we apply the same concept to the 5G LDPC code (including rate matching).

For this, we need to define a new model.


```python
class WeightedBP5G(tf.keras.Model):
    """System model for BER simulations of weighted BP decoding for 5G instruction_answer codes.
    This model uses `GaussianPriorSource` to mimic the LLRs after demapping of
    QPSK symbols transmitted over an AWGN channel.
    Parameters
    ----------
        k: int
            Number of information bits per codeword.
        n: int
            Codeword length.
        num_iter: int
            Number of BP decoding iterations.
    Input
    -----
        batch_size: int or tf.int
            The batch_size used for the simulation.
        ebno_db: float or tf.float
            A float defining the simulation SNR.
    Output
    ------
        (u, u_hat, loss):
            Tuple:
        u: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the transmitted information bits.
        u_hat: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the estimated information bits.
        loss: tf.float32
            Binary cross-entropy loss between `u` and `u_hat`.
    """
    def __init__(self, k, n, num_iter=20):
        super().__init__()
        # we need to initialize an encoder for the 5G parameters
        self.encoder = LDPC5GEncoder(k, n)
        self.decoder = LDPC5GDecoder(self.encoder,
                                     num_iter=1, # iterations are done via outer loop (to access intermediate results for multi-loss)
                                     stateful=True,
                                     hard_out=False,
                                     cn_type="boxplus",
                                     trainable=True)
        self.llr_source = GaussianPriorSource()
        self._num_iter = num_iter
        self._coderate = k/n
        self._bce = BinaryCrossentropy(from_logits=True)
    def call(self, batch_size, ebno_db):
        noise_var = ebnodb2no(ebno_db,
                              num_bits_per_symbol=2, # QPSK
                              coderate=self._coderate)
        # BPSK modulated all-zero CW
        c = tf.zeros([batch_size, k]) # decoder only returns info bits
        # use fake llrs from GA
        # works as BP is symmetric
        llr = self.llr_source([[batch_size, n], noise_var])
        # --- implement multi-loss is proposed by Nachmani et al. ---
        loss = 0
        msg_vn = None
        for i in range(self._num_iter):
            c_hat, msg_vn = self.decoder((llr, msg_vn)) # perform one decoding iteration; decoder returns soft-values
            loss += self._bce(c, c_hat)  # add loss after each iteration
        return c, c_hat, loss
```

```python
# generate model
num_iter = 10
k = 400
n = 800
model5G = WeightedBP5G(k, n, num_iter=num_iter)
# generate baseline BER
ebno_dbs = np.array(np.arange(0, 4, 0.25))
mc_iters = 100 # number of monte carlo iterations
ber_plot_5G = PlotBER("Weighted BP for 5G instruction_answer")
# simulate the untrained performance
ber_plot_5G.simulate(model5G,
                     ebno_dbs=ebno_dbs,
                     batch_size=1000,
                     num_target_bit_errors=2000, # stop sim after 2000 bit errors
                     legend="Untrained",
                     soft_estimates=True,
                     max_mc_iter=mc_iters);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 1.6660e-01 | 1.0000e+00 |       66640 |      400000 |         1000 |        1000 |         0.2 |reached target bit errors
     0.25 | 1.4864e-01 | 1.0000e+00 |       59455 |      400000 |         1000 |        1000 |         0.2 |reached target bit errors
      0.5 | 1.2470e-01 | 9.9700e-01 |       49880 |      400000 |          997 |        1000 |         0.2 |reached target bit errors
     0.75 | 9.4408e-02 | 9.8000e-01 |       37763 |      400000 |          980 |        1000 |         0.2 |reached target bit errors
      1.0 | 6.6635e-02 | 9.3900e-01 |       26654 |      400000 |          939 |        1000 |         0.2 |reached target bit errors
     1.25 | 4.1078e-02 | 8.1100e-01 |       16431 |      400000 |          811 |        1000 |         0.2 |reached target bit errors
      1.5 | 2.1237e-02 | 6.1200e-01 |        8495 |      400000 |          612 |        1000 |         0.2 |reached target bit errors
     1.75 | 9.2050e-03 | 3.7600e-01 |        3682 |      400000 |          376 |        1000 |         0.2 |reached target bit errors
      2.0 | 2.7175e-03 | 1.7050e-01 |        2174 |      800000 |          341 |        2000 |         0.5 |reached target bit errors
     2.25 | 8.8167e-04 | 6.3833e-02 |        2116 |     2400000 |          383 |        6000 |         1.5 |reached target bit errors
      2.5 | 2.1781e-04 | 2.1875e-02 |        2091 |     9600000 |          525 |       24000 |         5.9 |reached target bit errors
     2.75 | 4.2950e-05 | 4.9600e-03 |        1718 |    40000000 |          496 |      100000 |        24.5 |reached max iter
      3.0 | 6.8000e-06 | 9.1000e-04 |         272 |    40000000 |           91 |      100000 |        24.5 |reached max iter
     3.25 | 9.2500e-07 | 1.8000e-04 |          37 |    40000000 |           18 |      100000 |        24.5 |reached max iter
      3.5 | 2.5000e-08 | 1.0000e-05 |           1 |    40000000 |            1 |      100000 |        24.5 |reached max iter
     3.75 | 0.0000e+00 | 0.0000e+00 |           0 |    40000000 |            0 |      100000 |        24.1 |reached max iter
Simulation stopped as no error occurred @ EbNo = 3.8 dB.

```


And lets train this new model.


```python
# training parameters
batch_size = 1000
train_iter = 200
clip_value_grad = 10 # gradient clipping seems to be important
# smaller training SNR as the new code is longer (=stronger) than before
ebno_db = 1.5 # rule of thumb: train at ber = 1e-2
# only used as metric
bmi = BitwiseMutualInformation()
# try also different optimizers or different hyperparameters
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
# and let's go
for it in range(0, train_iter):
    with tf.GradientTape() as tape:
        b, llr, loss = model5G(batch_size, ebno_db)
    grads = tape.gradient(loss, model5G.trainable_variables)
    grads = tf.clip_by_value(grads, -clip_value_grad, clip_value_grad, name=None)
    optimizer.apply_gradients(zip(grads, model5G.trainable_weights))
    # calculate and print intermediate metrics
    if it%10==0:
        # calculate ber
        b_hat = hard_decisions(llr)
        ber = compute_ber(b, b_hat)
        # and print results
        mi = bmi(b, llr).numpy()
        l = loss.numpy()
        print(f"Current loss: {l:3f} ber: {ber:.4f} bmi: {mi:.3f}".format())
        bmi.reset_states()
```


```python
Current loss: 1.708751 ber: 0.0204 bmi: 0.925
Current loss: 1.745474 ber: 0.0219 bmi: 0.918
Current loss: 1.741312 ber: 0.0224 bmi: 0.917
Current loss: 1.707712 ber: 0.0208 bmi: 0.923
Current loss: 1.705274 ber: 0.0209 bmi: 0.923
Current loss: 1.706761 ber: 0.0211 bmi: 0.922
Current loss: 1.711995 ber: 0.0212 bmi: 0.921
Current loss: 1.729707 ber: 0.0223 bmi: 0.917
Current loss: 1.692947 ber: 0.0205 bmi: 0.924
Current loss: 1.703924 ber: 0.0203 bmi: 0.924
Current loss: 1.743640 ber: 0.0220 bmi: 0.919
Current loss: 1.719159 ber: 0.0220 bmi: 0.919
Current loss: 1.728399 ber: 0.0221 bmi: 0.920
Current loss: 1.717423 ber: 0.0211 bmi: 0.922
Current loss: 1.743661 ber: 0.0225 bmi: 0.918
Current loss: 1.704675 ber: 0.0212 bmi: 0.923
Current loss: 1.690425 ber: 0.0206 bmi: 0.924
Current loss: 1.728023 ber: 0.0212 bmi: 0.922
Current loss: 1.724549 ber: 0.0212 bmi: 0.922
Current loss: 1.739966 ber: 0.0224 bmi: 0.917
```


We now simulate the new results and compare it to the untrained results.


```python
ebno_dbs = np.array(np.arange(0, 4, 0.25))
batch_size = 1000
mc_iters = 100
ber_plot_5G.simulate(model5G,
                     ebno_dbs=ebno_dbs,
                     batch_size=batch_size,
                     num_target_bit_errors=2000, # stop sim after 2000 bit errors
                     legend="Trained",
                     max_mc_iter=mc_iters,
                     soft_estimates=True);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 1.6568e-01 | 1.0000e+00 |       66273 |      400000 |         1000 |        1000 |         0.2 |reached target bit errors
     0.25 | 1.4965e-01 | 9.9900e-01 |       59858 |      400000 |          999 |        1000 |         0.2 |reached target bit errors
      0.5 | 1.2336e-01 | 9.9900e-01 |       49342 |      400000 |          999 |        1000 |         0.2 |reached target bit errors
     0.75 | 9.6135e-02 | 9.9100e-01 |       38454 |      400000 |          991 |        1000 |         0.3 |reached target bit errors
      1.0 | 6.8543e-02 | 9.4500e-01 |       27417 |      400000 |          945 |        1000 |         0.2 |reached target bit errors
     1.25 | 3.9152e-02 | 8.3300e-01 |       15661 |      400000 |          833 |        1000 |         0.2 |reached target bit errors
      1.5 | 2.2040e-02 | 6.2400e-01 |        8816 |      400000 |          624 |        1000 |         0.2 |reached target bit errors
     1.75 | 9.1300e-03 | 3.8400e-01 |        3652 |      400000 |          384 |        1000 |         0.2 |reached target bit errors
      2.0 | 2.8075e-03 | 1.6600e-01 |        2246 |      800000 |          332 |        2000 |         0.5 |reached target bit errors
     2.25 | 8.5500e-04 | 6.2000e-02 |        2052 |     2400000 |          372 |        6000 |         1.4 |reached target bit errors
      2.5 | 1.9837e-04 | 2.1115e-02 |        2063 |    10400000 |          549 |       26000 |         6.3 |reached target bit errors
     2.75 | 2.9600e-05 | 4.1000e-03 |        1184 |    40000000 |          410 |      100000 |        24.3 |reached max iter
      3.0 | 6.5750e-06 | 9.1000e-04 |         263 |    40000000 |           91 |      100000 |        24.3 |reached max iter
     3.25 | 5.5000e-07 | 1.4000e-04 |          22 |    40000000 |           14 |      100000 |        24.3 |reached max iter
      3.5 | 7.5000e-08 | 3.0000e-05 |           3 |    40000000 |            3 |      100000 |        24.5 |reached max iter
     3.75 | 2.5000e-08 | 1.0000e-05 |           1 |    40000000 |            1 |      100000 |        24.3 |reached max iter
```


Unfortunately, we observe only very minor gains for the 5G LDPC code. We empirically observed that gain vanishes for more iterations and longer codewords, i.e., for most practical use-cases of the 5G LDPC code the gains are only minor.

However, there may be other `codes` `on` `graphs` that benefit from the principle idea of weighted BP - or other channel setups? Feel free to adjust this notebook and train for your favorite code / channel.

Other ideas for own experiments:

- Implement weighted BP with unique weights per iteration.
- Apply the concept to (scaled) min-sum decoding as in [5].
- Can you replace the complete CN update by a neural network?
- Verify the results from all-zero simulations for a *real* system simulation with explicit encoder and random data
- What happens in combination with higher order modulation?

## References

[1]E. Nachmani, Y. Beery and D. Burshtein, Learning to Decode Linear Codes Using Deep Learning, IEEE Annual Allerton Conference on Communication, Control, and Computing (Allerton), pp.341-346., 2016. [https://arxiv.org/pdf/1607.04793.pdf](https://arxiv.org/pdf/1607.04793.pdf)

[2] M. Lian, C. Hger, and H. Pfister, What can machine learning teach us about communications? IEEE Information Theory Workshop (ITW), pp.1-5. 2018.

[3] ] M. Pretti, A message passing algorithm with damping, J. Statist. Mech.: Theory Practice, p.11008, Nov.2005.

[4] J.S. Yedidia, W.T. Freeman and Y. Weiss, Constructing free energy approximations and Generalized Belief Propagation algorithms, IEEE Transactions on Information Theory, 2005.

[5] E. Nachmani, E. Marciano, L. Lugosch, W. Gross, D. Burshtein and Y. Beery, Deep learning methods for improved decoding of linear codes, IEEE Journal of Selected Topics in Signal Processing, vol.12, no. 1, pp.119-131, 2018.
[5] E. Nachmani, E. Marciano, L. Lugosch, W. Gross, D. Burshtein and Y. Beery, Deep learning methods for improved decoding of linear codes, IEEE Journal of Selected Topics in Signal Processing, vol.12, no. 1, pp.119-131, 2018.
