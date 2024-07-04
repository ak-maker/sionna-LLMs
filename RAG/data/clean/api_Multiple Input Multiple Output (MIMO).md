# Multiple-Input Multiple-Output (MIMO)

This module provides layers and functions to support simulation of multicell
MIMO transmissions.

## Stream Management

Stream management determines which transmitter is sending which stream to
which receiver. Transmitters and receivers can be user terminals or base
stations, depending on whether uplink or downlink transmissions are considered.
The [`StreamManagement`](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement) class has various properties that
are needed to recover desired or interfering channel coefficients for precoding
and equalization. In order to understand how the various properties of
[`StreamManagement`](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement) can be used, we recommend to have a look
at the source code of the [`LMMSEEqualizer`](ofdm.html#sionna.ofdm.LMMSEEqualizer) or
[`ZFPrecoder`](ofdm.html#sionna.ofdm.ZFPrecoder).

The following code snippet shows how to configure
[`StreamManagement`](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement) for a simple uplink scenario, where
four transmitters send each one stream to a receiver. Note that
[`StreamManagement`](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement) is independent of the actual number of
antennas at the transmitters and receivers.
```python
num_tx = 4
num_rx = 1
num_streams_per_tx = 1
# Indicate which transmitter is associated with which receiver
# rx_tx_association[i,j] = 1 means that transmitter j sends one
# or mutiple streams to receiver i.
rx_tx_association = np.zeros([num_rx, num_tx])
rx_tx_association[0,0] = 1
rx_tx_association[0,1] = 1
rx_tx_association[0,2] = 1
rx_tx_association[0,3] = 1
sm = StreamManagement(rx_tx_association, num_streams_per_tx)
```
`class` `sionna.mimo.``StreamManagement`(*`rx_tx_association`*, *`num_streams_per_tx`*)[`[source]`](../_modules/sionna/mimo/stream_management.html#StreamManagement)

Class for management of streams in multi-cell MIMO networks.
Parameters

- **rx_tx_association** (*[**num_rx**, **num_tx**]**, **np.int*)  A binary NumPy array where `rx_tx_association[i,j]=1` means
that receiver <cite>i</cite> gets one or multiple streams from
transmitter <cite>j</cite>.
- **num_streams_per_tx** (*int*)  Indicates the number of streams that are transmitted by each
transmitter.


**Note**

Several symmetry constraints on `rx_tx_association` are imposed
to ensure efficient processing. All row sums and all column sums
must be equal, i.e., all receivers have the same number of associated
transmitters and all transmitters have the same number of associated
receivers. It is also assumed that all transmitters send the same
number of streams `num_streams_per_tx`.

`property` `detection_desired_ind`

Indices needed to gather desired channels for receive processing.

A NumPy array of shape <cite>[num_rx*num_streams_per_rx]</cite> that
can be used to gather desired channels from the flattened
channel tensor of shape
<cite>[,num_rx, num_tx, num_streams_per_tx,]</cite>.
The result of the gather operation can be reshaped to
<cite>[,num_rx, num_streams_per_rx,]</cite>.


`property` `detection_undesired_ind`

Indices needed to gather undesired channels for receive processing.

A NumPy array of shape <cite>[num_rx*num_streams_per_rx]</cite> that
can be used to gather undesired channels from the flattened
channel tensor of shape <cite>[,num_rx, num_tx, num_streams_per_tx,]</cite>.
The result of the gather operation can be reshaped to
<cite>[,num_rx, num_interfering_streams_per_rx,]</cite>.


`property` `num_interfering_streams_per_rx`

Number of interfering streams received at each eceiver.


`property` `num_rx`

Number of receivers.


`property` `num_rx_per_tx`

Number of receivers communicating with a transmitter.


`property` `num_streams_per_rx`

Number of streams transmitted to each receiver.


`property` `num_streams_per_tx`

Number of streams per transmitter.


`property` `num_tx`

Number of transmitters.


`property` `num_tx_per_rx`

Number of transmitters communicating with a receiver.


`property` `precoding_ind`

Indices needed to gather channels for precoding.

A NumPy array of shape <cite>[num_tx, num_rx_per_tx]</cite>,
where `precoding_ind[i,:]` contains the indices of the
receivers to which transmitter <cite>i</cite> is sending streams.


`property` `rx_stream_ids`

Mapping of streams to receivers.

A Numpy array of shape <cite>[num_rx, num_streams_per_rx]</cite>.
This array is obtained from `tx_stream_ids` together with
the `rx_tx_association`. `rx_stream_ids[i,:]` contains
the indices of streams that are supposed to be decoded by receiver <cite>i</cite>.


`property` `rx_tx_association`

Association between receivers and transmitters.

A binary NumPy array of shape <cite>[num_rx, num_tx]</cite>,
where `rx_tx_association[i,j]=1` means that receiver <cite>i</cite>
gets one ore multiple streams from transmitter <cite>j</cite>.


`property` `stream_association`

Association between receivers, transmitters, and streams.

A binary NumPy array of shape
<cite>[num_rx, num_tx, num_streams_per_tx]</cite>, where
`stream_association[i,j,k]=1` means that receiver <cite>i</cite> gets
the <cite>k</cite> th stream from transmitter <cite>j</cite>.


`property` `stream_ind`

Indices needed to gather received streams in the correct order.

A NumPy array of shape <cite>[num_rx*num_streams_per_rx]</cite> that can be
used to gather streams from the flattened tensor of received streams
of shape <cite>[,num_rx, num_streams_per_rx,]</cite>. The result of the
gather operation is then reshaped to
<cite>[,num_tx, num_streams_per_tx,]</cite>.


`property` `tx_stream_ids`

Mapping of streams to transmitters.

A NumPy array of shape <cite>[num_tx, num_streams_per_tx]</cite>.
Streams are numbered from 0,1, and assiged to transmitters in
increasing order, i.e., transmitter 0 gets the first
<cite>num_streams_per_tx</cite> and so on.


## Precoding

### zero_forcing_precoder

`sionna.mimo.``zero_forcing_precoder`(*`x`*, *`h`*, *`return_precoding_matrix``=``False`*)[`[source]`](../_modules/sionna/mimo/precoding.html#zero_forcing_precoder)

Zero-Forcing (ZF) Precoder

This function implements ZF precoding for a MIMO link, assuming the
following model:

$$
\mathbf{y} = \mathbf{H}\mathbf{G}\mathbf{x} + \mathbf{n}
$$

where $\mathbf{y}\in\mathbb{C}^K$ is the received signal vector,
$\mathbf{H}\in\mathbb{C}^{K\times M}$ is the known channel matrix,
$\mathbf{G}\in\mathbb{C}^{M\times K}$ is the precoding matrix,
$\mathbf{x}\in\mathbb{C}^K$ is the symbol vector to be precoded,
and $\mathbf{n}\in\mathbb{C}^K$ is a noise vector. It is assumed that
$K\le M$.

The precoding matrix $\mathbf{G}$ is defined as (Eq. 4.37) [[BHS2017]](channel.wireless.html#bhs2017) :

$$
\mathbf{G} = \mathbf{V}\mathbf{D}
$$

where

$$
\begin{split}\mathbf{V} &= \mathbf{H}^{\mathsf{H}}\left(\mathbf{H} \mathbf{H}^{\mathsf{H}}\right)^{-1}\\
\mathbf{D} &= \mathop{\text{diag}}\left( \lVert \mathbf{v}_{k} \rVert_2^{-1}, k=0,\dots,K-1 \right).\end{split}
$$

This ensures that each stream is precoded with a unit-norm vector,
i.e., $\mathop{\text{tr}}\left(\mathbf{G}\mathbf{G}^{\mathsf{H}}\right)=K$.
The function returns the precoded vector $\mathbf{G}\mathbf{x}$.
Input

- **x** (*[,K], tf.complex*)  1+D tensor containing the symbol vectors to be precoded.
- **h** (*[,K,M], tf.complex*)  2+D tensor containing the channel matrices
- **return_precoding_matrices** (*bool*)  Indicates if the precoding matrices should be returned or not.
Defaults to False.


Output

- **x_precoded** (*[,M], tf.complex*)  Tensor of the same shape and dtype as `x` apart from the last
dimensions that has changed from <cite>K</cite> to <cite>M</cite>. It contains the
precoded symbol vectors.
- **g** (*[,M,K], tf.complex*)  2+D tensor containing the precoding matrices. It is only returned
if `return_precoding_matrices=True`.


**Note**

If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See [`xla_compat`](config.html#sionna.Config.xla_compat).

## Equalization

### lmmse_equalizer

`sionna.mimo.``lmmse_equalizer`(*`y`*, *`h`*, *`s`*, *`whiten_interference``=``True`*)[`[source]`](../_modules/sionna/mimo/equalization.html#lmmse_equalizer)

MIMO LMMSE Equalizer

This function implements LMMSE equalization for a MIMO link, assuming the
following model:

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$

where $\mathbf{y}\in\mathbb{C}^M$ is the received signal vector,
$\mathbf{x}\in\mathbb{C}^K$ is the vector of transmitted symbols,
$\mathbf{H}\in\mathbb{C}^{M\times K}$ is the known channel matrix,
and $\mathbf{n}\in\mathbb{C}^M$ is a noise vector.
It is assumed that $\mathbb{E}\left[\mathbf{x}\right]=\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}$,
$\mathbb{E}\left[\mathbf{x}\mathbf{x}^{\mathsf{H}}\right]=\mathbf{I}_K$ and
$\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}$.

The estimated symbol vector $\hat{\mathbf{x}}\in\mathbb{C}^K$ is given as
(Lemma B.19) [[BHS2017]](channel.wireless.html#bhs2017) :

$$
\hat{\mathbf{x}} = \mathop{\text{diag}}\left(\mathbf{G}\mathbf{H}\right)^{-1}\mathbf{G}\mathbf{y}
$$

where

$$
\mathbf{G} = \mathbf{H}^{\mathsf{H}} \left(\mathbf{H}\mathbf{H}^{\mathsf{H}} + \mathbf{S}\right)^{-1}.
$$

This leads to the post-equalized per-symbol model:

$$
\hat{x}_k = x_k + e_k,\quad k=0,\dots,K-1
$$

where the variances $\sigma^2_k$ of the effective residual noise
terms $e_k$ are given by the diagonal elements of

$$
\mathop{\text{diag}}\left(\mathbb{E}\left[\mathbf{e}\mathbf{e}^{\mathsf{H}}\right]\right)
= \mathop{\text{diag}}\left(\mathbf{G}\mathbf{H} \right)^{-1} - \mathbf{I}.
$$

Note that the scaling by $\mathop{\text{diag}}\left(\mathbf{G}\mathbf{H}\right)^{-1}$
is important for the [`Demapper`](mapping.html#sionna.mapping.Demapper) although it does
not change the signal-to-noise ratio.

The function returns $\hat{\mathbf{x}}$ and
$\boldsymbol{\sigma}^2=\left[\sigma^2_0,\dots, \sigma^2_{K-1}\right]^{\mathsf{T}}$.
Input

- **y** (*[,M], tf.complex*)  1+D tensor containing the received signals.
- **h** (*[,M,K], tf.complex*)  2+D tensor containing the channel matrices.
- **s** (*[,M,M], tf.complex*)  2+D tensor containing the noise covariance matrices.
- **whiten_interference** (*bool*)  If <cite>True</cite> (default), the interference is first whitened before equalization.
In this case, an alternative expression for the receive filter is used that
can be numerically more stable. Defaults to <cite>True</cite>.


Output

- **x_hat** (*[,K], tf.complex*)  1+D tensor representing the estimated symbol vectors.
- **no_eff** (*tf.float*)  Tensor of the same shape as `x_hat` containing the effective noise
variance estimates.


**Note**

If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See [`xla_compat`](config.html#sionna.Config.xla_compat).

### mf_equalizer

`sionna.mimo.``mf_equalizer`(*`y`*, *`h`*, *`s`*)[`[source]`](../_modules/sionna/mimo/equalization.html#mf_equalizer)

MIMO MF Equalizer

This function implements matched filter (MF) equalization for a
MIMO link, assuming the following model:

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$

where $\mathbf{y}\in\mathbb{C}^M$ is the received signal vector,
$\mathbf{x}\in\mathbb{C}^K$ is the vector of transmitted symbols,
$\mathbf{H}\in\mathbb{C}^{M\times K}$ is the known channel matrix,
and $\mathbf{n}\in\mathbb{C}^M$ is a noise vector.
It is assumed that $\mathbb{E}\left[\mathbf{x}\right]=\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}$,
$\mathbb{E}\left[\mathbf{x}\mathbf{x}^{\mathsf{H}}\right]=\mathbf{I}_K$ and
$\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}$.

The estimated symbol vector $\hat{\mathbf{x}}\in\mathbb{C}^K$ is given as
(Eq. 4.11) [[BHS2017]](channel.wireless.html#bhs2017) :

$$
\hat{\mathbf{x}} = \mathbf{G}\mathbf{y}
$$

where

$$
\mathbf{G} = \mathop{\text{diag}}\left(\mathbf{H}^{\mathsf{H}}\mathbf{H}\right)^{-1}\mathbf{H}^{\mathsf{H}}.
$$

This leads to the post-equalized per-symbol model:

$$
\hat{x}_k = x_k + e_k,\quad k=0,\dots,K-1
$$

where the variances $\sigma^2_k$ of the effective residual noise
terms $e_k$ are given by the diagonal elements of the matrix

$$
\mathbb{E}\left[\mathbf{e}\mathbf{e}^{\mathsf{H}}\right]
= \left(\mathbf{I}-\mathbf{G}\mathbf{H} \right)\left(\mathbf{I}-\mathbf{G}\mathbf{H} \right)^{\mathsf{H}} + \mathbf{G}\mathbf{S}\mathbf{G}^{\mathsf{H}}.
$$

Note that the scaling by $\mathop{\text{diag}}\left(\mathbf{H}^{\mathsf{H}}\mathbf{H}\right)^{-1}$
in the definition of $\mathbf{G}$
is important for the [`Demapper`](mapping.html#sionna.mapping.Demapper) although it does
not change the signal-to-noise ratio.

The function returns $\hat{\mathbf{x}}$ and
$\boldsymbol{\sigma}^2=\left[\sigma^2_0,\dots, \sigma^2_{K-1}\right]^{\mathsf{T}}$.
Input

- **y** (*[,M], tf.complex*)  1+D tensor containing the received signals.
- **h** (*[,M,K], tf.complex*)  2+D tensor containing the channel matrices.
- **s** (*[,M,M], tf.complex*)  2+D tensor containing the noise covariance matrices.


Output

- **x_hat** (*[,K], tf.complex*)  1+D tensor representing the estimated symbol vectors.
- **no_eff** (*tf.float*)  Tensor of the same shape as `x_hat` containing the effective noise
variance estimates.


### zf_equalizer

`sionna.mimo.``zf_equalizer`(*`y`*, *`h`*, *`s`*)[`[source]`](../_modules/sionna/mimo/equalization.html#zf_equalizer)

MIMO ZF Equalizer

This function implements zero-forcing (ZF) equalization for a MIMO link, assuming the
following model:

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$

where $\mathbf{y}\in\mathbb{C}^M$ is the received signal vector,
$\mathbf{x}\in\mathbb{C}^K$ is the vector of transmitted symbols,
$\mathbf{H}\in\mathbb{C}^{M\times K}$ is the known channel matrix,
and $\mathbf{n}\in\mathbb{C}^M$ is a noise vector.
It is assumed that $\mathbb{E}\left[\mathbf{x}\right]=\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}$,
$\mathbb{E}\left[\mathbf{x}\mathbf{x}^{\mathsf{H}}\right]=\mathbf{I}_K$ and
$\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}$.

The estimated symbol vector $\hat{\mathbf{x}}\in\mathbb{C}^K$ is given as
(Eq. 4.10) [[BHS2017]](channel.wireless.html#bhs2017) :

$$
\hat{\mathbf{x}} = \mathbf{G}\mathbf{y}
$$

where

$$
\mathbf{G} = \left(\mathbf{H}^{\mathsf{H}}\mathbf{H}\right)^{-1}\mathbf{H}^{\mathsf{H}}.
$$

This leads to the post-equalized per-symbol model:

$$
\hat{x}_k = x_k + e_k,\quad k=0,\dots,K-1
$$

where the variances $\sigma^2_k$ of the effective residual noise
terms $e_k$ are given by the diagonal elements of the matrix

$$
\mathbb{E}\left[\mathbf{e}\mathbf{e}^{\mathsf{H}}\right]
= \mathbf{G}\mathbf{S}\mathbf{G}^{\mathsf{H}}.
$$

The function returns $\hat{\mathbf{x}}$ and
$\boldsymbol{\sigma}^2=\left[\sigma^2_0,\dots, \sigma^2_{K-1}\right]^{\mathsf{T}}$.
Input

- **y** (*[,M], tf.complex*)  1+D tensor containing the received signals.
- **h** (*[,M,K], tf.complex*)  2+D tensor containing the channel matrices.
- **s** (*[,M,M], tf.complex*)  2+D tensor containing the noise covariance matrices.


Output

- **x_hat** (*[,K], tf.complex*)  1+D tensor representing the estimated symbol vectors.
- **no_eff** (*tf.float*)  Tensor of the same shape as `x_hat` containing the effective noise
variance estimates.


**Note**

If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See [`xla_compat`](config.html#sionna.Config.xla_compat).

## Detection

### EPDetector

`class` `sionna.mimo.``EPDetector`(*`output`*, *`num_bits_per_symbol`*, *`hard_out``=``False`*, *`l``=``10`*, *`beta``=``0.9`*, *`dtype``=``tf.complex64`*)[`[source]`](../_modules/sionna/mimo/detection.html#EPDetector)

MIMO Expectation Propagation (EP) detector

This layer implements Expectation Propagation (EP) MIMO detection as described
in [[EP2014]](https://nvlabs.github.io/sionna/api/mimo.html#ep2014). It can generate hard- or soft-decisions for symbols or bits.

This layer assumes the following channel model:

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$

where $\mathbf{y}\in\mathbb{C}^M$ is the received signal vector,
$\mathbf{x}\in\mathcal{C}^S$ is the vector of transmitted symbols which
are uniformly and independently drawn from the constellation $\mathcal{C}$,
$\mathbf{H}\in\mathbb{C}^{M\times S}$ is the known channel matrix,
and $\mathbf{n}\in\mathbb{C}^M$ is a complex Gaussian noise vector.
It is assumed that $\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}$ and
$\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}$,
where $\mathbf{S}$ has full rank.

The channel model is first whitened using [`whiten_channel()`](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.whiten_channel)
and then converted to its real-valued equivalent,
see [`complex2real_channel()`](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_channel), prior to MIMO detection.

The computation of LLRs is done by converting the symbol logits
that naturally arise in the algorithm to LLRs using
[`PAM2QAM()`](mapping.html#sionna.mapping.PAM2QAM). Custom conversions of symbol logits to LLRs
can be implemented by using the soft-symbol output.
Parameters

- **output** (*One of** [**"bit"**, **"symbol"**]**, **str*)  The type of output, either bits or symbols. Whether soft- or
hard-decisions are returned can be configured with the
`hard_out` flag.
- **num_bits_per_symbol** (*int*)  The number of bits per QAM constellation symbol, e.g., 4 for QAM16.
- **hard_out** (*bool*)  If <cite>True</cite>, the detector computes hard-decided bit values or
constellation point indices instead of soft-values.
Defaults to <cite>False</cite>.
- **l** (*int*)  Number of iterations. Defaults to 10.
- **beta** (*float*)  Parameter $\beta\in[0,1]$ for update smoothing.
Defaults to 0.9.
- **dtype** (*One of** [**tf.complex64**, **tf.complex128**] **tf.DType** (**dtype**)*)  Precision used for internal computations. Defaults to `tf.complex64`.
Especially for large MIMO setups, the precision can make a significant
performance difference.


Input

- **(y, h, s)**  Tuple:
- **y** (*[,M], tf.complex*)  1+D tensor containing the received signals
- **h** (*[,M,num_streams], tf.complex*)  2+D tensor containing the channel matrices
- **s** (*[,M,M], tf.complex*)  2+D tensor containing the noise covariance matrices


Output

- **One of**
- *[,num_streams,num_bits_per_symbol], tf.float*  LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>bit</cite>
- *[,num_streams,2**num_bits_per_symbol], tf.float or [,num_streams], tf.int*  Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>symbol</cite>


**Note**

For numerical stability, we do not recommend to use this function in Graph
mode with XLA, i.e., within a function that is decorated with
`@tf.function(jit_compile=True)`.
However, it is possible to do so by setting
`sionna.Config.xla_compat=true`.
See [`xla_compat`](config.html#sionna.Config.xla_compat).

### KBestDetector

`class` `sionna.mimo.``KBestDetector`(*`output`*, *`num_streams`*, *`k`*, *`constellation_type``=``None`*, *`num_bits_per_symbol``=``None`*, *`constellation``=``None`*, *`hard_out``=``False`*, *`use_real_rep``=``False`*, *`list2llr``=``None`*, *`dtype``=``tf.complex64`*)[`[source]`](../_modules/sionna/mimo/detection.html#KBestDetector)

MIMO K-Best detector

This layer implements K-Best MIMO detection as described
in (Eq. 4-5) [[FT2015]](https://nvlabs.github.io/sionna/api/mimo.html#ft2015). It can either generate hard decisions (for symbols
or bits) or compute LLRs.

The algorithm operates in either the complex or real-valued domain.
Although both options produce identical results, the former has the advantage
that it can be applied to arbitrary non-QAM constellations. It also reduces
the number of streams (or depth) by a factor of two.

The way soft-outputs (i.e., LLRs) are computed is determined by the
`list2llr` function. The default solution
[`List2LLRSimple`](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.List2LLRSimple) assigns a predetermined
value to all LLRs without counter-hypothesis.

This layer assumes the following channel model:

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$

where $\mathbf{y}\in\mathbb{C}^M$ is the received signal vector,
$\mathbf{x}\in\mathcal{C}^S$ is the vector of transmitted symbols which
are uniformly and independently drawn from the constellation $\mathcal{C}$,
$\mathbf{H}\in\mathbb{C}^{M\times S}$ is the known channel matrix,
and $\mathbf{n}\in\mathbb{C}^M$ is a complex Gaussian noise vector.
It is assumed that $\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}$ and
$\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}$,
where $\mathbf{S}$ has full rank.

In a first optional step, the channel model is converted to its real-valued equivalent,
see [`complex2real_channel()`](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_channel). We assume in the sequel the complex-valued
representation. Then, the channel is whitened using [`whiten_channel()`](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.whiten_channel):

$$
\begin{split}\tilde{\mathbf{y}} &= \mathbf{S}^{-\frac{1}{2}}\mathbf{y}\\
&=  \mathbf{S}^{-\frac{1}{2}}\mathbf{H}\mathbf{x} + \mathbf{S}^{-\frac{1}{2}}\mathbf{n}\\
&= \tilde{\mathbf{H}}\mathbf{x} + \tilde{\mathbf{n}}.\end{split}
$$

Next, the columns of $\tilde{\mathbf{H}}$ are sorted according
to their norm in descending order. Then, the QR decomposition of the
resulting channel matrix is computed:

$$
\tilde{\mathbf{H}} = \mathbf{Q}\mathbf{R}
$$

where $\mathbf{Q}\in\mathbb{C}^{M\times S}$ is unitary and
$\mathbf{R}\in\mathbb{C}^{S\times S}$ is upper-triangular.
The channel outputs are then pre-multiplied by $\mathbf{Q}^{\mathsf{H}}$.
This leads to the final channel model on which the K-Best detection algorithm operates:

$$
\bar{\mathbf{y}} = \mathbf{R}\bar{\mathbf{x}} + \bar{\mathbf{n}}
$$

where $\bar{\mathbf{y}}\in\mathbb{C}^S$,
$\bar{\mathbf{x}}\in\mathbb{C}^S$, and $\bar{\mathbf{n}}\in\mathbb{C}^S$
with $\mathbb{E}\left[\bar{\mathbf{n}}\right]=\mathbf{0}$ and
$\mathbb{E}\left[\bar{\mathbf{n}}\bar{\mathbf{n}}^{\mathsf{H}}\right]=\mathbf{I}$.

**LLR Computation**

The K-Best algorithm produces $K$ candidate solutions $\bar{\mathbf{x}}_k\in\mathcal{C}^S$
and their associated distance metrics $d_k=\lVert \bar{\mathbf{y}} - \mathbf{R}\bar{\mathbf{x}}_k \rVert^2$
for $k=1,\dots,K$. If the real-valued channel representation is used, the distance
metrics are scaled by 0.5 to account for the reduced noise power in each complex dimension.
A hard-decision is simply the candidate with the shortest distance.
Various ways to compute LLRs from this list (and possibly
additional side-information) are possible. The (sub-optimal) default solution
is [`List2LLRSimple`](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.List2LLRSimple). Custom solutions can be provided.
Parameters

- **output** (*One of** [**"bit"**, **"symbol"**]**, **str*)  The type of output, either bits or symbols. Whether soft- or
hard-decisions are returned can be configured with the
`hard_out` flag.
- **num_streams** (*tf.int*)  Number of transmitted streams
- **k** (*tf.int*)  The number of paths to keep. Cannot be larger than the
number of constellation points to the power of the number of
streams.
- **constellation_type** (*One of** [**"qam"**, **"pam"**, **"custom"**]**, **str*)  For custom, an instance of [`Constellation`](mapping.html#sionna.mapping.Constellation)
must be provided.
- **num_bits_per_symbol** (*int*)  The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [qam, pam].
- **constellation** ()  An instance of [`Constellation`](mapping.html#sionna.mapping.Constellation) or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (*bool*)  If <cite>True</cite>, the detector computes hard-decided bit values or
constellation point indices instead of soft-values.
Defaults to <cite>False</cite>. The detector cannot compute soft-symbols.
- **use_real_rep** (*bool*)  If <cite>True</cite>, the detector use the real-valued equivalent representation
of the channel. Note that this only works with a QAM constellation.
Defaults to <cite>False</cite>.
- **list2llr** (<cite>None</cite> or instance of [`List2LLR`](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.List2LLR))  The function to be used to compute LLRs from a list of candidate solutions.
If <cite>None</cite>, the default solution [`List2LLRSimple`](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.List2LLRSimple)
is used.
- **dtype** (*One of** [**tf.complex64**, **tf.complex128**] **tf.DType** (**dtype**)*)  The dtype of `y`. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input

- **(y, h, s)**  Tuple:
- **y** (*[,M], tf.complex*)  1+D tensor containing the received signals
- **h** (*[,M,num_streams], tf.complex*)  2+D tensor containing the channel matrices
- **s** (*[,M,M], tf.complex*)  2+D tensor containing the noise covariance matrices


Output

- **One of**
- *[,num_streams,num_bits_per_symbol], tf.float*  LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>bit</cite>
- *[,num_streams,2**num_points], tf.float or [,num_streams], tf.int*  Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>symbol</cite>
Hard-decisions correspond to the symbol indices.


**Note**

If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See [`xla_compat`](config.html#sionna.Config.xla_compat).

### LinearDetector

`class` `sionna.mimo.``LinearDetector`(*`equalizer`*, *`output`*, *`demapping_method`*, *`constellation_type``=``None`*, *`num_bits_per_symbol``=``None`*, *`constellation``=``None`*, *`hard_out``=``False`*, *`dtype``=``tf.complex64`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/mimo/detection.html#LinearDetector)

Convenience class that combines an equalizer,
such as [`lmmse_equalizer()`](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.lmmse_equalizer), and a [`Demapper`](mapping.html#sionna.mapping.Demapper).
Parameters

- **equalizer** (*str**, **one of** [**"lmmse"**, **"zf"**, **"mf"**]**, or **an equalizer function*)  The equalizer to be used. Either one of the existing equalizers
[`lmmse_equalizer()`](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.lmmse_equalizer), [`zf_equalizer()`](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.zf_equalizer), or
[`mf_equalizer()`](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.mf_equalizer) can be used, or a custom equalizer
callable provided that has the same input/output specification.
- **output** (*One of** [**"bit"**, **"symbol"**]**, **str*)  The type of output, either LLRs on bits or logits on constellation symbols.
- **demapping_method** (*One of** [**"app"**, **"maxlog"**]**, **str*)  The demapping method used.
- **constellation_type** (*One of** [**"qam"**, **"pam"**, **"custom"**]**, **str*)  For custom, an instance of [`Constellation`](mapping.html#sionna.mapping.Constellation)
must be provided.
- **num_bits_per_symbol** (*int*)  The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [qam, pam].
- **constellation** ()  An instance of [`Constellation`](mapping.html#sionna.mapping.Constellation) or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (*bool*)  If <cite>True</cite>, the detector computes hard-decided bit values or
constellation point indices instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (*One of** [**tf.complex64**, **tf.complex128**] **tf.DType** (**dtype**)*)  The dtype of `y`. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input

- **(y, h, s)**  Tuple:
- **y** (*[,M], tf.complex*)  1+D tensor containing the received signals
- **h** (*[,M,num_streams], tf.complex*)  2+D tensor containing the channel matrices
- **s** (*[,M,M], tf.complex*)  2+D tensor containing the noise covariance matrices


Output

- **One of**
- *[, num_streams, num_bits_per_symbol], tf.float*  LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>bit</cite>
- *[, num_streams, num_points], tf.float or [, num_streams], tf.int*  Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>symbol</cite>
Hard-decisions correspond to the symbol indices.


**Note**

If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you might need to set `sionna.Config.xla_compat=true`. This depends on the
chosen equalizer function. See [`xla_compat`](config.html#sionna.Config.xla_compat).

### MaximumLikelihoodDetector

`class` `sionna.mimo.``MaximumLikelihoodDetector`(*`output`*, *`demapping_method`*, *`num_streams`*, *`constellation_type``=``None`*, *`num_bits_per_symbol``=``None`*, *`constellation``=``None`*, *`hard_out``=``False`*, *`with_prior``=``False`*, *`dtype``=``tf.complex64`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/mimo/detection.html#MaximumLikelihoodDetector)

MIMO maximum-likelihood (ML) detector.
If the `with_prior` flag is set, prior knowledge on the bits or constellation points is assumed to be available.

This layer implements MIMO maximum-likelihood (ML) detection assuming the
following channel model:

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$

where $\mathbf{y}\in\mathbb{C}^M$ is the received signal vector,
$\mathbf{x}\in\mathcal{C}^K$ is the vector of transmitted symbols which
are uniformly and independently drawn from the constellation $\mathcal{C}$,
$\mathbf{H}\in\mathbb{C}^{M\times K}$ is the known channel matrix,
and $\mathbf{n}\in\mathbb{C}^M$ is a complex Gaussian noise vector.
It is assumed that $\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}$ and
$\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}$,
where $\mathbf{S}$ has full rank.
If the `with_prior` flag is set, it is assumed that prior information of the transmitted signal $\mathbf{x}$ is available,
provided either as LLRs on the bits mapped onto $\mathbf{x}$ or as logits on the individual
constellation points forming $\mathbf{x}$.

Prior to demapping, the received signal is whitened:

$$
\begin{split}\tilde{\mathbf{y}} &= \mathbf{S}^{-\frac{1}{2}}\mathbf{y}\\
&=  \mathbf{S}^{-\frac{1}{2}}\mathbf{H}\mathbf{x} + \mathbf{S}^{-\frac{1}{2}}\mathbf{n}\\
&= \tilde{\mathbf{H}}\mathbf{x} + \tilde{\mathbf{n}}\end{split}
$$

The layer can compute ML detection of symbols or bits with either
soft- or hard-decisions. Note that decisions are computed symbol-/bit-wise
and not jointly for the entire vector $\textbf{x}$ (or the underlying vector
of bits).

**ML detection of bits:**

Soft-decisions on bits are called log-likelihood ratios (LLR).
With the app demapping method, the LLR for the $i\text{th}$ bit
of the $k\text{th}$ user is then computed according to

$$
\begin{split}\begin{align}
    LLR(k,i)&= \ln\left(\frac{\Pr\left(b_{k,i}=1\lvert \mathbf{y},\mathbf{H}\right)}{\Pr\left(b_{k,i}=0\lvert \mathbf{y},\mathbf{H}\right)}\right)\\
            &=\ln\left(\frac{
            \sum_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \exp\left(
                -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                \right) \Pr\left( \mathbf{x} \right)
            }{
            \sum_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \exp\left(
                -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                \right) \Pr\left( \mathbf{x} \right)
            }\right)
\end{align}\end{split}
$$

where $\mathcal{C}_{k,i,1}$ and $\mathcal{C}_{k,i,0}$ are the
sets of vectors of constellation points for which the $i\text{th}$ bit
of the $k\text{th}$ user is equal to 1 and 0, respectively.
$\Pr\left( \mathbf{x} \right)$ is the prior distribution of the vector of
constellation points $\mathbf{x}$. Assuming that the constellation points and
bit levels are independent, it is computed from the prior of the bits according to

$$
\Pr\left( \mathbf{x} \right) = \prod_{k=1}^K \prod_{i=1}^{I} \sigma \left( LLR_p(k,i) \right)
$$

where $LLR_p(k,i)$ is the prior knowledge of the $i\text{th}$ bit of the
$k\text{th}$ user given as an LLR and which is set to $0$ if no prior knowledge is assumed to be available,
and $\sigma\left(\cdot\right)$ is the sigmoid function.
The definition of the LLR has been chosen such that it is equivalent with that of logit. This is
different from many textbooks in communications, where the LLR is
defined as $LLR(k,i) = \ln\left(\frac{\Pr\left(b_{k,i}=0\lvert \mathbf{y},\mathbf{H}\right)}{\Pr\left(b_{k,i}=1\lvert \mathbf{y},\mathbf{H}\right)}\right)$.

With the maxlog demapping method, the LLR for the $i\text{th}$ bit
of the $k\text{th}$ user is approximated like

$$
\begin{split}\begin{align}
    LLR(k,i) \approx&\ln\left(\frac{
        \max_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \left( \exp\left(
            -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
            \right) \Pr\left( \mathbf{x} \right) \right)
        }{
        \max_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \left( \exp\left(
            -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
            \right) \Pr\left( \mathbf{x} \right) \right)
        }\right)\\
        = &\min_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \left( \left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 - \ln \left(\Pr\left( \mathbf{x} \right) \right) \right) -
            \min_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \left( \left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 - \ln \left( \Pr\left( \mathbf{x} \right) \right) \right).
    \end{align}\end{split}
$$

**ML detection of symbols:**

Soft-decisions on symbols are called logits (i.e., unnormalized log-probability).

With the app demapping method, the logit for the
constellation point $c \in \mathcal{C}$ of the $k\text{th}$ user  is computed according to

$$
\begin{align}
    \text{logit}(k,c) &= \ln\left(\sum_{\mathbf{x} : x_k = c} \exp\left(
                -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                \right)\Pr\left( \mathbf{x} \right)\right).
\end{align}
$$

With the maxlog demapping method, the logit for the constellation point $c \in \mathcal{C}$
of the $k\text{th}$ user  is approximated like

$$
\text{logit}(k,c) \approx \max_{\mathbf{x} : x_k = c} \left(
        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 + \ln \left( \Pr\left( \mathbf{x} \right) \right)
        \right).
$$

When hard decisions are requested, this layer returns for the $k$ th stream

$$
\hat{c}_k = \underset{c \in \mathcal{C}}{\text{argmax}} \left( \sum_{\mathbf{x} : x_k = c} \exp\left(
                -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                \right)\Pr\left( \mathbf{x} \right) \right)
$$

where $\mathcal{C}$ is the set of constellation points.
Parameters

- **output** (*One of** [**"bit"**, **"symbol"**]**, **str*)  The type of output, either LLRs on bits or logits on constellation symbols.
- **demapping_method** (*One of** [**"app"**, **"maxlog"**]**, **str*)  The demapping method used.
- **num_streams** (*tf.int*)  Number of transmitted streams
- **constellation_type** (*One of** [**"qam"**, **"pam"**, **"custom"**]**, **str*)  For custom, an instance of [`Constellation`](mapping.html#sionna.mapping.Constellation)
must be provided.
- **num_bits_per_symbol** (*int*)  The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [qam, pam].
- **constellation** ()  An instance of [`Constellation`](mapping.html#sionna.mapping.Constellation) or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (*bool*)  If <cite>True</cite>, the detector computes hard-decided bit values or
constellation point indices instead of soft-values.
Defaults to <cite>False</cite>.
- **with_prior** (*bool*)  If <cite>True</cite>, it is assumed that prior knowledge on the bits or constellation points is available.
This prior information is given as LLRs (for bits) or log-probabilities (for constellation points) as an
additional input to the layer.
Defaults to <cite>False</cite>.
- **dtype** (*One of** [**tf.complex64**, **tf.complex128**] **tf.DType** (**dtype**)*)  The dtype of `y`. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input

- **(y, h, s) or (y, h, prior, s)**  Tuple:
- **y** (*[,M], tf.complex*)  1+D tensor containing the received signals.
- **h** (*[,M,num_streams], tf.complex*)  2+D tensor containing the channel matrices.
- **prior** (*[,num_streams,num_bits_per_symbol] or [,num_streams,num_points], tf.float*)  Prior of the transmitted signals.
If `output` equals bit, then LLRs of the transmitted bits are expected.
If `output` equals symbol, then logits of the transmitted constellation points are expected.
Only required if the `with_prior` flag is set.
- **s** (*[,M,M], tf.complex*)  2+D tensor containing the noise covariance matrices.


Output

- **One of**
- *[, num_streams, num_bits_per_symbol], tf.float*  LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>bit</cite>.
- *[, num_streams, num_points], tf.float or [, num_streams], tf.int*  Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>symbol</cite>.
Hard-decisions correspond to the symbol indices.


**Note**

If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See [`xla_compat`](config.html#sionna.Config.xla_compat).

### MaximumLikelihoodDetectorWithPrior

`class` `sionna.mimo.``MaximumLikelihoodDetectorWithPrior`(*`output`*, *`demapping_method`*, *`num_streams`*, *`constellation_type``=``None`*, *`num_bits_per_symbol``=``None`*, *`constellation``=``None`*, *`hard_out``=``False`*, *`dtype``=``tf.complex64`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/mimo/detection.html#MaximumLikelihoodDetectorWithPrior)

MIMO maximum-likelihood (ML) detector, assuming prior
knowledge on the bits or constellation points is available.

This class is deprecated as the functionality has been integrated
into [`MaximumLikelihoodDetector`](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.MaximumLikelihoodDetector).

This layer implements MIMO maximum-likelihood (ML) detection assuming the
following channel model:

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$

where $\mathbf{y}\in\mathbb{C}^M$ is the received signal vector,
$\mathbf{x}\in\mathcal{C}^K$ is the vector of transmitted symbols which
are uniformly and independently drawn from the constellation $\mathcal{C}$,
$\mathbf{H}\in\mathbb{C}^{M\times K}$ is the known channel matrix,
and $\mathbf{n}\in\mathbb{C}^M$ is a complex Gaussian noise vector.
It is assumed that $\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}$ and
$\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}$,
where $\mathbf{S}$ has full rank.
It is assumed that prior information of the transmitted signal $\mathbf{x}$ is available,
provided either as LLRs on the bits modulated onto $\mathbf{x}$ or as logits on the individual
constellation points forming $\mathbf{x}$.

Prior to demapping, the received signal is whitened:

$$
\begin{split}\tilde{\mathbf{y}} &= \mathbf{S}^{-\frac{1}{2}}\mathbf{y}\\
&=  \mathbf{S}^{-\frac{1}{2}}\mathbf{H}\mathbf{x} + \mathbf{S}^{-\frac{1}{2}}\mathbf{n}\\
&= \tilde{\mathbf{H}}\mathbf{x} + \tilde{\mathbf{n}}\end{split}
$$

The layer can compute ML detection of symbols or bits with either
soft- or hard-decisions. Note that decisions are computed symbol-/bit-wise
and not jointly for the entire vector $\textbf{x}$ (or the underlying vector
of bits).

**ML detection of bits:**

Soft-decisions on bits are called log-likelihood ratios (LLR).
With the app demapping method, the LLR for the $i\text{th}$ bit
of the $k\text{th}$ user is then computed according to

$$
\begin{split}\begin{align}
    LLR(k,i)&= \ln\left(\frac{\Pr\left(b_{k,i}=1\lvert \mathbf{y},\mathbf{H}\right)}{\Pr\left(b_{k,i}=0\lvert \mathbf{y},\mathbf{H}\right)}\right)\\
            &=\ln\left(\frac{
            \sum_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \exp\left(
                -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                \right) \Pr\left( \mathbf{x} \right)
            }{
            \sum_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \exp\left(
                -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                \right) \Pr\left( \mathbf{x} \right)
            }\right)
\end{align}\end{split}
$$

where $\mathcal{C}_{k,i,1}$ and $\mathcal{C}_{k,i,0}$ are the
sets of vectors of constellation points for which the $i\text{th}$ bit
of the $k\text{th}$ user is equal to 1 and 0, respectively.
$\Pr\left( \mathbf{x} \right)$ is the prior distribution of the vector of
constellation points $\mathbf{x}$. Assuming that the constellation points and
bit levels are independent, it is computed from the prior of the bits according to

$$
\Pr\left( \mathbf{x} \right) = \prod_{k=1}^K \prod_{i=1}^{I} \sigma \left( LLR_p(k,i) \right)
$$

where $LLR_p(k,i)$ is the prior knowledge of the $i\text{th}$ bit of the
$k\text{th}$ user given as an LLR, and $\sigma\left(\cdot\right)$ is the sigmoid function.
The definition of the LLR has been chosen such that it is equivalent with that of logit. This is
different from many textbooks in communications, where the LLR is
defined as $LLR(k,i) = \ln\left(\frac{\Pr\left(b_{k,i}=0\lvert \mathbf{y},\mathbf{H}\right)}{\Pr\left(b_{k,i}=1\lvert \mathbf{y},\mathbf{H}\right)}\right)$.

With the maxlog demapping method, the LLR for the $i\text{th}$ bit
of the $k\text{th}$ user is approximated like

$$
\begin{split}\begin{align}
    LLR(k,i) \approx&\ln\left(\frac{
        \max_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \left( \exp\left(
            -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
            \right) \Pr\left( \mathbf{x} \right) \right)
        }{
        \max_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \left( \exp\left(
            -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
            \right) \Pr\left( \mathbf{x} \right) \right)
        }\right)\\
        = &\min_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \left( \left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 - \ln \left(\Pr\left( \mathbf{x} \right) \right) \right) -
            \min_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \left( \left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 - \ln \left( \Pr\left( \mathbf{x} \right) \right) \right).
    \end{align}\end{split}
$$

**ML detection of symbols:**

Soft-decisions on symbols are called logits (i.e., unnormalized log-probability).

With the app demapping method, the logit for the
constellation point $c \in \mathcal{C}$ of the $k\text{th}$ user  is computed according to

$$
\begin{align}
    \text{logit}(k,c) &= \ln\left(\sum_{\mathbf{x} : x_k = c} \exp\left(
                -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                \right)\Pr\left( \mathbf{x} \right)\right).
\end{align}
$$

With the maxlog demapping method, the logit for the constellation point $c \in \mathcal{C}$
of the $k\text{th}$ user  is approximated like

$$
\text{logit}(k,c) \approx \max_{\mathbf{x} : x_k = c} \left(
        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 + \ln \left( \Pr\left( \mathbf{x} \right) \right)
        \right).
$$

When hard decisions are requested, this layer returns for the $k$ th stream

$$
\hat{c}_k = \underset{c \in \mathcal{C}}{\text{argmax}} \left( \sum_{\mathbf{x} : x_k = c} \exp\left(
                -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                \right)\Pr\left( \mathbf{x} \right) \right)
$$

where $\mathcal{C}$ is the set of constellation points.
Parameters

- **output** (*One of** [**"bit"**, **"symbol"**]**, **str*)  The type of output, either LLRs on bits or logits on constellation symbols.
- **demapping_method** (*One of** [**"app"**, **"maxlog"**]**, **str*)  The demapping method used.
- **num_streams** (*tf.int*)  Number of transmitted streams
- **constellation_type** (*One of** [**"qam"**, **"pam"**, **"custom"**]**, **str*)  For custom, an instance of [`Constellation`](mapping.html#sionna.mapping.Constellation)
must be provided.
- **num_bits_per_symbol** (*int*)  The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [qam, pam].
- **constellation** ()  An instance of [`Constellation`](mapping.html#sionna.mapping.Constellation) or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (*bool*)  If <cite>True</cite>, the detector computes hard-decided bit values or
constellation point indices instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (*One of** [**tf.complex64**, **tf.complex128**] **tf.DType** (**dtype**)*)  The dtype of `y`. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input

- **(y, h, prior, s)**  Tuple:
- **y** (*[,M], tf.complex*)  1+D tensor containing the received signals.
- **h** (*[,M,num_streams], tf.complex*)  2+D tensor containing the channel matrices.
- **prior** (*[,num_streams,num_bits_per_symbol] or [,num_streams,num_points], tf.float*)  Prior of the transmitted signals.
If `output` equals bit, then LLRs of the transmitted bits are expected.
If `output` equals symbol, then logits of the transmitted constellation points are expected.
- **s** (*[,M,M], tf.complex*)  2+D tensor containing the noise covariance matrices.


Output

- **One of**
- *[, num_streams, num_bits_per_symbol], tf.float*  LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>bit</cite>.
- *[, num_streams, num_points], tf.float or [, num_streams], tf.int*  Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>symbol</cite>.
Hard-decisions correspond to the symbol indices.


**Note**

If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See [`xla_compat`](config.html#sionna.Config.xla_compat).

### MMSE-PIC

`class` `sionna.mimo.``MMSEPICDetector`(*`output`*, *`demapping_method``=``'maxlog'`*, *`num_iter``=``1`*, *`constellation_type``=``None`*, *`num_bits_per_symbol``=``None`*, *`constellation``=``None`*, *`hard_out``=``False`*, *`dtype``=``tf.complex64`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/mimo/detection.html#MMSEPICDetector)

Minimum mean square error (MMSE) with parallel interference cancellation (PIC) detector

This layer implements the MMSE PIC detector, as proposed in [[CST2011]](https://nvlabs.github.io/sionna/api/mimo.html#cst2011).
For `num_iter`>1, this implementation performs MMSE PIC self-iterations.
MMSE PIC self-iterations can be understood as a concatenation of MMSE PIC
detectors from [[CST2011]](https://nvlabs.github.io/sionna/api/mimo.html#cst2011), which forward intrinsic LLRs to the next
self-iteration.

Compared to [[CST2011]](https://nvlabs.github.io/sionna/api/mimo.html#cst2011), this implementation also accepts priors on the
constellation symbols as an alternative to priors on the bits.

This layer assumes the following channel model:

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$

where $\mathbf{y}\in\mathbb{C}^M$ is the received signal vector,
$\mathbf{x}\in\mathcal{C}^S$ is the vector of transmitted symbols which
are uniformly and independently drawn from the constellation $\mathcal{C}$,
$\mathbf{H}\in\mathbb{C}^{M\times S}$ is the known channel matrix,
and $\mathbf{n}\in\mathbb{C}^M$ is a complex Gaussian noise vector.
It is assumed that $\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}$ and
$\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}$,
where $\mathbf{S}$ has full rank.

The algorithm starts by computing the soft symbols
$\bar{x}_s=\mathbb{E}\left[ x_s \right]$ and
variances $v_s=\mathbb{E}\left[ |e_s|^2\right]$ from the priors,
where $e_s = x_s - \bar{x}_s$, for all $s=1,\dots,S$.

Next, for each stream, the interference caused by all other streams is cancelled
from the observation $\mathbf{y}$, leading to

$$
\hat{\mathbf{y}}_s = \mathbf{y} - \sum_{j\neq s} \mathbf{h}_j x_j = \mathbf{h}_s x_s + \tilde{\mathbf{n}}_s,\quad s=1,\dots,S
$$

where $\tilde{\mathbf{n}}_s=\sum_{j\neq s} \mathbf{h}_j e_j + \mathbf{n}$.

Then, a linear MMSE filter $\mathbf{w}_s$ is computed to reduce the resdiual noise
for each observation $\hat{\mathbf{y}}_s$, which is given as

$$
\mathbf{w}_s = \mathbf{h}_s^{\mathsf{H}}\left( \mathbf{H} \mathbf{D}_s\mathbf{H}^{\mathsf{H}} +\mathbf{S} \right)^{-1}
$$

where $\mathbf{D}_s \in \mathbb{C}^{S\times S}$ is diagonal with entries

$$
\begin{split}\left[\mathbf{D}_s\right]_{i,i} = \begin{cases}
                                    v_i & i\neq s \\
                                    1 & i=s.
                                  \end{cases}\end{split}
$$

The filtered observations

$$
\tilde{z}_s = \mathbf{w}_s^{\mathsf{H}} \hat{\mathbf{y}}_s = \tilde{\mu}_s x_s + \mathbf{w}_s^{\mathsf{H}}\tilde{\mathbf{n}}_s
$$

where $\tilde{\mu}_s=\mathbf{w}_s^{\mathsf{H}} \mathbf{h}_s$, are then demapped to either symbol logits or LLRs, assuming that the remaining noise is Gaussian with variance

$$
\nu_s^2 = \mathop{\text{Var}}\left[\tilde{z}_s\right] = \mathbf{w}_s^{\mathsf{H}} \left(\sum_{j\neq s} \mathbf{h}_j \mathbf{h}_j^{\mathsf{H}} v_j +\mathbf{S} \right)\mathbf{w}_s.
$$

The resulting soft-symbols can then be used for the next self-iteration of the algorithm.

Note that this algorithm can be substantially simplified as described in [[CST2011]](https://nvlabs.github.io/sionna/api/mimo.html#cst2011) to avoid
the computation of different matrix inverses for each stream. This is the version which is
implemented.
Parameters

- **output** (*One of** [**"bit"**, **"symbol"**]**, **str*)  The type of output, either LLRs on bits or logits on constellation
symbols.
- **demapping_method** (*One of** [**"app"**, **"maxlog"**]**, **str*)  The demapping method used.
Defaults to maxlog.
- **num_iter** (*int*)  Number of MMSE PIC iterations.
Defaults to 1.
- **constellation_type** (*One of** [**"qam"**, **"pam"**, **"custom"**]**, **str*)  For custom, an instance of [`Constellation`](mapping.html#sionna.mapping.Constellation)
must be provided.
- **num_bits_per_symbol** (*int*)  The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [qam, pam].
- **constellation** ()  An instance of [`Constellation`](mapping.html#sionna.mapping.Constellation) or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (*bool*)  If <cite>True</cite>, the detector computes hard-decided bit values or
constellation point indices instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (*One of** [**tf.complex64**, **tf.complex128**] **tf.DType** (**dtype**)*)  The dtype of `y`. Defaults to tf.complex64.
The output dtype is the corresponding real dtype
(tf.float32 or tf.float64).


Input

- **(y, h, prior, s)**  Tuple:
- **y** (*[,M], tf.complex*)  1+D tensor containing the received signals
- **h** (*[,M,S], tf.complex*)  2+D tensor containing the channel matrices
- **prior** (*[,S,num_bits_per_symbol] or [,S,num_points], tf.float*)  Prior of the transmitted signals.
If `output` equals bit, then LLRs of the transmitted bits are expected.
If `output` equals symbol, then logits of the transmitted constellation points are expected.
- **s** (*[,M,M], tf.complex*)  2+D tensor containing the noise covariance matrices


Output

- **One of**
- *[,S,num_bits_per_symbol], tf.float*  LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>bit</cite>
- *[,S,2**num_bits_per_symbol], tf.float or [,S], tf.int*  Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>symbol</cite>


**Note**

For numerical stability, we do not recommend to use this function in Graph
mode with XLA, i.e., within a function that is decorated with
`@tf.function(jit_compile=True)`.
However, it is possible to do so by setting
`sionna.Config.xla_compat=true`.
See [`xla_compat`](config.html#sionna.Config.xla_compat).

## Utility Functions

### List2LLR

`class` `sionna.mimo.``List2LLR`[`[source]`](../_modules/sionna/mimo/utils.html#List2LLR)

Abstract class defining a callable to compute LLRs from a list of
candidate vectors (or paths) provided by a MIMO detector.

The following channel model is assumed

$$
\bar{\mathbf{y}} = \mathbf{R}\bar{\mathbf{x}} + \bar{\mathbf{n}}
$$

where $\bar{\mathbf{y}}\in\mathbb{C}^S$ are the channel outputs,
$\mathbf{R}\in\mathbb{C}^{S\times S}$ is an upper-triangular matrix,
$\bar{\mathbf{x}}\in\mathbb{C}^S$ is the transmitted vector whose entries
are uniformly and independently drawn from the constellation $\mathcal{C}$,
and $\bar{\mathbf{n}}\in\mathbb{C}^S$ is white noise
with $\mathbb{E}\left[\bar{\mathbf{n}}\right]=\mathbf{0}$ and
$\mathbb{E}\left[\bar{\mathbf{n}}\bar{\mathbf{n}}^{\mathsf{H}}\right]=\mathbf{I}$.

It is assumed that a MIMO detector such as [`KBestDetector`](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.KBestDetector)
produces $K$ candidate solutions $\bar{\mathbf{x}}_k\in\mathcal{C}^S$
and their associated distance metrics $d_k=\lVert \bar{\mathbf{y}} - \mathbf{R}\bar{\mathbf{x}}_k \rVert^2$
for $k=1,\dots,K$. This layer can also be used with the real-valued representation of the channel.
Input

- **(y, r, dists, path_inds, path_syms)**  Tuple:
- **y** (*[,M], tf.complex or tf.float*)  Channel outputs of the whitened channel
- **r** ([,num_streams, num_streams], same dtype as `y`)  Upper triangular channel matrix of the whitened channel
- **dists** (*[,num_paths], tf.float*)  Distance metric for each path (or candidate)
- **path_inds** (*[,num_paths,num_streams], tf.int32*)  Symbol indices for every stream of every path (or candidate)
- **path_syms** ([,num_path,num_streams], same dtype as `y`)  Constellation symbol for every stream of every path (or candidate)


Output

**llr** (*[num_streams,num_bits_per_symbol], tf.float*)  LLRs for all bits of every stream


**Note**

An implementation of this class does not need to make use of all of
the provided inputs which enable various different implementations.

### List2LLRSimple

`class` `sionna.mimo.``List2LLRSimple`(*`num_bits_per_symbol`*, *`llr_clip_val``=``20.0`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/mimo/utils.html#List2LLRSimple)

Computes LLRs from a list of candidate vectors (or paths) provided by a MIMO detector.

The following channel model is assumed:

$$
\bar{\mathbf{y}} = \mathbf{R}\bar{\mathbf{x}} + \bar{\mathbf{n}}
$$

where $\bar{\mathbf{y}}\in\mathbb{C}^S$ are the channel outputs,
$\mathbf{R}\in\mathbb{C}^{S\times S}$ is an upper-triangular matrix,
$\bar{\mathbf{x}}\in\mathbb{C}^S$ is the transmitted vector whose entries
are uniformly and independently drawn from the constellation $\mathcal{C}$,
and $\bar{\mathbf{n}}\in\mathbb{C}^S$ is white noise
with $\mathbb{E}\left[\bar{\mathbf{n}}\right]=\mathbf{0}$ and
$\mathbb{E}\left[\bar{\mathbf{n}}\bar{\mathbf{n}}^{\mathsf{H}}\right]=\mathbf{I}$.

It is assumed that a MIMO detector such as [`KBestDetector`](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.KBestDetector)
produces $K$ candidate solutions $\bar{\mathbf{x}}_k\in\mathcal{C}^S$
and their associated distance metrics $d_k=\lVert \bar{\mathbf{y}} - \mathbf{R}\bar{\mathbf{x}}_k \rVert^2$
for $k=1,\dots,K$. This layer can also be used with the real-valued representation of the channel.

The LLR for the $i\text{th}$ bit of the $k\text{th}$ stream is computed as

$$
\begin{split}\begin{align}
    LLR(k,i) &= \log\left(\frac{\Pr(b_{k,i}=1|\bar{\mathbf{y}},\mathbf{R})}{\Pr(b_{k,i}=0|\bar{\mathbf{y}},\mathbf{R})}\right)\\
        &\approx \min_{j \in  \mathcal{C}_{k,i,0}}d_j - \min_{j \in  \mathcal{C}_{k,i,1}}d_j
\end{align}\end{split}
$$

where $\mathcal{C}_{k,i,1}$ and $\mathcal{C}_{k,i,0}$ are the set of indices
in the list of candidates for which the $i\text{th}$ bit of the $k\text{th}$
stream is equal to 1 and 0, respectively. The LLRs are clipped to $\pm LLR_\text{clip}$
which can be configured through the parameter `llr_clip_val`.

If $\mathcal{C}_{k,i,0}$ is empty, $LLR(k,i)=LLR_\text{clip}$;
if $\mathcal{C}_{k,i,1}$ is empty, $LLR(k,i)=-LLR_\text{clip}$.
Parameters

- **num_bits_per_symbol** (*int*)  Number of bits per constellation symbol
- **llr_clip_val** (*float*)  The absolute values of LLRs are clipped to this value.
Defaults to 20.0. Can also be a trainable variable.


Input

- **(y, r, dists, path_inds, path_syms)**  Tuple:
- **y** (*[,M], tf.complex or tf.float*)  Channel outputs of the whitened channel
- **r** ([,num_streams, num_streams], same dtype as `y`)  Upper triangular channel matrix of the whitened channel
- **dists** (*[,num_paths], tf.float*)  Distance metric for each path (or candidate)
- **path_inds** (*[,num_paths,num_streams], tf.int32*)  Symbol indices for every stream of every path (or candidate)
- **path_syms** ([,num_path,num_streams], same dtype as `y`)  Constellation symbol for every stream of every path (or candidate)


Output

**llr** (*[num_streams,num_bits_per_symbol], tf.float*)  LLRs for all bits of every stream


### complex2real_vector

`sionna.mimo.``complex2real_vector`(*`z`*)[`[source]`](../_modules/sionna/mimo/utils.html#complex2real_vector)

Transforms a complex-valued vector into its real-valued equivalent.

Transforms the last dimension of a complex-valued tensor into
its real-valued equivalent by stacking the real and imaginary
parts on top of each other.

For a vector $\mathbf{z}\in \mathbb{C}^M$ with real and imaginary
parts $\mathbf{x}\in \mathbb{R}^M$ and
$\mathbf{y}\in \mathbb{R}^M$, respectively, this function returns
the vector $\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in\mathbb{R}^{2M}$.
Input

*[,M], tf.complex*

Output

*[,2M], tf.complex.real_dtype*


### real2complex_vector

`sionna.mimo.``real2complex_vector`(*`z`*)[`[source]`](../_modules/sionna/mimo/utils.html#real2complex_vector)

Transforms a real-valued vector into its complex-valued equivalent.

Transforms the last dimension of a real-valued tensor into
its complex-valued equivalent by interpreting the first half
as the real and the second half as the imaginary part.

For a vector $\mathbf{z}=\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in \mathbb{R}^{2M}$
with $\mathbf{x}\in \mathbb{R}^M$ and $\mathbf{y}\in \mathbb{R}^M$,
this function returns
the vector $\mathbf{x}+j\mathbf{y}\in\mathbb{C}^M$.
Input

*[,2M], tf.float*

Output

*[,M], tf.complex*


### complex2real_matrix

`sionna.mimo.``complex2real_matrix`(*`z`*)[`[source]`](../_modules/sionna/mimo/utils.html#complex2real_matrix)

Transforms a complex-valued matrix into its real-valued equivalent.

Transforms the last two dimensions of a complex-valued tensor into
their real-valued matrix equivalent representation.

For a matrix $\mathbf{Z}\in \mathbb{C}^{M\times K}$ with real and imaginary
parts $\mathbf{X}\in \mathbb{R}^{M\times K}$ and
$\mathbf{Y}\in \mathbb{R}^{M\times K}$, respectively, this function returns
the matrix $\tilde{\mathbf{Z}}\in \mathbb{R}^{2M\times 2K}$, given as

$$
\begin{split}\tilde{\mathbf{Z}} = \begin{pmatrix}
                        \mathbf{X} & -\mathbf{Y}\\
                        \mathbf{Y} & \mathbf{X}
                     \end{pmatrix}.\end{split}
$$

Input

*[,M,K], tf.complex*

Output

*[,2M, 2K], tf.complex.real_dtype*


### real2complex_matrix

`sionna.mimo.``real2complex_matrix`(*`z`*)[`[source]`](../_modules/sionna/mimo/utils.html#real2complex_matrix)

Transforms a real-valued matrix into its complex-valued equivalent.

Transforms the last two dimensions of a real-valued tensor into
their complex-valued matrix equivalent representation.

For a matrix $\tilde{\mathbf{Z}}\in \mathbb{R}^{2M\times 2K}$,
satisfying

$$
\begin{split}\tilde{\mathbf{Z}} = \begin{pmatrix}
                        \mathbf{X} & -\mathbf{Y}\\
                        \mathbf{Y} & \mathbf{X}
                     \end{pmatrix}\end{split}
$$

with $\mathbf{X}\in \mathbb{R}^{M\times K}$ and
$\mathbf{Y}\in \mathbb{R}^{M\times K}$, this function returns
the matrix $\mathbf{Z}=\mathbf{X}+j\mathbf{Y}\in\mathbb{C}^{M\times K}$.
Input

*[,2M,2K], tf.float*

Output

*[,M, 2], tf.complex*


### complex2real_covariance

`sionna.mimo.``complex2real_covariance`(*`r`*)[`[source]`](../_modules/sionna/mimo/utils.html#complex2real_covariance)

Transforms a complex-valued covariance matrix to its real-valued equivalent.

Assume a proper complex random variable $\mathbf{z}\in\mathbb{C}^M$ [[ProperRV]](https://nvlabs.github.io/sionna/api/mimo.html#properrv)
with covariance matrix $\mathbf{R}= \in\mathbb{C}^{M\times M}$
and real and imaginary parts $\mathbf{x}\in \mathbb{R}^M$ and
$\mathbf{y}\in \mathbb{R}^M$, respectively.
This function transforms the given $\mathbf{R}$ into the covariance matrix of the real-valued equivalent
vector $\tilde{\mathbf{z}}=\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in\mathbb{R}^{2M}$, which
is computed as [[CovProperRV]](https://nvlabs.github.io/sionna/api/mimo.html#covproperrv)

$$
\begin{split}\mathbb{E}\left[\tilde{\mathbf{z}}\tilde{\mathbf{z}}^{\mathsf{H}} \right] =
\begin{pmatrix}
    \frac12\Re\{\mathbf{R}\} & -\frac12\Im\{\mathbf{R}\}\\
    \frac12\Im\{\mathbf{R}\} & \frac12\Re\{\mathbf{R}\}
\end{pmatrix}.\end{split}
$$

Input

*[,M,M], tf.complex*

Output

*[,2M, 2M], tf.complex.real_dtype*


### real2complex_covariance

`sionna.mimo.``real2complex_covariance`(*`q`*)[`[source]`](../_modules/sionna/mimo/utils.html#real2complex_covariance)

Transforms a real-valued covariance matrix to its complex-valued equivalent.

Assume a proper complex random variable $\mathbf{z}\in\mathbb{C}^M$ [[ProperRV]](https://nvlabs.github.io/sionna/api/mimo.html#properrv)
with covariance matrix $\mathbf{R}= \in\mathbb{C}^{M\times M}$
and real and imaginary parts $\mathbf{x}\in \mathbb{R}^M$ and
$\mathbf{y}\in \mathbb{R}^M$, respectively.
This function transforms the given covariance matrix of the real-valued equivalent
vector $\tilde{\mathbf{z}}=\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in\mathbb{R}^{2M}$, which
is given as [[CovProperRV]](https://nvlabs.github.io/sionna/api/mimo.html#covproperrv)

$$
\begin{split}\mathbb{E}\left[\tilde{\mathbf{z}}\tilde{\mathbf{z}}^{\mathsf{H}} \right] =
\begin{pmatrix}
    \frac12\Re\{\mathbf{R}\} & -\frac12\Im\{\mathbf{R}\}\\
    \frac12\Im\{\mathbf{R}\} & \frac12\Re\{\mathbf{R}\}
\end{pmatrix},\end{split}
$$

into is complex-valued equivalent $\mathbf{R}$.
Input

*[,2M,2M], tf.float*

Output

*[,M, M], tf.complex*


### complex2real_channel

`sionna.mimo.``complex2real_channel`(*`y`*, *`h`*, *`s`*)[`[source]`](../_modules/sionna/mimo/utils.html#complex2real_channel)

Transforms a complex-valued MIMO channel into its real-valued equivalent.

Assume the canonical MIMO channel model

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$

where $\mathbf{y}\in\mathbb{C}^M$ is the received signal vector,
$\mathbf{x}\in\mathbb{C}^K$ is the vector of transmitted symbols,
$\mathbf{H}\in\mathbb{C}^{M\times K}$ is the known channel matrix,
and $\mathbf{n}\in\mathbb{C}^M$ is a noise vector with covariance
matrix $\mathbf{S}\in\mathbb{C}^{M\times M}$.

This function returns the real-valued equivalent representations of
$\mathbf{y}$, $\mathbf{H}$, and $\mathbf{S}$,
which are used by a wide variety of MIMO detection algorithms (Section VII) [[YH2015]](https://nvlabs.github.io/sionna/api/mimo.html#yh2015).
These are obtained by applying [`complex2real_vector()`](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_vector) to $\mathbf{y}$,
[`complex2real_matrix()`](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_matrix) to $\mathbf{H}$,
and [`complex2real_covariance()`](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_covariance) to $\mathbf{S}$.
Input

- **y** (*[,M], tf.complex*)  1+D tensor containing the received signals.
- **h** (*[,M,K], tf.complex*)  2+D tensor containing the channel matrices.
- **s** (*[,M,M], tf.complex*)  2+D tensor containing the noise covariance matrices.


Output

- *[,2M], tf.complex.real_dtype*  1+D tensor containing the real-valued equivalent received signals.
- *[,2M,2K], tf.complex.real_dtype*  2+D tensor containing the real-valued equivalent channel matrices.
- *[,2M,2M], tf.complex.real_dtype*  2+D tensor containing the real-valued equivalent noise covariance matrices.


### real2complex_channel

`sionna.mimo.``real2complex_channel`(*`y`*, *`h`*, *`s`*)[`[source]`](../_modules/sionna/mimo/utils.html#real2complex_channel)

Transforms a real-valued MIMO channel into its complex-valued equivalent.

Assume the canonical MIMO channel model

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$

where $\mathbf{y}\in\mathbb{C}^M$ is the received signal vector,
$\mathbf{x}\in\mathbb{C}^K$ is the vector of transmitted symbols,
$\mathbf{H}\in\mathbb{C}^{M\times K}$ is the known channel matrix,
and $\mathbf{n}\in\mathbb{C}^M$ is a noise vector with covariance
matrix $\mathbf{S}\in\mathbb{C}^{M\times M}$.

This function transforms the real-valued equivalent representations of
$\mathbf{y}$, $\mathbf{H}$, and $\mathbf{S}$, as, e.g.,
obtained with the function [`complex2real_channel()`](https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_channel),
back to their complex-valued equivalents (Section VII) [[YH2015]](https://nvlabs.github.io/sionna/api/mimo.html#yh2015).
Input

- **y** (*[,2M], tf.float*)  1+D tensor containing the real-valued received signals.
- **h** (*[,2M,2K], tf.float*)  2+D tensor containing the real-valued channel matrices.
- **s** (*[,2M,2M], tf.float*)  2+D tensor containing the real-valued noise covariance matrices.


Output

- *[,M], tf.complex*  1+D tensor containing the complex-valued equivalent received signals.
- *[,M,K], tf.complex*  2+D tensor containing the complex-valued equivalent channel matrices.
- *[,M,M], tf.complex*  2+D tensor containing the complex-valued equivalent noise covariance matrices.


### whiten_channel

`sionna.mimo.``whiten_channel`(*`y`*, *`h`*, *`s`*, *`return_s``=``True`*)[`[source]`](../_modules/sionna/mimo/utils.html#whiten_channel)

Whitens a canonical MIMO channel.

Assume the canonical MIMO channel model

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$

where $\mathbf{y}\in\mathbb{C}^M(\mathbb{R}^M)$ is the received signal vector,
$\mathbf{x}\in\mathbb{C}^K(\mathbb{R}^K)$ is the vector of transmitted symbols,
$\mathbf{H}\in\mathbb{C}^{M\times K}(\mathbb{R}^{M\times K})$ is the known channel matrix,
and $\mathbf{n}\in\mathbb{C}^M(\mathbb{R}^M)$ is a noise vector with covariance
matrix $\mathbf{S}\in\mathbb{C}^{M\times M}(\mathbb{R}^{M\times M})$.

This function whitens this channel by multiplying $\mathbf{y}$ and
$\mathbf{H}$ from the left by $\mathbf{S}^{-\frac{1}{2}}$.
Optionally, the whitened noise covariance matrix $\mathbf{I}_M$
can be returned.
Input

- **y** (*[,M], tf.float or tf.complex*)  1+D tensor containing the received signals.
- **h** (*[,M,K], tf.float or tf.complex*)  2+D tensor containing the  channel matrices.
- **s** (*[,M,M], tf.float or complex*)  2+D tensor containing the noise covariance matrices.
- **return_s** (*bool*)  If <cite>True</cite>, the whitened covariance matrix is returned.
Defaults to <cite>True</cite>.


Output

- *[,M], tf.float or tf.complex*  1+D tensor containing the whitened received signals.
- *[,M,K], tf.float or tf.complex*  2+D tensor containing the whitened channel matrices.
- *[,M,M], tf.float or tf.complex*  2+D tensor containing the whitened noise covariance matrices.
Only returned if `return_s` is <cite>True</cite>.


References:
ProperRV([1](https://nvlabs.github.io/sionna/api/mimo.html#id11),[2](https://nvlabs.github.io/sionna/api/mimo.html#id13))

[Proper complex random variables](https://en.wikipedia.org/wiki/Complex_random_variable#Proper_complex_random_variables),
Wikipedia, accessed 11 September, 2022.

CovProperRV([1](https://nvlabs.github.io/sionna/api/mimo.html#id12),[2](https://nvlabs.github.io/sionna/api/mimo.html#id14))

[Covariance matrices of real and imaginary parts](https://en.wikipedia.org/wiki/Complex_random_vector#Covariance_matrices_of_real_and_imaginary_parts),
Wikipedia, accessed 11 September, 2022.

YH2015([1](https://nvlabs.github.io/sionna/api/mimo.html#id15),[2](https://nvlabs.github.io/sionna/api/mimo.html#id16))

S. Yang and L. Hanzo, [Fifty Years of MIMO Detection: The Road to Large-Scale MIMOs](https://ieeexplore.ieee.org/abstract/document/7244171),
IEEE Communications Surveys & Tutorials, vol. 17, no. 4, pp. 1941-1988, 2015.

[FT2015](https://nvlabs.github.io/sionna/api/mimo.html#id6)

W. Fu and J. S. Thompson, [Performance analysis of K-best detection with adaptive modulation](https://ieeexplore.ieee.org/abstract/document/7454351), IEEE Int. Symp. Wirel. Commun. Sys. (ISWCS), 2015.

[EP2014](https://nvlabs.github.io/sionna/api/mimo.html#id5)

J. Cspedes, P. M. Olmos, M. Snchez-Fernndez, and F. Perez-Cruz,
[Expectation Propagation Detection for High-Order High-Dimensional MIMO Systems](https://ieeexplore.ieee.org/abstract/document/6841617),
IEEE Trans. Commun., vol. 62, no. 8, pp. 2840-2849, Aug. 2014.

CST2011([1](https://nvlabs.github.io/sionna/api/mimo.html#id7),[2](https://nvlabs.github.io/sionna/api/mimo.html#id8),[3](https://nvlabs.github.io/sionna/api/mimo.html#id9),[4](https://nvlabs.github.io/sionna/api/mimo.html#id10))

C. Studer, S. Fateh, and D. Seethaler,
[ASIC Implementation of Soft-Input Soft-Output MIMO Detection Using MMSE Parallel Interference Cancellation](https://ieeexplore.ieee.org/abstract/document/5779722),
IEEE Journal of Solid-State Circuits, vol. 46, no. 7, pp. 17541765, July 2011.



