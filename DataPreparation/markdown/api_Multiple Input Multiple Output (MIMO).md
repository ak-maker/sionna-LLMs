
# Multiple-Input Multiple-Output (MIMO)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#multiple-input-multiple-output-mimo" title="Permalink to this headline"></a>
    
This module provides layers and functions to support simulation of multicell
MIMO transmissions.

## Stream Management<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#stream-management" title="Permalink to this headline"></a>
    
Stream management determines which transmitter is sending which stream to
which receiver. Transmitters and receivers can be user terminals or base
stations, depending on whether uplink or downlink transmissions are considered.
The <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> class has various properties that
are needed to recover desired or interfering channel coefficients for precoding
and equalization. In order to understand how the various properties of
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> can be used, we recommend to have a look
at the source code of the <a class="reference internal" href="ofdm.html#sionna.ofdm.LMMSEEqualizer" title="sionna.ofdm.LMMSEEqualizer">`LMMSEEqualizer`</a> or
<a class="reference internal" href="ofdm.html#sionna.ofdm.ZFPrecoder" title="sionna.ofdm.ZFPrecoder">`ZFPrecoder`</a>.
    
The following code snippet shows how to configure
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> for a simple uplink scenario, where
four transmitters send each one stream to a receiver. Note that
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> is independent of the actual number of
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
<em class="property">`class` </em>`sionna.mimo.``StreamManagement`(<em class="sig-param">`rx_tx_association`</em>, <em class="sig-param">`num_streams_per_tx`</em>)<a class="reference internal" href="../_modules/sionna/mimo/stream_management.html#StreamManagement">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement" title="Permalink to this definition"></a>
    
Class for management of streams in multi-cell MIMO networks.
Parameters
 
- **rx_tx_association** (<em>[</em><em>num_rx</em><em>, </em><em>num_tx</em><em>]</em><em>, </em><em>np.int</em>) – A binary NumPy array where `rx_tx_association[i,j]=1` means
that receiver <cite>i</cite> gets one or multiple streams from
transmitter <cite>j</cite>.
- **num_streams_per_tx** (<em>int</em>) – Indicates the number of streams that are transmitted by each
transmitter.




**Note**
    
Several symmetry constraints on `rx_tx_association` are imposed
to ensure efficient processing. All row sums and all column sums
must be equal, i.e., all receivers have the same number of associated
transmitters and all transmitters have the same number of associated
receivers. It is also assumed that all transmitters send the same
number of streams `num_streams_per_tx`.

<em class="property">`property` </em>`detection_desired_ind`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.detection_desired_ind" title="Permalink to this definition"></a>
    
Indices needed to gather desired channels for receive processing.
    
A NumPy array of shape <cite>[num_rx*num_streams_per_rx]</cite> that
can be used to gather desired channels from the flattened
channel tensor of shape
<cite>[…,num_rx, num_tx, num_streams_per_tx,…]</cite>.
The result of the gather operation can be reshaped to
<cite>[…,num_rx, num_streams_per_rx,…]</cite>.


<em class="property">`property` </em>`detection_undesired_ind`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.detection_undesired_ind" title="Permalink to this definition"></a>
    
Indices needed to gather undesired channels for receive processing.
    
A NumPy array of shape <cite>[num_rx*num_streams_per_rx]</cite> that
can be used to gather undesired channels from the flattened
channel tensor of shape <cite>[…,num_rx, num_tx, num_streams_per_tx,…]</cite>.
The result of the gather operation can be reshaped to
<cite>[…,num_rx, num_interfering_streams_per_rx,…]</cite>.


<em class="property">`property` </em>`num_interfering_streams_per_rx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.num_interfering_streams_per_rx" title="Permalink to this definition"></a>
    
Number of interfering streams received at each eceiver.


<em class="property">`property` </em>`num_rx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.num_rx" title="Permalink to this definition"></a>
    
Number of receivers.


<em class="property">`property` </em>`num_rx_per_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.num_rx_per_tx" title="Permalink to this definition"></a>
    
Number of receivers communicating with a transmitter.


<em class="property">`property` </em>`num_streams_per_rx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.num_streams_per_rx" title="Permalink to this definition"></a>
    
Number of streams transmitted to each receiver.


<em class="property">`property` </em>`num_streams_per_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.num_streams_per_tx" title="Permalink to this definition"></a>
    
Number of streams per transmitter.


<em class="property">`property` </em>`num_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.num_tx" title="Permalink to this definition"></a>
    
Number of transmitters.


<em class="property">`property` </em>`num_tx_per_rx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.num_tx_per_rx" title="Permalink to this definition"></a>
    
Number of transmitters communicating with a receiver.


<em class="property">`property` </em>`precoding_ind`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.precoding_ind" title="Permalink to this definition"></a>
    
Indices needed to gather channels for precoding.
    
A NumPy array of shape <cite>[num_tx, num_rx_per_tx]</cite>,
where `precoding_ind[i,:]` contains the indices of the
receivers to which transmitter <cite>i</cite> is sending streams.


<em class="property">`property` </em>`rx_stream_ids`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.rx_stream_ids" title="Permalink to this definition"></a>
    
Mapping of streams to receivers.
    
A Numpy array of shape <cite>[num_rx, num_streams_per_rx]</cite>.
This array is obtained from `tx_stream_ids` together with
the `rx_tx_association`. `rx_stream_ids[i,:]` contains
the indices of streams that are supposed to be decoded by receiver <cite>i</cite>.


<em class="property">`property` </em>`rx_tx_association`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.rx_tx_association" title="Permalink to this definition"></a>
    
Association between receivers and transmitters.
    
A binary NumPy array of shape <cite>[num_rx, num_tx]</cite>,
where `rx_tx_association[i,j]=1` means that receiver <cite>i</cite>
gets one ore multiple streams from transmitter <cite>j</cite>.


<em class="property">`property` </em>`stream_association`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.stream_association" title="Permalink to this definition"></a>
    
Association between receivers, transmitters, and streams.
    
A binary NumPy array of shape
<cite>[num_rx, num_tx, num_streams_per_tx]</cite>, where
`stream_association[i,j,k]=1` means that receiver <cite>i</cite> gets
the <cite>k</cite> th stream from transmitter <cite>j</cite>.


<em class="property">`property` </em>`stream_ind`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.stream_ind" title="Permalink to this definition"></a>
    
Indices needed to gather received streams in the correct order.
    
A NumPy array of shape <cite>[num_rx*num_streams_per_rx]</cite> that can be
used to gather streams from the flattened tensor of received streams
of shape <cite>[…,num_rx, num_streams_per_rx,…]</cite>. The result of the
gather operation is then reshaped to
<cite>[…,num_tx, num_streams_per_tx,…]</cite>.


<em class="property">`property` </em>`tx_stream_ids`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.tx_stream_ids" title="Permalink to this definition"></a>
    
Mapping of streams to transmitters.
    
A NumPy array of shape <cite>[num_tx, num_streams_per_tx]</cite>.
Streams are numbered from 0,1,… and assiged to transmitters in
increasing order, i.e., transmitter 0 gets the first
<cite>num_streams_per_tx</cite> and so on.


## Precoding<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#precoding" title="Permalink to this headline"></a>

### zero_forcing_precoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#zero-forcing-precoder" title="Permalink to this headline"></a>

`sionna.mimo.``zero_forcing_precoder`(<em class="sig-param">`x`</em>, <em class="sig-param">`h`</em>, <em class="sig-param">`return_precoding_matrix``=``False`</em>)<a class="reference internal" href="../_modules/sionna/mimo/precoding.html#zero_forcing_precoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.zero_forcing_precoder" title="Permalink to this definition"></a>
    
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
    
The precoding matrix $\mathbf{G}$ is defined as (Eq. 4.37) <a class="reference internal" href="channel.wireless.html#bhs2017" id="id1">[BHS2017]</a> :

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
 
- **x** (<em>[…,K], tf.complex</em>) – 1+D tensor containing the symbol vectors to be precoded.
- **h** (<em>[…,K,M], tf.complex</em>) – 2+D tensor containing the channel matrices
- **return_precoding_matrices** (<em>bool</em>) – Indicates if the precoding matrices should be returned or not.
Defaults to False.


Output
 
- **x_precoded** (<em>[…,M], tf.complex</em>) – Tensor of the same shape and dtype as `x` apart from the last
dimensions that has changed from <cite>K</cite> to <cite>M</cite>. It contains the
precoded symbol vectors.
- **g** (<em>[…,M,K], tf.complex</em>) – 2+D tensor containing the precoding matrices. It is only returned
if `return_precoding_matrices=True`.




**Note**
    
If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

## Equalization<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#equalization" title="Permalink to this headline"></a>

### lmmse_equalizer<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#lmmse-equalizer" title="Permalink to this headline"></a>

`sionna.mimo.``lmmse_equalizer`(<em class="sig-param">`y`</em>, <em class="sig-param">`h`</em>, <em class="sig-param">`s`</em>, <em class="sig-param">`whiten_interference``=``True`</em>)<a class="reference internal" href="../_modules/sionna/mimo/equalization.html#lmmse_equalizer">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.lmmse_equalizer" title="Permalink to this definition"></a>
    
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
(Lemma B.19) <a class="reference internal" href="channel.wireless.html#bhs2017" id="id2">[BHS2017]</a> :

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
is important for the <a class="reference internal" href="mapping.html#sionna.mapping.Demapper" title="sionna.mapping.Demapper">`Demapper`</a> although it does
not change the signal-to-noise ratio.
    
The function returns $\hat{\mathbf{x}}$ and
$\boldsymbol{\sigma}^2=\left[\sigma^2_0,\dots, \sigma^2_{K-1}\right]^{\mathsf{T}}$.
Input
 
- **y** (<em>[…,M], tf.complex</em>) – 1+D tensor containing the received signals.
- **h** (<em>[…,M,K], tf.complex</em>) – 2+D tensor containing the channel matrices.
- **s** (<em>[…,M,M], tf.complex</em>) – 2+D tensor containing the noise covariance matrices.
- **whiten_interference** (<em>bool</em>) – If <cite>True</cite> (default), the interference is first whitened before equalization.
In this case, an alternative expression for the receive filter is used that
can be numerically more stable. Defaults to <cite>True</cite>.


Output
 
- **x_hat** (<em>[…,K], tf.complex</em>) – 1+D tensor representing the estimated symbol vectors.
- **no_eff** (<em>tf.float</em>) – Tensor of the same shape as `x_hat` containing the effective noise
variance estimates.




**Note**
    
If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

### mf_equalizer<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#mf-equalizer" title="Permalink to this headline"></a>

`sionna.mimo.``mf_equalizer`(<em class="sig-param">`y`</em>, <em class="sig-param">`h`</em>, <em class="sig-param">`s`</em>)<a class="reference internal" href="../_modules/sionna/mimo/equalization.html#mf_equalizer">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.mf_equalizer" title="Permalink to this definition"></a>
    
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
(Eq. 4.11) <a class="reference internal" href="channel.wireless.html#bhs2017" id="id3">[BHS2017]</a> :

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
is important for the <a class="reference internal" href="mapping.html#sionna.mapping.Demapper" title="sionna.mapping.Demapper">`Demapper`</a> although it does
not change the signal-to-noise ratio.
    
The function returns $\hat{\mathbf{x}}$ and
$\boldsymbol{\sigma}^2=\left[\sigma^2_0,\dots, \sigma^2_{K-1}\right]^{\mathsf{T}}$.
Input
 
- **y** (<em>[…,M], tf.complex</em>) – 1+D tensor containing the received signals.
- **h** (<em>[…,M,K], tf.complex</em>) – 2+D tensor containing the channel matrices.
- **s** (<em>[…,M,M], tf.complex</em>) – 2+D tensor containing the noise covariance matrices.


Output
 
- **x_hat** (<em>[…,K], tf.complex</em>) – 1+D tensor representing the estimated symbol vectors.
- **no_eff** (<em>tf.float</em>) – Tensor of the same shape as `x_hat` containing the effective noise
variance estimates.




### zf_equalizer<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#zf-equalizer" title="Permalink to this headline"></a>

`sionna.mimo.``zf_equalizer`(<em class="sig-param">`y`</em>, <em class="sig-param">`h`</em>, <em class="sig-param">`s`</em>)<a class="reference internal" href="../_modules/sionna/mimo/equalization.html#zf_equalizer">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.zf_equalizer" title="Permalink to this definition"></a>
    
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
(Eq. 4.10) <a class="reference internal" href="channel.wireless.html#bhs2017" id="id4">[BHS2017]</a> :

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
 
- **y** (<em>[…,M], tf.complex</em>) – 1+D tensor containing the received signals.
- **h** (<em>[…,M,K], tf.complex</em>) – 2+D tensor containing the channel matrices.
- **s** (<em>[…,M,M], tf.complex</em>) – 2+D tensor containing the noise covariance matrices.


Output
 
- **x_hat** (<em>[…,K], tf.complex</em>) – 1+D tensor representing the estimated symbol vectors.
- **no_eff** (<em>tf.float</em>) – Tensor of the same shape as `x_hat` containing the effective noise
variance estimates.




**Note**
    
If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

## Detection<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#detection" title="Permalink to this headline"></a>

### EPDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#epdetector" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mimo.``EPDetector`(<em class="sig-param">`output`</em>, <em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`l``=``10`</em>, <em class="sig-param">`beta``=``0.9`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/mimo/detection.html#EPDetector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.EPDetector" title="Permalink to this definition"></a>
    
MIMO Expectation Propagation (EP) detector
    
This layer implements Expectation Propagation (EP) MIMO detection as described
in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#ep2014" id="id5">[EP2014]</a>. It can generate hard- or soft-decisions for symbols or bits.
    
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
    
The channel model is first whitened using <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.whiten_channel" title="sionna.mimo.whiten_channel">`whiten_channel()`</a>
and then converted to its real-valued equivalent,
see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_channel" title="sionna.mimo.complex2real_channel">`complex2real_channel()`</a>, prior to MIMO detection.
    
The computation of LLRs is done by converting the symbol logits
that naturally arise in the algorithm to LLRs using
<a class="reference internal" href="mapping.html#sionna.mapping.PAM2QAM" title="sionna.mapping.PAM2QAM">`PAM2QAM()`</a>. Custom conversions of symbol logits to LLRs
can be implemented by using the soft-symbol output.
Parameters
 
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – The type of output, either bits or symbols. Whether soft- or
hard-decisions are returned can be configured with the
`hard_out` flag.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per QAM constellation symbol, e.g., 4 for QAM16.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the detector computes hard-decided bit values or
constellation point indices instead of soft-values.
Defaults to <cite>False</cite>.
- **l** (<em>int</em>) – Number of iterations. Defaults to 10.
- **beta** (<em>float</em>) – Parameter $\beta\in[0,1]$ for update smoothing.
Defaults to 0.9.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – Precision used for internal computations. Defaults to `tf.complex64`.
Especially for large MIMO setups, the precision can make a significant
performance difference.


Input
 
- **(y, h, s)** – Tuple:
- **y** (<em>[…,M], tf.complex</em>) – 1+D tensor containing the received signals
- **h** (<em>[…,M,num_streams], tf.complex</em>) – 2+D tensor containing the channel matrices
- **s** (<em>[…,M,M], tf.complex</em>) – 2+D tensor containing the noise covariance matrices


Output
 
- **One of**
- <em>[…,num_streams,num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>
- <em>[…,num_streams,2**num_bits_per_symbol], tf.float or […,num_streams], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>




**Note**
    
For numerical stability, we do not recommend to use this function in Graph
mode with XLA, i.e., within a function that is decorated with
`@tf.function(jit_compile=True)`.
However, it is possible to do so by setting
`sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

### KBestDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#kbestdetector" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mimo.``KBestDetector`(<em class="sig-param">`output`</em>, <em class="sig-param">`num_streams`</em>, <em class="sig-param">`k`</em>, <em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`use_real_rep``=``False`</em>, <em class="sig-param">`list2llr``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/mimo/detection.html#KBestDetector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.KBestDetector" title="Permalink to this definition"></a>
    
MIMO K-Best detector
    
This layer implements K-Best MIMO detection as described
in (Eq. 4-5) <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#ft2015" id="id6">[FT2015]</a>. It can either generate hard decisions (for symbols
or bits) or compute LLRs.
    
The algorithm operates in either the complex or real-valued domain.
Although both options produce identical results, the former has the advantage
that it can be applied to arbitrary non-QAM constellations. It also reduces
the number of streams (or depth) by a factor of two.
    
The way soft-outputs (i.e., LLRs) are computed is determined by the
`list2llr` function. The default solution
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.List2LLRSimple" title="sionna.mimo.List2LLRSimple">`List2LLRSimple`</a> assigns a predetermined
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
see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_channel" title="sionna.mimo.complex2real_channel">`complex2real_channel()`</a>. We assume in the sequel the complex-valued
representation. Then, the channel is whitened using <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.whiten_channel" title="sionna.mimo.whiten_channel">`whiten_channel()`</a>:

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
is <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.List2LLRSimple" title="sionna.mimo.List2LLRSimple">`List2LLRSimple`</a>. Custom solutions can be provided.
Parameters
 
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – The type of output, either bits or symbols. Whether soft- or
hard-decisions are returned can be configured with the
`hard_out` flag.
- **num_streams** (<em>tf.int</em>) – Number of transmitted streams
- **k** (<em>tf.int</em>) – The number of paths to keep. Cannot be larger than the
number of constellation points to the power of the number of
streams.
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – An instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the detector computes hard-decided bit values or
constellation point indices instead of soft-values.
Defaults to <cite>False</cite>. The detector cannot compute soft-symbols.
- **use_real_rep** (<em>bool</em>) – If <cite>True</cite>, the detector use the real-valued equivalent representation
of the channel. Note that this only works with a QAM constellation.
Defaults to <cite>False</cite>.
- **list2llr** (<cite>None</cite> or instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.List2LLR" title="sionna.mimo.List2LLR">`List2LLR`</a>) – The function to be used to compute LLRs from a list of candidate solutions.
If <cite>None</cite>, the default solution <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.List2LLRSimple" title="sionna.mimo.List2LLRSimple">`List2LLRSimple`</a>
is used.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of `y`. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y, h, s)** – Tuple:
- **y** (<em>[…,M], tf.complex</em>) – 1+D tensor containing the received signals
- **h** (<em>[…,M,num_streams], tf.complex</em>) – 2+D tensor containing the channel matrices
- **s** (<em>[…,M,M], tf.complex</em>) – 2+D tensor containing the noise covariance matrices


Output
 
- **One of**
- <em>[…,num_streams,num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>
- <em>[…,num_streams,2**num_points], tf.float or […,num_streams], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>
Hard-decisions correspond to the symbol indices.




**Note**
    
If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

### LinearDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#lineardetector" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mimo.``LinearDetector`(<em class="sig-param">`equalizer`</em>, <em class="sig-param">`output`</em>, <em class="sig-param">`demapping_method`</em>, <em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mimo/detection.html#LinearDetector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.LinearDetector" title="Permalink to this definition"></a>
    
Convenience class that combines an equalizer,
such as <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.lmmse_equalizer" title="sionna.mimo.lmmse_equalizer">`lmmse_equalizer()`</a>, and a <a class="reference internal" href="mapping.html#sionna.mapping.Demapper" title="sionna.mapping.Demapper">`Demapper`</a>.
Parameters
 
- **equalizer** (<em>str</em><em>, </em><em>one of</em><em> [</em><em>"lmmse"</em><em>, </em><em>"zf"</em><em>, </em><em>"mf"</em><em>]</em><em>, or </em><em>an equalizer function</em>) – The equalizer to be used. Either one of the existing equalizers
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.lmmse_equalizer" title="sionna.mimo.lmmse_equalizer">`lmmse_equalizer()`</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.zf_equalizer" title="sionna.mimo.zf_equalizer">`zf_equalizer()`</a>, or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.mf_equalizer" title="sionna.mimo.mf_equalizer">`mf_equalizer()`</a> can be used, or a custom equalizer
callable provided that has the same input/output specification.
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – The type of output, either LLRs on bits or logits on constellation symbols.
- **demapping_method** (<em>One of</em><em> [</em><em>"app"</em><em>, </em><em>"maxlog"</em><em>]</em><em>, </em><em>str</em>) – The demapping method used.
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – An instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the detector computes hard-decided bit values or
constellation point indices instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of `y`. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y, h, s)** – Tuple:
- **y** (<em>[…,M], tf.complex</em>) – 1+D tensor containing the received signals
- **h** (<em>[…,M,num_streams], tf.complex</em>) – 2+D tensor containing the channel matrices
- **s** (<em>[…,M,M], tf.complex</em>) – 2+D tensor containing the noise covariance matrices


Output
 
- **One of**
- <em>[…, num_streams, num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>
- <em>[…, num_streams, num_points], tf.float or […, num_streams], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>
Hard-decisions correspond to the symbol indices.




**Note**
    
If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you might need to set `sionna.Config.xla_compat=true`. This depends on the
chosen equalizer function. See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

### MaximumLikelihoodDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#maximumlikelihooddetector" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mimo.``MaximumLikelihoodDetector`(<em class="sig-param">`output`</em>, <em class="sig-param">`demapping_method`</em>, <em class="sig-param">`num_streams`</em>, <em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`with_prior``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mimo/detection.html#MaximumLikelihoodDetector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.MaximumLikelihoodDetector" title="Permalink to this definition"></a>
    
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
With the “app” demapping method, the LLR for the $i\text{th}$ bit
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
    
With the “maxlog” demapping method, the LLR for the $i\text{th}$ bit
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
    
With the “app” demapping method, the logit for the
constellation point $c \in \mathcal{C}$ of the $k\text{th}$ user  is computed according to

$$
\begin{align}
    \text{logit}(k,c) &= \ln\left(\sum_{\mathbf{x} : x_k = c} \exp\left(
                -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                \right)\Pr\left( \mathbf{x} \right)\right).
\end{align}
$$
    
With the “maxlog” demapping method, the logit for the constellation point $c \in \mathcal{C}$
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
 
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – The type of output, either LLRs on bits or logits on constellation symbols.
- **demapping_method** (<em>One of</em><em> [</em><em>"app"</em><em>, </em><em>"maxlog"</em><em>]</em><em>, </em><em>str</em>) – The demapping method used.
- **num_streams** (<em>tf.int</em>) – Number of transmitted streams
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – An instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the detector computes hard-decided bit values or
constellation point indices instead of soft-values.
Defaults to <cite>False</cite>.
- **with_prior** (<em>bool</em>) – If <cite>True</cite>, it is assumed that prior knowledge on the bits or constellation points is available.
This prior information is given as LLRs (for bits) or log-probabilities (for constellation points) as an
additional input to the layer.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of `y`. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y, h, s) or (y, h, prior, s)** – Tuple:
- **y** (<em>[…,M], tf.complex</em>) – 1+D tensor containing the received signals.
- **h** (<em>[…,M,num_streams], tf.complex</em>) – 2+D tensor containing the channel matrices.
- **prior** (<em>[…,num_streams,num_bits_per_symbol] or […,num_streams,num_points], tf.float</em>) – Prior of the transmitted signals.
If `output` equals “bit”, then LLRs of the transmitted bits are expected.
If `output` equals “symbol”, then logits of the transmitted constellation points are expected.
Only required if the `with_prior` flag is set.
- **s** (<em>[…,M,M], tf.complex</em>) – 2+D tensor containing the noise covariance matrices.


Output
 
- **One of**
- <em>[…, num_streams, num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>.
- <em>[…, num_streams, num_points], tf.float or […, num_streams], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>.
Hard-decisions correspond to the symbol indices.




**Note**
    
If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

### MaximumLikelihoodDetectorWithPrior<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#maximumlikelihooddetectorwithprior" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mimo.``MaximumLikelihoodDetectorWithPrior`(<em class="sig-param">`output`</em>, <em class="sig-param">`demapping_method`</em>, <em class="sig-param">`num_streams`</em>, <em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mimo/detection.html#MaximumLikelihoodDetectorWithPrior">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.MaximumLikelihoodDetectorWithPrior" title="Permalink to this definition"></a>
    
MIMO maximum-likelihood (ML) detector, assuming prior
knowledge on the bits or constellation points is available.
    
This class is deprecated as the functionality has been integrated
into <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.MaximumLikelihoodDetector" title="sionna.mimo.MaximumLikelihoodDetector">`MaximumLikelihoodDetector`</a>.
    
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
With the “app” demapping method, the LLR for the $i\text{th}$ bit
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
    
With the “maxlog” demapping method, the LLR for the $i\text{th}$ bit
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
    
With the “app” demapping method, the logit for the
constellation point $c \in \mathcal{C}$ of the $k\text{th}$ user  is computed according to

$$
\begin{align}
    \text{logit}(k,c) &= \ln\left(\sum_{\mathbf{x} : x_k = c} \exp\left(
                -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                \right)\Pr\left( \mathbf{x} \right)\right).
\end{align}
$$
    
With the “maxlog” demapping method, the logit for the constellation point $c \in \mathcal{C}$
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
 
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – The type of output, either LLRs on bits or logits on constellation symbols.
- **demapping_method** (<em>One of</em><em> [</em><em>"app"</em><em>, </em><em>"maxlog"</em><em>]</em><em>, </em><em>str</em>) – The demapping method used.
- **num_streams** (<em>tf.int</em>) – Number of transmitted streams
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – An instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the detector computes hard-decided bit values or
constellation point indices instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of `y`. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y, h, prior, s)** – Tuple:
- **y** (<em>[…,M], tf.complex</em>) – 1+D tensor containing the received signals.
- **h** (<em>[…,M,num_streams], tf.complex</em>) – 2+D tensor containing the channel matrices.
- **prior** (<em>[…,num_streams,num_bits_per_symbol] or […,num_streams,num_points], tf.float</em>) – Prior of the transmitted signals.
If `output` equals “bit”, then LLRs of the transmitted bits are expected.
If `output` equals “symbol”, then logits of the transmitted constellation points are expected.
- **s** (<em>[…,M,M], tf.complex</em>) – 2+D tensor containing the noise covariance matrices.


Output
 
- **One of**
- <em>[…, num_streams, num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>.
- <em>[…, num_streams, num_points], tf.float or […, num_streams], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>.
Hard-decisions correspond to the symbol indices.




**Note**
    
If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

### MMSE-PIC<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#mmse-pic" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mimo.``MMSEPICDetector`(<em class="sig-param">`output`</em>, <em class="sig-param">`demapping_method``=``'maxlog'`</em>, <em class="sig-param">`num_iter``=``1`</em>, <em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mimo/detection.html#MMSEPICDetector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.MMSEPICDetector" title="Permalink to this definition"></a>
    
Minimum mean square error (MMSE) with parallel interference cancellation (PIC) detector
    
This layer implements the MMSE PIC detector, as proposed in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#cst2011" id="id7">[CST2011]</a>.
For `num_iter`>1, this implementation performs MMSE PIC self-iterations.
MMSE PIC self-iterations can be understood as a concatenation of MMSE PIC
detectors from <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#cst2011" id="id8">[CST2011]</a>, which forward intrinsic LLRs to the next
self-iteration.
    
Compared to <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#cst2011" id="id9">[CST2011]</a>, this implementation also accepts priors on the
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
    
Note that this algorithm can be substantially simplified as described in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#cst2011" id="id10">[CST2011]</a> to avoid
the computation of different matrix inverses for each stream. This is the version which is
implemented.
Parameters
 
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – The type of output, either LLRs on bits or logits on constellation
symbols.
- **demapping_method** (<em>One of</em><em> [</em><em>"app"</em><em>, </em><em>"maxlog"</em><em>]</em><em>, </em><em>str</em>) – The demapping method used.
Defaults to “maxlog”.
- **num_iter** (<em>int</em>) – Number of MMSE PIC iterations.
Defaults to 1.
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – An instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the detector computes hard-decided bit values or
constellation point indices instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of `y`. Defaults to tf.complex64.
The output dtype is the corresponding real dtype
(tf.float32 or tf.float64).


Input
 
- **(y, h, prior, s)** – Tuple:
- **y** (<em>[…,M], tf.complex</em>) – 1+D tensor containing the received signals
- **h** (<em>[…,M,S], tf.complex</em>) – 2+D tensor containing the channel matrices
- **prior** (<em>[…,S,num_bits_per_symbol] or […,S,num_points], tf.float</em>) – Prior of the transmitted signals.
If `output` equals “bit”, then LLRs of the transmitted bits are expected.
If `output` equals “symbol”, then logits of the transmitted constellation points are expected.
- **s** (<em>[…,M,M], tf.complex</em>) – 2+D tensor containing the noise covariance matrices


Output
 
- **One of**
- <em>[…,S,num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>
- <em>[…,S,2**num_bits_per_symbol], tf.float or […,S], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>




**Note**
    
For numerical stability, we do not recommend to use this function in Graph
mode with XLA, i.e., within a function that is decorated with
`@tf.function(jit_compile=True)`.
However, it is possible to do so by setting
`sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

## Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#utility-functions" title="Permalink to this headline"></a>

### List2LLR<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#list2llr" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mimo.``List2LLR`<a class="reference internal" href="../_modules/sionna/mimo/utils.html#List2LLR">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.List2LLR" title="Permalink to this definition"></a>
    
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
    
It is assumed that a MIMO detector such as <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.KBestDetector" title="sionna.mimo.KBestDetector">`KBestDetector`</a>
produces $K$ candidate solutions $\bar{\mathbf{x}}_k\in\mathcal{C}^S$
and their associated distance metrics $d_k=\lVert \bar{\mathbf{y}} - \mathbf{R}\bar{\mathbf{x}}_k \rVert^2$
for $k=1,\dots,K$. This layer can also be used with the real-valued representation of the channel.
Input
 
- **(y, r, dists, path_inds, path_syms)** – Tuple:
- **y** (<em>[…,M], tf.complex or tf.float</em>) – Channel outputs of the whitened channel
- **r** ([…,num_streams, num_streams], same dtype as `y`) – Upper triangular channel matrix of the whitened channel
- **dists** (<em>[…,num_paths], tf.float</em>) – Distance metric for each path (or candidate)
- **path_inds** (<em>[…,num_paths,num_streams], tf.int32</em>) – Symbol indices for every stream of every path (or candidate)
- **path_syms** ([…,num_path,num_streams], same dtype as `y`) – Constellation symbol for every stream of every path (or candidate)


Output
    
**llr** (<em>[…num_streams,num_bits_per_symbol], tf.float</em>) – LLRs for all bits of every stream



**Note**
    
An implementation of this class does not need to make use of all of
the provided inputs which enable various different implementations.

### List2LLRSimple<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#list2llrsimple" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mimo.``List2LLRSimple`(<em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`llr_clip_val``=``20.0`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mimo/utils.html#List2LLRSimple">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.List2LLRSimple" title="Permalink to this definition"></a>
    
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
    
It is assumed that a MIMO detector such as <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.KBestDetector" title="sionna.mimo.KBestDetector">`KBestDetector`</a>
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
 
- **num_bits_per_symbol** (<em>int</em>) – Number of bits per constellation symbol
- **llr_clip_val** (<em>float</em>) – The absolute values of LLRs are clipped to this value.
Defaults to 20.0. Can also be a trainable variable.


Input
 
- **(y, r, dists, path_inds, path_syms)** – Tuple:
- **y** (<em>[…,M], tf.complex or tf.float</em>) – Channel outputs of the whitened channel
- **r** ([…,num_streams, num_streams], same dtype as `y`) – Upper triangular channel matrix of the whitened channel
- **dists** (<em>[…,num_paths], tf.float</em>) – Distance metric for each path (or candidate)
- **path_inds** (<em>[…,num_paths,num_streams], tf.int32</em>) – Symbol indices for every stream of every path (or candidate)
- **path_syms** ([…,num_path,num_streams], same dtype as `y`) – Constellation symbol for every stream of every path (or candidate)


Output
    
**llr** (<em>[…num_streams,num_bits_per_symbol], tf.float</em>) – LLRs for all bits of every stream



### complex2real_vector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#complex2real-vector" title="Permalink to this headline"></a>

`sionna.mimo.``complex2real_vector`(<em class="sig-param">`z`</em>)<a class="reference internal" href="../_modules/sionna/mimo/utils.html#complex2real_vector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_vector" title="Permalink to this definition"></a>
    
Transforms a complex-valued vector into its real-valued equivalent.
    
Transforms the last dimension of a complex-valued tensor into
its real-valued equivalent by stacking the real and imaginary
parts on top of each other.
    
For a vector $\mathbf{z}\in \mathbb{C}^M$ with real and imaginary
parts $\mathbf{x}\in \mathbb{R}^M$ and
$\mathbf{y}\in \mathbb{R}^M$, respectively, this function returns
the vector $\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in\mathbb{R}^{2M}$.
Input
    
<em>[…,M], tf.complex</em>

Output
    
<em>[…,2M], tf.complex.real_dtype</em>



### real2complex_vector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#real2complex-vector" title="Permalink to this headline"></a>

`sionna.mimo.``real2complex_vector`(<em class="sig-param">`z`</em>)<a class="reference internal" href="../_modules/sionna/mimo/utils.html#real2complex_vector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.real2complex_vector" title="Permalink to this definition"></a>
    
Transforms a real-valued vector into its complex-valued equivalent.
    
Transforms the last dimension of a real-valued tensor into
its complex-valued equivalent by interpreting the first half
as the real and the second half as the imaginary part.
    
For a vector $\mathbf{z}=\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in \mathbb{R}^{2M}$
with $\mathbf{x}\in \mathbb{R}^M$ and $\mathbf{y}\in \mathbb{R}^M$,
this function returns
the vector $\mathbf{x}+j\mathbf{y}\in\mathbb{C}^M$.
Input
    
<em>[…,2M], tf.float</em>

Output
    
<em>[…,M], tf.complex</em>



### complex2real_matrix<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#complex2real-matrix" title="Permalink to this headline"></a>

`sionna.mimo.``complex2real_matrix`(<em class="sig-param">`z`</em>)<a class="reference internal" href="../_modules/sionna/mimo/utils.html#complex2real_matrix">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_matrix" title="Permalink to this definition"></a>
    
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
    
<em>[…,M,K], tf.complex</em>

Output
    
<em>[…,2M, 2K], tf.complex.real_dtype</em>



### real2complex_matrix<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#real2complex-matrix" title="Permalink to this headline"></a>

`sionna.mimo.``real2complex_matrix`(<em class="sig-param">`z`</em>)<a class="reference internal" href="../_modules/sionna/mimo/utils.html#real2complex_matrix">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.real2complex_matrix" title="Permalink to this definition"></a>
    
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
    
<em>[…,2M,2K], tf.float</em>

Output
    
<em>[…,M, 2], tf.complex</em>



### complex2real_covariance<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#complex2real-covariance" title="Permalink to this headline"></a>

`sionna.mimo.``complex2real_covariance`(<em class="sig-param">`r`</em>)<a class="reference internal" href="../_modules/sionna/mimo/utils.html#complex2real_covariance">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_covariance" title="Permalink to this definition"></a>
    
Transforms a complex-valued covariance matrix to its real-valued equivalent.
    
Assume a proper complex random variable $\mathbf{z}\in\mathbb{C}^M$ <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#properrv" id="id11">[ProperRV]</a>
with covariance matrix $\mathbf{R}= \in\mathbb{C}^{M\times M}$
and real and imaginary parts $\mathbf{x}\in \mathbb{R}^M$ and
$\mathbf{y}\in \mathbb{R}^M$, respectively.
This function transforms the given $\mathbf{R}$ into the covariance matrix of the real-valued equivalent
vector $\tilde{\mathbf{z}}=\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in\mathbb{R}^{2M}$, which
is computed as <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#covproperrv" id="id12">[CovProperRV]</a>

$$
\begin{split}\mathbb{E}\left[\tilde{\mathbf{z}}\tilde{\mathbf{z}}^{\mathsf{H}} \right] =
\begin{pmatrix}
    \frac12\Re\{\mathbf{R}\} & -\frac12\Im\{\mathbf{R}\}\\
    \frac12\Im\{\mathbf{R}\} & \frac12\Re\{\mathbf{R}\}
\end{pmatrix}.\end{split}
$$

Input
    
<em>[…,M,M], tf.complex</em>

Output
    
<em>[…,2M, 2M], tf.complex.real_dtype</em>



### real2complex_covariance<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#real2complex-covariance" title="Permalink to this headline"></a>

`sionna.mimo.``real2complex_covariance`(<em class="sig-param">`q`</em>)<a class="reference internal" href="../_modules/sionna/mimo/utils.html#real2complex_covariance">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.real2complex_covariance" title="Permalink to this definition"></a>
    
Transforms a real-valued covariance matrix to its complex-valued equivalent.
    
Assume a proper complex random variable $\mathbf{z}\in\mathbb{C}^M$ <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#properrv" id="id13">[ProperRV]</a>
with covariance matrix $\mathbf{R}= \in\mathbb{C}^{M\times M}$
and real and imaginary parts $\mathbf{x}\in \mathbb{R}^M$ and
$\mathbf{y}\in \mathbb{R}^M$, respectively.
This function transforms the given covariance matrix of the real-valued equivalent
vector $\tilde{\mathbf{z}}=\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in\mathbb{R}^{2M}$, which
is given as <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#covproperrv" id="id14">[CovProperRV]</a>

$$
\begin{split}\mathbb{E}\left[\tilde{\mathbf{z}}\tilde{\mathbf{z}}^{\mathsf{H}} \right] =
\begin{pmatrix}
    \frac12\Re\{\mathbf{R}\} & -\frac12\Im\{\mathbf{R}\}\\
    \frac12\Im\{\mathbf{R}\} & \frac12\Re\{\mathbf{R}\}
\end{pmatrix},\end{split}
$$
    
into is complex-valued equivalent $\mathbf{R}$.
Input
    
<em>[…,2M,2M], tf.float</em>

Output
    
<em>[…,M, M], tf.complex</em>



### complex2real_channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#complex2real-channel" title="Permalink to this headline"></a>

`sionna.mimo.``complex2real_channel`(<em class="sig-param">`y`</em>, <em class="sig-param">`h`</em>, <em class="sig-param">`s`</em>)<a class="reference internal" href="../_modules/sionna/mimo/utils.html#complex2real_channel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_channel" title="Permalink to this definition"></a>
    
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
which are used by a wide variety of MIMO detection algorithms (Section VII) <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#yh2015" id="id15">[YH2015]</a>.
These are obtained by applying <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_vector" title="sionna.mimo.complex2real_vector">`complex2real_vector()`</a> to $\mathbf{y}$,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_matrix" title="sionna.mimo.complex2real_matrix">`complex2real_matrix()`</a> to $\mathbf{H}$,
and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_covariance" title="sionna.mimo.complex2real_covariance">`complex2real_covariance()`</a> to $\mathbf{S}$.
Input
 
- **y** (<em>[…,M], tf.complex</em>) – 1+D tensor containing the received signals.
- **h** (<em>[…,M,K], tf.complex</em>) – 2+D tensor containing the channel matrices.
- **s** (<em>[…,M,M], tf.complex</em>) – 2+D tensor containing the noise covariance matrices.


Output
 
- <em>[…,2M], tf.complex.real_dtype</em> – 1+D tensor containing the real-valued equivalent received signals.
- <em>[…,2M,2K], tf.complex.real_dtype</em> – 2+D tensor containing the real-valued equivalent channel matrices.
- <em>[…,2M,2M], tf.complex.real_dtype</em> – 2+D tensor containing the real-valued equivalent noise covariance matrices.




### real2complex_channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#real2complex-channel" title="Permalink to this headline"></a>

`sionna.mimo.``real2complex_channel`(<em class="sig-param">`y`</em>, <em class="sig-param">`h`</em>, <em class="sig-param">`s`</em>)<a class="reference internal" href="../_modules/sionna/mimo/utils.html#real2complex_channel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.real2complex_channel" title="Permalink to this definition"></a>
    
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
obtained with the function <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_channel" title="sionna.mimo.complex2real_channel">`complex2real_channel()`</a>,
back to their complex-valued equivalents (Section VII) <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#yh2015" id="id16">[YH2015]</a>.
Input
 
- **y** (<em>[…,2M], tf.float</em>) – 1+D tensor containing the real-valued received signals.
- **h** (<em>[…,2M,2K], tf.float</em>) – 2+D tensor containing the real-valued channel matrices.
- **s** (<em>[…,2M,2M], tf.float</em>) – 2+D tensor containing the real-valued noise covariance matrices.


Output
 
- <em>[…,M], tf.complex</em> – 1+D tensor containing the complex-valued equivalent received signals.
- <em>[…,M,K], tf.complex</em> – 2+D tensor containing the complex-valued equivalent channel matrices.
- <em>[…,M,M], tf.complex</em> – 2+D tensor containing the complex-valued equivalent noise covariance matrices.




### whiten_channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#whiten-channel" title="Permalink to this headline"></a>

`sionna.mimo.``whiten_channel`(<em class="sig-param">`y`</em>, <em class="sig-param">`h`</em>, <em class="sig-param">`s`</em>, <em class="sig-param">`return_s``=``True`</em>)<a class="reference internal" href="../_modules/sionna/mimo/utils.html#whiten_channel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.whiten_channel" title="Permalink to this definition"></a>
    
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
 
- **y** (<em>[…,M], tf.float or tf.complex</em>) – 1+D tensor containing the received signals.
- **h** (<em>[…,M,K], tf.float or tf.complex</em>) – 2+D tensor containing the  channel matrices.
- **s** (<em>[…,M,M], tf.float or complex</em>) – 2+D tensor containing the noise covariance matrices.
- **return_s** (<em>bool</em>) – If <cite>True</cite>, the whitened covariance matrix is returned.
Defaults to <cite>True</cite>.


Output
 
- <em>[…,M], tf.float or tf.complex</em> – 1+D tensor containing the whitened received signals.
- <em>[…,M,K], tf.float or tf.complex</em> – 2+D tensor containing the whitened channel matrices.
- <em>[…,M,M], tf.float or tf.complex</em> – 2+D tensor containing the whitened noise covariance matrices.
Only returned if `return_s` is <cite>True</cite>.





References:
ProperRV(<a href="https://nvlabs.github.io/sionna/api/mimo.html#id11">1</a>,<a href="https://nvlabs.github.io/sionna/api/mimo.html#id13">2</a>)
    
<a class="reference external" href="https://en.wikipedia.org/wiki/Complex_random_variable#Proper_complex_random_variables">Proper complex random variables</a>,
Wikipedia, accessed 11 September, 2022.

CovProperRV(<a href="https://nvlabs.github.io/sionna/api/mimo.html#id12">1</a>,<a href="https://nvlabs.github.io/sionna/api/mimo.html#id14">2</a>)
    
<a class="reference external" href="https://en.wikipedia.org/wiki/Complex_random_vector#Covariance_matrices_of_real_and_imaginary_parts">Covariance matrices of real and imaginary parts</a>,
Wikipedia, accessed 11 September, 2022.

YH2015(<a href="https://nvlabs.github.io/sionna/api/mimo.html#id15">1</a>,<a href="https://nvlabs.github.io/sionna/api/mimo.html#id16">2</a>)
    
S. Yang and L. Hanzo, <a class="reference external" href="https://ieeexplore.ieee.org/abstract/document/7244171">“Fifty Years of MIMO Detection: The Road to Large-Scale MIMOs”</a>,
IEEE Communications Surveys & Tutorials, vol. 17, no. 4, pp. 1941-1988, 2015.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/mimo.html#id6">FT2015</a>
    
W. Fu and J. S. Thompson, <a class="reference external" href="https://ieeexplore.ieee.org/abstract/document/7454351">“Performance analysis of K-best detection with adaptive modulation”</a>, IEEE Int. Symp. Wirel. Commun. Sys. (ISWCS), 2015.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/mimo.html#id5">EP2014</a>
    
J. Céspedes, P. M. Olmos, M. Sánchez-Fernández, and F. Perez-Cruz,
<a class="reference external" href="https://ieeexplore.ieee.org/abstract/document/6841617">“Expectation Propagation Detection for High-Order High-Dimensional MIMO Systems”</a>,
IEEE Trans. Commun., vol. 62, no. 8, pp. 2840-2849, Aug. 2014.

CST2011(<a href="https://nvlabs.github.io/sionna/api/mimo.html#id7">1</a>,<a href="https://nvlabs.github.io/sionna/api/mimo.html#id8">2</a>,<a href="https://nvlabs.github.io/sionna/api/mimo.html#id9">3</a>,<a href="https://nvlabs.github.io/sionna/api/mimo.html#id10">4</a>)
    
C. Studer, S. Fateh, and D. Seethaler,
<a class="reference external" href="https://ieeexplore.ieee.org/abstract/document/5779722">“ASIC Implementation of Soft-Input Soft-Output MIMO Detection Using MMSE Parallel Interference Cancellation”</a>,
IEEE Journal of Solid-State Circuits, vol. 46, no. 7, pp. 1754–1765, July 2011.



