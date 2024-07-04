
# Wireless<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#wireless" title="Permalink to this headline"></a>
    
This module provides layers and functions that implement wireless channel models.
Models currently available include <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.AWGN" title="sionna.channel.AWGN">`AWGN`</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#flat-fading">flat-fading</a> with (optional) <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.SpatialCorrelation" title="sionna.channel.SpatialCorrelation">`SpatialCorrelation`</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.RayleighBlockFading" title="sionna.channel.RayleighBlockFading">`RayleighBlockFading`</a>, as well as models from the 3rd Generation Partnership Project (3GPP) <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id1">[TR38901]</a>: <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tdl">TDL</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#cdl">CDL</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#umi">UMi</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#uma">UMa</a>, and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#rma">RMa</a>. It is also possible to <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#external-datasets">use externally generated CIRs</a>.
    
Apart from <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#flat-fading">flat-fading</a>, all of these models generate channel impulse responses (CIRs) that can then be used to
implement a channel transfer function in the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#time-domain">time domain</a> or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#ofdm-waveform">assuming an OFDM waveform</a>.
    
This is achieved using the different functions, classes, and Keras layers which
operate as shown in the figures below.
<img alt="../_images/channel_arch_time.png" src="https://nvlabs.github.io/sionna/_images/channel_arch_time.png" />
<p class="caption">Fig. 7 Channel module architecture for time domain simulations.<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id30" title="Permalink to this image"></a>

<img alt="../_images/channel_arch_freq.png" src="https://nvlabs.github.io/sionna/_images/channel_arch_freq.png" />
<p class="caption">Fig. 8 Channel module architecture for simulations assuming OFDM waveform.<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id31" title="Permalink to this image"></a>
    
A channel model generate CIRs from which channel responses in the time domain
or in the frequency domain are computed using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.cir_to_time_channel" title="sionna.channel.cir_to_time_channel">`cir_to_time_channel()`</a> or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.cir_to_ofdm_channel" title="sionna.channel.cir_to_ofdm_channel">`cir_to_ofdm_channel()`</a> functions, respectively.
If one does not need access to the raw CIRs, the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateTimeChannel" title="sionna.channel.GenerateTimeChannel">`GenerateTimeChannel`</a> and
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateOFDMChannel" title="sionna.channel.GenerateOFDMChannel">`GenerateOFDMChannel`</a> classes can be used to conveniently
sample CIRs and generate channel responses in the desired domain.
    
Once the channel responses in the time or frequency domain are computed, they
can be applied to the channel input using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyTimeChannel" title="sionna.channel.ApplyTimeChannel">`ApplyTimeChannel`</a> or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyOFDMChannel" title="sionna.channel.ApplyOFDMChannel">`ApplyOFDMChannel`</a> Keras layers.
    
The following code snippets show how to setup and run a Rayleigh block fading
model assuming an OFDM waveform, and without accessing the CIRs or
channel responses.
This is the easiest way to setup a channel model.
Setting-up other models is done in a similar way, except for
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.AWGN" title="sionna.channel.AWGN">`AWGN`</a> (see the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.AWGN" title="sionna.channel.AWGN">`AWGN`</a>
class documentation).
```python
rayleigh = RayleighBlockFading(num_rx = 1,
                               num_rx_ant = 32,
                               num_tx = 4,
                               num_tx_ant = 2)
channel  = OFDMChannel(channel_model = rayleigh,
                       resource_grid = rg)
```

    
where `rg` is an instance of <a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
    
Running the channel model is done as follows:
```python
# x is the channel input
# no is the noise variance
y = channel([x, no])
```

    
To use the time domain representation of the channel, one can use
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.TimeChannel" title="sionna.channel.TimeChannel">`TimeChannel`</a> instead of
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.OFDMChannel" title="sionna.channel.OFDMChannel">`OFDMChannel`</a>.
    
If access to the channel responses is needed, one can separate their
generation from their application to the channel input by setting up the channel
model as follows:
```python
rayleigh = RayleighBlockFading(num_rx = 1,
                               num_rx_ant = 32,
                               num_tx = 4,
                               num_tx_ant = 2)
generate_channel = GenerateOFDMChannel(channel_model = rayleigh,
                                       resource_grid = rg)
apply_channel = ApplyOFDMChannel()
```

    
where `rg` is an instance of <a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
Running the channel model is done as follows:
```python
# Generate a batch of channel responses
h = generate_channel(batch_size)
# Apply the channel
# x is the channel input
# no is the noise variance
y = apply_channel([x, h, no])
```

    
Generating and applying the channel in the time domain can be achieved by using
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateTimeChannel" title="sionna.channel.GenerateTimeChannel">`GenerateTimeChannel`</a> and
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyTimeChannel" title="sionna.channel.ApplyTimeChannel">`ApplyTimeChannel`</a> instead of
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateOFDMChannel" title="sionna.channel.GenerateOFDMChannel">`GenerateOFDMChannel`</a> and
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyOFDMChannel" title="sionna.channel.ApplyOFDMChannel">`ApplyOFDMChannel`</a>, respectively.
    
To access the CIRs, setting up the channel can be done as follows:
```python
rayleigh = RayleighBlockFading(num_rx = 1,
                               num_rx_ant = 32,
                               num_tx = 4,
                               num_tx_ant = 2)
apply_channel = ApplyOFDMChannel()
```

    
and running the channel model as follows:
```python
cir = rayleigh(batch_size)
h = cir_to_ofdm_channel(frequencies, *cir)
y = apply_channel([x, h, no])
```

    
where `frequencies` are the subcarrier frequencies in the baseband, which can
be computed using the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.subcarrier_frequencies" title="sionna.channel.subcarrier_frequencies">`subcarrier_frequencies()`</a> utility
function.
    
Applying the channel in the time domain can be done by using
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.cir_to_time_channel" title="sionna.channel.cir_to_time_channel">`cir_to_time_channel()`</a> and
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyTimeChannel" title="sionna.channel.ApplyTimeChannel">`ApplyTimeChannel`</a> instead of
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.cir_to_ofdm_channel" title="sionna.channel.cir_to_ofdm_channel">`cir_to_ofdm_channel()`</a> and
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyOFDMChannel" title="sionna.channel.ApplyOFDMChannel">`ApplyOFDMChannel`</a>, respectively.
    
For the purpose of the present document, the following symbols apply:
<table class="docutils align-default">
<colgroup>
<col style="width: 24%" />
<col style="width: 76%" />
</colgroup>
<tbody>
<tr class="row-odd"><td>    
$N_T (u)$</td>
<td>    
Number of transmitters (transmitter index)</td>
</tr>
<tr class="row-even"><td>    
$N_R (v)$</td>
<td>    
Number of receivers (receiver index)</td>
</tr>
<tr class="row-odd"><td>    
$N_{TA} (k)$</td>
<td>    
Number of antennas per transmitter (transmit antenna index)</td>
</tr>
<tr class="row-even"><td>    
$N_{RA} (l)$</td>
<td>    
Number of antennas per receiver (receive antenna index)</td>
</tr>
<tr class="row-odd"><td>    
$N_S (s)$</td>
<td>    
Number of OFDM symbols (OFDM symbol index)</td>
</tr>
<tr class="row-even"><td>    
$N_F (n)$</td>
<td>    
Number of subcarriers (subcarrier index)</td>
</tr>
<tr class="row-odd"><td>    
$N_B (b)$</td>
<td>    
Number of time samples forming the channel input (baseband symbol index)</td>
</tr>
<tr class="row-even"><td>    
$L_{\text{min}}$</td>
<td>    
Smallest time-lag for the discrete complex baseband channel</td>
</tr>
<tr class="row-odd"><td>    
$L_{\text{max}}$</td>
<td>    
Largest time-lag for the discrete complex baseband channel</td>
</tr>
<tr class="row-even"><td>    
$M (m)$</td>
<td>    
Number of paths (clusters) forming a power delay profile (path index)</td>
</tr>
<tr class="row-odd"><td>    
$\tau_m(t)$</td>
<td>    
$m^{th}$ path (cluster) delay at time step $t$</td>
</tr>
<tr class="row-even"><td>    
$a_m(t)$</td>
<td>    
$m^{th}$ path (cluster) complex coefficient at time step $t$</td>
</tr>
<tr class="row-odd"><td>    
$\Delta_f$</td>
<td>    
Subcarrier spacing</td>
</tr>
<tr class="row-even"><td>    
$W$</td>
<td>    
Bandwidth</td>
</tr>
<tr class="row-odd"><td>    
$N_0$</td>
<td>    
Noise variance</td>
</tr>
</tbody>
</table>
    
All transmitters are equipped with $N_{TA}$ antennas and all receivers
with $N_{RA}$ antennas.
    
A channel model, such as <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.RayleighBlockFading" title="sionna.channel.RayleighBlockFading">`RayleighBlockFading`</a> or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi" title="sionna.channel.tr38901.UMi">`UMi`</a>, is used to generate for each link between
antenna $k$ of transmitter $u$ and antenna $l$ of receiver
$v$ a power delay profile
$(a_{u, k, v, l, m}(t), \tau_{u, v, m}), 0 \leq m \leq M-1$.
The delays are assumed not to depend on time $t$, and transmit and receive
antennas $k$ and $l$.
Such a power delay profile corresponds to the channel impulse response

$$
h_{u, k, v, l}(t,\tau) =
\sum_{m=0}^{M-1} a_{u, k, v, l,m}(t) \delta(\tau - \tau_{u, v, m})
$$
    
where $\delta(\cdot)$ is the Dirac delta measure.
For example, in the case of Rayleigh block fading, the power delay profiles are
time-invariant and such that for every link $(u, k, v, l)$

$$
\begin{split}\begin{align}
   M                     &= 1\\
   \tau_{u, v, 0}  &= 0\\
   a_{u, k, v, l, 0}     &\sim \mathcal{CN}(0,1).
\end{align}\end{split}
$$
    
3GPP channel models use the procedure depicted in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id2">[TR38901]</a> to generate power
delay profiles. With these models, the power delay profiles are time-<em>variant</em>
in the event of mobility.

## AWGN<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#awgn" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``AWGN`(<em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/channel/awgn.html#AWGN">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.AWGN" title="Permalink to this definition"></a>
    
Add complex AWGN to the inputs with a certain variance.
    
This class inherits from the Keras <cite>Layer</cite> class and can be used as layer in
a Keras model.
    
This layer adds complex AWGN noise with variance `no` to the input.
The noise has variance `no/2` per real dimension.
It can be either a scalar or a tensor which can be broadcast to the shape
of the input.
<p class="rubric">Example
    
Setting-up:
```python
>>> awgn_channel = AWGN()
```

    
Running:
```python
>>> # x is the channel input
>>> # no is the noise variance
>>> y = awgn_channel((x, no))
```

Parameters
    
**dtype** (<em>Complex tf.DType</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.

Input
 
- **(x, no)** – Tuple:
- **x** (<em>Tensor, tf.complex</em>) – Channel input
- **no** (<em>Scalar or Tensor, tf.float</em>) – Scalar or tensor whose shape can be broadcast to the shape of `x`.
The noise power `no` is per complex dimension. If `no` is a
scalar, noise of the same variance will be added to the input.
If `no` is a tensor, it must have a shape that can be broadcast to
the shape of `x`. This allows, e.g., adding noise of different
variance to each example in a batch. If `no` has a lower rank than
`x`, then `no` will be broadcast to the shape of `x` by adding
dummy dimensions after the last axis.


Output
    
**y** (Tensor with same shape as `x`, tf.complex) – Channel output



## Flat-fading channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#flat-fading-channel" title="Permalink to this headline"></a>

### FlatFadingChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#flatfadingchannel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``FlatFadingChannel`(<em class="sig-param">`num_tx_ant`</em>, <em class="sig-param">`num_rx_ant`</em>, <em class="sig-param">`spatial_corr``=``None`</em>, <em class="sig-param">`add_awgn``=``True`</em>, <em class="sig-param">`return_channel``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/channel/flat_fading_channel.html#FlatFadingChannel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.FlatFadingChannel" title="Permalink to this definition"></a>
    
Applies random channel matrices to a vector input and adds AWGN.
    
This class combines <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateFlatFadingChannel" title="sionna.channel.GenerateFlatFadingChannel">`GenerateFlatFadingChannel`</a> and
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyFlatFadingChannel" title="sionna.channel.ApplyFlatFadingChannel">`ApplyFlatFadingChannel`</a> and computes the output of
a flat-fading channel with AWGN.
    
For a given batch of input vectors $\mathbf{x}\in\mathbb{C}^{K}$,
the output is

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$
    
where $\mathbf{H}\in\mathbb{C}^{M\times K}$ are randomly generated
flat-fading channel matrices and
$\mathbf{n}\in\mathbb{C}^{M}\sim\mathcal{CN}(0, N_o\mathbf{I})$
is an AWGN vector that is optionally added.
    
A <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.SpatialCorrelation" title="sionna.channel.SpatialCorrelation">`SpatialCorrelation`</a> can be configured and the
channel realizations optionally returned. This is useful to simulate
receiver algorithms with perfect channel knowledge.
Parameters
 
- **num_tx_ant** (<em>int</em>) – Number of transmit antennas.
- **num_rx_ant** (<em>int</em>) – Number of receive antennas.
- **spatial_corr** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.SpatialCorrelation" title="sionna.channel.SpatialCorrelation"><em>SpatialCorrelation</em></a><em>, </em><em>None</em>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.SpatialCorrelation" title="sionna.channel.SpatialCorrelation">`SpatialCorrelation`</a> or <cite>None</cite>.
Defaults to <cite>None</cite>.
- **add_awgn** (<em>bool</em>) – Indicates if AWGN noise should be added to the output.
Defaults to <cite>True</cite>.
- **return_channel** (<em>bool</em>) – Indicates if the channel realizations should be returned.
Defaults  to <cite>False</cite>.
- **dtype** (<em>tf.complex64</em><em>, </em><em>tf.complex128</em>) – The dtype of the output. Defaults to <cite>tf.complex64</cite>.


Input
 
- **(x, no)** – Tuple or Tensor:
- **x** (<em>[batch_size, num_tx_ant], tf.complex</em>) – Tensor of transmit vectors.
- **no** (<em>Scalar of Tensor, tf.float</em>) – The noise power `no` is per complex dimension.
Only required if `add_awgn==True`.
Will be broadcast to the dimensions of the channel output if needed.
For more details, see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.AWGN" title="sionna.channel.AWGN">`AWGN`</a>.


Output
 
- **(y, h)** – Tuple or Tensor:
- **y** ([batch_size, num_rx_ant, num_tx_ant], `dtype`) – Channel output.
- **h** ([batch_size, num_rx_ant, num_tx_ant], `dtype`) – Channel realizations. Will only be returned if
`return_channel==True`.




<em class="property">`property` </em>`apply`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.FlatFadingChannel.apply" title="Permalink to this definition"></a>
    
Calls the internal <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyFlatFadingChannel" title="sionna.channel.ApplyFlatFadingChannel">`ApplyFlatFadingChannel`</a>.


<em class="property">`property` </em>`generate`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.FlatFadingChannel.generate" title="Permalink to this definition"></a>
    
Calls the internal <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateFlatFadingChannel" title="sionna.channel.GenerateFlatFadingChannel">`GenerateFlatFadingChannel`</a>.


<em class="property">`property` </em>`spatial_corr`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.FlatFadingChannel.spatial_corr" title="Permalink to this definition"></a>
    
The <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.SpatialCorrelation" title="sionna.channel.SpatialCorrelation">`SpatialCorrelation`</a> to be used.


### GenerateFlatFadingChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#generateflatfadingchannel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``GenerateFlatFadingChannel`(<em class="sig-param">`num_tx_ant`</em>, <em class="sig-param">`num_rx_ant`</em>, <em class="sig-param">`spatial_corr``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/channel/flat_fading_channel.html#GenerateFlatFadingChannel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateFlatFadingChannel" title="Permalink to this definition"></a>
    
Generates tensors of flat-fading channel realizations.
    
This class generates batches of random flat-fading channel matrices.
A spatial correlation can be applied.
Parameters
 
- **num_tx_ant** (<em>int</em>) – Number of transmit antennas.
- **num_rx_ant** (<em>int</em>) – Number of receive antennas.
- **spatial_corr** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.SpatialCorrelation" title="sionna.channel.SpatialCorrelation"><em>SpatialCorrelation</em></a><em>, </em><em>None</em>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.SpatialCorrelation" title="sionna.channel.SpatialCorrelation">`SpatialCorrelation`</a> or <cite>None</cite>.
Defaults to <cite>None</cite>.
- **dtype** (<em>tf.complex64</em><em>, </em><em>tf.complex128</em>) – The dtype of the output. Defaults to <cite>tf.complex64</cite>.


Input
    
**batch_size** (<em>int</em>) – The batch size, i.e., the number of channel matrices to generate.

Output
    
**h** ([batch_size, num_rx_ant, num_tx_ant], `dtype`) – Batch of random flat fading channel matrices.



<em class="property">`property` </em>`spatial_corr`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateFlatFadingChannel.spatial_corr" title="Permalink to this definition"></a>
    
The <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.SpatialCorrelation" title="sionna.channel.SpatialCorrelation">`SpatialCorrelation`</a> to be used.


### ApplyFlatFadingChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#applyflatfadingchannel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``ApplyFlatFadingChannel`(<em class="sig-param">`add_awgn``=``True`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/channel/flat_fading_channel.html#ApplyFlatFadingChannel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyFlatFadingChannel" title="Permalink to this definition"></a>
    
Applies given channel matrices to a vector input and adds AWGN.
    
This class applies a given tensor of flat-fading channel matrices
to an input tensor. AWGN noise can be optionally added.
Mathematically, for channel matrices
$\mathbf{H}\in\mathbb{C}^{M\times K}$
and input $\mathbf{x}\in\mathbb{C}^{K}$, the output is

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$
    
where $\mathbf{n}\in\mathbb{C}^{M}\sim\mathcal{CN}(0, N_o\mathbf{I})$
is an AWGN vector that is optionally added.
Parameters
 
- **add_awgn** (<em>bool</em>) – Indicates if AWGN noise should be added to the output.
Defaults to <cite>True</cite>.
- **dtype** (<em>tf.complex64</em><em>, </em><em>tf.complex128</em>) – The dtype of the output. Defaults to <cite>tf.complex64</cite>.


Input
 
- **(x, h, no)** – Tuple:
- **x** (<em>[batch_size, num_tx_ant], tf.complex</em>) – Tensor of transmit vectors.
- **h** (<em>[batch_size, num_rx_ant, num_tx_ant], tf.complex</em>) – Tensor of channel realizations. Will be broadcast to the
dimensions of `x` if needed.
- **no** (<em>Scalar or Tensor, tf.float</em>) – The noise power `no` is per complex dimension.
Only required if `add_awgn==True`.
Will be broadcast to the shape of `y`.
For more details, see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.AWGN" title="sionna.channel.AWGN">`AWGN`</a>.


Output
    
**y** ([batch_size, num_rx_ant, num_tx_ant], `dtype`) – Channel output.



### SpatialCorrelation<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#spatialcorrelation" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``SpatialCorrelation`<a class="reference internal" href="../_modules/sionna/channel/spatial_correlation.html#SpatialCorrelation">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.SpatialCorrelation" title="Permalink to this definition"></a>
    
Abstract class that defines an interface for spatial correlation functions.
    
The <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.FlatFadingChannel" title="sionna.channel.FlatFadingChannel">`FlatFadingChannel`</a> model can be configured with a
spatial correlation model.
Input
    
**h** (<em>tf.complex</em>) – Tensor of arbitrary shape containing spatially uncorrelated
channel coefficients

Output
    
**h_corr** (<em>tf.complex</em>) – Tensor of the same shape and dtype as `h` containing the spatially
correlated channel coefficients.



### KroneckerModel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#kroneckermodel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``KroneckerModel`(<em class="sig-param">`r_tx``=``None`</em>, <em class="sig-param">`r_rx``=``None`</em>)<a class="reference internal" href="../_modules/sionna/channel/spatial_correlation.html#KroneckerModel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.KroneckerModel" title="Permalink to this definition"></a>
    
Kronecker model for spatial correlation.
    
Given a batch of matrices $\mathbf{H}\in\mathbb{C}^{M\times K}$,
$\mathbf{R}_\text{tx}\in\mathbb{C}^{K\times K}$, and
$\mathbf{R}_\text{rx}\in\mathbb{C}^{M\times M}$, this function
will generate the following output:

$$
\mathbf{H}_\text{corr} = \mathbf{R}^{\frac12}_\text{rx} \mathbf{H} \mathbf{R}^{\frac12}_\text{tx}
$$
    
Note that $\mathbf{R}_\text{tx}\in\mathbb{C}^{K\times K}$ and $\mathbf{R}_\text{rx}\in\mathbb{C}^{M\times M}$
must be positive semi-definite, such as the ones generated by
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.exp_corr_mat" title="sionna.channel.exp_corr_mat">`exp_corr_mat()`</a>.
Parameters
 
- **r_tx** (<em>[</em><em>...</em><em>, </em><em>K</em><em>, </em><em>K</em><em>]</em><em>, </em><em>tf.complex</em>) – Tensor containing the transmit correlation matrices. If
the rank of `r_tx` is smaller than that of the input `h`,
it will be broadcast.
- **r_rx** (<em>[</em><em>...</em><em>, </em><em>M</em><em>, </em><em>M</em><em>]</em><em>, </em><em>tf.complex</em>) – Tensor containing the receive correlation matrices. If
the rank of `r_rx` is smaller than that of the input `h`,
it will be broadcast.


Input
    
**h** (<em>[…, M, K], tf.complex</em>) – Tensor containing spatially uncorrelated
channel coeffficients.

Output
    
**h_corr** (<em>[…, M, K], tf.complex</em>) – Tensor containing the spatially
correlated channel coefficients.



<em class="property">`property` </em>`r_rx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.KroneckerModel.r_rx" title="Permalink to this definition"></a>
    
Tensor containing the receive correlation matrices.

**Note**
    
If you want to set this property in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.


<em class="property">`property` </em>`r_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.KroneckerModel.r_tx" title="Permalink to this definition"></a>
    
Tensor containing the transmit correlation matrices.

**Note**
    
If you want to set this property in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.


### PerColumnModel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#percolumnmodel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``PerColumnModel`(<em class="sig-param">`r_rx`</em>)<a class="reference internal" href="../_modules/sionna/channel/spatial_correlation.html#PerColumnModel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.PerColumnModel" title="Permalink to this definition"></a>
    
Per-column model for spatial correlation.
    
Given a batch of matrices $\mathbf{H}\in\mathbb{C}^{M\times K}$
and correlation matrices $\mathbf{R}_k\in\mathbb{C}^{M\times M}, k=1,\dots,K$,
this function will generate the output $\mathbf{H}_\text{corr}\in\mathbb{C}^{M\times K}$,
with columns

$$
\mathbf{h}^\text{corr}_k = \mathbf{R}^{\frac12}_k \mathbf{h}_k,\quad k=1, \dots, K
$$
    
where $\mathbf{h}_k$ is the kth column of $\mathbf{H}$.
Note that all $\mathbf{R}_k\in\mathbb{C}^{M\times M}$ must
be positive semi-definite, such as the ones generated
by <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.one_ring_corr_mat" title="sionna.channel.one_ring_corr_mat">`one_ring_corr_mat()`</a>.
    
This model is typically used to simulate a MIMO channel between multiple
single-antenna users and a base station with multiple antennas.
The resulting SIMO channel for each user has a different spatial correlation.
Parameters
    
**r_rx** (<em>[</em><em>...</em><em>, </em><em>M</em><em>, </em><em>M</em><em>]</em><em>, </em><em>tf.complex</em>) – Tensor containing the receive correlation matrices. If
the rank of `r_rx` is smaller than that of the input `h`,
it will be broadcast. For a typically use of this model, `r_rx`
has shape […, K, M, M], i.e., a different correlation matrix for each
column of `h`.

Input
    
**h** (<em>[…, M, K], tf.complex</em>) – Tensor containing spatially uncorrelated
channel coeffficients.

Output
    
**h_corr** (<em>[…, M, K], tf.complex</em>) – Tensor containing the spatially
correlated channel coefficients.



<em class="property">`property` </em>`r_rx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.PerColumnModel.r_rx" title="Permalink to this definition"></a>
    
Tensor containing the receive correlation matrices.

**Note**
    
If you want to set this property in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.


## Channel model interface<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#channel-model-interface" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``ChannelModel`<a class="reference internal" href="../_modules/sionna/channel/channel_model.html#ChannelModel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ChannelModel" title="Permalink to this definition"></a>
    
Abstract class that defines an interface for channel models.
    
Any channel model which generates channel impulse responses must implement this interface.
All the channel models available in Sionna, such as <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.RayleighBlockFading" title="sionna.channel.RayleighBlockFading">`RayleighBlockFading`</a> or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.TDL" title="sionna.channel.tr38901.TDL">`TDL`</a>, implement this interface.
    
<em>Remark:</em> Some channel models only require a subset of the input parameters.
Input
 
- **batch_size** (<em>int</em>) – Batch size
- **num_time_steps** (<em>int</em>) – Number of time steps
- **sampling_frequency** (<em>float</em>) – Sampling frequency [Hz]


Output
 
- **a** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex</em>) – Path coefficients
- **tau** (<em>[batch size, num_rx, num_tx, num_paths], tf.float</em>) – Path delays [s]




## Time domain channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#time-domain-channel" title="Permalink to this headline"></a>
    
The model of the channel in the time domain assumes pulse shaping and receive
filtering are performed using a conventional sinc filter (see, e.g., <a class="reference internal" href="../em_primer.html#tse" id="id3">[Tse]</a>).
Using sinc for transmit and receive filtering, the discrete-time domain received
signal at time step $b$ is

$$
y_{v, l, b} = \sum_{u=0}^{N_{T}-1}\sum_{k=0}^{N_{TA}-1}
   \sum_{\ell = L_{\text{min}}}^{L_{\text{max}}}
   \bar{h}_{u, k, v, l, b, \ell} x_{u, k, b-\ell}
   + w_{v, l, b}
$$
    
where $x_{u, k, b}$ is the baseband symbol transmitted by transmitter
$u$ on antenna $k$ and at time step $b$,
$w_{v, l, b} \sim \mathcal{CN}\left(0,N_0\right)$ the additive white
Gaussian noise, and $\bar{h}_{u, k, v, l, b, \ell}$ the channel filter tap
at time step $b$ and for time-lag $\ell$, which is given by

$$
\bar{h}_{u, k, v, l, b, \ell}
= \sum_{m=0}^{M-1} a_{u, k, v, l, m}\left(\frac{b}{W}\right)
   \text{sinc}\left( \ell - W\tau_{u, v, m} \right).
$$

**Note**
    
The two parameters $L_{\text{min}}$ and $L_{\text{max}}$ control the smallest
and largest time-lag for the discrete-time channel model, respectively.
They are set when instantiating <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.TimeChannel" title="sionna.channel.TimeChannel">`TimeChannel`</a>,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateTimeChannel" title="sionna.channel.GenerateTimeChannel">`GenerateTimeChannel`</a>, and when calling the utility
function <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.cir_to_time_channel" title="sionna.channel.cir_to_time_channel">`cir_to_time_channel()`</a>.
Because the sinc filter is neither time-limited nor causal, the discrete-time
channel model is not causal. Therefore, ideally, one would set
$L_{\text{min}} = -\infty$ and $L_{\text{max}} = +\infty$.
In practice, however, these two parameters need to be set to reasonable
finite values. Values for these two parameters can be computed using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.time_lag_discrete_time_channel" title="sionna.channel.time_lag_discrete_time_channel">`time_lag_discrete_time_channel()`</a> utility function from
a given bandwidth and maximum delay spread.
This function returns $-6$ for $L_{\text{min}}$. $L_{\text{max}}$ is computed
from the specified bandwidth and maximum delay spread, which default value is
$3 \mu s$. These values for $L_{\text{min}}$ and the maximum delay spread
were found to be valid for all the models available in Sionna when an RMS delay
spread of 100ns is assumed.

### TimeChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#timechannel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``TimeChannel`(<em class="sig-param">`channel_model`</em>, <em class="sig-param">`bandwidth`</em>, <em class="sig-param">`num_time_samples`</em>, <em class="sig-param">`maximum_delay_spread``=``3e-6`</em>, <em class="sig-param">`l_min``=``None`</em>, <em class="sig-param">`l_max``=``None`</em>, <em class="sig-param">`normalize_channel``=``False`</em>, <em class="sig-param">`add_awgn``=``True`</em>, <em class="sig-param">`return_channel``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/channel/time_channel.html#TimeChannel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.TimeChannel" title="Permalink to this definition"></a>
    
Generate channel responses and apply them to channel inputs in the time domain.
    
This class inherits from the Keras <cite>Layer</cite> class and can be used as layer
in a Keras model.
    
The channel output consists of `num_time_samples` + `l_max` - `l_min`
time samples, as it is the result of filtering the channel input of length
`num_time_samples` with the time-variant channel filter  of length
`l_max` - `l_min` + 1. In the case of a single-input single-output link and given a sequence of channel
inputs $x_0,\cdots,x_{N_B}$, where $N_B$ is `num_time_samples`, this
layer outputs

$$
y_b = \sum_{\ell = L_{\text{min}}}^{L_{\text{max}}} x_{b-\ell} \bar{h}_{b,\ell} + w_b
$$
    
where $L_{\text{min}}$ corresponds `l_min`, $L_{\text{max}}$ to `l_max`, $w_b$ to
the additive noise, and $\bar{h}_{b,\ell}$ to the
$\ell^{th}$ tap of the $b^{th}$ channel sample.
This layer outputs $y_b$ for $b$ ranging from $L_{\text{min}}$ to
$N_B + L_{\text{max}} - 1$, and $x_{b}$ is set to 0 for $b < 0$ or $b \geq N_B$.
The channel taps $\bar{h}_{b,\ell}$ are computed assuming a sinc filter
is used for pulse shaping and receive filtering. Therefore, given a channel impulse response
$(a_{m}(t), \tau_{m}), 0 \leq m \leq M-1$, generated by the `channel_model`,
the channel taps are computed as follows:

$$
\bar{h}_{b, \ell}
= \sum_{m=0}^{M-1} a_{m}\left(\frac{b}{W}\right)
    \text{sinc}\left( \ell - W\tau_{m} \right)
$$
    
for $\ell$ ranging from `l_min` to `l_max`, and where $W$ is
the `bandwidth`.
    
For multiple-input multiple-output (MIMO) links, the channel output is computed for each antenna of each receiver and by summing over all the antennas of all transmitters.
Parameters
 
- **channel_model** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ChannelModel" title="sionna.channel.ChannelModel">`ChannelModel`</a> object) – An instance of a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ChannelModel" title="sionna.channel.ChannelModel">`ChannelModel`</a>, such as
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.RayleighBlockFading" title="sionna.channel.RayleighBlockFading">`RayleighBlockFading`</a> or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi" title="sionna.channel.tr38901.UMi">`UMi`</a>.
- **bandwidth** (<em>float</em>) – Bandwidth ($W$) [Hz]
- **num_time_samples** (<em>int</em>) – Number of time samples forming the channel input ($N_B$)
- **maximum_delay_spread** (<em>float</em>) – Maximum delay spread [s].
Used to compute the default value of `l_max` if `l_max` is set to
<cite>None</cite>. If a value is given for `l_max`, this parameter is not used.
It defaults to 3us, which was found
to be large enough to include most significant paths with all channel
models included in Sionna assuming a nominal delay spread of 100ns.
- **l_min** (<em>int</em>) – Smallest time-lag for the discrete complex baseband channel ($L_{\text{min}}$).
If set to <cite>None</cite>, defaults to the value given by <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.time_lag_discrete_time_channel" title="sionna.channel.time_lag_discrete_time_channel">`time_lag_discrete_time_channel()`</a>.
- **l_max** (<em>int</em>) – Largest time-lag for the discrete complex baseband channel ($L_{\text{max}}$).
If set to <cite>None</cite>, it is computed from `bandwidth` and `maximum_delay_spread`
using <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.time_lag_discrete_time_channel" title="sionna.channel.time_lag_discrete_time_channel">`time_lag_discrete_time_channel()`</a>. If it is not set to <cite>None</cite>,
then the parameter `maximum_delay_spread` is not used.
- **add_awgn** (<em>bool</em>) – If set to <cite>False</cite>, no white Gaussian noise is added.
Defaults to <cite>True</cite>.
- **normalize_channel** (<em>bool</em>) – If set to <cite>True</cite>, the channel is normalized over the block size
to ensure unit average energy per time step. Defaults to <cite>False</cite>.
- **return_channel** (<em>bool</em>) – If set to <cite>True</cite>, the channel response is returned in addition to the
channel output. Defaults to <cite>False</cite>.
- **dtype** (<em>tf.DType</em>) – Complex datatype to use for internal processing and output.
Defaults to <cite>tf.complex64</cite>.


Input
 
- **(x, no) or x** – Tuple or Tensor:
- **x** (<em>[batch size, num_tx, num_tx_ant, num_time_samples], tf.complex</em>) – Channel inputs
- **no** (<em>Scalar or Tensor, tf.float</em>) – Scalar or tensor whose shape can be broadcast to the shape of the
channel outputs: [batch size, num_rx, num_rx_ant, num_time_samples].
Only required if `add_awgn` is set to <cite>True</cite>.
The noise power `no` is per complex dimension. If `no` is a scalar,
noise of the same variance will be added to the outputs.
If `no` is a tensor, it must have a shape that can be broadcast to
the shape of the channel outputs. This allows, e.g., adding noise of
different variance to each example in a batch. If `no` has a lower
rank than the channel outputs, then `no` will be broadcast to the
shape of the channel outputs by adding dummy dimensions after the last
axis.


Output
 
- **y** (<em>[batch size, num_rx, num_rx_ant, num_time_samples + l_max - l_min], tf.complex</em>) – Channel outputs
The channel output consists of `num_time_samples` + `l_max` - `l_min`
time samples, as it is the result of filtering the channel input of length
`num_time_samples` with the time-variant channel filter  of length
`l_max` - `l_min` + 1.
- **h_time** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_samples + l_max - l_min, l_max - l_min + 1], tf.complex</em>) – (Optional) Channel responses. Returned only if `return_channel`
is set to <cite>True</cite>.
For each batch example, `num_time_samples` + `l_max` - `l_min` time
steps of the channel realizations are generated to filter the channel input.




### GenerateTimeChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#generatetimechannel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``GenerateTimeChannel`(<em class="sig-param">`channel_model`</em>, <em class="sig-param">`bandwidth`</em>, <em class="sig-param">`num_time_samples`</em>, <em class="sig-param">`l_min`</em>, <em class="sig-param">`l_max`</em>, <em class="sig-param">`normalize_channel``=``False`</em>)<a class="reference internal" href="../_modules/sionna/channel/generate_time_channel.html#GenerateTimeChannel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateTimeChannel" title="Permalink to this definition"></a>
    
Generate channel responses in the time domain.
    
For each batch example, `num_time_samples` + `l_max` - `l_min` time steps of a
channel realization are generated by this layer.
These can be used to filter a channel input of length `num_time_samples` using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyTimeChannel" title="sionna.channel.ApplyTimeChannel">`ApplyTimeChannel`</a> layer.
    
The channel taps $\bar{h}_{b,\ell}$ (`h_time`) returned by this layer
are computed assuming a sinc filter is used for pulse shaping and receive filtering.
Therefore, given a channel impulse response
$(a_{m}(t), \tau_{m}), 0 \leq m \leq M-1$, generated by the `channel_model`,
the channel taps are computed as follows:

$$
\bar{h}_{b, \ell}
= \sum_{m=0}^{M-1} a_{m}\left(\frac{b}{W}\right)
    \text{sinc}\left( \ell - W\tau_{m} \right)
$$
    
for $\ell$ ranging from `l_min` to `l_max`, and where $W$ is
the `bandwidth`.
Parameters
 
- **channel_model** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ChannelModel" title="sionna.channel.ChannelModel">`ChannelModel`</a> object) – An instance of a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ChannelModel" title="sionna.channel.ChannelModel">`ChannelModel`</a>, such as
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.RayleighBlockFading" title="sionna.channel.RayleighBlockFading">`RayleighBlockFading`</a> or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi" title="sionna.channel.tr38901.UMi">`UMi`</a>.
- **bandwidth** (<em>float</em>) – Bandwidth ($W$) [Hz]
- **num_time_samples** (<em>int</em>) – Number of time samples forming the channel input ($N_B$)
- **l_min** (<em>int</em>) – Smallest time-lag for the discrete complex baseband channel ($L_{\text{min}}$)
- **l_max** (<em>int</em>) – Largest time-lag for the discrete complex baseband channel ($L_{\text{max}}$)
- **normalize_channel** (<em>bool</em>) – If set to <cite>True</cite>, the channel is normalized over the block size
to ensure unit average energy per time step. Defaults to <cite>False</cite>.


Input
    
**batch_size** (<em>int</em>) – Batch size. Defaults to <cite>None</cite> for channel models that do not require this paranmeter.

Output
    
**h_time** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_samples + l_max - l_min, l_max - l_min + 1], tf.complex</em>) – Channel responses.
For each batch example, `num_time_samples` + `l_max` - `l_min` time steps of a
channel realization are generated by this layer.
These can be used to filter a channel input of length `num_time_samples` using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyTimeChannel" title="sionna.channel.ApplyTimeChannel">`ApplyTimeChannel`</a> layer.



### ApplyTimeChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#applytimechannel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``ApplyTimeChannel`(<em class="sig-param">`num_time_samples`</em>, <em class="sig-param">`l_tot`</em>, <em class="sig-param">`add_awgn``=``True`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/channel/apply_time_channel.html#ApplyTimeChannel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyTimeChannel" title="Permalink to this definition"></a>
    
Apply time domain channel responses `h_time` to channel inputs `x`,
by filtering the channel inputs with time-variant channel responses.
    
This class inherits from the Keras <cite>Layer</cite> class and can be used as layer
in a Keras model.
    
For each batch example, `num_time_samples` + `l_tot` - 1 time steps of a
channel realization are required to filter the channel inputs.
    
The channel output consists of `num_time_samples` + `l_tot` - 1
time samples, as it is the result of filtering the channel input of length
`num_time_samples` with the time-variant channel filter  of length
`l_tot`. In the case of a single-input single-output link and given a sequence of channel
inputs $x_0,\cdots,x_{N_B}$, where $N_B$ is `num_time_samples`, this
layer outputs

$$
y_b = \sum_{\ell = 0}^{L_{\text{tot}}} x_{b-\ell} \bar{h}_{b,\ell} + w_b
$$
    
where $L_{\text{tot}}$ corresponds `l_tot`, $w_b$ to the additive noise, and
$\bar{h}_{b,\ell}$ to the $\ell^{th}$ tap of the $b^{th}$ channel sample.
This layer outputs $y_b$ for $b$ ranging from 0 to
$N_B + L_{\text{tot}} - 1$, and $x_{b}$ is set to 0 for $b \geq N_B$.
    
For multiple-input multiple-output (MIMO) links, the channel output is computed for each antenna
of each receiver and by summing over all the antennas of all transmitters.
Parameters
 
- **num_time_samples** (<em>int</em>) – Number of time samples forming the channel input ($N_B$)
- **l_tot** (<em>int</em>) – Length of the channel filter ($L_{\text{tot}} = L_{\text{max}} - L_{\text{min}} + 1$)
- **add_awgn** (<em>bool</em>) – If set to <cite>False</cite>, no white Gaussian noise is added.
Defaults to <cite>True</cite>.
- **dtype** (<em>tf.DType</em>) – Complex datatype to use for internal processing and output.
Defaults to <cite>tf.complex64</cite>.


Input
 
- **(x, h_time, no) or (x, h_time)** – Tuple:
- **x** (<em>[batch size, num_tx, num_tx_ant, num_time_samples], tf.complex</em>) – Channel inputs
- **h_time** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_samples + l_tot - 1, l_tot], tf.complex</em>) – Channel responses.
For each batch example, `num_time_samples` + `l_tot` - 1 time steps of a
channel realization are required to filter the channel inputs.
- **no** (<em>Scalar or Tensor, tf.float</em>) – Scalar or tensor whose shape can be broadcast to the shape of the channel outputs: [batch size, num_rx, num_rx_ant, num_time_samples + l_tot - 1].
Only required if `add_awgn` is set to <cite>True</cite>.
The noise power `no` is per complex dimension. If `no` is a
scalar, noise of the same variance will be added to the outputs.
If `no` is a tensor, it must have a shape that can be broadcast to
the shape of the channel outputs. This allows, e.g., adding noise of
different variance to each example in a batch. If `no` has a lower
rank than the channel outputs, then `no` will be broadcast to the
shape of the channel outputs by adding dummy dimensions after the
last axis.


Output
    
**y** (<em>[batch size, num_rx, num_rx_ant, num_time_samples + l_tot - 1], tf.complex</em>) – Channel outputs.
The channel output consists of `num_time_samples` + `l_tot` - 1
time samples, as it is the result of filtering the channel input of length
`num_time_samples` with the time-variant channel filter  of length
`l_tot`.



### cir_to_time_channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#cir-to-time-channel" title="Permalink to this headline"></a>

`sionna.channel.``cir_to_time_channel`(<em class="sig-param">`bandwidth`</em>, <em class="sig-param">`a`</em>, <em class="sig-param">`tau`</em>, <em class="sig-param">`l_min`</em>, <em class="sig-param">`l_max`</em>, <em class="sig-param">`normalize``=``False`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#cir_to_time_channel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.cir_to_time_channel" title="Permalink to this definition"></a>
    
Compute the channel taps forming the discrete complex-baseband
representation of the channel from the channel impulse response
(`a`, `tau`).
    
This function assumes that a sinc filter is used for pulse shaping and receive
filtering. Therefore, given a channel impulse response
$(a_{m}(t), \tau_{m}), 0 \leq m \leq M-1$, the channel taps
are computed as follows:

$$
\bar{h}_{b, \ell}
= \sum_{m=0}^{M-1} a_{m}\left(\frac{b}{W}\right)
    \text{sinc}\left( \ell - W\tau_{m} \right)
$$
    
for $\ell$ ranging from `l_min` to `l_max`, and where $W$ is
the `bandwidth`.
Input
 
- **bandwidth** (<em>float</em>) – Bandwidth [Hz]
- **a** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex</em>) – Path coefficients
- **tau** (<em>[batch size, num_rx, num_tx, num_paths] or [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths], tf.float</em>) – Path delays [s]
- **l_min** (<em>int</em>) – Smallest time-lag for the discrete complex baseband channel ($L_{\text{min}}$)
- **l_max** (<em>int</em>) – Largest time-lag for the discrete complex baseband channel ($L_{\text{max}}$)
- **normalize** (<em>bool</em>) – If set to <cite>True</cite>, the channel is normalized over the block size
to ensure unit average energy per time step. Defaults to <cite>False</cite>.


Output
    
**hm** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1], tf.complex</em>) – Channel taps coefficients



### time_to_ofdm_channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#time-to-ofdm-channel" title="Permalink to this headline"></a>

`sionna.channel.``time_to_ofdm_channel`(<em class="sig-param">`h_t`</em>, <em class="sig-param">`rg`</em>, <em class="sig-param">`l_min`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#time_to_ofdm_channel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.time_to_ofdm_channel" title="Permalink to this definition"></a>
    
Compute the channel frequency response from the discrete complex-baseband
channel impulse response.
    
Given a discrete complex-baseband channel impulse response
$\bar{h}_{b,\ell}$, for $\ell$ ranging from $L_\text{min}\le 0$
to $L_\text{max}$, the discrete channel frequency response is computed as

$$
\hat{h}_{b,n} = \sum_{k=0}^{L_\text{max}} \bar{h}_{b,k} e^{-j \frac{2\pi kn}{N}} + \sum_{k=L_\text{min}}^{-1} \bar{h}_{b,k} e^{-j \frac{2\pi n(N+k)}{N}}, \quad n=0,\dots,N-1
$$
    
where $N$ is the FFT size and $b$ is the time step.
    
This function only produces one channel frequency response per OFDM symbol, i.e.,
only values of $b$ corresponding to the start of an OFDM symbol (after
cyclic prefix removal) are considered.
Input
 
- **h_t** (<em>[…num_time_steps,l_max-l_min+1], tf.complex</em>) – Tensor of discrete complex-baseband channel impulse responses
- **resource_grid** (<a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>) – Resource grid
- **l_min** (<em>int</em>) – Smallest time-lag for the discrete complex baseband
channel impulse response ($L_{\text{min}}$)


Output
    
**h_f** (<em>[…,num_ofdm_symbols,fft_size], tf.complex</em>) – Tensor of discrete complex-baseband channel frequency responses



**Note**
    
Note that the result of this function is generally different from the
output of `cir_to_ofdm_channel()` because
the discrete complex-baseband channel impulse response is truncated
(see `cir_to_time_channel()`). This effect
can be observed in the example below.
<p class="rubric">Examples
```python
# Setup resource grid and channel model
tf.random.set_seed(4)
sm = StreamManagement(np.array([[1]]), 1)
rg = ResourceGrid(num_ofdm_symbols=1,
                  fft_size=1024,
                  subcarrier_spacing=15e3)
tdl = TDL("A", 100e-9, 3.5e9)
# Generate CIR
cir = tdl(batch_size=1, num_time_steps=1, sampling_frequency=rg.bandwidth)
# Generate OFDM channel from CIR
frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
h_freq = tf.squeeze(cir_to_ofdm_channel(frequencies, *cir, normalize=True))
# Generate time channel from CIR
l_min, l_max = time_lag_discrete_time_channel(rg.bandwidth)
h_time = cir_to_time_channel(rg.bandwidth, *cir, l_min=l_min, l_max=l_max, normalize=True)
# Generate OFDM channel from time channel
h_freq_hat = tf.squeeze(time_to_ofdm_channel(h_time, rg, l_min))
# Visualize results
plt.figure()
plt.plot(np.real(h_freq), "-")
plt.plot(np.real(h_freq_hat), "--")
plt.plot(np.imag(h_freq), "-")
plt.plot(np.imag(h_freq_hat), "--")
plt.xlabel("Subcarrier index")
plt.ylabel(r"Channel frequency response")
plt.legend(["OFDM Channel (real)", "OFDM Channel from time (real)", "OFDM Channel (imag)", "OFDM Channel from time (imag)"])
```

<img alt="../_images/time_to_ofdm_channel.png" src="https://nvlabs.github.io/sionna/_images/time_to_ofdm_channel.png" />

## Channel with OFDM waveform<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#channel-with-ofdm-waveform" title="Permalink to this headline"></a>
    
To implement the channel response assuming an OFDM waveform, it is assumed that
the power delay profiles are invariant over the duration of an OFDM symbol.
Moreover, it is assumed that the duration of the cyclic prefix (CP) equals at
least the maximum delay spread. These assumptions are common in the literature, as they
enable modeling of the channel transfer function in the frequency domain as a
single-tap channel.
    
For every link $(u, k, v, l)$ and resource element $(s,n)$,
the frequency channel response is obtained by computing the Fourier transform of
the channel response at the subcarrier frequencies, i.e.,

$$
\begin{split}\begin{align}
\widehat{h}_{u, k, v, l, s, n}
   &= \int_{-\infty}^{+\infty} h_{u, k, v, l}(s,\tau) e^{-j2\pi n \Delta_f \tau} d\tau\\
   &= \sum_{m=0}^{M-1} a_{u, k, v, l, m}(s)
   e^{-j2\pi n \Delta_f \tau_{u, k, v, l, m}}
\end{align}\end{split}
$$
    
where $s$ is used as time step to indicate that the channel response can
change from one OFDM symbol to the next in the event of mobility, even if it is
assumed static over the duration of an OFDM symbol.
    
For every receive antenna $l$ of every receiver $v$, the
received signal $y_{v, l, s, n}$ for resource element
$(s, n)$ is computed by

$$
y_{v, l, s, n} = \sum_{u=0}^{N_{T}-1}\sum_{k=0}^{N_{TA}-1}
   \widehat{h}_{u, k, v, l, s, n} x_{u, k, s, n}
   + w_{v, l, s, n}
$$
    
where $x_{u, k, s, n}$ is the baseband symbol transmitted by transmitter
$u$ on antenna $k$ and resource element $(s, n)$, and
$w_{v, l, s, n} \sim \mathcal{CN}\left(0,N_0\right)$ the additive white
Gaussian noise.

**Note**
    
This model does not account for intersymbol interference (ISI) nor
intercarrier interference (ICI). To model the ICI due to channel aging over
the duration of an OFDM symbol or the ISI due to a delay spread exceeding the
CP duration, one would need to simulate the channel in the time domain.
This can be achieved by using the <a class="reference internal" href="ofdm.html#sionna.ofdm.OFDMModulator" title="sionna.ofdm.OFDMModulator">`OFDMModulator`</a> and
<a class="reference internal" href="ofdm.html#sionna.ofdm.OFDMDemodulator" title="sionna.ofdm.OFDMDemodulator">`OFDMDemodulator`</a> layers, and the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#time-domain">time domain channel model</a>.
By doing so, one performs inverse discrete Fourier transform (IDFT) on
the transmitter side and discrete Fourier transform (DFT) on the receiver side
on top of a single-carrier sinc-shaped waveform.
This is equivalent to
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#ofdm-waveform">simulating the channel in the frequency domain</a> if no
ISI nor ICI is assumed, but allows the simulation of these effects in the
event of a non-stationary channel or long delay spreads.
Note that simulating the channel in the time domain is typically significantly
more computationally demanding that simulating the channel in the frequency
domain.

### OFDMChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#ofdmchannel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``OFDMChannel`(<em class="sig-param">`channel_model`</em>, <em class="sig-param">`resource_grid`</em>, <em class="sig-param">`add_awgn``=``True`</em>, <em class="sig-param">`normalize_channel``=``False`</em>, <em class="sig-param">`return_channel``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/channel/ofdm_channel.html#OFDMChannel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.OFDMChannel" title="Permalink to this definition"></a>
    
Generate channel frequency responses and apply them to channel inputs
assuming an OFDM waveform with no ICI nor ISI.
    
This class inherits from the Keras <cite>Layer</cite> class and can be used as layer
in a Keras model.
    
For each OFDM symbol $s$ and subcarrier $n$, the channel output is computed as follows:

$$
y_{s,n} = \widehat{h}_{s, n} x_{s,n} + w_{s,n}
$$
    
where $y_{s,n}$ is the channel output computed by this layer,
$\widehat{h}_{s, n}$ the frequency channel response,
$x_{s,n}$ the channel input `x`, and $w_{s,n}$ the additive noise.
    
For multiple-input multiple-output (MIMO) links, the channel output is computed for each antenna
of each receiver and by summing over all the antennas of all transmitters.
    
The channel frequency response for the $s^{th}$ OFDM symbol and
$n^{th}$ subcarrier is computed from a given channel impulse response
$(a_{m}(t), \tau_{m}), 0 \leq m \leq M-1$ generated by the `channel_model`
as follows:

$$
\widehat{h}_{s, n} = \sum_{m=0}^{M-1} a_{m}(s) e^{-j2\pi n \Delta_f \tau_{m}}
$$
    
where $\Delta_f$ is the subcarrier spacing, and $s$ is used as time
step to indicate that the channel impulse response can change from one OFDM symbol to the
next in the event of mobility, even if it is assumed static over the duration
of an OFDM symbol.
Parameters
 
- **channel_model** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ChannelModel" title="sionna.channel.ChannelModel">`ChannelModel`</a> object) – An instance of a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ChannelModel" title="sionna.channel.ChannelModel">`ChannelModel`</a> object, such as
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.RayleighBlockFading" title="sionna.channel.RayleighBlockFading">`RayleighBlockFading`</a> or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi" title="sionna.channel.tr38901.UMi">`UMi`</a>.
- **resource_grid** (<a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>) – Resource grid
- **add_awgn** (<em>bool</em>) – If set to <cite>False</cite>, no white Gaussian noise is added.
Defaults to <cite>True</cite>.
- **normalize_channel** (<em>bool</em>) – If set to <cite>True</cite>, the channel is normalized over the resource grid
to ensure unit average energy per resource element. Defaults to <cite>False</cite>.
- **return_channel** (<em>bool</em>) – If set to <cite>True</cite>, the channel response is returned in addition to the
channel output. Defaults to <cite>False</cite>.
- **dtype** (<em>tf.DType</em>) – Complex datatype to use for internal processing and output.
Defaults to tf.complex64.


Input
 
- **(x, no) or x** – Tuple or Tensor:
- **x** (<em>[batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Channel inputs
- **no** (<em>Scalar or Tensor, tf.float</em>) – Scalar or tensor whose shape can be broadcast to the shape of the
channel outputs:
[batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size].
Only required if `add_awgn` is set to <cite>True</cite>.
The noise power `no` is per complex dimension. If `no` is a scalar,
noise of the same variance will be added to the outputs.
If `no` is a tensor, it must have a shape that can be broadcast to
the shape of the channel outputs. This allows, e.g., adding noise of
different variance to each example in a batch. If `no` has a lower
rank than the channel outputs, then `no` will be broadcast to the
shape of the channel outputs by adding dummy dimensions after the last
axis.


Output
 
- **y** (<em>[batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Channel outputs
- **h_freq** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – (Optional) Channel frequency responses. Returned only if
`return_channel` is set to <cite>True</cite>.




### GenerateOFDMChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#generateofdmchannel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``GenerateOFDMChannel`(<em class="sig-param">`channel_model`</em>, <em class="sig-param">`resource_grid`</em>, <em class="sig-param">`normalize_channel``=``False`</em>)<a class="reference internal" href="../_modules/sionna/channel/generate_ofdm_channel.html#GenerateOFDMChannel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateOFDMChannel" title="Permalink to this definition"></a>
    
Generate channel frequency responses.
The channel impulse response is constant over the duration of an OFDM symbol.
    
Given a channel impulse response
$(a_{m}(t), \tau_{m}), 0 \leq m \leq M-1$, generated by the `channel_model`,
the channel frequency response for the $s^{th}$ OFDM symbol and
$n^{th}$ subcarrier is computed as follows:

$$
\widehat{h}_{s, n} = \sum_{m=0}^{M-1} a_{m}(s) e^{-j2\pi n \Delta_f \tau_{m}}
$$
    
where $\Delta_f$ is the subcarrier spacing, and $s$ is used as time
step to indicate that the channel impulse response can change from one OFDM symbol to the
next in the event of mobility, even if it is assumed static over the duration
of an OFDM symbol.
Parameters
 
- **channel_model** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ChannelModel" title="sionna.channel.ChannelModel">`ChannelModel`</a> object) – An instance of a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ChannelModel" title="sionna.channel.ChannelModel">`ChannelModel`</a> object, such as
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.RayleighBlockFading" title="sionna.channel.RayleighBlockFading">`RayleighBlockFading`</a> or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi" title="sionna.channel.tr38901.UMi">`UMi`</a>.
- **resource_grid** (<a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>) – Resource grid
- **normalize_channel** (<em>bool</em>) – If set to <cite>True</cite>, the channel is normalized over the resource grid
to ensure unit average energy per resource element. Defaults to <cite>False</cite>.
- **dtype** (<em>tf.DType</em>) – Complex datatype to use for internal processing and output.
Defaults to <cite>tf.complex64</cite>.


Input
    
**batch_size** (<em>int</em>) – Batch size. Defaults to <cite>None</cite> for channel models that do not require this paranmeter.

Output
    
**h_freq** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers], tf.complex</em>) – Channel frequency responses



### ApplyOFDMChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#applyofdmchannel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``ApplyOFDMChannel`(<em class="sig-param">`add_awgn``=``True`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/channel/apply_ofdm_channel.html#ApplyOFDMChannel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyOFDMChannel" title="Permalink to this definition"></a>
    
Apply single-tap channel frequency responses to channel inputs.
    
This class inherits from the Keras <cite>Layer</cite> class and can be used as layer
in a Keras model.
    
For each OFDM symbol $s$ and subcarrier $n$, the single-tap channel
is applied as follows:

$$
y_{s,n} = \widehat{h}_{s, n} x_{s,n} + w_{s,n}
$$
    
where $y_{s,n}$ is the channel output computed by this layer,
$\widehat{h}_{s, n}$ the frequency channel response (`h_freq`),
$x_{s,n}$ the channel input `x`, and $w_{s,n}$ the additive noise.
    
For multiple-input multiple-output (MIMO) links, the channel output is computed for each antenna
of each receiver and by summing over all the antennas of all transmitters.
Parameters
 
- **add_awgn** (<em>bool</em>) – If set to <cite>False</cite>, no white Gaussian noise is added.
Defaults to <cite>True</cite>.
- **dtype** (<em>tf.DType</em>) – Complex datatype to use for internal processing and output. Defaults to
<cite>tf.complex64</cite>.


Input
 
- **(x, h_freq, no) or (x, h_freq)** – Tuple:
- **x** (<em>[batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Channel inputs
- **h_freq** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Channel frequency responses
- **no** (<em>Scalar or Tensor, tf.float</em>) – Scalar or tensor whose shape can be broadcast to the shape of the
channel outputs:
[batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size].
Only required if `add_awgn` is set to <cite>True</cite>.
The noise power `no` is per complex dimension. If `no` is a
scalar, noise of the same variance will be added to the outputs.
If `no` is a tensor, it must have a shape that can be broadcast to
the shape of the channel outputs. This allows, e.g., adding noise of
different variance to each example in a batch. If `no` has a lower
rank than the channel outputs, then `no` will be broadcast to the
shape of the channel outputs by adding dummy dimensions after the
last axis.


Output
    
**y** (<em>[batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Channel outputs



### cir_to_ofdm_channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#cir-to-ofdm-channel" title="Permalink to this headline"></a>

`sionna.channel.``cir_to_ofdm_channel`(<em class="sig-param">`frequencies`</em>, <em class="sig-param">`a`</em>, <em class="sig-param">`tau`</em>, <em class="sig-param">`normalize``=``False`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#cir_to_ofdm_channel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.cir_to_ofdm_channel" title="Permalink to this definition"></a>
    
Compute the frequency response of the channel at `frequencies`.
    
Given a channel impulse response
$(a_{m}, \tau_{m}), 0 \leq m \leq M-1$ (inputs `a` and `tau`),
the channel frequency response for the frequency $f$
is computed as follows:

$$
\widehat{h}(f) = \sum_{m=0}^{M-1} a_{m} e^{-j2\pi f \tau_{m}}
$$

Input
 
- **frequencies** (<em>[fft_size], tf.float</em>) – Frequencies at which to compute the channel response
- **a** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex</em>) – Path coefficients
- **tau** (<em>[batch size, num_rx, num_tx, num_paths] or [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths], tf.float</em>) – Path delays
- **normalize** (<em>bool</em>) – If set to <cite>True</cite>, the channel is normalized over the resource grid
to ensure unit average energy per resource element. Defaults to <cite>False</cite>.


Output
    
**h_f** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size], tf.complex</em>) – Channel frequency responses at `frequencies`



## Rayleigh block fading<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#rayleigh-block-fading" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``RayleighBlockFading`(<em class="sig-param">`num_rx`</em>, <em class="sig-param">`num_rx_ant`</em>, <em class="sig-param">`num_tx`</em>, <em class="sig-param">`num_tx_ant`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/rayleigh_block_fading.html#RayleighBlockFading">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.RayleighBlockFading" title="Permalink to this definition"></a>
    
Generate channel impulse responses corresponding to a Rayleigh block
fading channel model.
    
The channel impulse responses generated are formed of a single path with
zero delay and a normally distributed fading coefficient.
All time steps of a batch example share the same channel coefficient
(block fading).
    
This class can be used in conjunction with the classes that simulate the
channel response in time or frequency domain, i.e.,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.OFDMChannel" title="sionna.channel.OFDMChannel">`OFDMChannel`</a>,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.TimeChannel" title="sionna.channel.TimeChannel">`TimeChannel`</a>,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateOFDMChannel" title="sionna.channel.GenerateOFDMChannel">`GenerateOFDMChannel`</a>,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyOFDMChannel" title="sionna.channel.ApplyOFDMChannel">`ApplyOFDMChannel`</a>,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateTimeChannel" title="sionna.channel.GenerateTimeChannel">`GenerateTimeChannel`</a>,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyTimeChannel" title="sionna.channel.ApplyTimeChannel">`ApplyTimeChannel`</a>.
Parameters
 
- **num_rx** (<em>int</em>) – Number of receivers ($N_R$)
- **num_rx_ant** (<em>int</em>) – Number of antennas per receiver ($N_{RA}$)
- **num_tx** (<em>int</em>) – Number of transmitters ($N_T$)
- **num_tx_ant** (<em>int</em>) – Number of antennas per transmitter ($N_{TA}$)
- **dtype** (<em>tf.DType</em>) – Complex datatype to use for internal processing and output.
Defaults to <cite>tf.complex64</cite>.


Input
 
- **batch_size** (<em>int</em>) – Batch size
- **num_time_steps** (<em>int</em>) – Number of time steps


Output
 
- **a** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths = 1, num_time_steps], tf.complex</em>) – Path coefficients
- **tau** (<em>[batch size, num_rx, num_tx, num_paths = 1], tf.float</em>) – Path delays [s]




## 3GPP 38.901 channel models<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#gpp-38-901-channel-models" title="Permalink to this headline"></a>
    
The submodule `tr38901` implements 3GPP channel models from <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id4">[TR38901]</a>.
    
The <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#cdl">CDL</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#umi">UMi</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#uma">UMa</a>, and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#rma">RMa</a>
models require setting-up antenna models for the transmitters and
receivers. This is achieved using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="sionna.channel.tr38901.PanelArray">`PanelArray`</a> class.
    
The <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#umi">UMi</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#uma">UMa</a>, and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#rma">RMa</a> models require
setting-up a network topology, specifying, e.g., the user terminals (UTs) and
base stations (BSs) locations, the UTs velocities, etc.
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#utility-functions">Utility functions</a> are available to help laying out
complex topologies or to quickly setup simple but widely used topologies.

### PanelArray<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#panelarray" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.tr38901.``PanelArray`(<em class="sig-param">`num_rows_per_panel`</em>, <em class="sig-param">`num_cols_per_panel`</em>, <em class="sig-param">`polarization`</em>, <em class="sig-param">`polarization_type`</em>, <em class="sig-param">`antenna_pattern`</em>, <em class="sig-param">`carrier_frequency`</em>, <em class="sig-param">`num_rows``=``1`</em>, <em class="sig-param">`num_cols``=``1`</em>, <em class="sig-param">`panel_vertical_spacing``=``None`</em>, <em class="sig-param">`panel_horizontal_spacing``=``None`</em>, <em class="sig-param">`element_vertical_spacing``=``None`</em>, <em class="sig-param">`element_horizontal_spacing``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/tr38901/antenna.html#PanelArray">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="Permalink to this definition"></a>
    
Antenna panel array following the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id5">[TR38901]</a> specification.
    
This class is used to create models of the panel arrays used by the
transmitters and receivers and that need to be specified when using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#cdl">CDL</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#umi">UMi</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#uma">UMa</a>, and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#rma">RMa</a>
models.
<p class="rubric">Example
```python
>>> array = PanelArray(num_rows_per_panel = 4,
...                    num_cols_per_panel = 4,
...                    polarization = 'dual',
...                    polarization_type = 'VH',
...                    antenna_pattern = '38.901',
...                    carrier_frequency = 3.5e9,
...                    num_cols = 2,
...                    panel_horizontal_spacing = 3.)
>>> array.show()
```

<img alt="../_images/panel_array.png" src="https://nvlabs.github.io/sionna/_images/panel_array.png" />
Parameters
 
- **num_rows_per_panel** (<em>int</em>) – Number of rows of elements per panel
- **num_cols_per_panel** (<em>int</em>) – Number of columns of elements per panel
- **polarization** (<em>str</em>) – Polarization, either “single” or “dual”
- **polarization_type** (<em>str</em>) – Type of polarization. For single polarization, must be “V” or “H”.
For dual polarization, must be “VH” or “cross”.
- **antenna_pattern** (<em>str</em>) – Element radiation pattern, either “omni” or “38.901”
- **carrier_frequency** (<em>float</em>) – Carrier frequency [Hz]
- **num_rows** (<em>int</em>) – Number of rows of panels. Defaults to 1.
- **num_cols** (<em>int</em>) – Number of columns of panels. Defaults to 1.
- **panel_vertical_spacing** (<cite>None</cite> or float) – Vertical spacing of panels [multiples of wavelength].
Must be greater than the panel width.
If set to <cite>None</cite> (default value), it is set to the panel width + 0.5.
- **panel_horizontal_spacing** (<cite>None</cite> or float) – Horizontal spacing of panels [in multiples of wavelength].
Must be greater than the panel height.
If set to <cite>None</cite> (default value), it is set to the panel height + 0.5.
- **element_vertical_spacing** (<cite>None</cite> or float) – Element vertical spacing [multiple of wavelength].
Defaults to 0.5 if set to <cite>None</cite>.
- **element_horizontal_spacing** (<cite>None</cite> or float) – Element horizontal spacing [multiple of wavelength].
Defaults to 0.5 if set to <cite>None</cite>.
- **dtype** (<em>Complex tf.DType</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.




<em class="property">`property` </em>`ant_ind_pol1`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.ant_ind_pol1" title="Permalink to this definition"></a>
    
Indices of antenna elements with the first polarization direction


<em class="property">`property` </em>`ant_ind_pol2`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.ant_ind_pol2" title="Permalink to this definition"></a>
    
Indices of antenna elements with the second polarization direction.
Only defined with dual polarization.


<em class="property">`property` </em>`ant_pol1`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.ant_pol1" title="Permalink to this definition"></a>
    
Field of an antenna element with the first polarization direction


<em class="property">`property` </em>`ant_pol2`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.ant_pol2" title="Permalink to this definition"></a>
    
Field of an antenna element with the second polarization direction.
Only defined with dual polarization.


<em class="property">`property` </em>`ant_pos`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.ant_pos" title="Permalink to this definition"></a>
    
Positions of the antennas


<em class="property">`property` </em>`ant_pos_pol1`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.ant_pos_pol1" title="Permalink to this definition"></a>
    
Positions of the antenna elements with the first polarization
direction


<em class="property">`property` </em>`ant_pos_pol2`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.ant_pos_pol2" title="Permalink to this definition"></a>
    
Positions of antenna elements with the second polarization direction.
Only defined with dual polarization.


<em class="property">`property` </em>`element_horizontal_spacing`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.element_horizontal_spacing" title="Permalink to this definition"></a>
    
Horizontal spacing between the antenna elements within a panel
[multiple of wavelength]


<em class="property">`property` </em>`element_vertical_spacing`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.element_vertical_spacing" title="Permalink to this definition"></a>
    
Vertical spacing between the antenna elements within a panel
[multiple of wavelength]


<em class="property">`property` </em>`num_ant`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.num_ant" title="Permalink to this definition"></a>
    
Total number of antenna elements


<em class="property">`property` </em>`num_cols`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.num_cols" title="Permalink to this definition"></a>
    
Number of columns of panels


<em class="property">`property` </em>`num_cols_per_panel`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.num_cols_per_panel" title="Permalink to this definition"></a>
    
Number of columns of elements per panel


<em class="property">`property` </em>`num_panels`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.num_panels" title="Permalink to this definition"></a>
    
Number of panels


<em class="property">`property` </em>`num_panels_ant`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.num_panels_ant" title="Permalink to this definition"></a>
    
Number of antenna elements per panel


<em class="property">`property` </em>`num_rows`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.num_rows" title="Permalink to this definition"></a>
    
Number of rows of panels


<em class="property">`property` </em>`num_rows_per_panel`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.num_rows_per_panel" title="Permalink to this definition"></a>
    
Number of rows of elements per panel


<em class="property">`property` </em>`panel_horizontal_spacing`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.panel_horizontal_spacing" title="Permalink to this definition"></a>
    
Horizontal spacing between the panels [multiple of wavelength]


<em class="property">`property` </em>`panel_vertical_spacing`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.panel_vertical_spacing" title="Permalink to this definition"></a>
    
Vertical spacing between the panels [multiple of wavelength]


<em class="property">`property` </em>`polarization`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.polarization" title="Permalink to this definition"></a>
    
Polarization (“single” or “dual”)


<em class="property">`property` </em>`polarization_type`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.polarization_type" title="Permalink to this definition"></a>
    
Polarization type. “V” or “H” for single polarization.
“VH” or “cross” for dual polarization.


`show`()<a class="reference internal" href="../_modules/sionna/channel/tr38901/antenna.html#PanelArray.show">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.show" title="Permalink to this definition"></a>
    
Show the panel array geometry


`show_element_radiation_pattern`()<a class="reference internal" href="../_modules/sionna/channel/tr38901/antenna.html#PanelArray.show_element_radiation_pattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.show_element_radiation_pattern" title="Permalink to this definition"></a>
    
Show the radiation field of antenna elements forming the panel


### Antenna<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#antenna" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.tr38901.``Antenna`(<em class="sig-param">`polarization`</em>, <em class="sig-param">`polarization_type`</em>, <em class="sig-param">`antenna_pattern`</em>, <em class="sig-param">`carrier_frequency`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/tr38901/antenna.html#Antenna">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.Antenna" title="Permalink to this definition"></a>
    
Single antenna following the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id6">[TR38901]</a> specification.
    
This class is a special case of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="sionna.channel.tr38901.PanelArray">`PanelArray`</a>,
and can be used in lieu of it.
Parameters
 
- **polarization** (<em>str</em>) – Polarization, either “single” or “dual”
- **polarization_type** (<em>str</em>) – Type of polarization. For single polarization, must be “V” or “H”.
For dual polarization, must be “VH” or “cross”.
- **antenna_pattern** (<em>str</em>) – Element radiation pattern, either “omni” or “38.901”
- **carrier_frequency** (<em>float</em>) – Carrier frequency [Hz]
- **dtype** (<em>Complex tf.DType</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.




### AntennaArray<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#antennaarray" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.tr38901.``AntennaArray`(<em class="sig-param">`num_rows`</em>, <em class="sig-param">`num_cols`</em>, <em class="sig-param">`polarization`</em>, <em class="sig-param">`polarization_type`</em>, <em class="sig-param">`antenna_pattern`</em>, <em class="sig-param">`carrier_frequency`</em>, <em class="sig-param">`vertical_spacing`</em>, <em class="sig-param">`horizontal_spacing`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/tr38901/antenna.html#AntennaArray">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.AntennaArray" title="Permalink to this definition"></a>
    
Antenna array following the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id7">[TR38901]</a> specification.
    
This class is a special case of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="sionna.channel.tr38901.PanelArray">`PanelArray`</a>,
and can used in lieu of it.
Parameters
 
- **num_rows** (<em>int</em>) – Number of rows of elements
- **num_cols** (<em>int</em>) – Number of columns of elements
- **polarization** (<em>str</em>) – Polarization, either “single” or “dual”
- **polarization_type** (<em>str</em>) – Type of polarization. For single polarization, must be “V” or “H”.
For dual polarization, must be “VH” or “cross”.
- **antenna_pattern** (<em>str</em>) – Element radiation pattern, either “omni” or “38.901”
- **carrier_frequency** (<em>float</em>) – Carrier frequency [Hz]
- **vertical_spacing** (<cite>None</cite> or float) – Element vertical spacing [multiple of wavelength].
Defaults to 0.5 if set to <cite>None</cite>.
- **horizontal_spacing** (<cite>None</cite> or float) – Element horizontal spacing [multiple of wavelength].
Defaults to 0.5 if set to <cite>None</cite>.
- **dtype** (<em>Complex tf.DType</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.




### Tapped delay line (TDL)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tapped-delay-line-tdl" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.tr38901.``TDL`(<em class="sig-param">`model`</em>, <em class="sig-param">`delay_spread`</em>, <em class="sig-param">`carrier_frequency`</em>, <em class="sig-param">`num_sinusoids``=``20`</em>, <em class="sig-param">`los_angle_of_arrival``=``PI` `/` `4.`</em>, <em class="sig-param">`min_speed``=``0.`</em>, <em class="sig-param">`max_speed``=``None`</em>, <em class="sig-param">`num_rx_ant``=``1`</em>, <em class="sig-param">`num_tx_ant``=``1`</em>, <em class="sig-param">`spatial_corr_mat``=``None`</em>, <em class="sig-param">`rx_corr_mat``=``None`</em>, <em class="sig-param">`tx_corr_mat``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/tr38901/tdl.html#TDL">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.TDL" title="Permalink to this definition"></a>
    
Tapped delay line (TDL) channel model from the 3GPP <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id8">[TR38901]</a> specification.
    
The power delay profiles (PDPs) are normalized to have a total energy of one.
    
Channel coefficients are generated using a sum-of-sinusoids model <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sos" id="id9">[SoS]</a>.
Channel aging is simulated in the event of mobility.
    
If a minimum speed and a maximum speed are specified such that the
maximum speed is greater than the minimum speed, then speeds are randomly
and uniformly sampled from the specified interval for each link and each
batch example.
    
The TDL model only works for systems with a single transmitter and a single
receiver. The transmitter and receiver can be equipped with multiple
antennas. Spatial correlation is simulated through filtering by specified
correlation matrices.
    
The `spatial_corr_mat` parameter can be used to specify an arbitrary
spatial correlation matrix. In particular, it can be used to model
correlated cross-polarized transmit and receive antennas as follows
(see, e.g., Annex G.2.3.2.1 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#ts38141-1" id="id10">[TS38141-1]</a>):

$$
\mathbf{R} = \mathbf{R}_{\text{rx}} \otimes \mathbf{\Gamma} \otimes \mathbf{R}_{\text{tx}}
$$
    
where $\mathbf{R}$ is the spatial correlation matrix `spatial_corr_mat`,
$\mathbf{R}_{\text{rx}}$ the spatial correlation matrix at the receiver
with same polarization, $\mathbf{R}_{\text{tx}}$ the spatial correlation
matrix at the transmitter with same polarization, and $\mathbf{\Gamma}$
the polarization correlation matrix. $\mathbf{\Gamma}$ is 1x1 for single-polarized
antennas, 2x2 when only the transmit or receive antennas are cross-polarized, and 4x4 when
transmit and receive antennas are cross-polarized.
    
It is also possible not to specify `spatial_corr_mat`, but instead the correlation matrices
at the receiver and transmitter, using the `rx_corr_mat` and `tx_corr_mat`
parameters, respectively.
This can be useful when single polarized antennas are simulated, and it is also
more computationally efficient.
This is equivalent to setting `spatial_corr_mat` to :

$$
\mathbf{R} = \mathbf{R}_{\text{rx}} \otimes \mathbf{R}_{\text{tx}}
$$
    
where $\mathbf{R}_{\text{rx}}$ is the correlation matrix at the receiver
`rx_corr_mat` and  $\mathbf{R}_{\text{tx}}$ the correlation matrix at
the transmitter `tx_corr_mat`.
<p class="rubric">Example
    
The following code snippet shows how to setup a TDL channel model assuming
an OFDM waveform:
```python
>>> tdl = TDL(model = "A",
...           delay_spread = 300e-9,
...           carrier_frequency = 3.5e9,
...           min_speed = 0.0,
...           max_speed = 3.0)
>>>
>>> channel = OFDMChannel(channel_model = tdl,
...                       resource_grid = rg)
```

    
where `rg` is an instance of <a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
<p class="rubric">Notes
    
The following tables from <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id11">[TR38901]</a> provide typical values for the delay
spread.
<table class="docutils align-default">
<colgroup>
<col style="width: 58%" />
<col style="width: 42%" />
</colgroup>
<thead>
<tr class="row-odd"><th class="head">    
Model</th>
<th class="head">    
Delay spread [ns]</th>
</tr>
</thead>
<tbody>
<tr class="row-even"><td>    
Very short delay spread</td>
<td>    
$10$</td>
</tr>
<tr class="row-odd"><td>    
Short short delay spread</td>
<td>    
$10$</td>
</tr>
<tr class="row-even"><td>    
Nominal delay spread</td>
<td>    
$100$</td>
</tr>
<tr class="row-odd"><td>    
Long delay spread</td>
<td>    
$300$</td>
</tr>
<tr class="row-even"><td>    
Very long delay spread</td>
<td>    
$1000$</td>
</tr>
</tbody>
</table>
<table class="docutils align-default">
<colgroup>
<col style="width: 30%" />
<col style="width: 27%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 5%" />
<col style="width: 6%" />
<col style="width: 6%" />
<col style="width: 5%" />
<col style="width: 6%" />
</colgroup>
<thead>
<tr class="row-odd"><th class="head" colspan="2" rowspan="2">    
Delay spread [ns]</th>
<th class="head" colspan="7">    
Frequency [GHz]</th>
</tr>
<tr class="row-even"><th class="head">    
2</th>
<th class="head">    
6</th>
<th class="head">    
15</th>
<th class="head">    
28</th>
<th class="head">    
39</th>
<th class="head">    
60</th>
<th class="head">    
70</th>
</tr>
</thead>
<tbody>
<tr class="row-odd"><td rowspan="3">    
Indoor office</td>
<td>    
Short delay profile</td>
<td>    
20</td>
<td>    
16</td>
<td>    
16</td>
<td>    
16</td>
<td>    
16</td>
<td>    
16</td>
<td>    
16</td>
</tr>
<tr class="row-even"><td>    
Normal delay profile</td>
<td>    
39</td>
<td>    
30</td>
<td>    
24</td>
<td>    
20</td>
<td>    
18</td>
<td>    
16</td>
<td>    
16</td>
</tr>
<tr class="row-odd"><td>    
Long delay profile</td>
<td>    
59</td>
<td>    
53</td>
<td>    
47</td>
<td>    
43</td>
<td>    
41</td>
<td>    
38</td>
<td>    
37</td>
</tr>
<tr class="row-even"><td rowspan="3">    
UMi Street-canyon</td>
<td>    
Short delay profile</td>
<td>    
65</td>
<td>    
45</td>
<td>    
37</td>
<td>    
32</td>
<td>    
30</td>
<td>    
27</td>
<td>    
26</td>
</tr>
<tr class="row-odd"><td>    
Normal delay profile</td>
<td>    
129</td>
<td>    
93</td>
<td>    
76</td>
<td>    
66</td>
<td>    
61</td>
<td>    
55</td>
<td>    
53</td>
</tr>
<tr class="row-even"><td>    
Long delay profile</td>
<td>    
634</td>
<td>    
316</td>
<td>    
307</td>
<td>    
301</td>
<td>    
297</td>
<td>    
293</td>
<td>    
291</td>
</tr>
<tr class="row-odd"><td rowspan="3">    
UMa</td>
<td>    
Short delay profile</td>
<td>    
93</td>
<td>    
93</td>
<td>    
85</td>
<td>    
80</td>
<td>    
78</td>
<td>    
75</td>
<td>    
74</td>
</tr>
<tr class="row-even"><td>    
Normal delay profile</td>
<td>    
363</td>
<td>    
363</td>
<td>    
302</td>
<td>    
266</td>
<td>    
249</td>
<td>    
228</td>
<td>    
221</td>
</tr>
<tr class="row-odd"><td>    
Long delay profile</td>
<td>    
1148</td>
<td>    
1148</td>
<td>    
955</td>
<td>    
841</td>
<td>    
786</td>
<td>    
720</td>
<td>    
698</td>
</tr>
<tr class="row-even"><td rowspan="3">    
RMa / RMa O2I</td>
<td>    
Short delay profile</td>
<td>    
32</td>
<td>    
32</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
</tr>
<tr class="row-odd"><td>    
Normal delay profile</td>
<td>    
37</td>
<td>    
37</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
</tr>
<tr class="row-even"><td>    
Long delay profile</td>
<td>    
153</td>
<td>    
153</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
</tr>
<tr class="row-odd"><td rowspan="2">    
UMi / UMa O2I</td>
<td>    
Normal delay profile</td>
<td colspan="7">    
242</td>
</tr>
<tr class="row-even"><td>    
Long delay profile</td>
<td colspan="7">    
616</td>
</tr>
</tbody>
</table>
Parameters
 
- **model** (<em>str</em>) – TDL model to use. Must be one of “A”, “B”, “C”, “D”, “E”, “A30”, “B100”, or “C300”.
- **delay_spread** (<em>float</em>) – RMS delay spread [s].
For the “A30”, “B100”, and “C300” models, the delay spread must be set
to 30ns, 100ns, and 300ns, respectively.
- **carrier_frequency** (<em>float</em>) – Carrier frequency [Hz]
- **num_sinusoids** (<em>int</em>) – Number of sinusoids for the sum-of-sinusoids model. Defaults to 20.
- **los_angle_of_arrival** (<em>float</em>) – Angle-of-arrival for LoS path [radian]. Only used with LoS models.
Defaults to $\pi/4$.
- **min_speed** (<em>float</em>) – Minimum speed [m/s]. Defaults to 0.
- **max_speed** (<em>None</em><em> or </em><em>float</em>) – Maximum speed [m/s]. If set to <cite>None</cite>,
then `max_speed` takes the same value as `min_speed`.
Defaults to <cite>None</cite>.
- **num_rx_ant** (<em>int</em>) – Number of receive antennas.
Defaults to 1.
- **num_tx_ant** (<em>int</em>) – Number of transmit antennas.
Defaults to 1.
- **spatial_corr_mat** ([num_rx_ant*num_tx_ant,num_rx_ant*num_tx_ant], tf.complex or <cite>None</cite>) – Spatial correlation matrix.
If not set to <cite>None</cite>, then `rx_corr_mat` and `tx_corr_mat` are ignored and
this matrix is used for spatial correlation.
If set to <cite>None</cite> and `rx_corr_mat` and `tx_corr_mat` are also set to <cite>None</cite>,
then no correlation is applied.
Defaults to <cite>None</cite>.
- **rx_corr_mat** ([num_rx_ant,num_rx_ant], tf.complex or <cite>None</cite>) – Spatial correlation matrix for the receiver.
If set to <cite>None</cite> and `spatial_corr_mat` is also set to <cite>None</cite>, then no receive
correlation is applied.
Defaults to <cite>None</cite>.
- **tx_corr_mat** ([num_tx_ant,num_tx_ant], tf.complex or <cite>None</cite>) – Spatial correlation matrix for the transmitter.
If set to <cite>None</cite> and `spatial_corr_mat` is also set to <cite>None</cite>, then no transmit
correlation is applied.
Defaults to <cite>None</cite>.
- **dtype** (<em>Complex tf.DType</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.


Input
 
- **batch_size** (<em>int</em>) – Batch size
- **num_time_steps** (<em>int</em>) – Number of time steps
- **sampling_frequency** (<em>float</em>) – Sampling frequency [Hz]


Output
 
- **a** (<em>[batch size, num_rx = 1, num_rx_ant = 1, num_tx = 1, num_tx_ant = 1, num_paths, num_time_steps], tf.complex</em>) – Path coefficients
- **tau** (<em>[batch size, num_rx = 1, num_tx = 1, num_paths], tf.float</em>) – Path delays [s]




<em class="property">`property` </em>`delay_spread`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.TDL.delay_spread" title="Permalink to this definition"></a>
    
RMS delay spread [s]


<em class="property">`property` </em>`delays`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.TDL.delays" title="Permalink to this definition"></a>
    
Path delays [s]


<em class="property">`property` </em>`k_factor`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.TDL.k_factor" title="Permalink to this definition"></a>
    
K-factor in linear scale. Only available with LoS models.


<em class="property">`property` </em>`los`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.TDL.los" title="Permalink to this definition"></a>
    
<cite>True</cite> if this is a LoS model. <cite>False</cite> otherwise.


<em class="property">`property` </em>`mean_power_los`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.TDL.mean_power_los" title="Permalink to this definition"></a>
    
LoS component power in linear scale.
Only available with LoS models.


<em class="property">`property` </em>`mean_powers`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.TDL.mean_powers" title="Permalink to this definition"></a>
    
Path powers in linear scale


<em class="property">`property` </em>`num_clusters`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.TDL.num_clusters" title="Permalink to this definition"></a>
    
Number of paths ($M$)


### Clustered delay line (CDL)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#clustered-delay-line-cdl" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.tr38901.``CDL`(<em class="sig-param">`model`</em>, <em class="sig-param">`delay_spread`</em>, <em class="sig-param">`carrier_frequency`</em>, <em class="sig-param">`ut_array`</em>, <em class="sig-param">`bs_array`</em>, <em class="sig-param">`direction`</em>, <em class="sig-param">`min_speed``=``0.`</em>, <em class="sig-param">`max_speed``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/tr38901/cdl.html#CDL">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.CDL" title="Permalink to this definition"></a>
    
Clustered delay line (CDL) channel model from the 3GPP <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id12">[TR38901]</a> specification.
    
The power delay profiles (PDPs) are normalized to have a total energy of one.
    
If a minimum speed and a maximum speed are specified such that the
maximum speed is greater than the minimum speed, then UTs speeds are
randomly and uniformly sampled from the specified interval for each link
and each batch example.
    
The CDL model only works for systems with a single transmitter and a single
receiver. The transmitter and receiver can be equipped with multiple
antennas.
<p class="rubric">Example
    
The following code snippet shows how to setup a CDL channel model assuming
an OFDM waveform:
```python
>>> # Panel array configuration for the transmitter and receiver
>>> bs_array = PanelArray(num_rows_per_panel = 4,
...                       num_cols_per_panel = 4,
...                       polarization = 'dual',
...                       polarization_type = 'cross',
...                       antenna_pattern = '38.901',
...                       carrier_frequency = 3.5e9)
>>> ut_array = PanelArray(num_rows_per_panel = 1,
...                       num_cols_per_panel = 1,
...                       polarization = 'single',
...                       polarization_type = 'V',
...                       antenna_pattern = 'omni',
...                       carrier_frequency = 3.5e9)
>>> # CDL channel model
>>> cdl = CDL(model = "A",
>>>           delay_spread = 300e-9,
...           carrier_frequency = 3.5e9,
...           ut_array = ut_array,
...           bs_array = bs_array,
...           direction = 'uplink')
>>> channel = OFDMChannel(channel_model = cdl,
...                       resource_grid = rg)
```

    
where `rg` is an instance of <a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
<p class="rubric">Notes
    
The following tables from <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id13">[TR38901]</a> provide typical values for the delay
spread.
<table class="docutils align-default">
<colgroup>
<col style="width: 58%" />
<col style="width: 42%" />
</colgroup>
<thead>
<tr class="row-odd"><th class="head">    
Model</th>
<th class="head">    
Delay spread [ns]</th>
</tr>
</thead>
<tbody>
<tr class="row-even"><td>    
Very short delay spread</td>
<td>    
$10$</td>
</tr>
<tr class="row-odd"><td>    
Short short delay spread</td>
<td>    
$10$</td>
</tr>
<tr class="row-even"><td>    
Nominal delay spread</td>
<td>    
$100$</td>
</tr>
<tr class="row-odd"><td>    
Long delay spread</td>
<td>    
$300$</td>
</tr>
<tr class="row-even"><td>    
Very long delay spread</td>
<td>    
$1000$</td>
</tr>
</tbody>
</table>
<table class="docutils align-default">
<colgroup>
<col style="width: 30%" />
<col style="width: 27%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 5%" />
<col style="width: 6%" />
<col style="width: 6%" />
<col style="width: 5%" />
<col style="width: 6%" />
</colgroup>
<thead>
<tr class="row-odd"><th class="head" colspan="2" rowspan="2">    
Delay spread [ns]</th>
<th class="head" colspan="7">    
Frequency [GHz]</th>
</tr>
<tr class="row-even"><th class="head">    
2</th>
<th class="head">    
6</th>
<th class="head">    
15</th>
<th class="head">    
28</th>
<th class="head">    
39</th>
<th class="head">    
60</th>
<th class="head">    
70</th>
</tr>
</thead>
<tbody>
<tr class="row-odd"><td rowspan="3">    
Indoor office</td>
<td>    
Short delay profile</td>
<td>    
20</td>
<td>    
16</td>
<td>    
16</td>
<td>    
16</td>
<td>    
16</td>
<td>    
16</td>
<td>    
16</td>
</tr>
<tr class="row-even"><td>    
Normal delay profile</td>
<td>    
39</td>
<td>    
30</td>
<td>    
24</td>
<td>    
20</td>
<td>    
18</td>
<td>    
16</td>
<td>    
16</td>
</tr>
<tr class="row-odd"><td>    
Long delay profile</td>
<td>    
59</td>
<td>    
53</td>
<td>    
47</td>
<td>    
43</td>
<td>    
41</td>
<td>    
38</td>
<td>    
37</td>
</tr>
<tr class="row-even"><td rowspan="3">    
UMi Street-canyon</td>
<td>    
Short delay profile</td>
<td>    
65</td>
<td>    
45</td>
<td>    
37</td>
<td>    
32</td>
<td>    
30</td>
<td>    
27</td>
<td>    
26</td>
</tr>
<tr class="row-odd"><td>    
Normal delay profile</td>
<td>    
129</td>
<td>    
93</td>
<td>    
76</td>
<td>    
66</td>
<td>    
61</td>
<td>    
55</td>
<td>    
53</td>
</tr>
<tr class="row-even"><td>    
Long delay profile</td>
<td>    
634</td>
<td>    
316</td>
<td>    
307</td>
<td>    
301</td>
<td>    
297</td>
<td>    
293</td>
<td>    
291</td>
</tr>
<tr class="row-odd"><td rowspan="3">    
UMa</td>
<td>    
Short delay profile</td>
<td>    
93</td>
<td>    
93</td>
<td>    
85</td>
<td>    
80</td>
<td>    
78</td>
<td>    
75</td>
<td>    
74</td>
</tr>
<tr class="row-even"><td>    
Normal delay profile</td>
<td>    
363</td>
<td>    
363</td>
<td>    
302</td>
<td>    
266</td>
<td>    
249</td>
<td>    
228</td>
<td>    
221</td>
</tr>
<tr class="row-odd"><td>    
Long delay profile</td>
<td>    
1148</td>
<td>    
1148</td>
<td>    
955</td>
<td>    
841</td>
<td>    
786</td>
<td>    
720</td>
<td>    
698</td>
</tr>
<tr class="row-even"><td rowspan="3">    
RMa / RMa O2I</td>
<td>    
Short delay profile</td>
<td>    
32</td>
<td>    
32</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
</tr>
<tr class="row-odd"><td>    
Normal delay profile</td>
<td>    
37</td>
<td>    
37</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
</tr>
<tr class="row-even"><td>    
Long delay profile</td>
<td>    
153</td>
<td>    
153</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
</tr>
<tr class="row-odd"><td rowspan="2">    
UMi / UMa O2I</td>
<td>    
Normal delay profile</td>
<td colspan="7">    
242</td>
</tr>
<tr class="row-even"><td>    
Long delay profile</td>
<td colspan="7">    
616</td>
</tr>
</tbody>
</table>
Parameters
 
- **model** (<em>str</em>) – CDL model to use. Must be one of “A”, “B”, “C”, “D” or “E”.
- **delay_spread** (<em>float</em>) – RMS delay spread [s].
- **carrier_frequency** (<em>float</em>) – Carrier frequency [Hz].
- **ut_array** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="sionna.channel.tr38901.PanelArray"><em>PanelArray</em></a>) – Panel array used by the UTs. All UTs share the same antenna array
configuration.
- **bs_array** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="sionna.channel.tr38901.PanelArray"><em>PanelArray</em></a>) – Panel array used by the Bs. All BSs share the same antenna array
configuration.
- **direction** (<em>str</em>) – Link direction. Must be either “uplink” or “downlink”.
- **ut_orientation** (<cite>None</cite> or Tensor of shape [3], tf.float) – Orientation of the UT. If set to <cite>None</cite>, [$\pi$, 0, 0] is used.
Defaults to <cite>None</cite>.
- **bs_orientation** (<cite>None</cite> or Tensor of shape [3], tf.float) – Orientation of the BS. If set to <cite>None</cite>, [0, 0, 0] is used.
Defaults to <cite>None</cite>.
- **min_speed** (<em>float</em>) – Minimum speed [m/s]. Defaults to 0.
- **max_speed** (<em>None</em><em> or </em><em>float</em>) – Maximum speed [m/s]. If set to <cite>None</cite>,
then `max_speed` takes the same value as `min_speed`.
Defaults to <cite>None</cite>.
- **dtype** (<em>Complex tf.DType</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.


Input
 
- **batch_size** (<em>int</em>) – Batch size
- **num_time_steps** (<em>int</em>) – Number of time steps
- **sampling_frequency** (<em>float</em>) – Sampling frequency [Hz]


Output
 
- **a** (<em>[batch size, num_rx = 1, num_rx_ant, num_tx = 1, num_tx_ant, num_paths, num_time_steps], tf.complex</em>) – Path coefficients
- **tau** (<em>[batch size, num_rx = 1, num_tx = 1, num_paths], tf.float</em>) – Path delays [s]




<em class="property">`property` </em>`delay_spread`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.CDL.delay_spread" title="Permalink to this definition"></a>
    
RMS delay spread [s]


<em class="property">`property` </em>`delays`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.CDL.delays" title="Permalink to this definition"></a>
    
Path delays [s]


<em class="property">`property` </em>`k_factor`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.CDL.k_factor" title="Permalink to this definition"></a>
    
K-factor in linear scale. Only available with LoS models.


<em class="property">`property` </em>`los`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.CDL.los" title="Permalink to this definition"></a>
    
<cite>True</cite> is this is a LoS model. <cite>False</cite> otherwise.


<em class="property">`property` </em>`num_clusters`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.CDL.num_clusters" title="Permalink to this definition"></a>
    
Number of paths ($M$)


<em class="property">`property` </em>`powers`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.CDL.powers" title="Permalink to this definition"></a>
    
Path powers in linear scale


### Urban microcell (UMi)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#urban-microcell-umi" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.tr38901.``UMi`(<em class="sig-param">`carrier_frequency`</em>, <em class="sig-param">`o2i_model`</em>, <em class="sig-param">`ut_array`</em>, <em class="sig-param">`bs_array`</em>, <em class="sig-param">`direction`</em>, <em class="sig-param">`enable_pathloss``=``True`</em>, <em class="sig-param">`enable_shadow_fading``=``True`</em>, <em class="sig-param">`always_generate_lsp``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/tr38901/umi.html#UMi">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi" title="Permalink to this definition"></a>
    
Urban microcell (UMi) channel model from 3GPP <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id14">[TR38901]</a> specification.
    
Setting up a UMi model requires configuring the network topology, i.e., the
UTs and BSs locations, UTs velocities, etc. This is achieved using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi.set_topology" title="sionna.channel.tr38901.UMi.set_topology">`set_topology()`</a> method. Setting a different
topology for each batch example is possible. The batch size used when setting up the network topology
is used for the link simulations.
    
The following code snippet shows how to setup a UMi channel model operating
in the frequency domain:
```python
>>> # UT and BS panel arrays
>>> bs_array = PanelArray(num_rows_per_panel = 4,
...                       num_cols_per_panel = 4,
...                       polarization = 'dual',
...                       polarization_type  = 'cross',
...                       antenna_pattern = '38.901',
...                       carrier_frequency = 3.5e9)
>>> ut_array = PanelArray(num_rows_per_panel = 1,
...                       num_cols_per_panel = 1,
...                       polarization = 'single',
...                       polarization_type = 'V',
...                       antenna_pattern = 'omni',
...                       carrier_frequency = 3.5e9)
>>> # Instantiating UMi channel model
>>> channel_model = UMi(carrier_frequency = 3.5e9,
...                     o2i_model = 'low',
...                     ut_array = ut_array,
...                     bs_array = bs_array,
...                     direction = 'uplink')
>>> # Setting up network topology
>>> # ut_loc: UTs locations
>>> # bs_loc: BSs locations
>>> # ut_orientations: UTs array orientations
>>> # bs_orientations: BSs array orientations
>>> # in_state: Indoor/outdoor states of UTs
>>> channel_model.set_topology(ut_loc,
...                            bs_loc,
...                            ut_orientations,
...                            bs_orientations,
...                            ut_velocities,
...                            in_state)
>>> # Instanting the frequency domain channel
>>> channel = OFDMChannel(channel_model = channel_model,
...                       resource_grid = rg)
```

    
where `rg` is an instance of <a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
Parameters
 
- **carrier_frequency** (<em>float</em>) – Carrier frequency in Hertz
- **o2i_model** (<em>str</em>) – Outdoor-to-indoor loss model for UTs located indoor.
Set this parameter to “low” to use the low-loss model, or to “high”
to use the high-loss model.
See section 7.4.3 of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id15">[TR38901]</a> for details.
- **rx_array** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="sionna.channel.tr38901.PanelArray"><em>PanelArray</em></a>) – Panel array used by the receivers. All receivers share the same
antenna array configuration.
- **tx_array** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="sionna.channel.tr38901.PanelArray"><em>PanelArray</em></a>) – Panel array used by the transmitters. All transmitters share the
same antenna array configuration.
- **direction** (<em>str</em>) – Link direction. Either “uplink” or “downlink”.
- **enable_pathloss** (<em>bool</em>) – If <cite>True</cite>, apply pathloss. Otherwise doesn’t. Defaults to <cite>True</cite>.
- **enable_shadow_fading** (<em>bool</em>) – If <cite>True</cite>, apply shadow fading. Otherwise doesn’t.
Defaults to <cite>True</cite>.
- **always_generate_lsp** (<em>bool</em>) – If <cite>True</cite>, new large scale parameters (LSPs) are generated for every
new generation of channel impulse responses. Otherwise, always reuse
the same LSPs, except if the topology is changed. Defaults to
<cite>False</cite>.
- **dtype** (<em>Complex tf.DType</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.


Input
 
- **num_time_steps** (<em>int</em>) – Number of time steps
- **sampling_frequency** (<em>float</em>) – Sampling frequency [Hz]


Output
 
- **a** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex</em>) – Path coefficients
- **tau** (<em>[batch size, num_rx, num_tx, num_paths], tf.float</em>) – Path delays [s]




`set_topology`(<em class="sig-param">`ut_loc``=``None`</em>, <em class="sig-param">`bs_loc``=``None`</em>, <em class="sig-param">`ut_orientations``=``None`</em>, <em class="sig-param">`bs_orientations``=``None`</em>, <em class="sig-param">`ut_velocities``=``None`</em>, <em class="sig-param">`in_state``=``None`</em>, <em class="sig-param">`los``=``None`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi.set_topology" title="Permalink to this definition"></a>
    
Set the network topology.
    
It is possible to set up a different network topology for each batch
example. The batch size used when setting up the network topology
is used for the link simulations.
    
When calling this function, not specifying a parameter leads to the
reuse of the previously given value. Not specifying a value that was not
set at a former call rises an error.
Input
 
- **ut_loc** (<em>[batch size,num_ut, 3], tf.float</em>) – Locations of the UTs
- **bs_loc** (<em>[batch size,num_bs, 3], tf.float</em>) – Locations of BSs
- **ut_orientations** (<em>[batch size,num_ut, 3], tf.float</em>) – Orientations of the UTs arrays [radian]
- **bs_orientations** (<em>[batch size,num_bs, 3], tf.float</em>) – Orientations of the BSs arrays [radian]
- **ut_velocities** (<em>[batch size,num_ut, 3], tf.float</em>) – Velocity vectors of UTs
- **in_state** (<em>[batch size,num_ut], tf.bool</em>) – Indoor/outdoor state of UTs. <cite>True</cite> means indoor and <cite>False</cite>
means outdoor.
- **los** (tf.bool or <cite>None</cite>) – If not <cite>None</cite> (default value), all UTs located outdoor are
forced to be in LoS if `los` is set to <cite>True</cite>, or in NLoS
if it is set to <cite>False</cite>. If set to <cite>None</cite>, the LoS/NLoS states
of UTs is set following 3GPP specification <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id16">[TR38901]</a>.




**Note**
    
If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.


`show_topology`(<em class="sig-param">`bs_index``=``0`</em>, <em class="sig-param">`batch_index``=``0`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi.show_topology" title="Permalink to this definition"></a>
    
Shows the network topology of the batch example with index
`batch_index`.
    
The `bs_index` parameter specifies with respect to which BS the
LoS/NLoS state of UTs is indicated.
Input
 
- **bs_index** (<em>int</em>) – BS index with respect to which the LoS/NLoS state of UTs is
indicated. Defaults to 0.
- **batch_index** (<em>int</em>) – Batch example for which the topology is shown. Defaults to 0.





### Urban macrocell (UMa)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#urban-macrocell-uma" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.tr38901.``UMa`(<em class="sig-param">`carrier_frequency`</em>, <em class="sig-param">`o2i_model`</em>, <em class="sig-param">`ut_array`</em>, <em class="sig-param">`bs_array`</em>, <em class="sig-param">`direction`</em>, <em class="sig-param">`enable_pathloss``=``True`</em>, <em class="sig-param">`enable_shadow_fading``=``True`</em>, <em class="sig-param">`always_generate_lsp``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/tr38901/uma.html#UMa">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMa" title="Permalink to this definition"></a>
    
Urban macrocell (UMa) channel model from 3GPP <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id17">[TR38901]</a> specification.
    
Setting up a UMa model requires configuring the network topology, i.e., the
UTs and BSs locations, UTs velocities, etc. This is achieved using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMa.set_topology" title="sionna.channel.tr38901.UMa.set_topology">`set_topology()`</a> method. Setting a different
topology for each batch example is possible. The batch size used when setting up the network topology
is used for the link simulations.
    
The following code snippet shows how to setup an UMa channel model assuming
an OFDM waveform:
```python
>>> # UT and BS panel arrays
>>> bs_array = PanelArray(num_rows_per_panel = 4,
...                       num_cols_per_panel = 4,
...                       polarization = 'dual',
...                       polarization_type = 'cross',
...                       antenna_pattern = '38.901',
...                       carrier_frequency = 3.5e9)
>>> ut_array = PanelArray(num_rows_per_panel = 1,
...                       num_cols_per_panel = 1,
...                       polarization = 'single',
...                       polarization_type = 'V',
...                       antenna_pattern = 'omni',
...                       carrier_frequency = 3.5e9)
>>> # Instantiating UMa channel model
>>> channel_model = UMa(carrier_frequency = 3.5e9,
...                     o2i_model = 'low',
...                     ut_array = ut_array,
...                     bs_array = bs_array,
...                     direction = 'uplink')
>>> # Setting up network topology
>>> # ut_loc: UTs locations
>>> # bs_loc: BSs locations
>>> # ut_orientations: UTs array orientations
>>> # bs_orientations: BSs array orientations
>>> # in_state: Indoor/outdoor states of UTs
>>> channel_model.set_topology(ut_loc,
...                            bs_loc,
...                            ut_orientations,
...                            bs_orientations,
...                            ut_velocities,
...                            in_state)
>>> # Instanting the OFDM channel
>>> channel = OFDMChannel(channel_model = channel_model,
...                       resource_grid = rg)
```

    
where `rg` is an instance of <a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
Parameters
 
- **carrier_frequency** (<em>float</em>) – Carrier frequency in Hertz
- **o2i_model** (<em>str</em>) – Outdoor-to-indoor loss model for UTs located indoor.
Set this parameter to “low” to use the low-loss model, or to “high”
to use the high-loss model.
See section 7.4.3 of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id18">[TR38901]</a> for details.
- **rx_array** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="sionna.channel.tr38901.PanelArray"><em>PanelArray</em></a>) – Panel array used by the receivers. All receivers share the same
antenna array configuration.
- **tx_array** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="sionna.channel.tr38901.PanelArray"><em>PanelArray</em></a>) – Panel array used by the transmitters. All transmitters share the
same antenna array configuration.
- **direction** (<em>str</em>) – Link direction. Either “uplink” or “downlink”.
- **enable_pathloss** (<em>bool</em>) – If <cite>True</cite>, apply pathloss. Otherwise doesn’t. Defaults to <cite>True</cite>.
- **enable_shadow_fading** (<em>bool</em>) – If <cite>True</cite>, apply shadow fading. Otherwise doesn’t.
Defaults to <cite>True</cite>.
- **always_generate_lsp** (<em>bool</em>) – If <cite>True</cite>, new large scale parameters (LSPs) are generated for every
new generation of channel impulse responses. Otherwise, always reuse
the same LSPs, except if the topology is changed. Defaults to
<cite>False</cite>.
- **dtype** (<em>Complex tf.DType</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.


Input
 
- **num_time_steps** (<em>int</em>) – Number of time steps
- **sampling_frequency** (<em>float</em>) – Sampling frequency [Hz]


Output
 
- **a** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex</em>) – Path coefficients
- **tau** (<em>[batch size, num_rx, num_tx, num_paths], tf.float</em>) – Path delays [s]




`set_topology`(<em class="sig-param">`ut_loc``=``None`</em>, <em class="sig-param">`bs_loc``=``None`</em>, <em class="sig-param">`ut_orientations``=``None`</em>, <em class="sig-param">`bs_orientations``=``None`</em>, <em class="sig-param">`ut_velocities``=``None`</em>, <em class="sig-param">`in_state``=``None`</em>, <em class="sig-param">`los``=``None`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMa.set_topology" title="Permalink to this definition"></a>
    
Set the network topology.
    
It is possible to set up a different network topology for each batch
example. The batch size used when setting up the network topology
is used for the link simulations.
    
When calling this function, not specifying a parameter leads to the
reuse of the previously given value. Not specifying a value that was not
set at a former call rises an error.
Input
 
- **ut_loc** (<em>[batch size,num_ut, 3], tf.float</em>) – Locations of the UTs
- **bs_loc** (<em>[batch size,num_bs, 3], tf.float</em>) – Locations of BSs
- **ut_orientations** (<em>[batch size,num_ut, 3], tf.float</em>) – Orientations of the UTs arrays [radian]
- **bs_orientations** (<em>[batch size,num_bs, 3], tf.float</em>) – Orientations of the BSs arrays [radian]
- **ut_velocities** (<em>[batch size,num_ut, 3], tf.float</em>) – Velocity vectors of UTs
- **in_state** (<em>[batch size,num_ut], tf.bool</em>) – Indoor/outdoor state of UTs. <cite>True</cite> means indoor and <cite>False</cite>
means outdoor.
- **los** (tf.bool or <cite>None</cite>) – If not <cite>None</cite> (default value), all UTs located outdoor are
forced to be in LoS if `los` is set to <cite>True</cite>, or in NLoS
if it is set to <cite>False</cite>. If set to <cite>None</cite>, the LoS/NLoS states
of UTs is set following 3GPP specification <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id19">[TR38901]</a>.




**Note**
    
If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.


`show_topology`(<em class="sig-param">`bs_index``=``0`</em>, <em class="sig-param">`batch_index``=``0`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMa.show_topology" title="Permalink to this definition"></a>
    
Shows the network topology of the batch example with index
`batch_index`.
    
The `bs_index` parameter specifies with respect to which BS the
LoS/NLoS state of UTs is indicated.
Input
 
- **bs_index** (<em>int</em>) – BS index with respect to which the LoS/NLoS state of UTs is
indicated. Defaults to 0.
- **batch_index** (<em>int</em>) – Batch example for which the topology is shown. Defaults to 0.





### Rural macrocell (RMa)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#rural-macrocell-rma" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.tr38901.``RMa`(<em class="sig-param">`carrier_frequency`</em>, <em class="sig-param">`ut_array`</em>, <em class="sig-param">`bs_array`</em>, <em class="sig-param">`direction`</em>, <em class="sig-param">`enable_pathloss``=``True`</em>, <em class="sig-param">`enable_shadow_fading``=``True`</em>, <em class="sig-param">`always_generate_lsp``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/tr38901/rma.html#RMa">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.RMa" title="Permalink to this definition"></a>
    
Rural macrocell (RMa) channel model from 3GPP <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id20">[TR38901]</a> specification.
    
Setting up a RMa model requires configuring the network topology, i.e., the
UTs and BSs locations, UTs velocities, etc. This is achieved using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.RMa.set_topology" title="sionna.channel.tr38901.RMa.set_topology">`set_topology()`</a> method. Setting a different
topology for each batch example is possible. The batch size used when setting up the network topology
is used for the link simulations.
    
The following code snippet shows how to setup an RMa channel model assuming
an OFDM waveform:
```python
>>> # UT and BS panel arrays
>>> bs_array = PanelArray(num_rows_per_panel = 4,
...                       num_cols_per_panel = 4,
...                       polarization = 'dual',
...                       polarization_type = 'cross',
...                       antenna_pattern = '38.901',
...                       carrier_frequency = 3.5e9)
>>> ut_array = PanelArray(num_rows_per_panel = 1,
...                       num_cols_per_panel = 1,
...                       polarization = 'single',
...                       polarization_type = 'V',
...                       antenna_pattern = 'omni',
...                       carrier_frequency = 3.5e9)
>>> # Instantiating RMa channel model
>>> channel_model = RMa(carrier_frequency = 3.5e9,
...                     ut_array = ut_array,
...                     bs_array = bs_array,
...                     direction = 'uplink')
>>> # Setting up network topology
>>> # ut_loc: UTs locations
>>> # bs_loc: BSs locations
>>> # ut_orientations: UTs array orientations
>>> # bs_orientations: BSs array orientations
>>> # in_state: Indoor/outdoor states of UTs
>>> channel_model.set_topology(ut_loc,
...                            bs_loc,
...                            ut_orientations,
...                            bs_orientations,
...                            ut_velocities,
...                            in_state)
>>> # Instanting the OFDM channel
>>> channel = OFDMChannel(channel_model = channel_model,
...                       resource_grid = rg)
```

    
where `rg` is an instance of <a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
Parameters
 
- **carrier_frequency** (<em>float</em>) – Carrier frequency [Hz]
- **rx_array** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="sionna.channel.tr38901.PanelArray"><em>PanelArray</em></a>) – Panel array used by the receivers. All receivers share the same
antenna array configuration.
- **tx_array** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="sionna.channel.tr38901.PanelArray"><em>PanelArray</em></a>) – Panel array used by the transmitters. All transmitters share the
same antenna array configuration.
- **direction** (<em>str</em>) – Link direction. Either “uplink” or “downlink”.
- **enable_pathloss** (<em>bool</em>) – If <cite>True</cite>, apply pathloss. Otherwise doesn’t. Defaults to <cite>True</cite>.
- **enable_shadow_fading** (<em>bool</em>) – If <cite>True</cite>, apply shadow fading. Otherwise doesn’t.
Defaults to <cite>True</cite>.
- **average_street_width** (<em>float</em>) – Average street width [m]. Defaults to 5m.
- **average_street_width** – Average building height [m]. Defaults to 20m.
- **always_generate_lsp** (<em>bool</em>) – If <cite>True</cite>, new large scale parameters (LSPs) are generated for every
new generation of channel impulse responses. Otherwise, always reuse
the same LSPs, except if the topology is changed. Defaults to
<cite>False</cite>.
- **dtype** (<em>Complex tf.DType</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.


Input
 
- **num_time_steps** (<em>int</em>) – Number of time steps
- **sampling_frequency** (<em>float</em>) – Sampling frequency [Hz]


Output
 
- **a** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex</em>) – Path coefficients
- **tau** (<em>[batch size, num_rx, num_tx, num_paths], tf.float</em>) – Path delays [s]




`set_topology`(<em class="sig-param">`ut_loc``=``None`</em>, <em class="sig-param">`bs_loc``=``None`</em>, <em class="sig-param">`ut_orientations``=``None`</em>, <em class="sig-param">`bs_orientations``=``None`</em>, <em class="sig-param">`ut_velocities``=``None`</em>, <em class="sig-param">`in_state``=``None`</em>, <em class="sig-param">`los``=``None`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.RMa.set_topology" title="Permalink to this definition"></a>
    
Set the network topology.
    
It is possible to set up a different network topology for each batch
example. The batch size used when setting up the network topology
is used for the link simulations.
    
When calling this function, not specifying a parameter leads to the
reuse of the previously given value. Not specifying a value that was not
set at a former call rises an error.
Input
 
- **ut_loc** (<em>[batch size,num_ut, 3], tf.float</em>) – Locations of the UTs
- **bs_loc** (<em>[batch size,num_bs, 3], tf.float</em>) – Locations of BSs
- **ut_orientations** (<em>[batch size,num_ut, 3], tf.float</em>) – Orientations of the UTs arrays [radian]
- **bs_orientations** (<em>[batch size,num_bs, 3], tf.float</em>) – Orientations of the BSs arrays [radian]
- **ut_velocities** (<em>[batch size,num_ut, 3], tf.float</em>) – Velocity vectors of UTs
- **in_state** (<em>[batch size,num_ut], tf.bool</em>) – Indoor/outdoor state of UTs. <cite>True</cite> means indoor and <cite>False</cite>
means outdoor.
- **los** (tf.bool or <cite>None</cite>) – If not <cite>None</cite> (default value), all UTs located outdoor are
forced to be in LoS if `los` is set to <cite>True</cite>, or in NLoS
if it is set to <cite>False</cite>. If set to <cite>None</cite>, the LoS/NLoS states
of UTs is set following 3GPP specification <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id21">[TR38901]</a>.




**Note**
    
If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.


`show_topology`(<em class="sig-param">`bs_index``=``0`</em>, <em class="sig-param">`batch_index``=``0`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.RMa.show_topology" title="Permalink to this definition"></a>
    
Shows the network topology of the batch example with index
`batch_index`.
    
The `bs_index` parameter specifies with respect to which BS the
LoS/NLoS state of UTs is indicated.
Input
 
- **bs_index** (<em>int</em>) – BS index with respect to which the LoS/NLoS state of UTs is
indicated. Defaults to 0.
- **batch_index** (<em>int</em>) – Batch example for which the topology is shown. Defaults to 0.





## External datasets<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#external-datasets" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``CIRDataset`(<em class="sig-param">`cir_generator`</em>, <em class="sig-param">`batch_size`</em>, <em class="sig-param">`num_rx`</em>, <em class="sig-param">`num_rx_ant`</em>, <em class="sig-param">`num_tx`</em>, <em class="sig-param">`num_tx_ant`</em>, <em class="sig-param">`num_paths`</em>, <em class="sig-param">`num_time_steps`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/cir_dataset.html#CIRDataset">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.CIRDataset" title="Permalink to this definition"></a>
    
Creates a channel model from a dataset that can be used with classes such as
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.TimeChannel" title="sionna.channel.TimeChannel">`TimeChannel`</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.OFDMChannel" title="sionna.channel.OFDMChannel">`OFDMChannel`</a>.
The dataset is defined by a <a class="reference external" href="https://wiki.python.org/moin/Generators">generator</a>.
    
The batch size is configured when instantiating the dataset or through the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.CIRDataset.batch_size" title="sionna.channel.CIRDataset.batch_size">`batch_size`</a> property.
The number of time steps (<cite>num_time_steps</cite>) and sampling frequency (<cite>sampling_frequency</cite>) can only be set when instantiating the dataset.
The specified values must be in accordance with the data.
<p class="rubric">Example
    
The following code snippet shows how to use this class as a channel model.
```python
>>> my_generator = MyGenerator(...)
>>> channel_model = sionna.channel.CIRDataset(my_generator,
...                                           batch_size,
...                                           num_rx,
...                                           num_rx_ant,
...                                           num_tx,
...                                           num_tx_ant,
...                                           num_paths,
...                                           num_time_steps+l_tot-1)
>>> channel = sionna.channel.TimeChannel(channel_model, bandwidth, num_time_steps)
```

    
where `MyGenerator` is a generator
```python
>>> class MyGenerator:
...
...     def __call__(self):
...         ...
...         yield a, tau
```

    
that returns complex-valued path coefficients `a` with shape
<cite>[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]</cite>
and real-valued path delays `tau` (in second)
<cite>[num_rx, num_tx, num_paths]</cite>.
Parameters
 
- **cir_generator** – Generator that returns channel impulse responses `(a,` `tau)` where
`a` is the tensor of channel coefficients of shape
<cite>[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]</cite>
and dtype `dtype`, and `tau` the tensor of path delays
of shape  <cite>[num_rx, num_tx, num_paths]</cite> and dtype `dtype.`
`real_dtype`.
- **batch_size** (<em>int</em>) – Batch size
- **num_rx** (<em>int</em>) – Number of receivers ($N_R$)
- **num_rx_ant** (<em>int</em>) – Number of antennas per receiver ($N_{RA}$)
- **num_tx** (<em>int</em>) – Number of transmitters ($N_T$)
- **num_tx_ant** (<em>int</em>) – Number of antennas per transmitter ($N_{TA}$)
- **num_paths** (<em>int</em>) – Number of paths ($M$)
- **num_time_steps** (<em>int</em>) – Number of time steps
- **dtype** (<em>tf.DType</em>) – Complex datatype to use for internal processing and output.
Defaults to <cite>tf.complex64</cite>.


Output
 
- **a** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex</em>) – Path coefficients
- **tau** (<em>[batch size, num_rx, num_tx, num_paths], tf.float</em>) – Path delays [s]




<em class="property">`property` </em>`batch_size`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.CIRDataset.batch_size" title="Permalink to this definition"></a>
    
Batch size


## Utility functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#utility-functions" title="Permalink to this headline"></a>

### subcarrier_frequencies<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#subcarrier-frequencies" title="Permalink to this headline"></a>

`sionna.channel.``subcarrier_frequencies`(<em class="sig-param">`num_subcarriers`</em>, <em class="sig-param">`subcarrier_spacing`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#subcarrier_frequencies">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.subcarrier_frequencies" title="Permalink to this definition"></a>
    
Compute the baseband frequencies of `num_subcarrier` subcarriers spaced by
`subcarrier_spacing`, i.e.,
```python
>>> # If num_subcarrier is even:
>>> frequencies = [-num_subcarrier/2, ..., 0, ..., num_subcarrier/2-1] * subcarrier_spacing
>>>
>>> # If num_subcarrier is odd:
>>> frequencies = [-(num_subcarrier-1)/2, ..., 0, ..., (num_subcarrier-1)/2] * subcarrier_spacing
```

Input
 
- **num_subcarriers** (<em>int</em>) – Number of subcarriers
- **subcarrier_spacing** (<em>float</em>) – Subcarrier spacing [Hz]
- **dtype** (<em>tf.DType</em>) – Datatype to use for internal processing and output.
If a complex datatype is provided, the corresponding precision of
real components is used.
Defaults to <cite>tf.complex64</cite> (<cite>tf.float32</cite>).


Output
    
**frequencies** ([`num_subcarrier`], tf.float) – Baseband frequencies of subcarriers



### time_lag_discrete_time_channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#time-lag-discrete-time-channel" title="Permalink to this headline"></a>

`sionna.channel.``time_lag_discrete_time_channel`(<em class="sig-param">`bandwidth`</em>, <em class="sig-param">`maximum_delay_spread``=``3e-06`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#time_lag_discrete_time_channel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.time_lag_discrete_time_channel" title="Permalink to this definition"></a>
    
Compute the smallest and largest time-lag for the descrete complex baseband
channel, i.e., $L_{\text{min}}$ and $L_{\text{max}}$.
    
The smallest time-lag ($L_{\text{min}}$) returned is always -6, as this value
was found small enough for all models included in Sionna.
    
The largest time-lag ($L_{\text{max}}$) is computed from the `bandwidth`
and `maximum_delay_spread` as follows:

$$
L_{\text{max}} = \lceil W \tau_{\text{max}} \rceil + 6
$$
    
where $L_{\text{max}}$ is the largest time-lag, $W$ the `bandwidth`,
and $\tau_{\text{max}}$ the `maximum_delay_spread`.
    
The default value for the `maximum_delay_spread` is 3us, which was found
to be large enough to include most significant paths with all channel models
included in Sionna assuming a nominal delay spread of 100ns.

**Note**
    
The values of $L_{\text{min}}$ and $L_{\text{max}}$ computed
by this function are only recommended values.
$L_{\text{min}}$ and $L_{\text{max}}$ should be set according to
the considered channel model. For OFDM systems, one also needs to be careful
that the effective length of the complex baseband channel is not larger than
the cyclic prefix length.

Input
 
- **bandwidth** (<em>float</em>) – Bandwith ($W$) [Hz]
- **maximum_delay_spread** (<em>float</em>) – Maximum delay spread [s]. Defaults to 3us.


Output
 
- **l_min** (<em>int</em>) – Smallest time-lag ($L_{\text{min}}$) for the descrete complex baseband
channel. Set to -6, , as this value was found small enough for all models
included in Sionna.
- **l_max** (<em>int</em>) – Largest time-lag ($L_{\text{max}}$) for the descrete complex baseband
channel




### deg_2_rad<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#deg-2-rad" title="Permalink to this headline"></a>

`sionna.channel.``deg_2_rad`(<em class="sig-param">`x`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#deg_2_rad">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.deg_2_rad" title="Permalink to this definition"></a>
    
Convert degree to radian
Input
    
**x** (<em>Tensor</em>) – Angles in degree

Output
    
**y** (<em>Tensor</em>) – Angles `x` converted to radian



### rad_2_deg<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#rad-2-deg" title="Permalink to this headline"></a>

`sionna.channel.``rad_2_deg`(<em class="sig-param">`x`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#rad_2_deg">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.rad_2_deg" title="Permalink to this definition"></a>
    
Convert radian to degree
Input
    
**x** (<em>Tensor</em>) – Angles in radian

Output
    
**y** (<em>Tensor</em>) – Angles `x` converted to degree



### wrap_angle_0_360<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#wrap-angle-0-360" title="Permalink to this headline"></a>

`sionna.channel.``wrap_angle_0_360`(<em class="sig-param">`angle`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#wrap_angle_0_360">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.wrap_angle_0_360" title="Permalink to this definition"></a>
    
Wrap `angle` to (0,360)
Input
    
**angle** (<em>Tensor</em>) – Input to wrap

Output
    
**y** (<em>Tensor</em>) – `angle` wrapped to (0,360)



### drop_uts_in_sector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#drop-uts-in-sector" title="Permalink to this headline"></a>

`sionna.channel.``drop_uts_in_sector`(<em class="sig-param">`batch_size`</em>, <em class="sig-param">`num_ut`</em>, <em class="sig-param">`min_bs_ut_dist`</em>, <em class="sig-param">`isd`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#drop_uts_in_sector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.drop_uts_in_sector" title="Permalink to this definition"></a>
    
Uniformly sample UT locations from a sector.
    
The sector from which UTs are sampled is shown in the following figure.
The BS is assumed to be located at the origin (0,0) of the coordinate
system.
<a class="reference internal image-reference" href="../_images/drop_uts_in_sector.png"><img alt="../_images/drop_uts_in_sector.png" src="https://nvlabs.github.io/sionna/_images/drop_uts_in_sector.png" style="width: 307.5px; height: 216.9px;" /></a>

Input
 
- **batch_size** (<em>int</em>) – Batch size
- **num_ut** (<em>int</em>) – Number of UTs to sample per batch example
- **min_bs_ut_dist** (<em>tf.float</em>) – Minimum BS-UT distance [m]
- **isd** (<em>tf.float</em>) – Inter-site distance, i.e., the distance between two adjacent BSs [m]
- **dtype** (<em>tf.DType</em>) – Datatype to use for internal processing and output.
If a complex datatype is provided, the corresponding precision of
real components is used.
Defaults to <cite>tf.complex64</cite> (<cite>tf.float32</cite>).


Output
    
**ut_loc** (<em>[batch_size, num_ut, 2], tf.float</em>) – UTs locations in the X-Y plan



### relocate_uts<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#relocate-uts" title="Permalink to this headline"></a>

`sionna.channel.``relocate_uts`(<em class="sig-param">`ut_loc`</em>, <em class="sig-param">`sector_id`</em>, <em class="sig-param">`cell_loc`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#relocate_uts">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.relocate_uts" title="Permalink to this definition"></a>
    
Relocate the UTs by rotating them into the sector with index `sector_id`
and transposing them to the cell centered on `cell_loc`.
    
`sector_id` gives the index of the sector to which the UTs are
rotated to. The picture below shows how the three sectors of a cell are
indexed.
<a class="reference internal image-reference" href="../_images/panel_array_sector_id.png"><img alt="../_images/panel_array_sector_id.png" src="https://nvlabs.github.io/sionna/_images/panel_array_sector_id.png" style="width: 188.1px; height: 162.9px;" /></a>
<p class="caption">Fig. 9 Indexing of sectors<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id32" title="Permalink to this image"></a>
    
If `sector_id` is a scalar, then all UTs are relocated to the same
sector indexed by `sector_id`.
If `sector_id` is a tensor, it should be broadcastable with
[`batch_size`, `num_ut`], and give the sector in which each UT or
batch example is relocated to.
    
When calling the function, `ut_loc` gives the locations of the UTs to
relocate, which are all assumed to be in sector with index 0, and in the
cell centered on the origin (0,0).
Input
 
- **ut_loc** (<em>[batch_size, num_ut, 2], tf.float</em>) – UTs locations in the X-Y plan
- **sector_id** (<em>Tensor broadcastable with [batch_size, num_ut], int</em>) – Indexes of the sector to which to relocate the UTs
- **cell_loc** (<em>Tensor broadcastable with [batch_size, num_ut], tf.float</em>) – Center of the cell to which to transpose the UTs


Output
    
**ut_loc** (<em>[batch_size, num_ut, 2], tf.float</em>) – Relocated UTs locations in the X-Y plan



### set_3gpp_scenario_parameters<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#set-3gpp-scenario-parameters" title="Permalink to this headline"></a>

`sionna.channel.``set_3gpp_scenario_parameters`(<em class="sig-param">`scenario`</em>, <em class="sig-param">`min_bs_ut_dist``=``None`</em>, <em class="sig-param">`isd``=``None`</em>, <em class="sig-param">`bs_height``=``None`</em>, <em class="sig-param">`min_ut_height``=``None`</em>, <em class="sig-param">`max_ut_height``=``None`</em>, <em class="sig-param">`indoor_probability``=``None`</em>, <em class="sig-param">`min_ut_velocity``=``None`</em>, <em class="sig-param">`max_ut_velocity``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#set_3gpp_scenario_parameters">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.set_3gpp_scenario_parameters" title="Permalink to this definition"></a>
    
Set valid parameters for a specified 3GPP system level `scenario`
(RMa, UMi, or UMa).
    
If a parameter is given, then it is returned. If it is set to <cite>None</cite>,
then a parameter valid according to the chosen scenario is returned
(see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id25">[TR38901]</a>).
Input
 
- **scenario** (<em>str</em>) – System level model scenario. Must be one of “rma”, “umi”, or “uma”.
- **min_bs_ut_dist** (<em>None or tf.float</em>) – Minimum BS-UT distance [m]
- **isd** (<em>None or tf.float</em>) – Inter-site distance [m]
- **bs_height** (<em>None or tf.float</em>) – BS elevation [m]
- **min_ut_height** (<em>None or tf.float</em>) – Minimum UT elevation [m]
- **max_ut_height** (<em>None or tf.float</em>) – Maximum UT elevation [m]
- **indoor_probability** (<em>None or tf.float</em>) – Probability of a UT to be indoor
- **min_ut_velocity** (<em>None or tf.float</em>) – Minimum UT velocity [m/s]
- **max_ut_velocity** (<em>None or tf.float</em>) – Maximim UT velocity [m/s]
- **dtype** (<em>tf.DType</em>) – Datatype to use for internal processing and output.
If a complex datatype is provided, the corresponding precision of
real components is used.
Defaults to <cite>tf.complex64</cite> (<cite>tf.float32</cite>).


Output
 
- **min_bs_ut_dist** (<em>tf.float</em>) – Minimum BS-UT distance [m]
- **isd** (<em>tf.float</em>) – Inter-site distance [m]
- **bs_height** (<em>tf.float</em>) – BS elevation [m]
- **min_ut_height** (<em>tf.float</em>) – Minimum UT elevation [m]
- **max_ut_height** (<em>tf.float</em>) – Maximum UT elevation [m]
- **indoor_probability** (<em>tf.float</em>) – Probability of a UT to be indoor
- **min_ut_velocity** (<em>tf.float</em>) – Minimum UT velocity [m/s]
- **max_ut_velocity** (<em>tf.float</em>) – Maximim UT velocity [m/s]




### gen_single_sector_topology<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#gen-single-sector-topology" title="Permalink to this headline"></a>

`sionna.channel.``gen_single_sector_topology`(<em class="sig-param">`batch_size`</em>, <em class="sig-param">`num_ut`</em>, <em class="sig-param">`scenario`</em>, <em class="sig-param">`min_bs_ut_dist``=``None`</em>, <em class="sig-param">`isd``=``None`</em>, <em class="sig-param">`bs_height``=``None`</em>, <em class="sig-param">`min_ut_height``=``None`</em>, <em class="sig-param">`max_ut_height``=``None`</em>, <em class="sig-param">`indoor_probability``=``None`</em>, <em class="sig-param">`min_ut_velocity``=``None`</em>, <em class="sig-param">`max_ut_velocity``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#gen_single_sector_topology">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.gen_single_sector_topology" title="Permalink to this definition"></a>
    
Generate a batch of topologies consisting of a single BS located at the
origin and `num_ut` UTs randomly and uniformly dropped in a cell sector.
    
The following picture shows the sector from which UTs are sampled.
<a class="reference internal image-reference" href="../_images/drop_uts_in_sector.png"><img alt="../_images/drop_uts_in_sector.png" src="https://nvlabs.github.io/sionna/_images/drop_uts_in_sector.png" style="width: 307.5px; height: 216.9px;" /></a>
    
UTs orientations are randomly and uniformly set, whereas the BS orientation
is set such that the it is oriented towards the center of the sector.
    
The drop configuration can be controlled through the optional parameters.
Parameters set to <cite>None</cite> are set to valid values according to the chosen
`scenario` (see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id26">[TR38901]</a>).
    
The returned batch of topologies can be used as-is with the
`set_topology()` method of the system level models, i.e.
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi" title="sionna.channel.tr38901.UMi">`UMi`</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMa" title="sionna.channel.tr38901.UMa">`UMa`</a>,
and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.RMa" title="sionna.channel.tr38901.RMa">`RMa`</a>.
<p class="rubric">Example
```python
>>> # Create antenna arrays
>>> bs_array = PanelArray(num_rows_per_panel = 4,
...                      num_cols_per_panel = 4,
...                      polarization = 'dual',
...                      polarization_type = 'VH',
...                      antenna_pattern = '38.901',
...                      carrier_frequency = 3.5e9)
>>>
>>> ut_array = PanelArray(num_rows_per_panel = 1,
...                       num_cols_per_panel = 1,
...                       polarization = 'single',
...                       polarization_type = 'V',
...                       antenna_pattern = 'omni',
...                       carrier_frequency = 3.5e9)
>>> # Create channel model
>>> channel_model = UMi(carrier_frequency = 3.5e9,
...                     o2i_model = 'low',
...                     ut_array = ut_array,
...                     bs_array = bs_array,
...                     direction = 'uplink')
>>> # Generate the topology
>>> topology = gen_single_sector_topology(batch_size = 100,
...                                       num_ut = 4,
...                                       scenario = 'umi')
>>> # Set the topology
>>> ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = topology
>>> channel_model.set_topology(ut_loc,
...                            bs_loc,
...                            ut_orientations,
...                            bs_orientations,
...                            ut_velocities,
...                            in_state)
>>> channel_model.show_topology()
```

<img alt="../_images/drop_uts_in_sector_topology.png" src="https://nvlabs.github.io/sionna/_images/drop_uts_in_sector_topology.png" />
Input
 
- **batch_size** (<em>int</em>) – Batch size
- **num_ut** (<em>int</em>) – Number of UTs to sample per batch example
- **scenario** (<em>str</em>) – System leven model scenario. Must be one of “rma”, “umi”, or “uma”.
- **min_bs_ut_dist** (<em>None or tf.float</em>) – Minimum BS-UT distance [m]
- **isd** (<em>None or tf.float</em>) – Inter-site distance [m]
- **bs_height** (<em>None or tf.float</em>) – BS elevation [m]
- **min_ut_height** (<em>None or tf.float</em>) – Minimum UT elevation [m]
- **max_ut_height** (<em>None or tf.float</em>) – Maximum UT elevation [m]
- **indoor_probability** (<em>None or tf.float</em>) – Probability of a UT to be indoor
- **min_ut_velocity** (<em>None or tf.float</em>) – Minimum UT velocity [m/s]
- **max_ut_velocity** (<em>None or tf.float</em>) – Maximim UT velocity [m/s]
- **dtype** (<em>tf.DType</em>) – Datatype to use for internal processing and output.
If a complex datatype is provided, the corresponding precision of
real components is used.
Defaults to <cite>tf.complex64</cite> (<cite>tf.float32</cite>).


Output
 
- **ut_loc** (<em>[batch_size, num_ut, 3], tf.float</em>) – UTs locations
- **bs_loc** (<em>[batch_size, 1, 3], tf.float</em>) – BS location. Set to (0,0,0) for all batch examples.
- **ut_orientations** (<em>[batch_size, num_ut, 3], tf.float</em>) – UTs orientations [radian]
- **bs_orientations** (<em>[batch_size, 1, 3], tf.float</em>) – BS orientations [radian]. Oriented towards the center of the sector.
- **ut_velocities** (<em>[batch_size, num_ut, 3], tf.float</em>) – UTs velocities [m/s]
- **in_state** (<em>[batch_size, num_ut], tf.float</em>) – Indoor/outdoor state of UTs. <cite>True</cite> means indoor, <cite>False</cite> means
outdoor.




### gen_single_sector_topology_interferers<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#gen-single-sector-topology-interferers" title="Permalink to this headline"></a>

`sionna.channel.``gen_single_sector_topology_interferers`(<em class="sig-param">`batch_size`</em>, <em class="sig-param">`num_ut`</em>, <em class="sig-param">`num_interferer`</em>, <em class="sig-param">`scenario`</em>, <em class="sig-param">`min_bs_ut_dist``=``None`</em>, <em class="sig-param">`isd``=``None`</em>, <em class="sig-param">`bs_height``=``None`</em>, <em class="sig-param">`min_ut_height``=``None`</em>, <em class="sig-param">`max_ut_height``=``None`</em>, <em class="sig-param">`indoor_probability``=``None`</em>, <em class="sig-param">`min_ut_velocity``=``None`</em>, <em class="sig-param">`max_ut_velocity``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#gen_single_sector_topology_interferers">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.gen_single_sector_topology_interferers" title="Permalink to this definition"></a>
    
Generate a batch of topologies consisting of a single BS located at the
origin, `num_ut` UTs randomly and uniformly dropped in a cell sector, and
`num_interferer` interfering UTs randomly dropped in the adjacent cells.
    
The following picture shows how UTs are sampled
<a class="reference internal image-reference" href="../_images/drop_uts_in_sector_interferers.png"><img alt="../_images/drop_uts_in_sector_interferers.png" src="https://nvlabs.github.io/sionna/_images/drop_uts_in_sector_interferers.png" style="width: 383.4px; height: 383.09999999999997px;" /></a>
    
UTs orientations are randomly and uniformly set, whereas the BS orientation
is set such that it is oriented towards the center of the sector it
serves.
    
The drop configuration can be controlled through the optional parameters.
Parameters set to <cite>None</cite> are set to valid values according to the chosen
`scenario` (see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id27">[TR38901]</a>).
    
The returned batch of topologies can be used as-is with the
`set_topology()` method of the system level models, i.e.
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi" title="sionna.channel.tr38901.UMi">`UMi`</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMa" title="sionna.channel.tr38901.UMa">`UMa`</a>,
and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.RMa" title="sionna.channel.tr38901.RMa">`RMa`</a>.
    
In the returned `ut_loc`, `ut_orientations`, `ut_velocities`, and
`in_state` tensors, the first `num_ut` items along the axis with index
1 correspond to the served UTs, whereas the remaining `num_interferer`
items correspond to the interfering UTs.
<p class="rubric">Example
```python
>>> # Create antenna arrays
>>> bs_array = PanelArray(num_rows_per_panel = 4,
...                      num_cols_per_panel = 4,
...                      polarization = 'dual',
...                      polarization_type = 'VH',
...                      antenna_pattern = '38.901',
...                      carrier_frequency = 3.5e9)
>>>
>>> ut_array = PanelArray(num_rows_per_panel = 1,
...                       num_cols_per_panel = 1,
...                       polarization = 'single',
...                       polarization_type = 'V',
...                       antenna_pattern = 'omni',
...                       carrier_frequency = 3.5e9)
>>> # Create channel model
>>> channel_model = UMi(carrier_frequency = 3.5e9,
...                     o2i_model = 'low',
...                     ut_array = ut_array,
...                     bs_array = bs_array,
...                     direction = 'uplink')
>>> # Generate the topology
>>> topology = gen_single_sector_topology_interferers(batch_size = 100,
...                                                   num_ut = 4,
...                                                   num_interferer = 4,
...                                                   scenario = 'umi')
>>> # Set the topology
>>> ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = topology
>>> channel_model.set_topology(ut_loc,
...                            bs_loc,
...                            ut_orientations,
...                            bs_orientations,
...                            ut_velocities,
...                            in_state)
>>> channel_model.show_topology()
```

<img alt="../_images/drop_uts_in_sector_topology_inter.png" src="https://nvlabs.github.io/sionna/_images/drop_uts_in_sector_topology_inter.png" />
Input
 
- **batch_size** (<em>int</em>) – Batch size
- **num_ut** (<em>int</em>) – Number of UTs to sample per batch example
- **num_interferer** (<em>int</em>) – Number of interfeering UTs per batch example
- **scenario** (<em>str</em>) – System leven model scenario. Must be one of “rma”, “umi”, or “uma”.
- **min_bs_ut_dist** (<em>None or tf.float</em>) – Minimum BS-UT distance [m]
- **isd** (<em>None or tf.float</em>) – Inter-site distance [m]
- **bs_height** (<em>None or tf.float</em>) – BS elevation [m]
- **min_ut_height** (<em>None or tf.float</em>) – Minimum UT elevation [m]
- **max_ut_height** (<em>None or tf.float</em>) – Maximum UT elevation [m]
- **indoor_probability** (<em>None or tf.float</em>) – Probability of a UT to be indoor
- **min_ut_velocity** (<em>None or tf.float</em>) – Minimum UT velocity [m/s]
- **max_ut_velocity** (<em>None or tf.float</em>) – Maximim UT velocity [m/s]
- **dtype** (<em>tf.DType</em>) – Datatype to use for internal processing and output.
If a complex datatype is provided, the corresponding precision of
real components is used.
Defaults to <cite>tf.complex64</cite> (<cite>tf.float32</cite>).


Output
 
- **ut_loc** (<em>[batch_size, num_ut, 3], tf.float</em>) – UTs locations. The first `num_ut` items along the axis with index
1 correspond to the served UTs, whereas the remaining
`num_interferer` items correspond to the interfeering UTs.
- **bs_loc** (<em>[batch_size, 1, 3], tf.float</em>) – BS location. Set to (0,0,0) for all batch examples.
- **ut_orientations** (<em>[batch_size, num_ut, 3], tf.float</em>) – UTs orientations [radian]. The first `num_ut` items along the
axis with index 1 correspond to the served UTs, whereas the
remaining `num_interferer` items correspond to the interfeering
UTs.
- **bs_orientations** (<em>[batch_size, 1, 3], tf.float</em>) – BS orientation [radian]. Oriented towards the center of the sector.
- **ut_velocities** (<em>[batch_size, num_ut, 3], tf.float</em>) – UTs velocities [m/s]. The first `num_ut` items along the axis
with index 1 correspond to the served UTs, whereas the remaining
`num_interferer` items correspond to the interfeering UTs.
- **in_state** (<em>[batch_size, num_ut], tf.float</em>) – Indoor/outdoor state of UTs. <cite>True</cite> means indoor, <cite>False</cite> means
outdoor. The first `num_ut` items along the axis with
index 1 correspond to the served UTs, whereas the remaining
`num_interferer` items correspond to the interfeering UTs.




### exp_corr_mat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#exp-corr-mat" title="Permalink to this headline"></a>

`sionna.channel.``exp_corr_mat`(<em class="sig-param">`a`</em>, <em class="sig-param">`n`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#exp_corr_mat">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.exp_corr_mat" title="Permalink to this definition"></a>
    
Generate exponential correlation matrices.
    
This function computes for every element $a$ of a complex-valued
tensor $\mathbf{a}$ the corresponding $n\times n$ exponential
correlation matrix $\mathbf{R}(a,n)$, defined as (Eq. 1, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#mal2018" id="id28">[MAL2018]</a>):

$$
\begin{split}\mathbf{R}(a,n)_{i,j} = \begin{cases}
            1 & \text{if } i=j\\
            a^{i-j}  & \text{if } i>j\\
            (a^\star)^{j-i}  & \text{if } j<i, j=1,\dots,n\\
          \end{cases}\end{split}
$$
    
where $|a|<1$ and $\mathbf{R}\in\mathbb{C}^{n\times n}$.
Input
 
- **a** (<em>[n_0, …, n_k], tf.complex</em>) – A tensor of arbitrary rank whose elements
have an absolute value smaller than one.
- **n** (<em>int</em>) – Number of dimensions of the output correlation matrices.
- **dtype** (<em>tf.complex64, tf.complex128</em>) – The dtype of the output.


Output
    
**R** (<em>[n_0, …, n_k, n, n], tf.complex</em>) – A tensor of the same dtype as the input tensor $\mathbf{a}$.



### one_ring_corr_mat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#one-ring-corr-mat" title="Permalink to this headline"></a>

`sionna.channel.``one_ring_corr_mat`(<em class="sig-param">`phi_deg`</em>, <em class="sig-param">`num_ant`</em>, <em class="sig-param">`d_h``=``0.5`</em>, <em class="sig-param">`sigma_phi_deg``=``15`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#one_ring_corr_mat">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.one_ring_corr_mat" title="Permalink to this definition"></a>
    
Generate covariance matrices from the one-ring model.
    
This function generates approximate covariance matrices for the
so-called <cite>one-ring</cite> model (Eq. 2.24) <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#bhs2017" id="id29">[BHS2017]</a>. A uniform
linear array (ULA) with uniform antenna spacing is assumed. The elements
of the covariance matrices are computed as:

$$
\mathbf{R}_{\ell,m} =
      \exp\left( j2\pi d_\text{H} (\ell -m)\sin(\varphi) \right)
      \exp\left( -\frac{\sigma_\varphi^2}{2}
      \left( 2\pi d_\text{H}(\ell -m)\cos(\varphi) \right)^2 \right)
$$
    
for $\ell,m = 1,\dots, M$, where $M$ is the number of antennas,
$\varphi$ is the angle of arrival, $d_\text{H}$ is the antenna
spacing in multiples of the wavelength,
and $\sigma^2_\varphi$ is the angular standard deviation.
Input
 
- **phi_deg** (<em>[n_0, …, n_k], tf.float</em>) – A tensor of arbitrary rank containing azimuth angles (deg) of arrival.
- **num_ant** (<em>int</em>) – Number of antennas
- **d_h** (<em>float</em>) – Antenna spacing in multiples of the wavelength. Defaults to 0.5.
- **sigma_phi_deg** (<em>float</em>) – Angular standard deviation (deg). Defaults to 15 (deg). Values greater
than 15 should not be used as the approximation becomes invalid.
- **dtype** (<em>tf.complex64, tf.complex128</em>) – The dtype of the output.


Output
    
**R** ([n_0, …, n_k, num_ant, nun_ant], <cite>dtype</cite>) – Tensor containing the covariance matrices of the desired dtype.




References:
TR38901(<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id1">1</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id2">2</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id4">3</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id5">4</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id6">5</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id7">6</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id8">7</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id11">8</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id12">9</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id13">10</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id14">11</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id15">12</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id16">13</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id17">14</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id18">15</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id19">16</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id20">17</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id21">18</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id25">19</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id26">20</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id27">21</a>)
    
3GPP TR 38.901,
“Study on channel model for frequencies from 0.5 to 100 GHz”, Release 16.1

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id10">TS38141-1</a>
    
3GPP TS 38.141-1
“Base Station (BS) conformance testing Part 1: Conducted conformance testing”,
Release 17

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id3">Tse</a>
    
D. Tse and P. Viswanath, “Fundamentals of wireless communication“,
Cambridge university press, 2005.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id9">SoS</a>
<ol class="upperalpha simple" start="3">
- Xiao, Y. R. Zheng and N. C. Beaulieu, “Novel Sum-of-Sinusoids Simulation Models for Rayleigh and Rician Fading Channels,” in IEEE Transactions on Wireless Communications, vol. 5, no. 12, pp. 3667-3679, December 2006, doi: 10.1109/TWC.2006.256990.
</ol>

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id28">MAL2018</a>
    
Ranjan K. Mallik,
“The exponential correlation matrix: Eigen-analysis and
applications”, IEEE Trans. Wireless Commun., vol. 17, no. 7,
pp. 4690-4705, Jul. 2018.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id29">BHS2017</a>
    
Emil Björnson, Jakob Hoydis and Luca Sanguinetti (2017),
<a class="reference external" href="https://massivemimobook.com">“Massive MIMO Networks: Spectral, Energy, and Hardware Efficiency”</a>,
Foundations and Trends in Signal Processing:
Vol. 11, No. 3-4, pp 154–655.



