
# Orthogonal Frequency-Division Multiplexing (OFDM)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#orthogonal-frequency-division-multiplexing-ofdm" title="Permalink to this headline"></a>
    
This module provides layers and functions to support
simulation of OFDM-based systems. The key component is the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> that defines how data and pilot symbols
are mapped onto a sequence of OFDM symbols with a given FFT size. The resource
grid can also define guard and DC carriers which are nulled. In 4G/5G parlance,
a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> would be a slot.
Once a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> is defined, one can use the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGridMapper" title="sionna.ofdm.ResourceGridMapper">`ResourceGridMapper`</a> to map a tensor of complex-valued
data symbols onto the resource grid, prior to OFDM modulation using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.OFDMModulator" title="sionna.ofdm.OFDMModulator">`OFDMModulator`</a> or further processing in the
frequency domain.
    
The <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a> allows for a fine-grained configuration
of how transmitters send pilots for each of their streams or antennas. As the
management of pilots in multi-cell MIMO setups can quickly become complicated,
the module provides the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.KroneckerPilotPattern" title="sionna.ofdm.KroneckerPilotPattern">`KroneckerPilotPattern`</a> class
that automatically generates orthogonal pilot transmissions for all transmitters
and streams.
    
Additionally, the module contains layers for channel estimation, precoding,
equalization, and detection,
such as the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LSChannelEstimator" title="sionna.ofdm.LSChannelEstimator">`LSChannelEstimator`</a>, the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ZFPrecoder" title="sionna.ofdm.ZFPrecoder">`ZFPrecoder`</a>, and the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LMMSEEqualizer" title="sionna.ofdm.LMMSEEqualizer">`LMMSEEqualizer`</a> and
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LinearDetector" title="sionna.ofdm.LinearDetector">`LinearDetector`</a>.
These are good starting points for the development of more advanced algorithms
and provide robust baselines for benchmarking.

## Resource Grid<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#resource-grid" title="Permalink to this headline"></a>
    
The following code snippet shows how to setup and visualize an instance of
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>:
```python
rg = ResourceGrid(num_ofdm_symbols = 14,
                  fft_size = 64,
                  subcarrier_spacing = 30e3,
                  num_tx = 1,
                  num_streams_per_tx = 1,
                  num_guard_carriers = [5, 6],
                  dc_null = True,
                  pilot_pattern = "kronecker",
                  pilot_ofdm_symbol_indices = [2, 11])
rg.show();
```

<img alt="../_images/resource_grid.png" src="https://nvlabs.github.io/sionna/_images/resource_grid.png" />
    
This code creates a resource grid consisting of 14 OFDM symbols with 64
subcarriers. The first five and last six subcarriers as well as the DC
subcarriers are nulled. The second and eleventh OFDM symbol are reserved
for pilot transmissions.
    
Subcarriers are numbered from $0$ to $N-1$, where $N$
is the FTT size. The index $0$ corresponds to the lowest frequency,
which is $-\frac{N}{2}\Delta_f$ (for $N$ even) or
$-\frac{N-1}{2}\Delta_f$ (for $N$ odd), where $\Delta_f$
is the subcarrier spacing which is irrelevant for the resource grid.
The index $N-1$ corresponds to the highest frequency,
which is $(\frac{N}{2}-1)\Delta_f$ (for $N$ even) or
$\frac{N-1}{2}\Delta_f$ (for $N$ odd).

### ResourceGrid<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#resourcegrid" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``ResourceGrid`(<em class="sig-param">`num_ofdm_symbols`</em>, <em class="sig-param">`fft_size`</em>, <em class="sig-param">`subcarrier_spacing`</em>, <em class="sig-param">`num_tx``=``1`</em>, <em class="sig-param">`num_streams_per_tx``=``1`</em>, <em class="sig-param">`cyclic_prefix_length``=``0`</em>, <em class="sig-param">`num_guard_carriers``=``(0,` `0)`</em>, <em class="sig-param">`dc_null``=``False`</em>, <em class="sig-param">`pilot_pattern``=``None`</em>, <em class="sig-param">`pilot_ofdm_symbol_indices``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/resource_grid.html#ResourceGrid">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="Permalink to this definition"></a>
    
Defines a <cite>ResourceGrid</cite> spanning multiple OFDM symbols and subcarriers.
Parameters
 
- **num_ofdm_symbols** (<em>int</em>) – Number of OFDM symbols.
- **fft_size** (<em>int</em>) – FFT size (, i.e., the number of subcarriers).
- **subcarrier_spacing** (<em>float</em>) – The subcarrier spacing in Hz.
- **num_tx** (<em>int</em>) – Number of transmitters.
- **num_streams_per_tx** (<em>int</em>) – Number of streams per transmitter.
- **cyclic_prefix_length** (<em>int</em>) – Length of the cyclic prefix.
- **num_guard_carriers** (<em>int</em>) – List of two integers defining the number of guardcarriers at the
left and right side of the resource grid.
- **dc_null** (<em>bool</em>) – Indicates if the DC carrier is nulled or not.
- **pilot_pattern** (<em>One of</em><em> [</em><em>None</em><em>, </em><em>"kronecker"</em><em>, </em><em>"empty"</em><em>, </em><a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern"><em>PilotPattern</em></a><em>]</em>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a>, a string
shorthand for the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.KroneckerPilotPattern" title="sionna.ofdm.KroneckerPilotPattern">`KroneckerPilotPattern`</a>
or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.EmptyPilotPattern" title="sionna.ofdm.EmptyPilotPattern">`EmptyPilotPattern`</a>, or <cite>None</cite>.
Defaults to <cite>None</cite> which is equivalent to <cite>“empty”</cite>.
- **pilot_ofdm_symbol_indices** (<em>List</em><em>, </em><em>int</em>) – List of indices of OFDM symbols reserved for pilot transmissions.
Only needed if `pilot_pattern="kronecker"`. Defaults to <cite>None</cite>.
- **dtype** (<em>tf.Dtype</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.




<em class="property">`property` </em>`bandwidth`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.bandwidth" title="Permalink to this definition"></a>
    
`fft_size*subcarrier_spacing`.
Type
    
The occupied bandwidth [Hz]




`build_type_grid`()<a class="reference internal" href="../_modules/sionna/ofdm/resource_grid.html#ResourceGrid.build_type_grid">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.build_type_grid" title="Permalink to this definition"></a>
    
Returns a tensor indicating the type of each resource element.
    
Resource elements can be one of
 
- 0 : Data symbol
- 1 : Pilot symbol
- 2 : Guard carrier symbol
- 3 : DC carrier symbol

Output
    
<em>[num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.int32</em> – Tensor indicating for each transmitter and stream the type of
the resource elements of the corresponding resource grid.
The type can be one of [0,1,2,3] as explained above.




<em class="property">`property` </em>`cyclic_prefix_length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.cyclic_prefix_length" title="Permalink to this definition"></a>
    
Length of the cyclic prefix.


<em class="property">`property` </em>`dc_ind`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.dc_ind" title="Permalink to this definition"></a>
    
Index of the DC subcarrier.
    
If `fft_size` is odd, the index is (`fft_size`-1)/2.
If `fft_size` is even, the index is `fft_size`/2.


<em class="property">`property` </em>`dc_null`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.dc_null" title="Permalink to this definition"></a>
    
Indicates if the DC carriers is nulled or not.


<em class="property">`property` </em>`effective_subcarrier_ind`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.effective_subcarrier_ind" title="Permalink to this definition"></a>
    
Returns the indices of the effective subcarriers.


<em class="property">`property` </em>`fft_size`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.fft_size" title="Permalink to this definition"></a>
    
The FFT size.


<em class="property">`property` </em>`num_data_symbols`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.num_data_symbols" title="Permalink to this definition"></a>
    
Number of resource elements used for data transmissions.


<em class="property">`property` </em>`num_effective_subcarriers`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.num_effective_subcarriers" title="Permalink to this definition"></a>
    
Number of subcarriers used for data and pilot transmissions.


<em class="property">`property` </em>`num_guard_carriers`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.num_guard_carriers" title="Permalink to this definition"></a>
    
Number of left and right guard carriers.


<em class="property">`property` </em>`num_ofdm_symbols`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.num_ofdm_symbols" title="Permalink to this definition"></a>
    
The number of OFDM symbols of the resource grid.


<em class="property">`property` </em>`num_pilot_symbols`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.num_pilot_symbols" title="Permalink to this definition"></a>
    
Number of resource elements used for pilot symbols.


<em class="property">`property` </em>`num_resource_elements`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.num_resource_elements" title="Permalink to this definition"></a>
    
Number of resource elements.


<em class="property">`property` </em>`num_streams_per_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.num_streams_per_tx" title="Permalink to this definition"></a>
    
Number of streams  per transmitter.


<em class="property">`property` </em>`num_time_samples`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.num_time_samples" title="Permalink to this definition"></a>
    
The number of time-domain samples occupied by the resource grid.


<em class="property">`property` </em>`num_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.num_tx" title="Permalink to this definition"></a>
    
Number of transmitters.


<em class="property">`property` </em>`num_zero_symbols`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.num_zero_symbols" title="Permalink to this definition"></a>
    
Number of empty resource elements.


<em class="property">`property` </em>`ofdm_symbol_duration`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.ofdm_symbol_duration" title="Permalink to this definition"></a>
    
Duration of an OFDM symbol with cyclic prefix [s].


<em class="property">`property` </em>`pilot_pattern`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.pilot_pattern" title="Permalink to this definition"></a>
    
The used PilotPattern.


`show`(<em class="sig-param">`tx_ind``=``0`</em>, <em class="sig-param">`tx_stream_ind``=``0`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/resource_grid.html#ResourceGrid.show">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.show" title="Permalink to this definition"></a>
    
Visualizes the resource grid for a specific transmitter and stream.
Input
 
- **tx_ind** (<em>int</em>) – Indicates the transmitter index.
- **tx_stream_ind** (<em>int</em>) – Indicates the index of the stream.


Output
    
<cite>matplotlib.figure</cite> – A handle to a matplot figure object.




<em class="property">`property` </em>`subcarrier_spacing`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.subcarrier_spacing" title="Permalink to this definition"></a>
    
The subcarrier spacing [Hz].


### ResourceGridMapper<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#resourcegridmapper" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``ResourceGridMapper`(<em class="sig-param">`resource_grid`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/resource_grid.html#ResourceGridMapper">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGridMapper" title="Permalink to this definition"></a>
    
Maps a tensor of modulated data symbols to a ResourceGrid.
    
This layer takes as input a tensor of modulated data symbols
and maps them together with pilot symbols onto an
OFDM <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>. The output can be
converted to a time-domain signal with the
`Modulator` or further processed in the
frequency domain.
Parameters
 
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
- **dtype** (<em>tf.Dtype</em>) – Datatype for internal calculations and the output dtype.
Defaults to <cite>tf.complex64</cite>.


Input
    
<em>[batch_size, num_tx, num_streams_per_tx, num_data_symbols], tf.complex</em> – The modulated data symbols to be mapped onto the resource grid.

Output
    
<em>[batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex</em> – The full OFDM resource grid in the frequency domain.



### ResourceGridDemapper<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#resourcegriddemapper" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``ResourceGridDemapper`(<em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/resource_grid.html#ResourceGridDemapper">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGridDemapper" title="Permalink to this definition"></a>
    
Extracts data-carrying resource elements from a resource grid.
    
This layer takes as input an OFDM <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> and
extracts the data-carrying resource elements. In other words, it implements
the reverse operation of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGridMapper" title="sionna.ofdm.ResourceGridMapper">`ResourceGridMapper`</a>.
Parameters
 
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – An instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>.
- **dtype** (<em>tf.Dtype</em>) – Datatype for internal calculations and the output dtype.
Defaults to <cite>tf.complex64</cite>.


Input
    
<em>[batch_size, num_rx, num_streams_per_rx, num_ofdm_symbols, fft_size, data_dim]</em> – The full OFDM resource grid in the frequency domain.
The last dimension <cite>data_dim</cite> is optional. If <cite>data_dim</cite>
is used, it refers to the dimensionality of the data that should be
demapped to individual streams. An example would be LLRs.

Output
    
<em>[batch_size, num_rx, num_streams_per_rx, num_data_symbols, data_dim]</em> – The data that were mapped into the resource grid.
The last dimension <cite>data_dim</cite> is only returned if it was used for the
input.



### RemoveNulledSubcarriers<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#removenulledsubcarriers" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``RemoveNulledSubcarriers`(<em class="sig-param">`resource_grid`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/resource_grid.html#RemoveNulledSubcarriers">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.RemoveNulledSubcarriers" title="Permalink to this definition"></a>
    
Removes nulled guard and/or DC subcarriers from a resource grid.
Parameters
    
**resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.

Input
    
<em>[batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex64</em> – Full resource grid.

Output
    
<em>[batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex64</em> – Resource grid without nulled subcarriers.



## Modulation & Demodulation<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#modulation-demodulation" title="Permalink to this headline"></a>

### OFDMModulator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#ofdmmodulator" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``OFDMModulator`(<em class="sig-param">`cyclic_prefix_length`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/modulator.html#OFDMModulator">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.OFDMModulator" title="Permalink to this definition"></a>
    
Computes the time-domain representation of an OFDM resource grid
with (optional) cyclic prefix.
Parameters
    
**cyclic_prefix_length** (<em>int</em>) – Integer indicating the length of the
cyclic prefix that it prepended to each OFDM symbol. It cannot
be longer than the FFT size.

Input
    
<em>[…,num_ofdm_symbols,fft_size], tf.complex</em> – A resource grid in the frequency domain.

Output
    
<em>[…,num_ofdm_symbols*(fft_size+cyclic_prefix_length)], tf.complex</em> – Time-domain OFDM signal.



### OFDMDemodulator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#ofdmdemodulator" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``OFDMDemodulator`(<em class="sig-param">`fft_size`</em>, <em class="sig-param">`l_min`</em>, <em class="sig-param">`cyclic_prefix_length`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/demodulator.html#OFDMDemodulator">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.OFDMDemodulator" title="Permalink to this definition"></a>
    
Computes the frequency-domain representation of an OFDM waveform
with cyclic prefix removal.
    
The demodulator assumes that the input sequence is generated by the
<a class="reference internal" href="channel.wireless.html#sionna.channel.TimeChannel" title="sionna.channel.TimeChannel">`TimeChannel`</a>. For a single pair of antennas,
the received signal sequence is given as:

$$
y_b = \sum_{\ell =L_\text{min}}^{L_\text{max}} \bar{h}_\ell x_{b-\ell} + w_b, \quad b \in[L_\text{min}, N_B+L_\text{max}-1]
$$
    
where $\bar{h}_\ell$ are the discrete-time channel taps,
$x_{b}$ is the the transmitted signal,
and $w_\ell$ Gaussian noise.
    
Starting from the first symbol, the demodulator cuts the input
sequence into pieces of size `cyclic_prefix_length` `+` `fft_size`,
and throws away any trailing symbols. For each piece, the cyclic
prefix is removed and the `fft_size`-point discrete Fourier
transform is computed.
    
Since the input sequence starts at time $L_\text{min}$,
the FFT-window has a timing offset of $L_\text{min}$ symbols,
which leads to a subcarrier-dependent phase shift of
$e^{\frac{j2\pi k L_\text{min}}{N}}$, where $k$
is the subcarrier index, $N$ is the FFT size,
and $L_\text{min} \le 0$ is the largest negative time lag of
the discrete-time channel impulse response. This phase shift
is removed in this layer, by explicitly multiplying
each subcarrier by  $e^{\frac{-j2\pi k L_\text{min}}{N}}$.
This is a very important step to enable channel estimation with
sparse pilot patterns that needs to interpolate the channel frequency
response accross subcarriers. It also ensures that the
channel frequency response <cite>seen</cite> by the time-domain channel
is close to the <a class="reference internal" href="channel.wireless.html#sionna.channel.OFDMChannel" title="sionna.channel.OFDMChannel">`OFDMChannel`</a>.
Parameters
 
- **fft_size** (<em>int</em>) – FFT size (, i.e., the number of subcarriers).
- **l_min** (<em>int</em>) – The largest negative time lag of the discrete-time channel
impulse response. It should be the same value as that used by the
<cite>cir_to_time_channel</cite> function.
- **cyclic_prefix_length** (<em>int</em>) – Integer indicating the length of the cyclic prefix that
is prepended to each OFDM symbol.


Input
    
<em>[…,num_ofdm_symbols*(fft_size+cyclic_prefix_length)+n], tf.complex</em> – Tensor containing the time-domain signal along the last dimension.
<cite>n</cite> is a nonnegative integer.

Output
    
<em>[…,num_ofdm_symbols,fft_size], tf.complex</em> – Tensor containing the OFDM resource grid along the last
two dimension.



## Pilot Pattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#pilot-pattern" title="Permalink to this headline"></a>
    
A <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a> defines how transmitters send pilot
sequences for each of their antennas or streams over an OFDM resource grid.
It consists of two components,
a `mask` and `pilots`. The `mask` indicates which resource elements are
reserved for pilot transmissions by each transmitter and its respective
streams. In some cases, the number of streams is equal to the number of
transmit antennas, but this does not need to be the case, e.g., for precoded
transmissions. The `pilots` contains the pilot symbols that are transmitted
at the positions indicated by the `mask`. Separating a pilot pattern into
`mask` and `pilots` enables the implementation of a wide range of pilot
configurations, including trainable pilot sequences.
    
The following code snippet shows how to define a simple custom
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a> for single transmitter, sending two streams
Note that `num_effective_subcarriers` is the number of subcarriers that
can be used for data or pilot transmissions. Due to guard
carriers or a nulled DC carrier, this number can be smaller than the
`fft_size` of the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
```python
num_tx = 1
num_streams_per_tx = 2
num_ofdm_symbols = 14
num_effective_subcarriers = 12
# Create a pilot mask
mask = np.zeros([num_tx,
                 num_streams_per_tx,
                 num_ofdm_symbols,
                 num_effective_subcarriers])
mask[0, :, [2,11], :] = 1
num_pilot_symbols = int(np.sum(mask[0,0]))
# Define pilot sequences
pilots = np.zeros([num_tx,
                   num_streams_per_tx,
                   num_pilot_symbols], np.complex64)
pilots[0, 0, 0:num_pilot_symbols:2] = (1+1j)/np.sqrt(2)
pilots[0, 1, 1:num_pilot_symbols:2] = (1+1j)/np.sqrt(2)
# Create a PilotPattern instance
pp = PilotPattern(mask, pilots)
# Visualize non-zero elements of the pilot sequence
pp.show(show_pilot_ind=True);
```

<img alt="../_images/pilot_pattern.png" src="https://nvlabs.github.io/sionna/_images/pilot_pattern.png" />
<img alt="../_images/pilot_pattern_2.png" src="https://nvlabs.github.io/sionna/_images/pilot_pattern_2.png" />
    
As shown in the figures above, the pilots are mapped onto the mask from
the smallest effective subcarrier and OFDM symbol index to the highest
effective subcarrier and OFDM symbol index. Here, boths stream have 24
pilot symbols, out of which only 12 are nonzero. It is important to keep
this order of mapping in mind when designing more complex pilot sequences.

### PilotPattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#pilotpattern" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``PilotPattern`(<em class="sig-param">`mask`</em>, <em class="sig-param">`pilots`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`normalize``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/pilot_pattern.html#PilotPattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="Permalink to this definition"></a>
    
Class defining a pilot pattern for an OFDM ResourceGrid.
    
This class defines a pilot pattern object that is used to configure
an OFDM <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
Parameters
 
- **mask** (<em>[</em><em>num_tx</em><em>, </em><em>num_streams_per_tx</em><em>, </em><em>num_ofdm_symbols</em><em>, </em><em>num_effective_subcarriers</em><em>]</em><em>, </em><em>bool</em>) – Tensor indicating resource elements that are reserved for pilot transmissions.
- **pilots** (<em>[</em><em>num_tx</em><em>, </em><em>num_streams_per_tx</em><em>, </em><em>num_pilots</em><em>]</em><em>, </em><em>tf.complex</em>) – The pilot symbols to be mapped onto the `mask`.
- **trainable** (<em>bool</em>) – Indicates if `pilots` is a trainable <cite>Variable</cite>.
Defaults to <cite>False</cite>.
- **normalize** (<em>bool</em>) – Indicates if the `pilots` should be normalized to an average
energy of one across the last dimension. This can be useful to
ensure that trainable `pilots` have a finite energy.
Defaults to <cite>False</cite>.
- **dtype** (<em>tf.Dtype</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.




<em class="property">`property` </em>`mask`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern.mask" title="Permalink to this definition"></a>
    
Mask of the pilot pattern


<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern.normalize" title="Permalink to this definition"></a>
    
Returns or sets the flag indicating if the pilots
are normalized or not


<em class="property">`property` </em>`num_data_symbols`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern.num_data_symbols" title="Permalink to this definition"></a>
    
Number of data symbols per transmit stream.


<em class="property">`property` </em>`num_effective_subcarriers`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern.num_effective_subcarriers" title="Permalink to this definition"></a>
    
Number of effectvie subcarriers


<em class="property">`property` </em>`num_ofdm_symbols`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern.num_ofdm_symbols" title="Permalink to this definition"></a>
    
Number of OFDM symbols


<em class="property">`property` </em>`num_pilot_symbols`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern.num_pilot_symbols" title="Permalink to this definition"></a>
    
Number of pilot symbols per transmit stream.


<em class="property">`property` </em>`num_streams_per_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern.num_streams_per_tx" title="Permalink to this definition"></a>
    
Number of streams per transmitter


<em class="property">`property` </em>`num_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern.num_tx" title="Permalink to this definition"></a>
    
Number of transmitters


<em class="property">`property` </em>`pilots`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern.pilots" title="Permalink to this definition"></a>
    
Returns or sets the possibly normalized tensor of pilot symbols.
If pilots are normalized, the normalization will be applied
after new values for pilots have been set. If this is
not the desired behavior, turn normalization off.


`show`(<em class="sig-param">`tx_ind``=``None`</em>, <em class="sig-param">`stream_ind``=``None`</em>, <em class="sig-param">`show_pilot_ind``=``False`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/pilot_pattern.html#PilotPattern.show">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern.show" title="Permalink to this definition"></a>
    
Visualizes the pilot patterns for some transmitters and streams.
Input
 
- **tx_ind** (<em>list, int</em>) – Indicates the indices of transmitters to be included.
Defaults to <cite>None</cite>, i.e., all transmitters included.
- **stream_ind** (<em>list, int</em>) – Indicates the indices of streams to be included.
Defaults to <cite>None</cite>, i.e., all streams included.
- **show_pilot_ind** (<em>bool</em>) – Indicates if the indices of the pilot symbols should be shown.


Output
    
**list** (<em>matplotlib.figure.Figure</em>) – List of matplot figure objects showing each the pilot pattern
from a specific transmitter and stream.




<em class="property">`property` </em>`trainable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern.trainable" title="Permalink to this definition"></a>
    
Returns if pilots are trainable or not


### EmptyPilotPattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#emptypilotpattern" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``EmptyPilotPattern`(<em class="sig-param">`num_tx`</em>, <em class="sig-param">`num_streams_per_tx`</em>, <em class="sig-param">`num_ofdm_symbols`</em>, <em class="sig-param">`num_effective_subcarriers`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/pilot_pattern.html#EmptyPilotPattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.EmptyPilotPattern" title="Permalink to this definition"></a>
    
Creates an empty pilot pattern.
    
Generates a instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a> with
an empty `mask` and `pilots`.
Parameters
 
- **num_tx** (<em>int</em>) – Number of transmitters.
- **num_streams_per_tx** (<em>int</em>) – Number of streams per transmitter.
- **num_ofdm_symbols** (<em>int</em>) – Number of OFDM symbols.
- **num_effective_subcarriers** (<em>int</em>) – Number of effective subcarriers
that are available for the transmission of data and pilots.
Note that this number is generally smaller than the `fft_size`
due to nulled subcarriers.
- **dtype** (<em>tf.Dtype</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.




### KroneckerPilotPattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#kroneckerpilotpattern" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``KroneckerPilotPattern`(<em class="sig-param">`resource_grid`</em>, <em class="sig-param">`pilot_ofdm_symbol_indices`</em>, <em class="sig-param">`normalize``=``True`</em>, <em class="sig-param">`seed``=``0`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/pilot_pattern.html#KroneckerPilotPattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.KroneckerPilotPattern" title="Permalink to this definition"></a>
    
Simple orthogonal pilot pattern with Kronecker structure.
    
This function generates an instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a>
that allocates non-overlapping pilot sequences for all transmitters and
streams on specified OFDM symbols. As the same pilot sequences are reused
across those OFDM symbols, the resulting pilot pattern has a frequency-time
Kronecker structure. This structure enables a very efficient implementation
of the LMMSE channel estimator. Each pilot sequence is constructed from
randomly drawn QPSK constellation points.
Parameters
 
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – An instance of a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
- **pilot_ofdm_symbol_indices** (<em>list</em><em>, </em><em>int</em>) – List of integers defining the OFDM symbol indices that are reserved
for pilots.
- **normalize** (<em>bool</em>) – Indicates if the `pilots` should be normalized to an average
energy of one across the last dimension.
Defaults to <cite>True</cite>.
- **seed** (<em>int</em>) – Seed for the generation of the pilot sequence. Different seed values
lead to different sequences. Defaults to 0.
- **dtype** (<em>tf.Dtype</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.




**Note**
    
It is required that the `resource_grid`’s property
`num_effective_subcarriers` is an
integer multiple of `num_tx` `*` `num_streams_per_tx`. This condition is
required to ensure that all transmitters and streams get
non-overlapping pilot sequences. For a large number of streams and/or
transmitters, the pilot pattern becomes very sparse in the frequency
domain.
<p class="rubric">Examples
```python
>>> rg = ResourceGrid(num_ofdm_symbols=14,
...                   fft_size=64,
...                   subcarrier_spacing = 30e3,
...                   num_tx=4,
...                   num_streams_per_tx=2,
...                   pilot_pattern = "kronecker",
...                   pilot_ofdm_symbol_indices = [2, 11])
>>> rg.pilot_pattern.show();
```

<img alt="../_images/kronecker_pilot_pattern.png" src="https://nvlabs.github.io/sionna/_images/kronecker_pilot_pattern.png" />

## Channel Estimation<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#channel-estimation" title="Permalink to this headline"></a>

### BaseChannelEstimator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#basechannelestimator" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``BaseChannelEstimator`(<em class="sig-param">`resource_grid`</em>, <em class="sig-param">`interpolation_type``=``'nn'`</em>, <em class="sig-param">`interpolator``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/channel_estimation.html#BaseChannelEstimator">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.BaseChannelEstimator" title="Permalink to this definition"></a>
    
Abstract layer for implementing an OFDM channel estimator.
    
Any layer that implements an OFDM channel estimator must implement this
class and its
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.BaseChannelEstimator.estimate_at_pilot_locations" title="sionna.ofdm.BaseChannelEstimator.estimate_at_pilot_locations">`estimate_at_pilot_locations()`</a>
abstract method.
    
This class extracts the pilots from the received resource grid `y`, calls
the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.BaseChannelEstimator.estimate_at_pilot_locations" title="sionna.ofdm.BaseChannelEstimator.estimate_at_pilot_locations">`estimate_at_pilot_locations()`</a>
method to estimate the channel for the pilot-carrying resource elements,
and then interpolates the channel to compute channel estimates for the
data-carrying resouce elements using the interpolation method specified by
`interpolation_type` or the `interpolator` object.
Parameters
 
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
- **interpolation_type** (<em>One of</em><em> [</em><em>"nn"</em><em>, </em><em>"lin"</em><em>, </em><em>"lin_time_avg"</em><em>]</em><em>, </em><em>string</em>) – The interpolation method to be used.
It is ignored if `interpolator` is not <cite>None</cite>.
Available options are <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.NearestNeighborInterpolator" title="sionna.ofdm.NearestNeighborInterpolator">`NearestNeighborInterpolator`</a> (<cite>“nn</cite>”)
or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LinearInterpolator" title="sionna.ofdm.LinearInterpolator">`LinearInterpolator`</a> without (<cite>“lin”</cite>) or with
averaging across OFDM symbols (<cite>“lin_time_avg”</cite>).
Defaults to “nn”.
- **interpolator** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.BaseChannelInterpolator" title="sionna.ofdm.BaseChannelInterpolator"><em>BaseChannelInterpolator</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.BaseChannelInterpolator" title="sionna.ofdm.BaseChannelInterpolator">`BaseChannelInterpolator`</a>,
such as <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LMMSEInterpolator" title="sionna.ofdm.LMMSEInterpolator">`LMMSEInterpolator`</a>,
or <cite>None</cite>. In the latter case, the interpolator specfied
by `interpolation_type` is used.
Otherwise, the `interpolator` is used and `interpolation_type`
is ignored.
Defaults to <cite>None</cite>.
- **dtype** (<em>tf.Dtype</em>) – Datatype for internal calculations and the output dtype.
Defaults to <cite>tf.complex64</cite>.


Input
 
- **(y, no)** – Tuple:
- **y** (<em>[batch_size, num_rx, num_rx_ant, num_ofdm_symbols,fft_size], tf.complex</em>) – Observed resource grid
- **no** (<em>[batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, tf.float</em>) – Variance of the AWGN


Output
 
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,fft_size], tf.complex</em>) – Channel estimates accross the entire resource grid for all
transmitters and streams
- **err_var** (Same shape as `h_hat`, tf.float) – Channel estimation error variance accross the entire resource grid
for all transmitters and streams




<em class="property">`abstract` </em>`estimate_at_pilot_locations`(<em class="sig-param">`y_pilots`</em>, <em class="sig-param">`no`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/channel_estimation.html#BaseChannelEstimator.estimate_at_pilot_locations">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.BaseChannelEstimator.estimate_at_pilot_locations" title="Permalink to this definition"></a>
    
Estimates the channel for the pilot-carrying resource elements.
    
This is an abstract method that must be implemented by a concrete
OFDM channel estimator that implement this class.
Input
 
- **y_pilots** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams, num_pilot_symbols], tf.complex</em>) – Observed signals for the pilot-carrying resource elements
- **no** (<em>[batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, tf.float</em>) – Variance of the AWGN


Output
 
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams, num_pilot_symbols], tf.complex</em>) – Channel estimates for the pilot-carrying resource elements
- **err_var** (Same shape as `h_hat`, tf.float) – Channel estimation error variance for the pilot-carrying
resource elements





### BaseChannelInterpolator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#basechannelinterpolator" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``BaseChannelInterpolator`<a class="reference internal" href="../_modules/sionna/ofdm/channel_estimation.html#BaseChannelInterpolator">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.BaseChannelInterpolator" title="Permalink to this definition"></a>
    
Abstract layer for implementing an OFDM channel interpolator.
    
Any layer that implements an OFDM channel interpolator must implement this
callable class.
    
A channel interpolator is used by an OFDM channel estimator
(<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.BaseChannelEstimator" title="sionna.ofdm.BaseChannelEstimator">`BaseChannelEstimator`</a>) to compute channel estimates
for the data-carrying resource elements from the channel estimates for the
pilot-carrying resource elements.
Input
 
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex</em>) – Channel estimates for the pilot-carrying resource elements
- **err_var** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex</em>) – Channel estimation error variances for the pilot-carrying resource elements


Output
 
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex</em>) – Channel estimates accross the entire resource grid for all
transmitters and streams
- **err_var** (Same shape as `h_hat`, tf.float) – Channel estimation error variance accross the entire resource grid
for all transmitters and streams




### LSChannelEstimator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#lschannelestimator" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``LSChannelEstimator`(<em class="sig-param">`resource_grid`</em>, <em class="sig-param">`interpolation_type``=``'nn'`</em>, <em class="sig-param">`interpolator``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/channel_estimation.html#LSChannelEstimator">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LSChannelEstimator" title="Permalink to this definition"></a>
    
Layer implementing least-squares (LS) channel estimation for OFDM MIMO systems.
    
After LS channel estimation at the pilot positions, the channel estimates
and error variances are interpolated accross the entire resource grid using
a specified interpolation function.
    
For simplicity, the underlying algorithm is described for a vectorized observation,
where we have a nonzero pilot for all elements to be estimated.
The actual implementation works on a full OFDM resource grid with sparse
pilot patterns. The following model is assumed:

$$
\mathbf{y} = \mathbf{h}\odot\mathbf{p} + \mathbf{n}
$$
    
where $\mathbf{y}\in\mathbb{C}^{M}$ is the received signal vector,
$\mathbf{p}\in\mathbb{C}^M$ is the vector of pilot symbols,
$\mathbf{h}\in\mathbb{C}^{M}$ is the channel vector to be estimated,
and $\mathbf{n}\in\mathbb{C}^M$ is a zero-mean noise vector whose
elements have variance $N_0$. The operator $\odot$ denotes
element-wise multiplication.
    
The channel estimate $\hat{\mathbf{h}}$ and error variances
$\sigma^2_i$, $i=0,\dots,M-1$, are computed as

$$
\begin{split}\hat{\mathbf{h}} &= \mathbf{y} \odot
                   \frac{\mathbf{p}^\star}{\left|\mathbf{p}\right|^2}
                 = \mathbf{h} + \tilde{\mathbf{h}}\\
     \sigma^2_i &= \mathbb{E}\left[\tilde{h}_i \tilde{h}_i^\star \right]
                 = \frac{N_0}{\left|p_i\right|^2}.\end{split}
$$
    
The channel estimates and error variances are then interpolated accross
the entire resource grid.
Parameters
 
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
- **interpolation_type** (<em>One of</em><em> [</em><em>"nn"</em><em>, </em><em>"lin"</em><em>, </em><em>"lin_time_avg"</em><em>]</em><em>, </em><em>string</em>) – The interpolation method to be used.
It is ignored if `interpolator` is not <cite>None</cite>.
Available options are <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.NearestNeighborInterpolator" title="sionna.ofdm.NearestNeighborInterpolator">`NearestNeighborInterpolator`</a> (<cite>“nn</cite>”)
or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LinearInterpolator" title="sionna.ofdm.LinearInterpolator">`LinearInterpolator`</a> without (<cite>“lin”</cite>) or with
averaging across OFDM symbols (<cite>“lin_time_avg”</cite>).
Defaults to “nn”.
- **interpolator** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.BaseChannelInterpolator" title="sionna.ofdm.BaseChannelInterpolator"><em>BaseChannelInterpolator</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.BaseChannelInterpolator" title="sionna.ofdm.BaseChannelInterpolator">`BaseChannelInterpolator`</a>,
such as <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LMMSEInterpolator" title="sionna.ofdm.LMMSEInterpolator">`LMMSEInterpolator`</a>,
or <cite>None</cite>. In the latter case, the interpolator specfied
by `interpolation_type` is used.
Otherwise, the `interpolator` is used and `interpolation_type`
is ignored.
Defaults to <cite>None</cite>.
- **dtype** (<em>tf.Dtype</em>) – Datatype for internal calculations and the output dtype.
Defaults to <cite>tf.complex64</cite>.


Input
 
- **(y, no)** – Tuple:
- **y** (<em>[batch_size, num_rx, num_rx_ant, num_ofdm_symbols,fft_size], tf.complex</em>) – Observed resource grid
- **no** (<em>[batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, tf.float</em>) – Variance of the AWGN


Output
 
- **h_ls** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,fft_size], tf.complex</em>) – Channel estimates accross the entire resource grid for all
transmitters and streams
- **err_var** (Same shape as `h_ls`, tf.float) – Channel estimation error variance accross the entire resource grid
for all transmitters and streams




### LinearInterpolator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#linearinterpolator" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``LinearInterpolator`(<em class="sig-param">`pilot_pattern`</em>, <em class="sig-param">`time_avg``=``False`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/channel_estimation.html#LinearInterpolator">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LinearInterpolator" title="Permalink to this definition"></a>
    
Linear channel estimate interpolation on a resource grid.
    
This class computes for each element of an OFDM resource grid
a channel estimate based on `num_pilots` provided channel estimates and
error variances through linear interpolation.
It is assumed that the measurements were taken at the nonzero positions
of a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a>.
    
The interpolation is done first across sub-carriers and then
across OFDM symbols.
Parameters
 
- **pilot_pattern** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern"><em>PilotPattern</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a>
- **time_avg** (<em>bool</em>) – If enabled, measurements will be averaged across OFDM symbols
(i.e., time). This is useful for channels that do not vary
substantially over the duration of an OFDM frame. Defaults to <cite>False</cite>.


Input
 
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex</em>) – Channel estimates for the pilot-carrying resource elements
- **err_var** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex</em>) – Channel estimation error variances for the pilot-carrying resource elements


Output
 
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex</em>) – Channel estimates accross the entire resource grid for all
transmitters and streams
- **err_var** (Same shape as `h_hat`, tf.float) – Channel estimation error variances accross the entire resource grid
for all transmitters and streams




### LMMSEInterpolator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#lmmseinterpolator" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``LMMSEInterpolator`(<em class="sig-param">`pilot_pattern`</em>, <em class="sig-param">`cov_mat_time`</em>, <em class="sig-param">`cov_mat_freq`</em>, <em class="sig-param">`cov_mat_space``=``None`</em>, <em class="sig-param">`order``=``'t-f'`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/channel_estimation.html#LMMSEInterpolator">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LMMSEInterpolator" title="Permalink to this definition"></a>
    
LMMSE interpolation on a resource grid with optional spatial smoothing.
    
This class computes for each element of an OFDM resource grid
a channel estimate and error variance
through linear minimum mean square error (LMMSE) interpolation/smoothing.
It is assumed that the measurements were taken at the nonzero positions
of a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a>.
    
Depending on the value of `order`, the interpolation is carried out
accross time (t), i.e., OFDM symbols, frequency (f), i.e., subcarriers,
and optionally space (s), i.e., receive antennas, in any desired order.
    
For simplicity, we describe the underlying algorithm assuming that interpolation
across the sub-carriers is performed first, followed by interpolation across
OFDM symbols, and finally by spatial smoothing across receive
antennas.
The algorithm is similar if interpolation and/or smoothing are performed in
a different order.
For clarity, antenna indices are omitted when describing frequency and time
interpolation, as the same process is applied to all the antennas.
    
The input `h_hat` is first reshaped to a resource grid
$\hat{\mathbf{H}} \in \mathbb{C}^{N \times M}$, by scattering the channel
estimates at pilot locations according to the `pilot_pattern`. $N$
denotes the number of OFDM symbols and $M$ the number of sub-carriers.
    
The first pass consists in interpolating across the sub-carriers:

$$
\hat{\mathbf{h}}_n^{(1)} = \mathbf{A}_n \hat{\mathbf{h}}_n
$$
    
where $1 \leq n \leq N$ is the OFDM symbol index and $\hat{\mathbf{h}}_n$ is
the $n^{\text{th}}$ (transposed) row of $\hat{\mathbf{H}}$.
$\mathbf{A}_n$ is the $M \times M$ matrix such that:

$$
\mathbf{A}_n = \bar{\mathbf{A}}_n \mathbf{\Pi}_n^\intercal
$$
    
where

$$
\bar{\mathbf{A}}_n = \underset{\mathbf{Z} \in \mathbb{C}^{M \times K_n}}{\text{argmin}} \left\lVert \mathbf{Z}\left( \mathbf{\Pi}_n^\intercal \mathbf{R^{(f)}} \mathbf{\Pi}_n + \mathbf{\Sigma}_n \right) - \mathbf{R^{(f)}} \mathbf{\Pi}_n \right\rVert_{\text{F}}^2
$$
    
and $\mathbf{R^{(f)}}$ is the $M \times M$ channel frequency covariance matrix,
$\mathbf{\Pi}_n$ the $M \times K_n$ matrix that spreads $K_n$
values to a vector of size $M$ according to the `pilot_pattern` for the $n^{\text{th}}$ OFDM symbol,
and $\mathbf{\Sigma}_n \in \mathbb{R}^{K_n \times K_n}$ is the channel estimation error covariance built from
`err_var` and assumed to be diagonal.
Computation of $\bar{\mathbf{A}}_n$ is done using an algorithm based on complete orthogonal decomposition.
This is done to avoid matrix inversion for badly conditioned covariance matrices.
    
The channel estimation error variances after the first interpolation pass are computed as

$$
\mathbf{\Sigma}^{(1)}_n = \text{diag} \left( \mathbf{R^{(f)}} - \mathbf{A}_n \mathbf{\Xi}_n \mathbf{R^{(f)}} \right)
$$
    
where $\mathbf{\Xi}_n$ is the diagonal matrix of size $M \times M$ that zeros the
columns corresponding to sub-carriers not carrying any pilots.
Note that interpolation is not performed for OFDM symbols which do not carry pilots.
    
**Remark**: The interpolation matrix differs across OFDM symbols as different
OFDM symbols may carry pilots on different sub-carriers and/or have different
estimation error variances.
    
Scaling of the estimates is then performed to ensure that their
variances match the ones expected by the next interpolation step, and the error variances are updated accordingly:

$$
\begin{split}\begin{align}
    \left[\hat{\mathbf{h}}_n^{(2)}\right]_m &= s_{n,m} \left[\hat{\mathbf{h}}_n^{(1)}\right]_m\\
    \left[\mathbf{\Sigma}^{(2)}_n\right]_{m,m}  &= s_{n,m}\left( s_{n,m}-1 \right) \left[\hat{\mathbf{\Sigma}}^{(1)}_n\right]_{m,m} + \left( 1 - s_{n,m} \right) \left[\mathbf{R^{(f)}}\right]_{m,m} + s_{n,m} \left[\mathbf{\Sigma}^{(1)}_n\right]_{m,m}
\end{align}\end{split}
$$
    
where the scaling factor $s_{n,m}$ is such that:

$$
\mathbb{E} \left\{ \left\lvert s_{n,m} \left[\hat{\mathbf{h}}_n^{(1)}\right]_m \right\rvert^2 \right\} = \left[\mathbf{R^{(f)}}\right]_{m,m} +  \mathbb{E} \left\{ \left\lvert s_{n,m} \left[\hat{\mathbf{h}}^{(1)}_n\right]_m - \left[\mathbf{h}_n\right]_m \right\rvert^2 \right\}
$$
    
which leads to:

$$
\begin{split}\begin{align}
    s_{n,m} &= \frac{2 \left[\mathbf{R^{(f)}}\right]_{m,m}}{\left[\mathbf{R^{(f)}}\right]_{m,m} - \left[\mathbf{\Sigma}^{(1)}_n\right]_{m,m} + \left[\hat{\mathbf{\Sigma}}^{(1)}_n\right]_{m,m}}\\
    \hat{\mathbf{\Sigma}}^{(1)}_n &= \mathbf{A}_n \mathbf{R^{(f)}} \mathbf{A}_n^{\mathrm{H}}.
\end{align}\end{split}
$$
    
The second pass consists in interpolating across the OFDM symbols:

$$
\hat{\mathbf{h}}_m^{(3)} = \mathbf{B}_m \tilde{\mathbf{h}}^{(2)}_m
$$
    
where $1 \leq m \leq M$ is the sub-carrier index and $\tilde{\mathbf{h}}^{(2)}_m$ is
the $m^{\text{th}}$ column of

$$
\begin{split}\hat{\mathbf{H}}^{(2)} = \begin{bmatrix}
                            {\hat{\mathbf{h}}_1^{(2)}}^\intercal\\
                            \vdots\\
                            {\hat{\mathbf{h}}_N^{(2)}}^\intercal
                         \end{bmatrix}\end{split}
$$
    
and $\mathbf{B}_m$ is the $N \times N$ interpolation LMMSE matrix:

$$
\mathbf{B}_m = \bar{\mathbf{B}}_m \tilde{\mathbf{\Pi}}_m^\intercal
$$
    
where

$$
\bar{\mathbf{B}}_m = \underset{\mathbf{Z} \in \mathbb{C}^{N \times L_m}}{\text{argmin}} \left\lVert \mathbf{Z} \left( \tilde{\mathbf{\Pi}}_m^\intercal \mathbf{R^{(t)}}\tilde{\mathbf{\Pi}}_m + \tilde{\mathbf{\Sigma}}^{(2)}_m \right) -  \mathbf{R^{(t)}}\tilde{\mathbf{\Pi}}_m \right\rVert_{\text{F}}^2
$$
    
where $\mathbf{R^{(t)}}$ is the $N \times N$ channel time covariance matrix,
$\tilde{\mathbf{\Pi}}_m$ the $N \times L_m$ matrix that spreads $L_m$
values to a vector of size $N$ according to the `pilot_pattern` for the $m^{\text{th}}$ sub-carrier,
and $\tilde{\mathbf{\Sigma}}^{(2)}_m \in \mathbb{R}^{L_m \times L_m}$ is the diagonal matrix of channel estimation error variances
built by gathering the error variances from ($\mathbf{\Sigma}^{(2)}_1,\dots,\mathbf{\Sigma}^{(2)}_N$) corresponding
to resource elements carried by the $m^{\text{th}}$ sub-carrier.
Computation of $\bar{\mathbf{B}}_m$ is done using an algorithm based on complete orthogonal decomposition.
This is done to avoid matrix inversion for badly conditioned covariance matrices.
    
The resulting channel estimate for the resource grid is

$$
\hat{\mathbf{H}}^{(3)} = \left[ \hat{\mathbf{h}}_1^{(3)} \dots \hat{\mathbf{h}}_M^{(3)} \right]
$$
    
The resulting channel estimation error variances are the diagonal coefficients of the matrices

$$
\mathbf{\Sigma}^{(3)}_m = \mathbf{R^{(t)}} - \mathbf{B}_m \tilde{\mathbf{\Xi}}_m \mathbf{R^{(t)}}, 1 \leq m \leq M
$$
    
where $\tilde{\mathbf{\Xi}}_m$ is the diagonal matrix of size $N \times N$ that zeros the
columns corresponding to OFDM symbols not carrying any pilots.
    
**Remark**: The interpolation matrix differs across sub-carriers as different
sub-carriers may have different estimation error variances computed by the first
pass.
However, all sub-carriers carry at least one channel estimate as a result of
the first pass, ensuring that a channel estimate is computed for all the resource
elements after the second pass.
    
**Remark:** LMMSE interpolation requires knowledge of the time and frequency
covariance matrices of the channel. The notebook <a class="reference internal" href="../examples/OFDM_MIMO_Detection.html">OFDM MIMO Channel Estimation and Detection</a> shows how to estimate
such matrices for arbitrary channel models.
Moreover, the functions <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.tdl_time_cov_mat" title="sionna.ofdm.tdl_time_cov_mat">`tdl_time_cov_mat()`</a>
and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.tdl_freq_cov_mat" title="sionna.ofdm.tdl_freq_cov_mat">`tdl_freq_cov_mat()`</a> compute the expected time and frequency
covariance matrices, respectively, for the <a class="reference internal" href="channel.wireless.html#sionna.channel.tr38901.TDL" title="sionna.channel.tr38901.TDL">`TDL`</a> channel models.
    
Scaling of the estimates is then performed to ensure that their
variances match the ones expected by the next smoothing step, and the
error variances are updated accordingly:

$$
\begin{split}\begin{align}
    \left[\hat{\mathbf{h}}_m^{(4)}\right]_n &= \gamma_{m,n} \left[\hat{\mathbf{h}}_m^{(3)}\right]_n\\
    \left[\mathbf{\Sigma}^{(4)}_m\right]_{n,n}  &= \gamma_{m,n}\left( \gamma_{m,n}-1 \right) \left[\hat{\mathbf{\Sigma}}^{(3)}_m\right]_{n,n} + \left( 1 - \gamma_{m,n} \right) \left[\mathbf{R^{(t)}}\right]_{n,n} + \gamma_{m,n} \left[\mathbf{\Sigma}^{(3)}_n\right]_{m,m}
\end{align}\end{split}
$$
    
where:

$$
\begin{split}\begin{align}
    \gamma_{m,n} &= \frac{2 \left[\mathbf{R^{(t)}}\right]_{n,n}}{\left[\mathbf{R^{(t)}}\right]_{n,n} - \left[\mathbf{\Sigma}^{(3)}_m\right]_{n,n} + \left[\hat{\mathbf{\Sigma}}^{(3)}_n\right]_{m,m}}\\
    \hat{\mathbf{\Sigma}}^{(3)}_m &= \mathbf{B}_m \mathbf{R^{(t)}} \mathbf{B}_m^{\mathrm{H}}
\end{align}\end{split}
$$
    
Finally, a spatial smoothing step is applied to every resource element carrying
a channel estimate.
For clarity, we drop the resource element indexing $(n,m)$.
We denote by $L$ the number of receive antennas, and by
$\mathbf{R^{(s)}}\in\mathbb{C}^{L \times L}$ the spatial covariance matrix.
    
LMMSE spatial smoothing consists in the following computations:

$$
\hat{\mathbf{h}}^{(5)} = \mathbf{C} \hat{\mathbf{h}}^{(4)}
$$
    
where

$$
\mathbf{C} = \mathbf{R^{(s)}} \left( \mathbf{R^{(s)}} + \mathbf{\Sigma}^{(4)} \right)^{-1}.
$$
    
The estimation error variances are the digonal coefficients of

$$
\mathbf{\Sigma}^{(5)} = \mathbf{R^{(s)}} - \mathbf{C}\mathbf{R^{(s)}}
$$
    
The smoothed channel estimate $\hat{\mathbf{h}}^{(5)}$ and corresponding
error variances $\text{diag}\left( \mathbf{\Sigma}^{(5)} \right)$ are
returned for every resource element $(m,n)$.
    
**Remark:** No scaling is performed after the last interpolation or smoothing
step.
    
**Remark:** All passes assume that the estimation error covariance matrix
($\mathbf{\Sigma}$, $\tilde{\mathbf{\Sigma}}^{(2)}$, or $\tilde{\mathbf{\Sigma}}^{(4)}$) is diagonal, which
may not be accurate. When this assumption does not hold, this interpolator is only
an approximation of LMMSE interpolation.
    
**Remark:** The order in which frequency interpolation, temporal
interpolation, and, optionally, spatial smoothing are applied, is controlled using the
`order` parameter.

**Note**
    
This layer does not support graph mode with XLA.

Parameters
 
- **pilot_pattern** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern"><em>PilotPattern</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a>
- **cov_mat_time** (<em>[</em><em>num_ofdm_symbols</em><em>, </em><em>num_ofdm_symbols</em><em>]</em><em>, </em><em>tf.complex</em>) – Time covariance matrix of the channel
- **cov_mat_freq** (<em>[</em><em>fft_size</em><em>, </em><em>fft_size</em><em>]</em><em>, </em><em>tf.complex</em>) – Frequency covariance matrix of the channel
- **cov_time_space** (<em>[</em><em>num_rx_ant</em><em>, </em><em>num_rx_ant</em><em>]</em><em>, </em><em>tf.complex</em>) – Spatial covariance matrix of the channel.
Defaults to <cite>None</cite>.
Only required if spatial smoothing is requested (see `order`).
- **order** (<em>str</em>) – Order in which to perform interpolation and optional smoothing.
For example, `"t-f-s"` means that interpolation across the OFDM symbols
is performed first (`"t"`: time), followed by interpolation across the
sub-carriers (`"f"`: frequency), and finally smoothing across the
receive antennas (`"s"`: space).
Similarly, `"f-t"` means interpolation across the sub-carriers followed
by interpolation across the OFDM symbols and no spatial smoothing.
The spatial covariance matrix (`cov_time_space`) is only required when
spatial smoothing is requested.
Time and frequency interpolation are not optional to ensure that a channel
estimate is computed for all resource elements.


Input
 
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex</em>) – Channel estimates for the pilot-carrying resource elements
- **err_var** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex</em>) – Channel estimation error variances for the pilot-carrying resource elements


Output
 
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex</em>) – Channel estimates accross the entire resource grid for all
transmitters and streams
- **err_var** (Same shape as `h_hat`, tf.float) – Channel estimation error variances accross the entire resource grid
for all transmitters and streams




### NearestNeighborInterpolator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#nearestneighborinterpolator" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``NearestNeighborInterpolator`(<em class="sig-param">`pilot_pattern`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/channel_estimation.html#NearestNeighborInterpolator">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.NearestNeighborInterpolator" title="Permalink to this definition"></a>
    
Nearest-neighbor channel estimate interpolation on a resource grid.
    
This class assigns to each element of an OFDM resource grid one of
`num_pilots` provided channel estimates and error
variances according to the nearest neighbor method. It is assumed
that the measurements were taken at the nonzero positions of a
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a>.
    
The figure below shows how four channel estimates are interpolated
accross a resource grid. Grey fields indicate measurement positions
while the colored regions show which resource elements are assigned
to the same measurement value.
<img alt="../_images/nearest_neighbor_interpolation.png" src="https://nvlabs.github.io/sionna/_images/nearest_neighbor_interpolation.png" />
Parameters
    
**pilot_pattern** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern"><em>PilotPattern</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a>

Input
 
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex</em>) – Channel estimates for the pilot-carrying resource elements
- **err_var** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex</em>) – Channel estimation error variances for the pilot-carrying resource elements


Output
 
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex</em>) – Channel estimates accross the entire resource grid for all
transmitters and streams
- **err_var** (Same shape as `h_hat`, tf.float) – Channel estimation error variances accross the entire resource grid
for all transmitters and streams




### tdl_time_cov_mat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#tdl-time-cov-mat" title="Permalink to this headline"></a>

`sionna.ofdm.``tdl_time_cov_mat`(<em class="sig-param">`model`</em>, <em class="sig-param">`speed`</em>, <em class="sig-param">`carrier_frequency`</em>, <em class="sig-param">`ofdm_symbol_duration`</em>, <em class="sig-param">`num_ofdm_symbols`</em>, <em class="sig-param">`los_angle_of_arrival``=``0.7853981633974483`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/channel_estimation.html#tdl_time_cov_mat">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.tdl_time_cov_mat" title="Permalink to this definition"></a>
    
Computes the time covariance matrix of a
<a class="reference internal" href="channel.wireless.html#sionna.channel.tr38901.TDL" title="sionna.channel.tr38901.TDL">`TDL`</a> channel model.
    
For non-line-of-sight (NLoS) model, the channel time covariance matrix
$\mathbf{R^{(t)}}$ of a TDL channel model is

$$
\mathbf{R^{(t)}}_{u,v} = J_0 \left( \nu \Delta_t \left( u-v \right) \right)
$$
    
where $J_0$ is the zero-order Bessel function of the first kind,
$\Delta_t$ the duration of an OFDM symbol, and $\nu$ the Doppler
spread defined by

$$
\nu = 2 \pi \frac{v}{c} f_c
$$
    
where $v$ is the movement speed, $c$ the speed of light, and
$f_c$ the carrier frequency.
    
For line-of-sight (LoS) channel models, the channel time covariance matrix
is

$$
\mathbf{R^{(t)}}_{u,v} = P_{\text{NLoS}} J_0 \left( \nu \Delta_t \left( u-v \right) \right) + P_{\text{LoS}}e^{j \nu \Delta_t \left( u-v \right) \cos{\alpha_{\text{LoS}}}}
$$
    
where $\alpha_{\text{LoS}}$ is the angle-of-arrival for the LoS path,
$P_{\text{NLoS}}$ the total power of NLoS paths, and
$P_{\text{LoS}}$ the power of the LoS path. The power delay profile
is assumed to have unit power, i.e., $P_{\text{NLoS}} + P_{\text{LoS}} = 1$.
Input
 
- **model** (<em>str</em>) – TDL model for which to return the covariance matrix.
Should be one of “A”, “B”, “C”, “D”, or “E”.
- **speed** (<em>float</em>) – Speed [m/s]
- **carrier_frequency** (<em>float</em>) – Carrier frequency [Hz]
- **ofdm_symbol_duration** (<em>float</em>) – Duration of an OFDM symbol [s]
- **num_ofdm_symbols** (<em>int</em>) – Number of OFDM symbols
- **los_angle_of_arrival** (<em>float</em>) – Angle-of-arrival for LoS path [radian]. Only used with LoS models.
Defaults to $\pi/4$.
- **dtype** (<em>tf.DType</em>) – Datatype to use for the output.
Should be one of <cite>tf.complex64</cite> or <cite>tf.complex128</cite>.
Defaults to <cite>tf.complex64</cite>.


Output
    
**cov_mat** (<em>[num_ofdm_symbols, num_ofdm_symbols], tf.complex</em>) – Channel time covariance matrix



### tdl_freq_cov_mat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#tdl-freq-cov-mat" title="Permalink to this headline"></a>

`sionna.ofdm.``tdl_freq_cov_mat`(<em class="sig-param">`model`</em>, <em class="sig-param">`subcarrier_spacing`</em>, <em class="sig-param">`fft_size`</em>, <em class="sig-param">`delay_spread`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/channel_estimation.html#tdl_freq_cov_mat">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.tdl_freq_cov_mat" title="Permalink to this definition"></a>
    
Computes the frequency covariance matrix of a
<a class="reference internal" href="channel.wireless.html#sionna.channel.tr38901.TDL" title="sionna.channel.tr38901.TDL">`TDL`</a> channel model.
    
The channel frequency covariance matrix $\mathbf{R}^{(f)}$ of a TDL channel model is

$$
\mathbf{R}^{(f)}_{u,v} = \sum_{\ell=1}^L P_\ell e^{-j 2 \pi \tau_\ell \Delta_f (u-v)}, 1 \leq u,v \leq M
$$
    
where $M$ is the FFT size, $L$ is the number of paths for the selected TDL model,
$P_\ell$ and $\tau_\ell$ are the average power and delay for the
$\ell^{\text{th}}$ path, respectively, and $\Delta_f$ is the sub-carrier spacing.
Input
 
- **model** (<em>str</em>) – TDL model for which to return the covariance matrix.
Should be one of “A”, “B”, “C”, “D”, or “E”.
- **subcarrier_spacing** (<em>float</em>) – Sub-carrier spacing [Hz]
- **fft_size** (<em>float</em>) – FFT size
- **delay_spread** (<em>float</em>) – Delay spread [s]
- **dtype** (<em>tf.DType</em>) – Datatype to use for the output.
Should be one of <cite>tf.complex64</cite> or <cite>tf.complex128</cite>.
Defaults to <cite>tf.complex64</cite>.


Output
    
**cov_mat** (<em>[fft_size, fft_size], tf.complex</em>) – Channel frequency covariance matrix



## Precoding<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#precoding" title="Permalink to this headline"></a>

### ZFPrecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#zfprecoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``ZFPrecoder`(<em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`return_effective_channel``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/precoding.html#ZFPrecoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ZFPrecoder" title="Permalink to this definition"></a>
    
Zero-forcing precoding for multi-antenna transmissions.
    
This layer precodes a tensor containing OFDM resource grids using
the <a class="reference internal" href="mimo.html#sionna.mimo.zero_forcing_precoder" title="sionna.mimo.zero_forcing_precoder">`zero_forcing_precoder()`</a>. For every
transmitter, the channels to all intended receivers are gathered
into a channel matrix, based on the which the precoding matrix
is computed and the input tensor is precoded. The layer also outputs
optionally the effective channel after precoding for each stream.
Parameters
 
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – An instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>.
- **return_effective_channel** (<em>bool</em>) – Indicates if the effective channel after precoding should be returned.
- **dtype** (<em>tf.Dtype</em>) – Datatype for internal calculations and the output dtype.
Defaults to <cite>tf.complex64</cite>.


Input
 
- **(x, h)** – Tuple:
- **x** (<em>[batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex</em>) – Tensor containing the resource grid to be precoded.
- **h** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm, fft_size], tf.complex</em>) – Tensor containing the channel knowledge based on which the precoding
is computed.


Output
 
- **x_precoded** (<em>[batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – The precoded resource grids.
- **h_eff** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm, num_effective_subcarriers], tf.complex</em>) – Only returned if `return_effective_channel=True`.
The effectice channels for all streams after precoding. Can be used to
simulate perfect channel state information (CSI) at the receivers.
Nulled subcarriers are automatically removed to be compliant with the
behavior of a channel estimator.




**Note**
    
If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

## Equalization<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#equalization" title="Permalink to this headline"></a>

### OFDMEqualizer<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#ofdmequalizer" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``OFDMEqualizer`(<em class="sig-param">`equalizer`</em>, <em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/equalization.html#OFDMEqualizer">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.OFDMEqualizer" title="Permalink to this definition"></a>
    
Layer that wraps a MIMO equalizer for use with the OFDM waveform.
    
The parameter `equalizer` is a callable (e.g., a function) that
implements a MIMO equalization algorithm for arbitrary batch dimensions.
    
This class pre-processes the received resource grid `y` and channel
estimate `h_hat`, and computes for each receiver the
noise-plus-interference covariance matrix according to the OFDM and stream
configuration provided by the `resource_grid` and
`stream_management`, which also accounts for the channel
estimation error variance `err_var`. These quantities serve as input
to the equalization algorithm that is implemented by the callable `equalizer`.
This layer computes soft-symbol estimates together with effective noise
variances for all streams which can, e.g., be used by a
<a class="reference internal" href="mapping.html#sionna.mapping.Demapper" title="sionna.mapping.Demapper">`Demapper`</a> to obtain LLRs.

**Note**
    
The callable `equalizer` must take three inputs:
 
- **y** ([…,num_rx_ant], tf.complex) – 1+D tensor containing the received signals.
- **h** ([…,num_rx_ant,num_streams_per_rx], tf.complex) – 2+D tensor containing the channel matrices.
- **s** ([…,num_rx_ant,num_rx_ant], tf.complex) – 2+D tensor containing the noise-plus-interference covariance matrices.

    
It must generate two outputs:
 
- **x_hat** ([…,num_streams_per_rx], tf.complex) – 1+D tensor representing the estimated symbol vectors.
- **no_eff** (tf.float) – Tensor of the same shape as `x_hat` containing the effective noise variance estimates.
Parameters
 
- **equalizer** (<em>Callable</em>) – Callable object (e.g., a function) that implements a MIMO equalization
algorithm for arbitrary batch dimensions
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – Instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>
- **dtype** (<em>tf.Dtype</em>) – Datatype for internal calculations and the output dtype.
Defaults to <cite>tf.complex64</cite>.


Input
 
- **(y, h_hat, err_var, no)** – Tuple:
- **y** (<em>[batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Received OFDM resource grid after cyclic prefix removal and FFT
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex</em>) – Channel estimates for all streams from all transmitters
- **err_var** ([Broadcastable to shape of `h_hat`], tf.float) – Variance of the channel estimation error
- **no** (<em>[batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float</em>) – Variance of the AWGN


Output
 
- **x_hat** (<em>[batch_size, num_tx, num_streams, num_data_symbols], tf.complex</em>) – Estimated symbols
- **no_eff** (<em>[batch_size, num_tx, num_streams, num_data_symbols], tf.float</em>) – Effective noise variance for each estimated symbol




### LMMSEEqualizer<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#lmmseequalizer" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``LMMSEEqualizer`(<em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`whiten_interference``=``True`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/equalization.html#LMMSEEqualizer">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LMMSEEqualizer" title="Permalink to this definition"></a>
    
LMMSE equalization for OFDM MIMO transmissions.
    
This layer computes linear minimum mean squared error (LMMSE) equalization
for OFDM MIMO transmissions. The OFDM and stream configuration are provided
by a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> and
<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> instance, respectively. The
detection algorithm is the <a class="reference internal" href="mimo.html#sionna.mimo.lmmse_equalizer" title="sionna.mimo.lmmse_equalizer">`lmmse_equalizer()`</a>. The layer
computes soft-symbol estimates together with effective noise variances
for all streams which can, e.g., be used by a
<a class="reference internal" href="mapping.html#sionna.mapping.Demapper" title="sionna.mapping.Demapper">`Demapper`</a> to obtain LLRs.
Parameters
 
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – Instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>
- **whiten_interference** (<em>bool</em>) – If <cite>True</cite> (default), the interference is first whitened before equalization.
In this case, an alternative expression for the receive filter is used which
can be numerically more stable.
- **dtype** (<em>tf.Dtype</em>) – Datatype for internal calculations and the output dtype.
Defaults to <cite>tf.complex64</cite>.


Input
 
- **(y, h_hat, err_var, no)** – Tuple:
- **y** (<em>[batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Received OFDM resource grid after cyclic prefix removal and FFT
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex</em>) – Channel estimates for all streams from all transmitters
- **err_var** ([Broadcastable to shape of `h_hat`], tf.float) – Variance of the channel estimation error
- **no** (<em>[batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float</em>) – Variance of the AWGN


Output
 
- **x_hat** (<em>[batch_size, num_tx, num_streams, num_data_symbols], tf.complex</em>) – Estimated symbols
- **no_eff** (<em>[batch_size, num_tx, num_streams, num_data_symbols], tf.float</em>) – Effective noise variance for each estimated symbol




**Note**
    
If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

### MFEqualizer<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#mfequalizer" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``MFEqualizer`(<em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/equalization.html#MFEqualizer">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.MFEqualizer" title="Permalink to this definition"></a>
    
MF equalization for OFDM MIMO transmissions.
    
This layer computes matched filter (MF) equalization
for OFDM MIMO transmissions. The OFDM and stream configuration are provided
by a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> and
<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> instance, respectively. The
detection algorithm is the <a class="reference internal" href="mimo.html#sionna.mimo.mf_equalizer" title="sionna.mimo.mf_equalizer">`mf_equalizer()`</a>. The layer
computes soft-symbol estimates together with effective noise variances
for all streams which can, e.g., be used by a
<a class="reference internal" href="mapping.html#sionna.mapping.Demapper" title="sionna.mapping.Demapper">`Demapper`</a> to obtain LLRs.
Parameters
 
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – An instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>.
- **dtype** (<em>tf.Dtype</em>) – Datatype for internal calculations and the output dtype.
Defaults to <cite>tf.complex64</cite>.


Input
 
- **(y, h_hat, err_var, no)** – Tuple:
- **y** (<em>[batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Received OFDM resource grid after cyclic prefix removal and FFT
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex</em>) – Channel estimates for all streams from all transmitters
- **err_var** ([Broadcastable to shape of `h_hat`], tf.float) – Variance of the channel estimation error
- **no** (<em>[batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float</em>) – Variance of the AWGN


Output
 
- **x_hat** (<em>[batch_size, num_tx, num_streams, num_data_symbols], tf.complex</em>) – Estimated symbols
- **no_eff** (<em>[batch_size, num_tx, num_streams, num_data_symbols], tf.float</em>) – Effective noise variance for each estimated symbol




**Note**
    
If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

### ZFEqualizer<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#zfequalizer" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``ZFEqualizer`(<em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/equalization.html#ZFEqualizer">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ZFEqualizer" title="Permalink to this definition"></a>
    
ZF equalization for OFDM MIMO transmissions.
    
This layer computes zero-forcing (ZF) equalization
for OFDM MIMO transmissions. The OFDM and stream configuration are provided
by a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> and
<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> instance, respectively. The
detection algorithm is the <a class="reference internal" href="mimo.html#sionna.mimo.zf_equalizer" title="sionna.mimo.zf_equalizer">`zf_equalizer()`</a>. The layer
computes soft-symbol estimates together with effective noise variances
for all streams which can, e.g., be used by a
<a class="reference internal" href="mapping.html#sionna.mapping.Demapper" title="sionna.mapping.Demapper">`Demapper`</a> to obtain LLRs.
Parameters
 
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – An instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>.
- **dtype** (<em>tf.Dtype</em>) – Datatype for internal calculations and the output dtype.
Defaults to <cite>tf.complex64</cite>.


Input
 
- **(y, h_hat, err_var, no)** – Tuple:
- **y** (<em>[batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Received OFDM resource grid after cyclic prefix removal and FFT
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex</em>) – Channel estimates for all streams from all transmitters
- **err_var** ([Broadcastable to shape of `h_hat`], tf.float) – Variance of the channel estimation error
- **no** (<em>[batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float</em>) – Variance of the AWGN


Output
 
- **x_hat** (<em>[batch_size, num_tx, num_streams, num_data_symbols], tf.complex</em>) – Estimated symbols
- **no_eff** (<em>[batch_size, num_tx, num_streams, num_data_symbols], tf.float</em>) – Effective noise variance for each estimated symbol




**Note**
    
If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

## Detection<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#detection" title="Permalink to this headline"></a>

### OFDMDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#ofdmdetector" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``OFDMDetector`(<em class="sig-param">`detector`</em>, <em class="sig-param">`output`</em>, <em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/detection.html#OFDMDetector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.OFDMDetector" title="Permalink to this definition"></a>
    
Layer that wraps a MIMO detector for use with the OFDM waveform.
    
The parameter `detector` is a callable (e.g., a function) that
implements a MIMO detection algorithm for arbitrary batch dimensions.
    
This class pre-processes the received resource grid `y` and channel
estimate `h_hat`, and computes for each receiver the
noise-plus-interference covariance matrix according to the OFDM and stream
configuration provided by the `resource_grid` and
`stream_management`, which also accounts for the channel
estimation error variance `err_var`. These quantities serve as input to the detection
algorithm that is implemented by `detector`.
Both detection of symbols or bits with either soft- or hard-decisions are supported.

**Note**
    
The callable `detector` must take as input a tuple $(\mathbf{y}, \mathbf{h}, \mathbf{s})$ such that:
 
- **y** ([…,num_rx_ant], tf.complex) – 1+D tensor containing the received signals.
- **h** ([…,num_rx_ant,num_streams_per_rx], tf.complex) – 2+D tensor containing the channel matrices.
- **s** ([…,num_rx_ant,num_rx_ant], tf.complex) – 2+D tensor containing the noise-plus-interference covariance matrices.

    
It must generate one of following outputs depending on the value of `output`:
 
- **b_hat** ([…, num_streams_per_rx, num_bits_per_symbol], tf.float) – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>.
- **x_hat** ([…, num_streams_per_rx, num_points], tf.float) or ([…, num_streams_per_rx], tf.int) – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>. Hard-decisions correspond to the symbol indices.
Parameters
 
- **detector** (<em>Callable</em>) – Callable object (e.g., a function) that implements a MIMO detection
algorithm for arbitrary batch dimensions. Either one of the existing detectors, e.g.,
<a class="reference internal" href="mimo.html#sionna.mimo.LinearDetector" title="sionna.mimo.LinearDetector">`LinearDetector`</a>, <a class="reference internal" href="mimo.html#sionna.mimo.MaximumLikelihoodDetector" title="sionna.mimo.MaximumLikelihoodDetector">`MaximumLikelihoodDetector`</a>, or
<a class="reference internal" href="mimo.html#sionna.mimo.KBestDetector" title="sionna.mimo.KBestDetector">`KBestDetector`</a> can be used, or a custom detector
callable provided that has the same input/output specification.
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – Type of output, either bits or symbols
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – Instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of <cite>y</cite>. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y, h_hat, err_var, no)** – Tuple:
- **y** (<em>[batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Received OFDM resource grid after cyclic prefix removal and FFT
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex</em>) – Channel estimates for all streams from all transmitters
- **err_var** ([Broadcastable to shape of `h_hat`], tf.float) – Variance of the channel estimation error
- **no** (<em>[batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float</em>) – Variance of the AWGN


Output
 
- **One of**
- <em>[batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>
- <em>[batch_size, num_tx, num_streams, num_data_symbols, num_points], tf.float or [batch_size, num_tx, num_streams, num_data_symbols], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>.
Hard-decisions correspond to the symbol indices.




### OFDMDetectorWithPrior<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#ofdmdetectorwithprior" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``OFDMDetectorWithPrior`(<em class="sig-param">`detector`</em>, <em class="sig-param">`output`</em>, <em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`constellation_type`</em>, <em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`constellation`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/detection.html#OFDMDetectorWithPrior">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.OFDMDetectorWithPrior" title="Permalink to this definition"></a>
    
Layer that wraps a MIMO detector that assumes prior knowledge of the bits or
constellation points is available, for use with the OFDM waveform.
    
The parameter `detector` is a callable (e.g., a function) that
implements a MIMO detection algorithm with prior for arbitrary batch
dimensions.
    
This class pre-processes the received resource grid `y`, channel
estimate `h_hat`, and the prior information `prior`, and computes for each receiver the
noise-plus-interference covariance matrix according to the OFDM and stream
configuration provided by the `resource_grid` and
`stream_management`, which also accounts for the channel
estimation error variance `err_var`. These quantities serve as input to the detection
algorithm that is implemented by `detector`.
Both detection of symbols or bits with either soft- or hard-decisions are supported.

**Note**
    
The callable `detector` must take as input a tuple $(\mathbf{y}, \mathbf{h}, \mathbf{prior}, \mathbf{s})$ such that:
 
- **y** ([…,num_rx_ant], tf.complex) – 1+D tensor containing the received signals.
- **h** ([…,num_rx_ant,num_streams_per_rx], tf.complex) – 2+D tensor containing the channel matrices.
- **prior** ([…,num_streams_per_rx,num_bits_per_symbol] or […,num_streams_per_rx,num_points], tf.float) – Prior for the transmitted signals. If `output` equals “bit”, then LLRs for the transmitted bits are expected. If `output` equals “symbol”, then logits for the transmitted constellation points are expected.
- **s** ([…,num_rx_ant,num_rx_ant], tf.complex) – 2+D tensor containing the noise-plus-interference covariance matrices.

    
It must generate one of the following outputs depending on the value of `output`:
 
- **b_hat** ([…, num_streams_per_rx, num_bits_per_symbol], tf.float) – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>.
- **x_hat** ([…, num_streams_per_rx, num_points], tf.float) or ([…, num_streams_per_rx], tf.int) – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>. Hard-decisions correspond to the symbol indices.
Parameters
 
- **detector** (<em>Callable</em>) – Callable object (e.g., a function) that implements a MIMO detection
algorithm with prior for arbitrary batch dimensions. Either the existing detector
<a class="reference internal" href="mimo.html#sionna.mimo.MaximumLikelihoodDetectorWithPrior" title="sionna.mimo.MaximumLikelihoodDetectorWithPrior">`MaximumLikelihoodDetectorWithPrior`</a> can be used, or a custom detector
callable provided that has the same input/output specification.
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – Type of output, either bits or symbols
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – Instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – Number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – Instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of <cite>y</cite>. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y, h_hat, prior, err_var, no)** – Tuple:
- **y** (<em>[batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Received OFDM resource grid after cyclic prefix removal and FFT
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex</em>) – Channel estimates for all streams from all transmitters
- **prior** (<em>[batch_size, num_tx, num_streams, num_data_symbols x num_bits_per_symbol] or [batch_size, num_tx, num_streams, num_data_symbols, num_points], tf.float</em>) – Prior of the transmitted signals.
If `output` equals “bit”, LLRs of the transmitted bits are expected.
If `output` equals “symbol”, logits of the transmitted constellation points are expected.
- **err_var** ([Broadcastable to shape of `h_hat`], tf.float) – Variance of the channel estimation error
- **no** (<em>[batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float</em>) – Variance of the AWGN


Output
 
- **One of**
- <em>[batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>.
- <em>[batch_size, num_tx, num_streams, num_data_symbols, num_points], tf.float or [batch_size, num_tx, num_streams, num_data_symbols], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>.
Hard-decisions correspond to the symbol indices.




### EPDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#epdetector" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``EPDetector`(<em class="sig-param">`output`</em>, <em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`l``=``10`</em>, <em class="sig-param">`beta``=``0.9`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/detection.html#EPDetector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.EPDetector" title="Permalink to this definition"></a>
    
This layer wraps the MIMO EP detector for use with the OFDM waveform.
    
Both detection of symbols or bits with either
soft- or hard-decisions are supported. The OFDM and stream configuration are provided
by a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> and
<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> instance, respectively. The
actual detector is an instance of <a class="reference internal" href="mimo.html#sionna.mimo.EPDetector" title="sionna.mimo.EPDetector">`EPDetector`</a>.
Parameters
 
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – Type of output, either bits or symbols. Whether soft- or
hard-decisions are returned can be configured with the
`hard_out` flag.
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – Instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>
- **num_bits_per_symbol** (<em>int</em>) – Number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
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
 
- **(y, h_hat, err_var, no)** – Tuple:
- **y** (<em>[batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Received OFDM resource grid after cyclic prefix removal and FFT
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex</em>) – Channel estimates for all streams from all transmitters
- **err_var** ([Broadcastable to shape of `h_hat`], tf.float) – Variance of the channel estimation error
- **no** (<em>[batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float</em>) – Variance of the AWGN


Output
 
- **One of**
- <em>[batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>.
- <em>[batch_size, num_tx, num_streams, num_data_symbols, num_points], tf.float or [batch_size, num_tx, num_streams, num_data_symbols], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>.
Hard-decisions correspond to the symbol indices.




**Note**
    
For numerical stability, we do not recommend to use this function in Graph
mode with XLA, i.e., within a function that is decorated with
`@tf.function(jit_compile=True)`.
However, it is possible to do so by setting
`sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

### KBestDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#kbestdetector" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``KBestDetector`(<em class="sig-param">`output`</em>, <em class="sig-param">`num_streams`</em>, <em class="sig-param">`k`</em>, <em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`use_real_rep``=``False`</em>, <em class="sig-param">`list2llr``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/detection.html#KBestDetector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.KBestDetector" title="Permalink to this definition"></a>
    
This layer wraps the MIMO K-Best detector for use with the OFDM waveform.
    
Both detection of symbols or bits with either
soft- or hard-decisions are supported. The OFDM and stream configuration are provided
by a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> and
<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> instance, respectively. The
actual detector is an instance of <a class="reference internal" href="mimo.html#sionna.mimo.KBestDetector" title="sionna.mimo.KBestDetector">`KBestDetector`</a>.
Parameters
 
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – Type of output, either bits or symbols. Whether soft- or
hard-decisions are returned can be configured with the
`hard_out` flag.
- **num_streams** (<em>tf.int</em>) – Number of transmitted streams
- **k** (<em>tf.int</em>) – Number of paths to keep. Cannot be larger than the
number of constellation points to the power of the number of
streams.
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – Instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – Number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – Instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the detector computes hard-decided bit values or
constellation point indices instead of soft-values.
Defaults to <cite>False</cite>.
- **use_real_rep** (<em>bool</em>) – If <cite>True</cite>, the detector use the real-valued equivalent representation
of the channel. Note that this only works with a QAM constellation.
Defaults to <cite>False</cite>.
- **list2llr** (<cite>None</cite> or instance of <a class="reference internal" href="mimo.html#sionna.mimo.List2LLR" title="sionna.mimo.List2LLR">`List2LLR`</a>) – The function to be used to compute LLRs from a list of candidate solutions.
If <cite>None</cite>, the default solution <a class="reference internal" href="mimo.html#sionna.mimo.List2LLRSimple" title="sionna.mimo.List2LLRSimple">`List2LLRSimple`</a>
is used.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of <cite>y</cite>. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y, h_hat, err_var, no)** – Tuple:
- **y** (<em>[batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Received OFDM resource grid after cyclic prefix removal and FFT
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex</em>) – Channel estimates for all streams from all transmitters
- **err_var** ([Broadcastable to shape of `h_hat`], tf.float) – Variance of the channel estimation error
- **no** (<em>[batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float</em>) – Variance of the AWGN


Output
 
- **One of**
- <em>[batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>.
- <em>[batch_size, num_tx, num_streams, num_data_symbols, num_points], tf.float or [batch_size, num_tx, num_streams, num_data_symbols], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>.
Hard-decisions correspond to the symbol indices.




**Note**
    
If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

### LinearDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#lineardetector" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``LinearDetector`(<em class="sig-param">`equalizer`</em>, <em class="sig-param">`output`</em>, <em class="sig-param">`demapping_method`</em>, <em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/detection.html#LinearDetector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LinearDetector" title="Permalink to this definition"></a>
    
This layer wraps a MIMO linear equalizer and a <a class="reference internal" href="mapping.html#sionna.mapping.Demapper" title="sionna.mapping.Demapper">`Demapper`</a>
for use with the OFDM waveform.
    
Both detection of symbols or bits with either
soft- or hard-decisions are supported. The OFDM and stream configuration are provided
by a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> and
<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> instance, respectively. The
actual detector is an instance of <a class="reference internal" href="mimo.html#sionna.mimo.LinearDetector" title="sionna.mimo.LinearDetector">`LinearDetector`</a>.
Parameters
 
- **equalizer** (<em>str</em><em>, </em><em>one of</em><em> [</em><em>"lmmse"</em><em>, </em><em>"zf"</em><em>, </em><em>"mf"</em><em>]</em><em>, or </em><em>an equalizer function</em>) – Equalizer to be used. Either one of the existing equalizers, e.g.,
<a class="reference internal" href="mimo.html#sionna.mimo.lmmse_equalizer" title="sionna.mimo.lmmse_equalizer">`lmmse_equalizer()`</a>, <a class="reference internal" href="mimo.html#sionna.mimo.zf_equalizer" title="sionna.mimo.zf_equalizer">`zf_equalizer()`</a>, or
<a class="reference internal" href="mimo.html#sionna.mimo.mf_equalizer" title="sionna.mimo.mf_equalizer">`mf_equalizer()`</a> can be used, or a custom equalizer
function provided that has the same input/output specification.
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – Type of output, either bits or symbols. Whether soft- or
hard-decisions are returned can be configured with the
`hard_out` flag.
- **demapping_method** (<em>One of</em><em> [</em><em>"app"</em><em>, </em><em>"maxlog"</em><em>]</em><em>, </em><em>str</em>) – Demapping method used
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – Instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – Number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – Instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the detector computes hard-decided bit values or
constellation point indices instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of <cite>y</cite>. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y, h_hat, err_var, no)** – Tuple:
- **y** (<em>[batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Received OFDM resource grid after cyclic prefix removal and FFT
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex</em>) – Channel estimates for all streams from all transmitters
- **err_var** ([Broadcastable to shape of `h_hat`], tf.float) – Variance of the channel estimation error
- **no** (<em>[batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float</em>) – Variance of the AWGN


Output
 
- **One of**
- <em>[batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>.
- <em>[batch_size, num_tx, num_streams, num_data_symbols, num_points], tf.float or [batch_size, num_tx, num_streams, num_data_symbols], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>.
Hard-decisions correspond to the symbol indices.




**Note**
    
If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

### MaximumLikelihoodDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#maximumlikelihooddetector" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``MaximumLikelihoodDetector`(<em class="sig-param">`output`</em>, <em class="sig-param">`demapping_method`</em>, <em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/detection.html#MaximumLikelihoodDetector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.MaximumLikelihoodDetector" title="Permalink to this definition"></a>
    
Maximum-likelihood (ML) detection for OFDM MIMO transmissions.
    
This layer implements maximum-likelihood (ML) detection
for OFDM MIMO transmissions. Both ML detection of symbols or bits with either
soft- or hard-decisions are supported. The OFDM and stream configuration are provided
by a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> and
<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> instance, respectively. The
actual detector is an instance of <a class="reference internal" href="mimo.html#sionna.mimo.MaximumLikelihoodDetector" title="sionna.mimo.MaximumLikelihoodDetector">`MaximumLikelihoodDetector`</a>.
Parameters
 
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – Type of output, either bits or symbols. Whether soft- or
hard-decisions are returned can be configured with the
`hard_out` flag.
- **demapping_method** (<em>One of</em><em> [</em><em>"app"</em><em>, </em><em>"maxlog"</em><em>]</em><em>, </em><em>str</em>) – Demapping method used
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – Instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – Number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – Instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the detector computes hard-decided bit values or
constellation point indices instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of <cite>y</cite>. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y, h_hat, err_var, no)** – Tuple:
- **y** (<em>[batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Received OFDM resource grid after cyclic prefix removal and FFT
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex</em>) – Channel estimates for all streams from all transmitters
- **err_var** ([Broadcastable to shape of `h_hat`], tf.float) – Variance of the channel estimation error
- **no** (<em>[batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float</em>) – Variance of the AWGN noise


Output
 
- **One of**
- <em>[batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>.
- <em>[batch_size, num_tx, num_streams, num_data_symbols, num_points], tf.float or [batch_size, num_tx, num_streams, num_data_symbols], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>.
Hard-decisions correspond to the symbol indices.




**Note**
    
If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

### MaximumLikelihoodDetectorWithPrior<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#maximumlikelihooddetectorwithprior" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``MaximumLikelihoodDetectorWithPrior`(<em class="sig-param">`output`</em>, <em class="sig-param">`demapping_method`</em>, <em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/detection.html#MaximumLikelihoodDetectorWithPrior">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.MaximumLikelihoodDetectorWithPrior" title="Permalink to this definition"></a>
    
Maximum-likelihood (ML) detection for OFDM MIMO transmissions, assuming prior
knowledge of the bits or constellation points is available.
    
This layer implements maximum-likelihood (ML) detection
for OFDM MIMO transmissions assuming prior knowledge on the transmitted data is available.
Both ML detection of symbols or bits with either
soft- or hard-decisions are supported. The OFDM and stream configuration are provided
by a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> and
<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> instance, respectively. The
actual detector is an instance of <a class="reference internal" href="mimo.html#sionna.mimo.MaximumLikelihoodDetectorWithPrior" title="sionna.mimo.MaximumLikelihoodDetectorWithPrior">`MaximumLikelihoodDetectorWithPrior`</a>.
Parameters
 
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – Type of output, either bits or symbols. Whether soft- or
hard-decisions are returned can be configured with the
`hard_out` flag.
- **demapping_method** (<em>One of</em><em> [</em><em>"app"</em><em>, </em><em>"maxlog"</em><em>]</em><em>, </em><em>str</em>) – Demapping method used
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – Instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – Number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – Instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the detector computes hard-decided bit values or
constellation point indices instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of <cite>y</cite>. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y, h_hat, prior, err_var, no)** – Tuple:
- **y** (<em>[batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Received OFDM resource grid after cyclic prefix removal and FFT
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex</em>) – Channel estimates for all streams from all transmitters
- **prior** (<em>[batch_size, num_tx, num_streams, num_data_symbols x num_bits_per_symbol] or [batch_size, num_tx, num_streams, num_data_symbols, num_points], tf.float</em>) – Prior of the transmitted signals.
If `output` equals “bit”, LLRs of the transmitted bits are expected.
If `output` equals “symbol”, logits of the transmitted constellation points are expected.
- **err_var** ([Broadcastable to shape of `h_hat`], tf.float) – Variance of the channel estimation error
- **no** (<em>[batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float</em>) – Variance of the AWGN noise


Output
 
- **One of**
- <em>[batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>.
- <em>[batch_size, num_tx, num_streams, num_data_symbols, num_points], tf.float or [batch_size, num_tx, num_streams, num_data_symbols], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>.
Hard-decisions correspond to the symbol indices.




**Note**
    
If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

### MMSEPICDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#mmsepicdetector" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``MMSEPICDetector`(<em class="sig-param">`output`</em>, <em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`demapping_method``=``'maxlog'`</em>, <em class="sig-param">`num_iter``=``1`</em>, <em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/detection.html#MMSEPICDetector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.MMSEPICDetector" title="Permalink to this definition"></a>
    
This layer wraps the MIMO MMSE PIC detector for use with the OFDM waveform.
    
Both detection of symbols or bits with either
soft- or hard-decisions are supported. The OFDM and stream configuration are provided
by a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> and
<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> instance, respectively. The
actual detector is an instance of <a class="reference internal" href="mimo.html#sionna.mimo.MMSEPICDetector" title="sionna.mimo.MMSEPICDetector">`MMSEPICDetector`</a>.
Parameters
 
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – Type of output, either bits or symbols. Whether soft- or
hard-decisions are returned can be configured with the
`hard_out` flag.
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – Instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>
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
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – Precision used for internal computations. Defaults to `tf.complex64`.
Especially for large MIMO setups, the precision can make a significant
performance difference.


Input
 
- **(y, h_hat, prior, err_var, no)** – Tuple:
- **y** (<em>[batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Received OFDM resource grid after cyclic prefix removal and FFT
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex</em>) – Channel estimates for all streams from all transmitters
- **prior** (<em>[batch_size, num_tx, num_streams, num_data_symbols x num_bits_per_symbol] or [batch_size, num_tx, num_streams, num_data_symbols, num_points], tf.float</em>) – Prior of the transmitted signals.
If `output` equals “bit”, LLRs of the transmitted bits are expected.
If `output` equals “symbol”, logits of the transmitted constellation points are expected.
- **err_var** ([Broadcastable to shape of `h_hat`], tf.float) – Variance of the channel estimation error
- **no** (<em>[batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float</em>) – Variance of the AWGN


Output
 
- **One of**
- <em>[batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>.
- <em>[batch_size, num_tx, num_streams, num_data_symbols, num_points], tf.float or [batch_size, num_tx, num_streams, num_data_symbols], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>.
Hard-decisions correspond to the symbol indices.




**Note**
    
For numerical stability, we do not recommend to use this function in Graph
mode with XLA, i.e., within a function that is decorated with
`@tf.function(jit_compile=True)`.
However, it is possible to do so by setting
`sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.
