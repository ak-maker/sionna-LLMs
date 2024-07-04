
# 5G NR<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#g-nr" title="Permalink to this headline"></a>
    
This module provides layers and functions to support simulations of
5G NR compliant features, in particular, the physical uplink shared channel (PUSCH). It provides implementations of a subset of the physical layer functionalities as described in the 3GPP specifications <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id1">[3GPP38211]</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38212" id="id2">[3GPP38212]</a>, and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id3">[3GPP38214]</a>.
    
The best way to discover this module’s components is by having a look at the <a class="reference external" href="../examples/5G_NR_PUSCH.html">5G NR PUSCH Tutorial</a>.
    
The following code snippet shows how you can make standard-compliant
simulations of the 5G NR PUSCH with a few lines of code:
```python
# Create a PUSCH configuration with default settings
pusch_config = PUSCHConfig()
# Instantiate a PUSCHTransmitter from the PUSCHConfig
pusch_transmitter = PUSCHTransmitter(pusch_config)
# Create a PUSCHReceiver using the PUSCHTransmitter
pusch_receiver = PUSCHReceiver(pusch_transmitter)
# AWGN channel
channel = AWGN()
# Simulate transmissions over the AWGN channel
batch_size = 16
no = 0.1 # Noise variance
x, b = pusch_transmitter(batch_size) # Generate transmit signal and info bits
y = channel([x, no]) # Simulate channel output
b_hat = pusch_receiver([x, no]) # Recover the info bits
# Compute BER
print("BER:", compute_ber(b, b_hat).numpy())
```

    
The <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHTransmitter" title="sionna.nr.PUSCHTransmitter">`PUSCHTransmitter`</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHReceiver" title="sionna.nr.PUSCHReceiver">`PUSCHReceiver`</a> provide high-level abstractions of all required processing blocks. You can easily modify them according to your needs.

## Carrier<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#carrier" title="Permalink to this headline"></a>

### CarrierConfig<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#carrierconfig" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``CarrierConfig`(<em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/carrier_config.html#CarrierConfig">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig" title="Permalink to this definition"></a>
    
The CarrierConfig objects sets parameters for a specific OFDM numerology,
as described in Section 4 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id4">[3GPP38211]</a>.
    
All configurable properties can be provided as keyword arguments during the
initialization or changed later.
<p class="rubric">Example
```python
>>> carrier_config = CarrierConfig(n_cell_id=41)
>>> carrier_config.subcarrier_spacing = 30
```
<em class="property">`property` </em>`cyclic_prefix`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.cyclic_prefix" title="Permalink to this definition"></a>
    
Cyclic prefix length
    
The option “normal” corresponds to 14 OFDM symbols per slot, while
“extended” corresponds to 12 OFDM symbols. The latter option is
only possible with a <cite>subcarrier_spacing</cite> of 60 kHz.
Type
    
str, “normal” (default) | “extended”




<em class="property">`property` </em>`cyclic_prefix_length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.cyclic_prefix_length" title="Permalink to this definition"></a>
    
Cyclic prefix length
$N_{\text{CP},l}^{\mu} \cdot T_{\text{c}}$ [s]
Type
    
float, read-only




<em class="property">`property` </em>`frame_duration`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.frame_duration" title="Permalink to this definition"></a>
    
Duration of a frame
$T_\text{f}$ [s]
Type
    
float, 10e-3 (default), read-only




<em class="property">`property` </em>`frame_number`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.frame_number" title="Permalink to this definition"></a>
    
System frame number $n_\text{f}$
Type
    
int, 0 (default), [0,…,1023]




<em class="property">`property` </em>`kappa`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.kappa" title="Permalink to this definition"></a>
    
The constant
$\kappa = T_\text{s}/T_\text{c}$
Type
    
float, 64, read-only




<em class="property">`property` </em>`mu`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.mu" title="Permalink to this definition"></a>
    
Subcarrier
spacing configuration, $\Delta f = 2^\mu 15$ kHz
Type
    
int, 0 (default) | 1 | 2 | 3 | 4 | 5 | 6, read-only




<em class="property">`property` </em>`n_cell_id`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.n_cell_id" title="Permalink to this definition"></a>
    
Physical layer cell identity
$N_\text{ID}^\text{cell}$
Type
    
int, 1 (default) | [0,…,1007]




<em class="property">`property` </em>`n_size_grid`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.n_size_grid" title="Permalink to this definition"></a>
    
Number of resource blocks in the
carrier resource grid $N^{\text{size},\mu}_{\text{grid},x}$
Type
    
int, 4 (default) | [1,…,275]




<em class="property">`property` </em>`n_start_grid`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.n_start_grid" title="Permalink to this definition"></a>
    
Start of resource grid relative to
common resource block (CRB) 0
$N^{\text{start},\mu}_{\text{grid},x}$
Type
    
int, 0 (default) | [0,…,2199]




<em class="property">`property` </em>`num_slots_per_frame`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.num_slots_per_frame" title="Permalink to this definition"></a>
    
Number
of slots per frame $N_\text{slot}^{\text{frame},\mu}$
    
Depends on the <cite>subcarrier_spacing</cite>.
Type
    
int, 10 (default) | 20 | 40 | 80 | 160 | 320 | 640, read-only




<em class="property">`property` </em>`num_slots_per_subframe`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.num_slots_per_subframe" title="Permalink to this definition"></a>
    
Number of
slots per subframe $N_\text{slot}^{\text{subframe},\mu}$
    
Depends on the <cite>subcarrier_spacing</cite>.
Type
    
int, 1 (default) | 2 | 4 | 8 | 16 | 32 | 64, read-only




<em class="property">`property` </em>`num_symbols_per_slot`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.num_symbols_per_slot" title="Permalink to this definition"></a>
    
Number of OFDM symbols per slot
$N_\text{symb}^\text{slot}$
    
Configured through the <cite>cyclic_prefix</cite>.
Type
    
int, 14 (default) | 12, read-only




<em class="property">`property` </em>`slot_number`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.slot_number" title="Permalink to this definition"></a>
    
Slot number within a frame
$n^\mu_{s,f}$
Type
    
int, 0 (default), [0,…,num_slots_per_frame]




<em class="property">`property` </em>`sub_frame_duration`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.sub_frame_duration" title="Permalink to this definition"></a>
    
Duration of a subframe
$T_\text{sf}$ [s]
Type
    
float, 1e-3 (default), read-only




<em class="property">`property` </em>`subcarrier_spacing`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.subcarrier_spacing" title="Permalink to this definition"></a>
    
Subcarrier
spacing $\Delta f$ [kHz]
Type
    
float, 15 (default) | 30 | 60 | 120 | 240 | 480 | 960




<em class="property">`property` </em>`t_c`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.t_c" title="Permalink to this definition"></a>
    
Sampling time $T_\text{c}$ for
subcarrier spacing 480kHz.
Type
    
float, 0.509e-9 [s], read-only




<em class="property">`property` </em>`t_s`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.t_s" title="Permalink to this definition"></a>
    
Sampling time $T_\text{s}$ for
subcarrier spacing 15kHz.
Type
    
float, 32.552e-9 [s], read-only




## Layer Mapping<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#layer-mapping" title="Permalink to this headline"></a>

### LayerMapper<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#layermapper" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``LayerMapper`(<em class="sig-param">`num_layers``=``1`</em>, <em class="sig-param">`verbose``=``False`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/layer_mapping.html#LayerMapper">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.LayerMapper" title="Permalink to this definition"></a>
    
Performs MIMO layer mapping of modulated symbols to layers as defined in
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id5">[3GPP38211]</a>.
    
The LayerMapper supports PUSCH and PDSCH channels and follows the procedure
as defined in Sec. 6.3.1.3 and Sec. 7.3.1.3 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id6">[3GPP38211]</a>, respectively.
    
As specified in Tab. 7.3.1.3.-1 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id7">[3GPP38211]</a>, the LayerMapper expects two
input streams for multiplexing if more than 4 layers are active (only
relevant for PDSCH).
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **num_layers** (<em>int</em><em>, </em><em>1</em><em> (</em><em>default</em><em>) </em><em>|</em><em> [</em><em>1</em><em>,</em><em>...</em><em>,</em><em>8</em><em>]</em>) – Number of MIMO layers. If
`num_layers` >=4, a list of two inputs is expected.
- **verbose** (<em>bool</em><em>, </em><em>False</em><em> (</em><em>default</em><em>)</em>) – If True, additional parameters are printed.


Input
    
**inputs** (<em>[…,n], or [[…,n1], […,n2]], tf.complex</em>) – 2+D tensor containing the sequence of symbols to be mapped. If
`num_layers` >=4, a list of two inputs is expected and <cite>n1</cite>/<cite>n2</cite>
must be chosen as defined in Tab. 7.3.1.3.-1 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id8">[3GPP38211]</a>.

Output
    
<em>[…,num_layers, n/num_layers], tf.complex</em> – 2+D tensor containing the sequence of symbols mapped to the MIMO
layers.



<em class="property">`property` </em>`num_codewords`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.LayerMapper.num_codewords" title="Permalink to this definition"></a>
    
Number of input codewords for layer mapping. Can be either 1 or 2.


<em class="property">`property` </em>`num_layers`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.LayerMapper.num_layers" title="Permalink to this definition"></a>
    
Number of MIMO layers


<em class="property">`property` </em>`num_layers0`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.LayerMapper.num_layers0" title="Permalink to this definition"></a>
    
Number of layers for first codeword (only relevant for
<cite>num_codewords</cite> =2)


<em class="property">`property` </em>`num_layers1`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.LayerMapper.num_layers1" title="Permalink to this definition"></a>
    
Number of layers for second codeword (only relevant for
<cite>num_codewords</cite> =2)


### LayerDemapper<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#layerdemapper" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``LayerDemapper`(<em class="sig-param">`layer_mapper`</em>, <em class="sig-param">`num_bits_per_symbol``=``1`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/layer_mapping.html#LayerDemapper">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.LayerDemapper" title="Permalink to this definition"></a>
    
Demaps MIMO layers to coded transport block(s) by following Sec. 6.3.1.3
and Sec. 7.3.1.3 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id9">[3GPP38211]</a>.
    
This layer must be associated to a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.LayerMapper" title="sionna.nr.LayerMapper">`LayerMapper`</a> and
performs the inverse operation.
    
It is assumed that `num_bits_per_symbol` consecutive LLRs belong to
a single symbol position. This allows to apply the LayerDemapper after
demapping symbols to LLR values.
    
If the layer mapper is configured for dual codeword transmission, a list of
both transport block streams is returned.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **layer_mapper** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.LayerMapper" title="sionna.nr.LayerMapper">`LayerMapper`</a>) – Associated LayerMapper.
- **num_bits_per_symbol** (<em>int</em><em>, </em><em>1</em><em> (</em><em>default</em><em>)</em>) – Modulation order. Defines how many consecutive LLRs are associated
to the same symbol position.


Input
    
**inputs** (<em>[…,num_layers, n/num_layers], tf.float</em>) – 2+D tensor containing MIMO layer data sequences.

Output
    
<em>[…,n], or [[…,n1], […,n2]], tf.float</em> – 2+D tensor containing the sequence of bits after layer demapping.
If `num_codewords` =2, a list of two transport blocks is returned.



**Note**
    
As it is more convenient to apply the layer demapper after demapping
symbols to LLRs, this layer groups the input sequence into groups of
`num_bits_per_symbol` LLRs before restoring the original symbol sequence.
This behavior can be deactivated by setting `num_bits_per_symbol` =1.

## PUSCH<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#pusch" title="Permalink to this headline"></a>

### PUSCHConfig<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#puschconfig" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``PUSCHConfig`(<em class="sig-param">`carrier_config``=``None`</em>, <em class="sig-param">`pusch_dmrs_config``=``None`</em>, <em class="sig-param">`tb_config``=``None`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/pusch_config.html#PUSCHConfig">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig" title="Permalink to this definition"></a>
    
The PUSCHConfig objects sets parameters for a physical uplink shared
channel (PUSCH), as described in Sections 6.3 and 6.4 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id10">[3GPP38211]</a>.
    
All configurable properties can be provided as keyword arguments during the
initialization or changed later.
Parameters
 
- **carrier_config** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig" title="sionna.nr.CarrierConfig">`CarrierConfig`</a> or <cite>None</cite>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig" title="sionna.nr.CarrierConfig">`CarrierConfig`</a>. If <cite>None</cite>, a
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig" title="sionna.nr.CarrierConfig">`CarrierConfig`</a> instance with default settings
will be created.
- **pusch_dmrs_config** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig" title="sionna.nr.PUSCHDMRSConfig">`PUSCHDMRSConfig`</a> or <cite>None</cite>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig" title="sionna.nr.PUSCHDMRSConfig">`PUSCHDMRSConfig`</a>. If <cite>None</cite>, a
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig" title="sionna.nr.PUSCHDMRSConfig">`PUSCHDMRSConfig`</a> instance with default settings
will be created.



<p class="rubric">Example
```python
>>> pusch_config = PUSCHConfig(mapping_type="B")
>>> pusch_config.dmrs.config_type = 2
>>> pusch_config.carrier.subcarrier_spacing = 30
```
`c_init`(<em class="sig-param">`l`</em>)<a class="reference internal" href="../_modules/sionna/nr/pusch_config.html#PUSCHConfig.c_init">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.c_init" title="Permalink to this definition"></a>
    
Compute RNG initialization $c_\text{init}$ as in Section 6.4.1.1.1.1 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id11">[3GPP38211]</a>
Input
    
**l** (<em>int</em>) – OFDM symbol index relative to a reference $l$

Output
    
**c_init** (<em>int</em>) – RNG initialization value




<em class="property">`property` </em>`carrier`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.carrier" title="Permalink to this definition"></a>
    
Carrier configuration
Type
    
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig" title="sionna.nr.CarrierConfig">`CarrierConfig`</a>




<em class="property">`property` </em>`dmrs`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.dmrs" title="Permalink to this definition"></a>
    
PUSCH DMRS configuration
Type
    
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig" title="sionna.nr.PUSCHDMRSConfig">`PUSCHDMRSConfig`</a>




<em class="property">`property` </em>`dmrs_grid`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.dmrs_grid" title="Permalink to this definition"></a>
    
Empty
resource grid for each DMRS port, filled with DMRS signals
    
This property returns for each configured DMRS port an empty
resource grid filled with DMRS signals as defined in
Section 6.4.1.1 [3GPP38211]. Not all possible options are implemented,
e.g., frequency hopping and transform precoding are not available.
    
This property provides the <em>unprecoded</em> DMRS for each configured DMRS port.
Precoding might be applied to map the DMRS to the antenna ports. However,
in this case, the number of DMRS ports cannot be larger than the number of
layers.
Type
    
complex, [num_dmrs_ports, num_subcarriers, num_symbols_per_slot], read-only




<em class="property">`property` </em>`dmrs_mask`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.dmrs_mask" title="Permalink to this definition"></a>
    
Masked
resource elements in the resource grid. <cite>True</cite> corresponds to
resource elements on which no data is transmitted.
Type
    
bool, [num_subcarriers, num_symbols_per_slot], read-only




<em class="property">`property` </em>`dmrs_symbol_indices`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.dmrs_symbol_indices" title="Permalink to this definition"></a>
    
Indices of DMRS symbols within a slot
Type
    
list, int, read-only




<em class="property">`property` </em>`frequency_hopping`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.frequency_hopping" title="Permalink to this definition"></a>
    
Frequency hopping configuration
Type
    
str, “neither” (default), read-only




<em class="property">`property` </em>`l_bar`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.l_bar" title="Permalink to this definition"></a>
    
List of possible values of
$\bar{l}$ used for DMRS generation
    
Defined in Tables 6.4.1.1.3-3 and 6.4.1.1.3-4 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id12">[3GPP38211]</a>.
Type
    
list, elements in [0,…,11], read-only




<em class="property">`property` </em>`mapping_type`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.mapping_type" title="Permalink to this definition"></a>
    
Mapping type
Type
    
string, “A” (default) | “B”




<em class="property">`property` </em>`n_rnti`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.n_rnti" title="Permalink to this definition"></a>
    
Radio network temporary identifier
$n_\text{RNTI}$
Type
    
int, 1 (default), [0,…,65535]




<em class="property">`property` </em>`n_size_bwp`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.n_size_bwp" title="Permalink to this definition"></a>
    
Number of resource blocks in the
bandwidth part (BWP) $N^{\text{size},\mu}_{\text{BWP},i}$
    
If set to <cite>None</cite>, the property
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.n_size_grid" title="sionna.nr.CarrierConfig.n_size_grid">`n_size_grid`</a> of
<cite>carrier</cite> will be used.
Type
    
int, None (default), [1,…,275]




<em class="property">`property` </em>`n_start_bwp`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.n_start_bwp" title="Permalink to this definition"></a>
    
Start of BWP relative to
common resource block (CRB) 0
$N^{\text{start},\mu}_{\text{BWP},i}$
Type
    
int, 0 (default) | [0,…,2199]




<em class="property">`property` </em>`num_antenna_ports`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.num_antenna_ports" title="Permalink to this definition"></a>
    
Number of antenna ports
    
Must be larger than or equal to
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.num_layers" title="sionna.nr.PUSCHConfig.num_layers">`num_layers`</a>.
Type
    
int, 1 (default) | 2 | 4




<em class="property">`property` </em>`num_coded_bits`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.num_coded_bits" title="Permalink to this definition"></a>
    
Number of coded bits that fit into one PUSCH slot.
Type
    
int, read-only




<em class="property">`property` </em>`num_layers`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.num_layers" title="Permalink to this definition"></a>
    
Number of transmission layers
$\nu$
    
Must be smaller than or equal to
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.num_antenna_ports" title="sionna.nr.PUSCHConfig.num_antenna_ports">`num_antenna_ports`</a>.
Type
    
int, 1 (default) | 2 | 3 | 4




<em class="property">`property` </em>`num_ov`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.num_ov" title="Permalink to this definition"></a>
    
Number of unused resource elements due to additional overhead as specified by higher layer.
Type
    
int, 0 (default), read-only




<em class="property">`property` </em>`num_res_per_prb`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.num_res_per_prb" title="Permalink to this definition"></a>
    
Number of resource elements per PRB
available for data
Type
    
int, read-only




<em class="property">`property` </em>`num_resource_blocks`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.num_resource_blocks" title="Permalink to this definition"></a>
    
Number of allocated resource blocks for the
PUSCH transmissions.
Type
    
int, read-only




<em class="property">`property` </em>`num_subcarriers`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.num_subcarriers" title="Permalink to this definition"></a>
    
Number of allocated subcarriers for the
PUSCH transmissions
Type
    
int, read-only




<em class="property">`property` </em>`precoding`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.precoding" title="Permalink to this definition"></a>
    
PUSCH
transmission scheme
Type
    
str, “non-codebook” (default), “codebook”




<em class="property">`property` </em>`precoding_matrix`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.precoding_matrix" title="Permalink to this definition"></a>
    
Precoding matrix
$\mathbf{W}$ as defined in
Tables 6.3.1.5-1 to 6.3.1.5-7 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id13">[3GPP38211]</a>.
    
Only relevant if `precoding`
is “codebook”.
Type
    
nd_array, complex, [num_antenna_ports, numLayers]




`show`()<a class="reference internal" href="../_modules/sionna/nr/pusch_config.html#PUSCHConfig.show">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.show" title="Permalink to this definition"></a>
    
Print all properties of the PUSCHConfig and children


<em class="property">`property` </em>`symbol_allocation`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.symbol_allocation" title="Permalink to this definition"></a>
    
PUSCH symbol allocation
    
The first elements denotes the start of the symbol allocation.
The second denotes the positive number of allocated OFDM symbols.
For <cite>mapping_type</cite> “A”, the first element must be zero.
For <cite>mapping_type</cite> “B”, the first element must be in
[0,…,13]. The second element must be such that the index
of the last allocated OFDM symbol is not larger than 13
(for “normal” cyclic prefix) or 11 (for “extended” cyclic prefix).
Type
    
2-tuple, int, [0, 14] (default)




<em class="property">`property` </em>`tb`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.tb" title="Permalink to this definition"></a>
    
Transport block configuration
Type
    
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBConfig" title="sionna.nr.TBConfig">`TBConfig`</a>




<em class="property">`property` </em>`tb_size`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.tb_size" title="Permalink to this definition"></a>
    
Transport block size, i.e., how many information bits can be encoded into a slot for the given slot configuration.
Type
    
int, read-only




<em class="property">`property` </em>`tpmi`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.tpmi" title="Permalink to this definition"></a>
    
Transmit precoding matrix indicator
    
The allowed value depends on the number of layers and
the number of antenna ports according to Table 6.3.1.5-1
until Table 6.3.1.5-7 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id14">[3GPP38211]</a>.
Type
    
int,  0 (default) | [0,…,27]




<em class="property">`property` </em>`transform_precoding`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.transform_precoding" title="Permalink to this definition"></a>
    
Use transform precoding
Type
    
bool, False (default)




### PUSCHDMRSConfig<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#puschdmrsconfig" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``PUSCHDMRSConfig`(<em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/pusch_dmrs_config.html#PUSCHDMRSConfig">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig" title="Permalink to this definition"></a>
    
The PUSCHDMRSConfig objects sets parameters related to the generation
of demodulation reference signals (DMRS) for a physical uplink shared
channel (PUSCH), as described in Section 6.4.1.1 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id15">[3GPP38211]</a>.
    
All configurable properties can be provided as keyword arguments during the
initialization or changed later.
<p class="rubric">Example
```python
>>> dmrs_config = PUSCHDMRSConfig(config_type=2)
>>> dmrs_config.additional_position = 1
```
<em class="property">`property` </em>`additional_position`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.additional_position" title="Permalink to this definition"></a>
    
Maximum number of additional DMRS positions
    
The actual number of used DMRS positions depends on
the length of the PUSCH symbol allocation.
Type
    
int, 0 (default) | 1 | 2 | 3




<em class="property">`property` </em>`allowed_dmrs_ports`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.allowed_dmrs_ports" title="Permalink to this definition"></a>
    
List of nominal antenna
ports
    
The maximum number of allowed antenna ports <cite>max_num_dmrs_ports</cite>
depends on the DMRS <cite>config_type</cite> and <cite>length</cite>. It can be
equal to 4, 6, 8, or 12.
Type
    
list, [0,…,max_num_dmrs_ports-1], read-only




<em class="property">`property` </em>`beta`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.beta" title="Permalink to this definition"></a>
    
Ratio of PUSCH energy per resource element
(EPRE) to DMRS EPRE $\beta^{\text{DMRS}}_\text{PUSCH}$
Table 6.2.2-1 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id16">[3GPP38214]</a>
Type
    
float, read-only




<em class="property">`property` </em>`cdm_groups`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.cdm_groups" title="Permalink to this definition"></a>
    
List of CDM groups
$\lambda$ for all ports
in the <cite>dmrs_port_set</cite> as defined in
Table 6.4.1.1.3-1 or 6.4.1.1.3-2 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id17">[3GPP38211]</a>
    
Depends on the <cite>config_type</cite>.
Type
    
list, elements in [0,1,2], read-only




<em class="property">`property` </em>`config_type`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.config_type" title="Permalink to this definition"></a>
    
DMRS configuration type
    
The configuration type determines the frequency density of
DMRS signals. With configuration type 1, six subcarriers per PRB are
used for each antenna port, with configuration type 2, four
subcarriers are used.
Type
    
int, 1 (default) | 2




<em class="property">`property` </em>`deltas`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.deltas" title="Permalink to this definition"></a>
    
List of delta (frequency)
shifts $\Delta$ for all ports in the <cite>port_set</cite> as defined in
Table 6.4.1.1.3-1 or 6.4.1.1.3-2 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id18">[3GPP38211]</a>
    
Depends on the <cite>config_type</cite>.
Type
    
list, elements in [0,1,2,4], read-only




<em class="property">`property` </em>`dmrs_port_set`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.dmrs_port_set" title="Permalink to this definition"></a>
    
List of used DMRS antenna ports
    
The elements in this list must all be from the list of
<cite>allowed_dmrs_ports</cite> which depends on the <cite>config_type</cite> as well as
the <cite>length</cite>. If set to <cite>[]</cite>, the port set will be equal to
[0,…,num_layers-1], where
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.num_layers" title="sionna.nr.PUSCHConfig.num_layers">`num_layers`</a> is a property of the
parent <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig" title="sionna.nr.PUSCHConfig">`PUSCHConfig`</a> instance.
Type
    
list, [] (default) | [0,…,11]




<em class="property">`property` </em>`length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.length" title="Permalink to this definition"></a>
    
Number of front-loaded DMRS symbols
A value of 1 corresponds to “single-symbol” DMRS, a value
of 2 corresponds to “double-symbol” DMRS.
Type
    
int, 1 (default) | 2




<em class="property">`property` </em>`n_id`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.n_id" title="Permalink to this definition"></a>
    
Scrambling
identities
    
Defines the scrambling identities $N_\text{ID}^0$ and
$N_\text{ID}^1$ as a 2-tuple of integers. If <cite>None</cite>,
the property <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.n_cell_id" title="sionna.nr.CarrierConfig.n_cell_id">`n_cell_id`</a> of the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig" title="sionna.nr.CarrierConfig">`CarrierConfig`</a> is used.
Type
    
2-tuple, None (default), [[0,…,65535], [0,…,65535]]




<em class="property">`property` </em>`n_scid`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.n_scid" title="Permalink to this definition"></a>
    
DMRS scrambling initialization
$n_\text{SCID}$
Type
    
int, 0 (default) | 1




<em class="property">`property` </em>`num_cdm_groups_without_data`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.num_cdm_groups_without_data" title="Permalink to this definition"></a>
    
Number of CDM groups without data
    
This parameter controls how many REs are available for data
transmission in a DMRS symbol. It should be greater or equal to
the maximum configured number of CDM groups. A value of
1 corresponds to CDM group 0, a value of 2 corresponds to
CDM groups 0 and 1, and a value of 3 corresponds to
CDM groups 0, 1, and 2.
Type
    
int, 2 (default) | 1 | 3




<em class="property">`property` </em>`type_a_position`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.type_a_position" title="Permalink to this definition"></a>
    
Position of first DMRS OFDM symbol
    
Defines the position of the first DMRS symbol within a slot.
This parameter only applies if the property
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.mapping_type" title="sionna.nr.PUSCHConfig.mapping_type">`mapping_type`</a> of
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig" title="sionna.nr.PUSCHConfig">`PUSCHConfig`</a> is equal to “A”.
Type
    
int, 2 (default) | 3




<em class="property">`property` </em>`w_f`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.w_f" title="Permalink to this definition"></a>
    
Frequency weight vectors
$w_f(k')$ for all ports in the port set as defined in
Table 6.4.1.1.3-1 or 6.4.1.1.3-2 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id19">[3GPP38211]</a>
Type
    
matrix, elements in [-1,1], read-only




<em class="property">`property` </em>`w_t`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.w_t" title="Permalink to this definition"></a>
    
Time weight vectors
$w_t(l')$ for all ports in the port set as defined in
Table 6.4.1.1.3-1 or 6.4.1.1.3-2 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id20">[3GPP38211]</a>
Type
    
matrix, elements in [-1,1], read-only




### PUSCHLSChannelEstimator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#puschlschannelestimator" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``PUSCHLSChannelEstimator`(<em class="sig-param">`resource_grid`</em>, <em class="sig-param">`dmrs_length`</em>, <em class="sig-param">`dmrs_additional_position`</em>, <em class="sig-param">`num_cdm_groups_without_data`</em>, <em class="sig-param">`interpolation_type``=``'nn'`</em>, <em class="sig-param">`interpolator``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/pusch_channel_estimation.html#PUSCHLSChannelEstimator">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHLSChannelEstimator" title="Permalink to this definition"></a>
    
Layer implementing least-squares (LS) channel estimation for NR PUSCH Transmissions.
    
After LS channel estimation at the pilot positions, the channel estimates
and error variances are interpolated accross the entire resource grid using
a specified interpolation function.
    
The implementation is similar to that of <a class="reference internal" href="ofdm.html#sionna.ofdm.LSChannelEstimator" title="sionna.ofdm.LSChannelEstimator">`LSChannelEstimator`</a>.
However, it additional takes into account the separation of streams in the same CDM group
as defined in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig" title="sionna.nr.PUSCHDMRSConfig">`PUSCHDMRSConfig`</a>. This is done through
frequency and time averaging of adjacent LS channel estimates.
Parameters
 
- **resource_grid** (<a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – An instance of <a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>
- **dmrs_length** (<em>int</em><em>, </em><em>[</em><em>1</em><em>,</em><em>2</em><em>]</em>) – Length of DMRS symbols. See <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig" title="sionna.nr.PUSCHDMRSConfig">`PUSCHDMRSConfig`</a>.
- **dmrs_additional_position** (<em>int</em><em>, </em><em>[</em><em>0</em><em>,</em><em>1</em><em>,</em><em>2</em><em>,</em><em>3</em><em>]</em>) – Number of additional DMRS symbols.
See <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig" title="sionna.nr.PUSCHDMRSConfig">`PUSCHDMRSConfig`</a>.
- **num_cdm_groups_without_data** (<em>int</em><em>, </em><em>[</em><em>1</em><em>,</em><em>2</em><em>,</em><em>3</em><em>]</em>) – Number of CDM groups masked for data transmissions.
See <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig" title="sionna.nr.PUSCHDMRSConfig">`PUSCHDMRSConfig`</a>.
- **interpolation_type** (<em>One of</em><em> [</em><em>"nn"</em><em>, </em><em>"lin"</em><em>, </em><em>"lin_time_avg"</em><em>]</em><em>, </em><em>string</em>) – The interpolation method to be used.
It is ignored if `interpolator` is not <cite>None</cite>.
Available options are <a class="reference internal" href="ofdm.html#sionna.ofdm.NearestNeighborInterpolator" title="sionna.ofdm.NearestNeighborInterpolator">`NearestNeighborInterpolator`</a> (<cite>“nn</cite>”)
or <a class="reference internal" href="ofdm.html#sionna.ofdm.LinearInterpolator" title="sionna.ofdm.LinearInterpolator">`LinearInterpolator`</a> without (<cite>“lin”</cite>) or with
averaging across OFDM symbols (<cite>“lin_time_avg”</cite>).
Defaults to “nn”.
- **interpolator** (<a class="reference internal" href="ofdm.html#sionna.ofdm.BaseChannelInterpolator" title="sionna.ofdm.BaseChannelInterpolator"><em>BaseChannelInterpolator</em></a>) – An instance of <a class="reference internal" href="ofdm.html#sionna.ofdm.BaseChannelInterpolator" title="sionna.ofdm.BaseChannelInterpolator">`BaseChannelInterpolator`</a>,
such as <a class="reference internal" href="ofdm.html#sionna.ofdm.LMMSEInterpolator" title="sionna.ofdm.LMMSEInterpolator">`LMMSEInterpolator`</a>,
or <cite>None</cite>. In the latter case, the interpolator specified
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
 
- **h_ls** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,fft_size], tf.complex</em>) – Channel estimates across the entire resource grid for all
transmitters and streams
- **err_var** (Same shape as `h_ls`, tf.float) – Channel estimation error variance across the entire resource grid
for all transmitters and streams




### PUSCHPilotPattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#puschpilotpattern" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``PUSCHPilotPattern`(<em class="sig-param">`pusch_configs`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/nr/pusch_pilot_pattern.html#PUSCHPilotPattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern" title="Permalink to this definition"></a>
    
Class defining a pilot pattern for NR PUSCH.
    
This class defines a <a class="reference internal" href="ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a>
that is used to configure an OFDM <a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
    
For every transmitter, a separte <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig" title="sionna.nr.PUSCHConfig">`PUSCHConfig`</a>
needs to be provided from which the pilot pattern will be created.
Parameters
 
- **pusch_configs** (instance or list of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig" title="sionna.nr.PUSCHConfig">`PUSCHConfig`</a>) – PUSCH Configurations according to which the pilot pattern
will created. One configuration is needed for each transmitter.
- **dtype** (<em>tf.Dtype</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.




<em class="property">`property` </em>`mask`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern.mask" title="Permalink to this definition"></a>
    
Mask of the pilot pattern


<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern.normalize" title="Permalink to this definition"></a>
    
Returns or sets the flag indicating if the pilots
are normalized or not


<em class="property">`property` </em>`num_data_symbols`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern.num_data_symbols" title="Permalink to this definition"></a>
    
Number of data symbols per transmit stream.


<em class="property">`property` </em>`num_effective_subcarriers`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern.num_effective_subcarriers" title="Permalink to this definition"></a>
    
Number of effectvie subcarriers


<em class="property">`property` </em>`num_ofdm_symbols`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern.num_ofdm_symbols" title="Permalink to this definition"></a>
    
Number of OFDM symbols


<em class="property">`property` </em>`num_pilot_symbols`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern.num_pilot_symbols" title="Permalink to this definition"></a>
    
Number of pilot symbols per transmit stream.


<em class="property">`property` </em>`num_streams_per_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern.num_streams_per_tx" title="Permalink to this definition"></a>
    
Number of streams per transmitter


<em class="property">`property` </em>`num_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern.num_tx" title="Permalink to this definition"></a>
    
Number of transmitters


<em class="property">`property` </em>`pilots`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern.pilots" title="Permalink to this definition"></a>
    
Returns or sets the possibly normalized tensor of pilot symbols.
If pilots are normalized, the normalization will be applied
after new values for pilots have been set. If this is
not the desired behavior, turn normalization off.


`show`(<em class="sig-param">`tx_ind``=``None`</em>, <em class="sig-param">`stream_ind``=``None`</em>, <em class="sig-param">`show_pilot_ind``=``False`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern.show" title="Permalink to this definition"></a>
    
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




<em class="property">`property` </em>`trainable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern.trainable" title="Permalink to this definition"></a>
    
Returns if pilots are trainable or not


### PUSCHPrecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#puschprecoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``PUSCHPrecoder`(<em class="sig-param">`precoding_matrices`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/pusch_precoder.html#PUSCHPrecoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPrecoder" title="Permalink to this definition"></a>
    
Precodes a batch of modulated symbols mapped onto a resource grid
for PUSCH transmissions. Each transmitter is assumed to have its
own precoding matrix.
Parameters
 
- **precoding_matrices** (<em>list</em><em>, </em><em>[</em><em>num_tx</em><em>, </em><em>num_antenna_ports</em><em>, </em><em>num_layers</em><em>]</em><em> tf.complex</em>) – List of precoding matrices, one for each transmitter.
All precoding matrices must have the same shape.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>]</em>) – Dtype of inputs and outputs. Defaults to tf.complex64.


Input
    
<em>[batch_size, num_tx, num_layers, num_symbols_per_slot, num_subcarriers]</em> – Batch of resource grids to be precoded

Output
    
<em>[batch_size, num_tx, num_antenna_ports, num_symbols_per_slot, num_subcarriers]</em> – Batch of precoded resource grids



### PUSCHReceiver<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#puschreceiver" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``PUSCHReceiver`(<em class="sig-param">`pusch_transmitter`</em>, <em class="sig-param">`channel_estimator``=``None`</em>, <em class="sig-param">`mimo_detector``=``None`</em>, <em class="sig-param">`tb_decoder``=``None`</em>, <em class="sig-param">`return_tb_crc_status``=``False`</em>, <em class="sig-param">`stream_management``=``None`</em>, <em class="sig-param">`input_domain``=``'freq'`</em>, <em class="sig-param">`l_min``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/pusch_receiver.html#PUSCHReceiver">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHReceiver" title="Permalink to this definition"></a>
    
This layer implements a full receiver for batches of 5G NR PUSCH slots sent
by multiple transmitters. Inputs can be in the time or frequency domain.
Perfect channel state information can be optionally provided.
Different channel estimatiors, MIMO detectors, and transport decoders
can be configured.
    
The layer combines multiple processing blocks into a single layer
as shown in the following figure. Blocks with dashed lines are
optional and depend on the configuration.
<a class="reference internal image-reference" href="../_images/pusch_receiver_block_diagram.png"><img alt="../_images/pusch_receiver_block_diagram.png" src="https://nvlabs.github.io/sionna/_images/pusch_receiver_block_diagram.png" style="width: 258.3px; height: 420.59999999999997px;" /></a>
    
If the `input_domain` equals “time”, the inputs $\mathbf{y}$ are first
transformed to resource grids with the <a class="reference internal" href="ofdm.html#sionna.ofdm.OFDMDemodulator" title="sionna.ofdm.OFDMDemodulator">`OFDMDemodulator`</a>.
Then channel estimation is performed, e.g., with the help of the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHLSChannelEstimator" title="sionna.nr.PUSCHLSChannelEstimator">`PUSCHLSChannelEstimator`</a>. If `channel_estimator`
is chosen to be “perfect”, this step is skipped and the input $\mathbf{h}$
is used instead.
Next, MIMO detection is carried out with an arbitrary <a class="reference internal" href="ofdm.html#sionna.ofdm.OFDMDetector" title="sionna.ofdm.OFDMDetector">`OFDMDetector`</a>.
The resulting LLRs for each layer are then combined to transport blocks
with the help of the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.LayerDemapper" title="sionna.nr.LayerDemapper">`LayerDemapper`</a>.
Finally, the transport blocks are decoded with the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBDecoder" title="sionna.nr.TBDecoder">`TBDecoder`</a>.
Parameters
 
- **pusch_transmitter** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHTransmitter" title="sionna.nr.PUSCHTransmitter">`PUSCHTransmitter`</a>) – Transmitter used for the generation of the transmit signals
- **channel_estimator** (<a class="reference internal" href="ofdm.html#sionna.ofdm.BaseChannelEstimator" title="sionna.ofdm.BaseChannelEstimator">`BaseChannelEstimator`</a>, “perfect”, or <cite>None</cite>) – Channel estimator to be used.
If <cite>None</cite>, the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHLSChannelEstimator" title="sionna.nr.PUSCHLSChannelEstimator">`PUSCHLSChannelEstimator`</a> with
linear interpolation is used.
If “perfect”, no channel estimation is performed and the channel state information
`h` must be provided as additional input.
Defaults to <cite>None</cite>.
- **mimo_detector** (<a class="reference internal" href="ofdm.html#sionna.ofdm.OFDMDetector" title="sionna.ofdm.OFDMDetector">`OFDMDetector`</a> or <cite>None</cite>) – MIMO Detector to be used.
If <cite>None</cite>, the <a class="reference internal" href="ofdm.html#sionna.ofdm.LinearDetector" title="sionna.ofdm.LinearDetector">`LinearDetector`</a> with
LMMSE detection is used.
Defaults to <cite>None</cite>.
- **tb_decoder** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBDecoder" title="sionna.nr.TBDecoder">`TBDecoder`</a> or <cite>None</cite>) – Transport block decoder to be used.
If <cite>None</cite>, the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBDecoder" title="sionna.nr.TBDecoder">`TBDecoder`</a> with its
default settings is used.
Defaults to <cite>None</cite>.
- **return_tb_crc_status** (<em>bool</em>) – If <cite>True</cite>, the status of the transport block CRC is returned
as additional output.
Defaults to <cite>False</cite>.
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> or <cite>None</cite>) – Stream management configuration to be used.
If <cite>None</cite>, it is assumed that there is a single receiver
which decodes all streams of all transmitters.
Defaults to <cite>None</cite>.
- **input_domain** (<em>str</em><em>, </em><em>one of</em><em> [</em><em>"freq"</em><em>, </em><em>"time"</em><em>]</em>) – Domain of the input signal.
Defaults to “freq”.
- **l_min** (int or <cite>None</cite>) – Smallest time-lag for the discrete complex baseband channel.
Only needed if `input_domain` equals “time”.
Defaults to <cite>None</cite>.
- **dtype** (<em>tf.Dtype</em>) – Datatype for internal calculations and the output dtype.
Defaults to <cite>tf.complex64</cite>.


Input
 
- **(y, h, no)** – Tuple:
- **y** (<em>[batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex or [batch size, num_rx, num_rx_ant, num_time_samples + l_max - l_min], tf.complex</em>) – Frequency- or time-domain input signal
- **h** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers], tf.complex or [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_samples + l_max - l_min, l_max - l_min + 1], tf.complex</em>) – Perfect channel state information in either frequency or time domain
(depending on `input_domain`) to be used for detection.
Only required if `channel_estimator` equals “perfect”.
- **no** (<em>[batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, tf.float</em>) – Variance of the AWGN


Output
 
- **b_hat** (<em>[batch_size, num_tx, tb_size], tf.float</em>) – Decoded information bits
- **tb_crc_status** (<em>[batch_size, num_tx], tf.bool</em>) – Transport block CRC status



<p class="rubric">Example
```python
>>> pusch_config = PUSCHConfig()
>>> pusch_transmitter = PUSCHTransmitter(pusch_config)
>>> pusch_receiver = PUSCHReceiver(pusch_transmitter)
>>> channel = AWGN()
>>> x, b = pusch_transmitter(16)
>>> no = 0.1
>>> y = channel([x, no])
>>> b_hat = pusch_receiver([x, no])
>>> compute_ber(b, b_hat)
<tf.Tensor: shape=(), dtype=float64, numpy=0.0>
```
<em class="property">`property` </em>`resource_grid`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHReceiver.resource_grid" title="Permalink to this definition"></a>
    
OFDM resource grid underlying the PUSCH transmissions


### PUSCHTransmitter<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#puschtransmitter" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``PUSCHTransmitter`(<em class="sig-param">`pusch_configs`</em>, <em class="sig-param">`return_bits``=``True`</em>, <em class="sig-param">`output_domain``=``'freq'`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`verbose``=``False`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/pusch_transmitter.html#PUSCHTransmitter">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHTransmitter" title="Permalink to this definition"></a>
    
This layer generates batches of 5G NR PUSCH slots for multiple transmitters
with random or provided payloads. Frequency- or time-domain outputs can be generated.
    
It combines multiple processing blocks into a single layer
as shown in the following figure. Blocks with dashed lines are
optional and depend on the configuration.
<a class="reference internal image-reference" href="../_images/pusch_transmitter_block_diagram.png"><img alt="../_images/pusch_transmitter_block_diagram.png" src="https://nvlabs.github.io/sionna/_images/pusch_transmitter_block_diagram.png" style="width: 364.2px; height: 543.0px;" /></a>
    
Information bits $\mathbf{b}$ that are either randomly generated or
provided as input are encoded into a transport block by the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder" title="sionna.nr.TBEncoder">`TBEncoder`</a>.
The encoded bits are then mapped to QAM constellation symbols by the <a class="reference internal" href="mapping.html#sionna.mapping.Mapper" title="sionna.mapping.Mapper">`Mapper`</a>.
The <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.LayerMapper" title="sionna.nr.LayerMapper">`LayerMapper`</a> splits the modulated symbols into different layers
which are then mapped onto OFDM resource grids by the <a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGridMapper" title="sionna.ofdm.ResourceGridMapper">`ResourceGridMapper`</a>.
If precoding is enabled in the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig" title="sionna.nr.PUSCHConfig">`PUSCHConfig`</a>, the resource grids
are further precoded so that there is one for each transmitter and antenna port.
If `output_domain` equals “freq”, these are the outputs $\mathbf{x}$.
If `output_domain` is chosen to be “time”, the resource grids are transformed into
time-domain signals by the <a class="reference internal" href="ofdm.html#sionna.ofdm.OFDMModulator" title="sionna.ofdm.OFDMModulator">`OFDMModulator`</a>.
Parameters
 
- **pusch_configs** (instance or list of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig" title="sionna.nr.PUSCHConfig">`PUSCHConfig`</a>) – PUSCH Configurations according to which the resource grid and pilot pattern
will created. One configuration is needed for each transmitter.
- **return_bits** (<em>bool</em>) – If set to <cite>True</cite>, the layer generates random information bits
to be transmitted and returns them together with the transmit signal.
Defaults to <cite>True</cite>.
- **output_domain** (<em>str</em><em>, </em><em>one of</em><em> [</em><em>"freq"</em><em>, </em><em>"time"</em><em>]</em>) – The domain of the output. Defaults to “freq”.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>]</em>) – Dtype of inputs and outputs. Defaults to tf.complex64.
- **verbose** (<em>bool</em>) – If <cite>True</cite>, additional parameters are printed during initialization.
Defaults to <cite>False</cite>.


Input
 
- **One of**
- **batch_size** (<em>int</em>) – Batch size of random transmit signals to be generated,
if `return_bits` is <cite>True</cite>.
- **b** (<em>[batch_size, num_tx, tb_size], tf.float</em>) – Information bits to be transmitted,
if `return_bits` is <cite>False</cite>.


Output
 
- **x** (<em>[batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex or [batch size, num_tx, num_tx_ant, num_time_samples], tf.complex</em>) – Transmit signal in either frequency or time domain, depending on `output_domain`.
- **b** (<em>[batch_size, num_tx, tb_size], tf.float</em>) – Transmitted information bits.
Only returned if `return_bits` is <cite>True</cite>.



<p class="rubric">Example
```python
>>> pusch_config = PUSCHConfig()
>>> pusch_transmitter = PUSCHTransmitter(pusch_config)
>>> x, b = pusch_transmitter(16)
>>> print("Shape of x:", x.shape)
Shape of x: (16, 1, 1, 14, 48)
>>> print("Shape of b:", b.shape)
Shape of b: (16, 1, 1352)
```
<em class="property">`property` </em>`pilot_pattern`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHTransmitter.pilot_pattern" title="Permalink to this definition"></a>
    
Aggregate pilot pattern of all transmitters


<em class="property">`property` </em>`resource_grid`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHTransmitter.resource_grid" title="Permalink to this definition"></a>
    
OFDM resource grid underlying the PUSCH transmissions


`show`()<a class="reference internal" href="../_modules/sionna/nr/pusch_transmitter.html#PUSCHTransmitter.show">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHTransmitter.show" title="Permalink to this definition"></a>
    
Print all properties of the PUSCHConfig and children


## Transport Block<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#transport-block" title="Permalink to this headline"></a>

### TBConfig<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#tbconfig" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``TBConfig`(<em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/tb_config.html#TBConfig">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBConfig" title="Permalink to this definition"></a>
    
The TBConfig objects sets parameters related to the transport block
encoding, as described in TS 38.214 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id21">[3GPP38214]</a>.
    
All configurable properties can be provided as keyword arguments during the
initialization or changed later.
    
The TBConfig is configured by selecting the modulation and coding scheme
(MCS) tables and index.
<p class="rubric">Example
```python
>>> tb_config = TBConfig(mcs_index=13)
>>> tb_config.mcs_table = 3
>>> tb_config.channel_type = "PUSCH"
>>> tb_config.show()
```

    
The following tables provide an overview of the corresponding coderates and
modulation orders.
<table class="docutils align-center" id="id46">
<caption>Table 1 MCS Index Table 1 (Table 5.1.3.1-1 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id22">[3GPP38214]</a>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#id46" title="Permalink to this table"></a></caption>
<colgroup>
<col style="width: 22%" />
<col style="width: 23%" />
<col style="width: 29%" />
<col style="width: 26%" />
</colgroup>
<thead>
<tr class="row-odd"><th class="head">
MCS Index
$I_{MCS}$
</th>
<th class="head">
Modulation Order
$Q_m$
</th>
<th class="head">
Target Coderate
$R\times[1024]$
</th>
<th class="head">
Spectral Efficiency

</th>
</tr>
</thead>
<tbody>
<tr class="row-even"><td>    
0</td>
<td>    
2</td>
<td>    
120</td>
<td>    
0.2344</td>
</tr>
<tr class="row-odd"><td>    
1</td>
<td>    
2</td>
<td>    
157</td>
<td>    
0.3066</td>
</tr>
<tr class="row-even"><td>    
2</td>
<td>    
2</td>
<td>    
193</td>
<td>    
0.3770</td>
</tr>
<tr class="row-odd"><td>    
3</td>
<td>    
2</td>
<td>    
251</td>
<td>    
0.4902</td>
</tr>
<tr class="row-even"><td>    
4</td>
<td>    
2</td>
<td>    
308</td>
<td>    
0.6016</td>
</tr>
<tr class="row-odd"><td>    
5</td>
<td>    
2</td>
<td>    
379</td>
<td>    
0.7402</td>
</tr>
<tr class="row-even"><td>    
6</td>
<td>    
2</td>
<td>    
449</td>
<td>    
0.8770</td>
</tr>
<tr class="row-odd"><td>    
7</td>
<td>    
2</td>
<td>    
526</td>
<td>    
1.0273</td>
</tr>
<tr class="row-even"><td>    
8</td>
<td>    
2</td>
<td>    
602</td>
<td>    
1.1758</td>
</tr>
<tr class="row-odd"><td>    
9</td>
<td>    
2</td>
<td>    
679</td>
<td>    
1.3262</td>
</tr>
<tr class="row-even"><td>    
10</td>
<td>    
4</td>
<td>    
340</td>
<td>    
1.3281</td>
</tr>
<tr class="row-odd"><td>    
11</td>
<td>    
4</td>
<td>    
378</td>
<td>    
1.4766</td>
</tr>
<tr class="row-even"><td>    
12</td>
<td>    
4</td>
<td>    
434</td>
<td>    
1.6953</td>
</tr>
<tr class="row-odd"><td>    
13</td>
<td>    
4</td>
<td>    
490</td>
<td>    
1.9141</td>
</tr>
<tr class="row-even"><td>    
14</td>
<td>    
4</td>
<td>    
553</td>
<td>    
2.1602</td>
</tr>
<tr class="row-odd"><td>    
15</td>
<td>    
4</td>
<td>    
616</td>
<td>    
2.4063</td>
</tr>
<tr class="row-even"><td>    
16</td>
<td>    
4</td>
<td>    
658</td>
<td>    
2.5703</td>
</tr>
<tr class="row-odd"><td>    
17</td>
<td>    
6</td>
<td>    
438</td>
<td>    
2.5664</td>
</tr>
<tr class="row-even"><td>    
18</td>
<td>    
6</td>
<td>    
466</td>
<td>    
2.7305</td>
</tr>
<tr class="row-odd"><td>    
19</td>
<td>    
6</td>
<td>    
517</td>
<td>    
3.0293</td>
</tr>
<tr class="row-even"><td>    
20</td>
<td>    
6</td>
<td>    
567</td>
<td>    
3.3223</td>
</tr>
<tr class="row-odd"><td>    
21</td>
<td>    
6</td>
<td>    
616</td>
<td>    
3.6094</td>
</tr>
<tr class="row-even"><td>    
22</td>
<td>    
6</td>
<td>    
666</td>
<td>    
3.9023</td>
</tr>
<tr class="row-odd"><td>    
23</td>
<td>    
6</td>
<td>    
719</td>
<td>    
4.2129</td>
</tr>
<tr class="row-even"><td>    
24</td>
<td>    
6</td>
<td>    
772</td>
<td>    
4.5234</td>
</tr>
<tr class="row-odd"><td>    
25</td>
<td>    
6</td>
<td>    
822</td>
<td>    
4.8164</td>
</tr>
<tr class="row-even"><td>    
26</td>
<td>    
6</td>
<td>    
873</td>
<td>    
5.1152</td>
</tr>
<tr class="row-odd"><td>    
27</td>
<td>    
6</td>
<td>    
910</td>
<td>    
5.3320</td>
</tr>
<tr class="row-even"><td>    
28</td>
<td>    
6</td>
<td>    
948</td>
<td>    
5.5547</td>
</tr>
</tbody>
</table>
<table class="docutils align-center" id="id47">
<caption>Table 2 MCS Index Table 2 (Table 5.1.3.1-2 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id23">[3GPP38214]</a>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#id47" title="Permalink to this table"></a></caption>
<colgroup>
<col style="width: 22%" />
<col style="width: 23%" />
<col style="width: 29%" />
<col style="width: 26%" />
</colgroup>
<thead>
<tr class="row-odd"><th class="head">
MCS Index
$I_{MCS}$
</th>
<th class="head">
Modulation Order
$Q_m$
</th>
<th class="head">
Target Coderate
$R\times[1024]$
</th>
<th class="head">
Spectral Efficiency

</th>
</tr>
</thead>
<tbody>
<tr class="row-even"><td>    
0</td>
<td>    
2</td>
<td>    
120</td>
<td>    
0.2344</td>
</tr>
<tr class="row-odd"><td>    
1</td>
<td>    
2</td>
<td>    
193</td>
<td>    
0.3770</td>
</tr>
<tr class="row-even"><td>    
2</td>
<td>    
2</td>
<td>    
308</td>
<td>    
0.6016</td>
</tr>
<tr class="row-odd"><td>    
3</td>
<td>    
2</td>
<td>    
449</td>
<td>    
0.8770</td>
</tr>
<tr class="row-even"><td>    
4</td>
<td>    
2</td>
<td>    
602</td>
<td>    
1.1758</td>
</tr>
<tr class="row-odd"><td>    
5</td>
<td>    
4</td>
<td>    
378</td>
<td>    
1.4766</td>
</tr>
<tr class="row-even"><td>    
6</td>
<td>    
4</td>
<td>    
434</td>
<td>    
1.6953</td>
</tr>
<tr class="row-odd"><td>    
7</td>
<td>    
4</td>
<td>    
490</td>
<td>    
1.9141</td>
</tr>
<tr class="row-even"><td>    
8</td>
<td>    
4</td>
<td>    
553</td>
<td>    
2.1602</td>
</tr>
<tr class="row-odd"><td>    
9</td>
<td>    
4</td>
<td>    
616</td>
<td>    
2.4063</td>
</tr>
<tr class="row-even"><td>    
10</td>
<td>    
4</td>
<td>    
658</td>
<td>    
2.5703</td>
</tr>
<tr class="row-odd"><td>    
11</td>
<td>    
6</td>
<td>    
466</td>
<td>    
2.7305</td>
</tr>
<tr class="row-even"><td>    
12</td>
<td>    
6</td>
<td>    
517</td>
<td>    
3.0293</td>
</tr>
<tr class="row-odd"><td>    
13</td>
<td>    
6</td>
<td>    
567</td>
<td>    
3.3223</td>
</tr>
<tr class="row-even"><td>    
14</td>
<td>    
6</td>
<td>    
616</td>
<td>    
3.6094</td>
</tr>
<tr class="row-odd"><td>    
15</td>
<td>    
6</td>
<td>    
666</td>
<td>    
3.9023</td>
</tr>
<tr class="row-even"><td>    
16</td>
<td>    
6</td>
<td>    
719</td>
<td>    
4.2129</td>
</tr>
<tr class="row-odd"><td>    
17</td>
<td>    
6</td>
<td>    
772</td>
<td>    
4.5234</td>
</tr>
<tr class="row-even"><td>    
18</td>
<td>    
6</td>
<td>    
822</td>
<td>    
4.8164</td>
</tr>
<tr class="row-odd"><td>    
19</td>
<td>    
6</td>
<td>    
873</td>
<td>    
5.1152</td>
</tr>
<tr class="row-even"><td>    
20</td>
<td>    
8</td>
<td>    
682.5</td>
<td>    
5.3320</td>
</tr>
<tr class="row-odd"><td>    
21</td>
<td>    
8</td>
<td>    
711</td>
<td>    
5.5547</td>
</tr>
<tr class="row-even"><td>    
22</td>
<td>    
8</td>
<td>    
754</td>
<td>    
5.8906</td>
</tr>
<tr class="row-odd"><td>    
23</td>
<td>    
8</td>
<td>    
797</td>
<td>    
6.2266</td>
</tr>
<tr class="row-even"><td>    
24</td>
<td>    
8</td>
<td>    
841</td>
<td>    
6.5703</td>
</tr>
<tr class="row-odd"><td>    
25</td>
<td>    
8</td>
<td>    
885</td>
<td>    
6.9141</td>
</tr>
<tr class="row-even"><td>    
26</td>
<td>    
8</td>
<td>    
916.5</td>
<td>    
7.1602</td>
</tr>
<tr class="row-odd"><td>    
27</td>
<td>    
8</td>
<td>    
948</td>
<td>    
7.4063</td>
</tr>
</tbody>
</table>
<table class="docutils align-center" id="id48">
<caption>Table 3 MCS Index Table 3 (Table 5.1.3.1-3 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id24">[3GPP38214]</a>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#id48" title="Permalink to this table"></a></caption>
<colgroup>
<col style="width: 22%" />
<col style="width: 23%" />
<col style="width: 29%" />
<col style="width: 26%" />
</colgroup>
<thead>
<tr class="row-odd"><th class="head">
MCS Index
$I_{MCS}$
</th>
<th class="head">
Modulation Order
$Q_m$
</th>
<th class="head">
Target Coderate
$R\times[1024]$
</th>
<th class="head">
Spectral Efficiency

</th>
</tr>
</thead>
<tbody>
<tr class="row-even"><td>    
0</td>
<td>    
2</td>
<td>    
30</td>
<td>    
0.0586</td>
</tr>
<tr class="row-odd"><td>    
1</td>
<td>    
2</td>
<td>    
40</td>
<td>    
0.0781</td>
</tr>
<tr class="row-even"><td>    
2</td>
<td>    
2</td>
<td>    
50</td>
<td>    
0.0977</td>
</tr>
<tr class="row-odd"><td>    
3</td>
<td>    
2</td>
<td>    
64</td>
<td>    
0.1250</td>
</tr>
<tr class="row-even"><td>    
4</td>
<td>    
2</td>
<td>    
78</td>
<td>    
0.1523</td>
</tr>
<tr class="row-odd"><td>    
5</td>
<td>    
2</td>
<td>    
99</td>
<td>    
0.1934</td>
</tr>
<tr class="row-even"><td>    
6</td>
<td>    
2</td>
<td>    
120</td>
<td>    
0.2344</td>
</tr>
<tr class="row-odd"><td>    
7</td>
<td>    
2</td>
<td>    
157</td>
<td>    
0.3066</td>
</tr>
<tr class="row-even"><td>    
8</td>
<td>    
2</td>
<td>    
193</td>
<td>    
0.3770</td>
</tr>
<tr class="row-odd"><td>    
9</td>
<td>    
2</td>
<td>    
251</td>
<td>    
0.4902</td>
</tr>
<tr class="row-even"><td>    
10</td>
<td>    
2</td>
<td>    
308</td>
<td>    
0.6016</td>
</tr>
<tr class="row-odd"><td>    
11</td>
<td>    
2</td>
<td>    
379</td>
<td>    
0.7402</td>
</tr>
<tr class="row-even"><td>    
12</td>
<td>    
2</td>
<td>    
449</td>
<td>    
0.8770</td>
</tr>
<tr class="row-odd"><td>    
13</td>
<td>    
2</td>
<td>    
526</td>
<td>    
1.0273</td>
</tr>
<tr class="row-even"><td>    
14</td>
<td>    
2</td>
<td>    
602</td>
<td>    
1.1758</td>
</tr>
<tr class="row-odd"><td>    
15</td>
<td>    
4</td>
<td>    
340</td>
<td>    
1.3281</td>
</tr>
<tr class="row-even"><td>    
16</td>
<td>    
4</td>
<td>    
378</td>
<td>    
1.4766</td>
</tr>
<tr class="row-odd"><td>    
17</td>
<td>    
4</td>
<td>    
434</td>
<td>    
1.6953</td>
</tr>
<tr class="row-even"><td>    
18</td>
<td>    
4</td>
<td>    
490</td>
<td>    
1.9141</td>
</tr>
<tr class="row-odd"><td>    
19</td>
<td>    
4</td>
<td>    
553</td>
<td>    
2.1602</td>
</tr>
<tr class="row-even"><td>    
20</td>
<td>    
4</td>
<td>    
616</td>
<td>    
2.4063</td>
</tr>
<tr class="row-odd"><td>    
21</td>
<td>    
6</td>
<td>    
438</td>
<td>    
2.5564</td>
</tr>
<tr class="row-even"><td>    
22</td>
<td>    
6</td>
<td>    
466</td>
<td>    
2.7305</td>
</tr>
<tr class="row-odd"><td>    
23</td>
<td>    
6</td>
<td>    
517</td>
<td>    
3.0293</td>
</tr>
<tr class="row-even"><td>    
24</td>
<td>    
6</td>
<td>    
567</td>
<td>    
3.3223</td>
</tr>
<tr class="row-odd"><td>    
25</td>
<td>    
6</td>
<td>    
616</td>
<td>    
3.6094</td>
</tr>
<tr class="row-even"><td>    
26</td>
<td>    
6</td>
<td>    
666</td>
<td>    
3.9023</td>
</tr>
<tr class="row-odd"><td>    
27</td>
<td>    
6</td>
<td>    
719</td>
<td>    
4.2129</td>
</tr>
<tr class="row-even"><td>    
28</td>
<td>    
6</td>
<td>    
772</td>
<td>    
4.5234</td>
</tr>
</tbody>
</table>
<table class="docutils align-center" id="id49">
<caption>Table 4 MCS Index Table 4 (Table 5.1.3.1-4 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id25">[3GPP38214]</a>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#id49" title="Permalink to this table"></a></caption>
<colgroup>
<col style="width: 22%" />
<col style="width: 23%" />
<col style="width: 29%" />
<col style="width: 26%" />
</colgroup>
<thead>
<tr class="row-odd"><th class="head">
MCS Index
$I_{MCS}$
</th>
<th class="head">
Modulation Order
$Q_m$
</th>
<th class="head">
Target Coderate
$R\times[1024]$
</th>
<th class="head">
Spectral Efficiency

</th>
</tr>
</thead>
<tbody>
<tr class="row-even"><td>    
0</td>
<td>    
2</td>
<td>    
120</td>
<td>    
0.2344</td>
</tr>
<tr class="row-odd"><td>    
1</td>
<td>    
2</td>
<td>    
193</td>
<td>    
0.3770</td>
</tr>
<tr class="row-even"><td>    
2</td>
<td>    
2</td>
<td>    
449</td>
<td>    
0.8770</td>
</tr>
<tr class="row-odd"><td>    
3</td>
<td>    
4</td>
<td>    
378</td>
<td>    
1.4766</td>
</tr>
<tr class="row-even"><td>    
4</td>
<td>    
4</td>
<td>    
490</td>
<td>    
1.9141</td>
</tr>
<tr class="row-odd"><td>    
5</td>
<td>    
4</td>
<td>    
616</td>
<td>    
2.4063</td>
</tr>
<tr class="row-even"><td>    
6</td>
<td>    
6</td>
<td>    
466</td>
<td>    
2.7305</td>
</tr>
<tr class="row-odd"><td>    
7</td>
<td>    
6</td>
<td>    
517</td>
<td>    
3.0293</td>
</tr>
<tr class="row-even"><td>    
8</td>
<td>    
6</td>
<td>    
567</td>
<td>    
3.3223</td>
</tr>
<tr class="row-odd"><td>    
9</td>
<td>    
6</td>
<td>    
616</td>
<td>    
3.6094</td>
</tr>
<tr class="row-even"><td>    
10</td>
<td>    
6</td>
<td>    
666</td>
<td>    
3.9023</td>
</tr>
<tr class="row-odd"><td>    
11</td>
<td>    
6</td>
<td>    
719</td>
<td>    
4.2129</td>
</tr>
<tr class="row-even"><td>    
12</td>
<td>    
6</td>
<td>    
772</td>
<td>    
4.5234</td>
</tr>
<tr class="row-odd"><td>    
13</td>
<td>    
6</td>
<td>    
822</td>
<td>    
4.8154</td>
</tr>
<tr class="row-even"><td>    
14</td>
<td>    
6</td>
<td>    
873</td>
<td>    
5.1152</td>
</tr>
<tr class="row-odd"><td>    
15</td>
<td>    
8</td>
<td>    
682.5</td>
<td>    
5.3320</td>
</tr>
<tr class="row-even"><td>    
16</td>
<td>    
8</td>
<td>    
711</td>
<td>    
5.5547</td>
</tr>
<tr class="row-odd"><td>    
17</td>
<td>    
8</td>
<td>    
754</td>
<td>    
5.8906</td>
</tr>
<tr class="row-even"><td>    
18</td>
<td>    
8</td>
<td>    
797</td>
<td>    
6.2266</td>
</tr>
<tr class="row-odd"><td>    
19</td>
<td>    
8</td>
<td>    
841</td>
<td>    
6.5703</td>
</tr>
<tr class="row-even"><td>    
20</td>
<td>    
8</td>
<td>    
885</td>
<td>    
6.9141</td>
</tr>
<tr class="row-odd"><td>    
21</td>
<td>    
8</td>
<td>    
916.5</td>
<td>    
7.1602</td>
</tr>
<tr class="row-even"><td>    
22</td>
<td>    
8</td>
<td>    
948</td>
<td>    
7.4063</td>
</tr>
<tr class="row-odd"><td>    
23</td>
<td>    
10</td>
<td>    
805.5</td>
<td>    
7.8662</td>
</tr>
<tr class="row-even"><td>    
24</td>
<td>    
10</td>
<td>    
853</td>
<td>    
8.3301</td>
</tr>
<tr class="row-odd"><td>    
25</td>
<td>    
10</td>
<td>    
900.5</td>
<td>    
8.7939</td>
</tr>
<tr class="row-even"><td>    
26</td>
<td>    
10</td>
<td>    
948</td>
<td>    
9.2578</td>
</tr>
</tbody>
</table>

<em class="property">`property` </em>`channel_type`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBConfig.channel_type" title="Permalink to this definition"></a>
    
5G NR physical channel type. Valid choices are “PDSCH” and “PUSCH”.


`check_config`()<a class="reference internal" href="../_modules/sionna/nr/tb_config.html#TBConfig.check_config">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBConfig.check_config" title="Permalink to this definition"></a>
    
Test if configuration is valid


<em class="property">`property` </em>`mcs_index`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBConfig.mcs_index" title="Permalink to this definition"></a>
    
Modulation and coding scheme (MCS) index (denoted as $I_{MCS}$
in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id26">[3GPP38214]</a>)


<em class="property">`property` </em>`mcs_table`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBConfig.mcs_table" title="Permalink to this definition"></a>
    
Indicates which MCS table from <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id27">[3GPP38214]</a> to use. Starts with “1”.


<em class="property">`property` </em>`n_id`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBConfig.n_id" title="Permalink to this definition"></a>
    
Data scrambling initialization
$n_\text{ID}$. Data Scrambling ID related to cell id and
provided by higher layer. If <cite>None</cite>, the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig" title="sionna.nr.PUSCHConfig">`PUSCHConfig`</a> will automatically set
$n_\text{ID}=N_\text{ID}^{cell}$.
Type
    
int, None (default), [0, 1023]




<em class="property">`property` </em>`num_bits_per_symbol`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBConfig.num_bits_per_symbol" title="Permalink to this definition"></a>
    
Modulation order as defined by the selected MCS
Type
    
int, read-only




<em class="property">`property` </em>`target_coderate`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBConfig.target_coderate" title="Permalink to this definition"></a>
    
Target coderate of the TB as defined by the selected
MCS
Type
    
float, read-only




<em class="property">`property` </em>`tb_scaling`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBConfig.tb_scaling" title="Permalink to this definition"></a>
    
TB scaling factor for PDSCH as
defined in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id28">[3GPP38214]</a> Tab. 5.1.3.2-2.
Type
    
float, 1. (default), read-only




### TBEncoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#tbencoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``TBEncoder`(<em class="sig-param">`target_tb_size`</em>, <em class="sig-param">`num_coded_bits`</em>, <em class="sig-param">`target_coderate`</em>, <em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`num_layers=1`</em>, <em class="sig-param">`n_rnti=1`</em>, <em class="sig-param">`n_id=1`</em>, <em class="sig-param">`channel_type="PUSCH"`</em>, <em class="sig-param">`codeword_index=0`</em>, <em class="sig-param">`use_scrambler=True`</em>, <em class="sig-param">`verbose=False`</em>, <em class="sig-param">`output_dtype=tf.float32`</em>, <em class="sig-param">`**kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/tb_encoder.html#TBEncoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder" title="Permalink to this definition"></a>
    
5G NR transport block (TB) encoder as defined in TS 38.214
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id29">[3GPP38214]</a> and TS 38.211 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id30">[3GPP38211]</a>
    
The transport block (TB) encoder takes as input a <cite>transport block</cite> of
information bits and generates a sequence of codewords for transmission.
For this, the information bit sequence is segmented into multiple codewords,
protected by additional CRC checks and FEC encoded. Further, interleaving
and scrambling is applied before a codeword concatenation generates the
final bit sequence. Fig. 1 provides an overview of the TB encoding
procedure and we refer the interested reader to <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id31">[3GPP38214]</a> and
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id32">[3GPP38211]</a> for further details.
<img alt="../_images/tb_encoding.png" src="https://nvlabs.github.io/sionna/_images/tb_encoding.png" />
<p class="caption">Fig. 10 Fig. 1: Overview TB encoding (CB CRC does not always apply).<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#id50" title="Permalink to this image"></a>
    
If `n_rnti` and `n_id` are given as list, the TBEncoder encodes
<cite>num_tx = len(</cite> `n_rnti` <cite>)</cite> parallel input streams with different
scrambling sequences per user.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **target_tb_size** (<em>int</em>) – Target transport block size, i.e., how many information bits are
encoded into the TB. Note that the effective TB size can be
slightly different due to quantization. If required, zero padding
is internally applied.
- **num_coded_bits** (<em>int</em>) – Number of coded bits after TB encoding.
- **target_coderate** (<em>float</em>) – Target coderate.
- **num_bits_per_symbol** (<em>int</em>) – Modulation order, i.e., number of bits per QAM symbol.
- **num_layers** (<em>int</em><em>, </em><em>1</em><em> (</em><em>default</em><em>) </em><em>|</em><em> [</em><em>1</em><em>,</em><em>...</em><em>,</em><em>8</em><em>]</em>) – Number of transmission layers.
- **n_rnti** (<em>int</em><em> or </em><em>list of ints</em><em>, </em><em>1</em><em> (</em><em>default</em><em>) </em><em>|</em><em> [</em><em>0</em><em>,</em><em>...</em><em>,</em><em>65335</em><em>]</em>) – RNTI identifier provided by higher layer. Defaults to 1 and must be
in range <cite>[0, 65335]</cite>. Defines a part of the random seed of the
scrambler. If provided as list, every list entry defines the RNTI
of an independent input stream.
- **n_id** (<em>int</em><em> or </em><em>list of ints</em><em>, </em><em>1</em><em> (</em><em>default</em><em>) </em><em>|</em><em> [</em><em>0</em><em>,</em><em>...</em><em>,</em><em>1023</em><em>]</em>) – Data scrambling ID $n_\text{ID}$ related to cell id and
provided by higher layer.
Defaults to 1 and must be in range <cite>[0, 1023]</cite>. If provided as
list, every list entry defines the scrambling id of an independent
input stream.
- **channel_type** (<em>str</em><em>, </em><em>"PUSCH"</em><em> (</em><em>default</em><em>) </em><em>| "PDSCH"</em>) – Can be either “PUSCH” or “PDSCH”.
- **codeword_index** (<em>int</em><em>, </em><em>0</em><em> (</em><em>default</em><em>) </em><em>| 1</em>) – Scrambler can be configured for two codeword transmission.
`codeword_index` can be either 0 or 1. Must be 0 for
`channel_type` = “PUSCH”.
- **use_scrambler** (<em>bool</em><em>, </em><em>True</em><em> (</em><em>default</em><em>)</em>) – If False, no data scrambling is applied (non standard-compliant).
- **verbose** (<em>bool</em><em>, </em><em>False</em><em> (</em><em>default</em><em>)</em>) – If <cite>True</cite>, additional parameters are printed during initialization.
- **dtype** (<em>tf.float32</em><em> (</em><em>default</em><em>)</em>) – Defines the datatype for internal calculations and the output dtype.


Input
    
**inputs** (<em>[…,target_tb_size] or […,num_tx,target_tb_size], tf.float</em>) – 2+D tensor containing the information bits to be encoded. If
`n_rnti` and `n_id` are a list of size <cite>num_tx</cite>, the input must
be of shape <cite>[…,num_tx,target_tb_size]</cite>.

Output
    
<em>[…,num_coded_bits], tf.float</em> – 2+D tensor containing the sequence of the encoded codeword bits of
the transport block.



**Note**
    
The parameters `tb_size` and `num_coded_bits` can be derived by the
`calculate_tb_size()` function or
by accessing the corresponding <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig" title="sionna.nr.PUSCHConfig">`PUSCHConfig`</a> attributes.

<em class="property">`property` </em>`cb_crc_encoder`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.cb_crc_encoder" title="Permalink to this definition"></a>
    
CB CRC encoder. <cite>None</cite> if no CB CRC is applied.


<em class="property">`property` </em>`coderate`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.coderate" title="Permalink to this definition"></a>
    
Effective coderate of the TB after rate-matching including overhead
for the CRC.


<em class="property">`property` </em>`cw_lengths`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.cw_lengths" title="Permalink to this definition"></a>
    
Each list element defines the codeword length of each of the
codewords after LDPC encoding and rate-matching. The total number of
coded bits is $\sum$ <cite>cw_lengths</cite>.


<em class="property">`property` </em>`k`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.k" title="Permalink to this definition"></a>
    
Number of input information bits. Equals <cite>tb_size</cite> except for zero
padding of the last positions if the `target_tb_size` is quantized.


<em class="property">`property` </em>`k_padding`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.k_padding" title="Permalink to this definition"></a>
    
Number of zero padded bits at the end of the TB.


<em class="property">`property` </em>`ldpc_encoder`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.ldpc_encoder" title="Permalink to this definition"></a>
    
LDPC encoder used for TB encoding.


<em class="property">`property` </em>`n`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.n" title="Permalink to this definition"></a>
    
Total number of output bits.


<em class="property">`property` </em>`num_cbs`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.num_cbs" title="Permalink to this definition"></a>
    
Number code blocks.


<em class="property">`property` </em>`num_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.num_tx" title="Permalink to this definition"></a>
    
Number of independent streams


<em class="property">`property` </em>`output_perm_inv`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.output_perm_inv" title="Permalink to this definition"></a>
    
Inverse interleaver pattern for output bit interleaver.


<em class="property">`property` </em>`scrambler`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.scrambler" title="Permalink to this definition"></a>
    
Scrambler used for TB scrambling. <cite>None</cite> if no scrambler is used.


<em class="property">`property` </em>`tb_crc_encoder`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.tb_crc_encoder" title="Permalink to this definition"></a>
    
TB CRC encoder


<em class="property">`property` </em>`tb_size`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.tb_size" title="Permalink to this definition"></a>
    
Effective number of information bits per TB.
Note that (if required) internal zero padding can be applied to match
the request exact `target_tb_size`.


### TBDecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#tbdecoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``TBDecoder`(<em class="sig-param">`encoder`</em>, <em class="sig-param">`num_bp_iter``=``20`</em>, <em class="sig-param">`cn_type``=``'boxplus-phi'`</em>, <em class="sig-param">`output_dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/tb_decoder.html#TBDecoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBDecoder" title="Permalink to this definition"></a>
    
5G NR transport block (TB) decoder as defined in TS 38.214
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id33">[3GPP38214]</a>.
    
The transport block decoder takes as input a sequence of noisy channel
observations and reconstructs the corresponding <cite>transport block</cite> of
information bits. The detailed procedure is described in TS 38.214
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id34">[3GPP38214]</a> and TS 38.211 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id35">[3GPP38211]</a>.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **encoder** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder" title="sionna.nr.TBEncoder">`TBEncoder`</a>) – Associated transport block encoder used for encoding of the signal.
- **num_bp_iter** (<em>int</em><em>, </em><em>20</em><em> (</em><em>default</em><em>)</em>) – Number of BP decoder iterations
- **cn_type** (<em>str</em><em>, </em><em>"boxplus-phi"</em><em> (</em><em>default</em><em>) </em><em>| "boxplus" | "minsum"</em>) – The check node processing function of the LDPC BP decoder.
One of {<cite>“boxplus”</cite>, <cite>“boxplus-phi”</cite>, <cite>“minsum”</cite>} where
‘“boxplus”’ implements the single-parity-check APP decoding rule.
‘“boxplus-phi”’ implements the numerical more stable version of
boxplus <a class="reference internal" href="fec.ldpc.html#ryan" id="id36">[Ryan]</a>.
‘“minsum”’ implements the min-approximation of the CN update rule
<a class="reference internal" href="fec.ldpc.html#ryan" id="id37">[Ryan]</a>.
- **output_dtype** (<em>tf.float32</em><em> (</em><em>default</em><em>)</em>) – Defines the datatype for internal calculations and the output dtype.


Input
    
**inputs** (<em>[…,num_coded_bits], tf.float</em>) – 2+D tensor containing channel logits/llr values of the (noisy)
channel observations.

Output
 
- **b_hat** (<em>[…,target_tb_size], tf.float</em>) – 2+D tensor containing hard decided bit estimates of all information
bits of the transport block.
- **tb_crc_status** (<em>[…], tf.bool</em>) – Transport block CRC status indicating if a transport block was
(most likely) correctly recovered. Note that false positives are
possible.




<em class="property">`property` </em>`k`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBDecoder.k" title="Permalink to this definition"></a>
    
Number of input information bits. Equals TB size.


<em class="property">`property` </em>`n`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBDecoder.n" title="Permalink to this definition"></a>
    
Total number of output codeword bits.


<em class="property">`property` </em>`tb_size`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBDecoder.tb_size" title="Permalink to this definition"></a>
    
Number of information bits per TB.


## Utils<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#utils" title="Permalink to this headline"></a>

### calculate_tb_size<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#calculate-tb-size" title="Permalink to this headline"></a>

`sionna.nr.utils.``calculate_tb_size`(<em class="sig-param">`modulation_order`</em>, <em class="sig-param">`target_coderate`</em>, <em class="sig-param">`target_tb_size``=``None`</em>, <em class="sig-param">`num_coded_bits``=``None`</em>, <em class="sig-param">`num_prbs``=``None`</em>, <em class="sig-param">`num_ofdm_symbols``=``None`</em>, <em class="sig-param">`num_dmrs_per_prb``=``None`</em>, <em class="sig-param">`num_layers``=``1`</em>, <em class="sig-param">`num_ov``=``0`</em>, <em class="sig-param">`tb_scaling``=``1.0`</em>, <em class="sig-param">`verbose``=``True`</em>)<a class="reference internal" href="../_modules/sionna/nr/utils.html#calculate_tb_size">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.utils.calculate_tb_size" title="Permalink to this definition"></a>
    
Calculates transport block (TB) size for given system parameters.
    
This function follows the basic procedure as defined in TS 38.214 Sec.
5.1.3.2 and Sec. 6.1.4.2 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id38">[3GPP38214]</a>.
Parameters
 
- **modulation_order** (<em>int</em>) – Modulation order, i.e., number of bits per QAM symbol.
- **target_coderate** (<em>float</em>) – Target coderate.
- **target_tb_size** (<em>None</em><em> (</em><em>default</em><em>) </em><em>| int</em>) – Target transport block size, i.e., how many information bits can be
encoded into a slot for the given slot configuration. If provided,
`num_prbs`, `num_ofdm_symbols` and `num_dmrs_per_prb` will be
ignored.
- **num_coded_bits** (<em>None</em><em> (</em><em>default</em><em>) </em><em>| int</em>) – How many coded bits can be fit into a given slot. If provided,
`num_prbs`, `num_ofdm_symbols` and `num_dmrs_per_prb` will be
ignored.
- **num_prbs** (<em>None</em><em> (</em><em>default</em><em>) </em><em>| int</em>) – Total number of allocated PRBs per OFDM symbol where 1 PRB equals 12
subcarriers.
- **num_ofdm_symbols** (<em>None</em><em> (</em><em>default</em><em>) </em><em>| int</em>) – Number of OFDM symbols allocated for transmission. Cannot be larger
than 14.
- **num_dmrs_per_prb** (<em>None</em><em> (</em><em>default</em><em>) </em><em>| int</em>) – Number of DMRS (i.e., pilot) symbols per PRB that are NOT used for data
transmission. Sum over all `num_ofdm_symbols` OFDM symbols.
- **num_layers** (<em>int</em><em>, </em><em>1</em><em> (</em><em>default</em><em>)</em>) – Number of MIMO layers.
- **num_ov** (<em>int</em><em>, </em><em>0</em><em> (</em><em>default</em><em>)</em>) – Number of unused resource elements due to additional
overhead as specified by higher layer.
- **tb_scaling** (<em>float</em><em>, </em><em>0.25 | 0.5 | 1</em><em> (</em><em>default</em><em>)</em>) – TB scaling factor for PDSCH as defined in TS 38.214 Tab. 5.1.3.2-2.
Valid choices are 0.25, 0.5 and 1.0.
- **verbose** (<em>bool</em><em>, </em><em>False</em><em> (</em><em>default</em><em>)</em>) – If True, additional information will be printed.


Returns
    
 
- <em>(tb_size, cb_size, num_cbs, cw_length, tb_crc_length, cb_crc_length, cw_lengths)</em> – Tuple:
- **tb_size** (<em>int</em>) – Transport block size, i.e., how many information bits can be encoded
into a slot for the given slot configuration.
- **cb_size** (<em>int</em>) – Code block (CB) size. Determines the number of
information bits (including TB/CB CRC parity bits) per codeword.
- **num_cbs** (<em>int</em>) – Number of code blocks. Determines into how many CBs the TB is segmented.
- **cw_lengths** (<em>list of ints</em>) – Each list element defines the codeword length of each of the `num_cbs`
codewords after LDPC encoding and rate-matching. The total number of
coded bits is $\sum$ `cw_lengths`.
- **tb_crc_length** (<em>int</em>) – Length of the TB CRC.
- **cb_crc_length** (<em>int</em>) – Length of each CB CRC.




**Note**
    
Due to rounding, `cw_lengths` (=length of each codeword after encoding),
can be slightly different within a transport block. Thus,
`cw_lengths` is given as a list of ints where each list elements denotes
the number of codeword bits of the corresponding codeword after
rate-matching.

### generate_prng_seq<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#generate-prng-seq" title="Permalink to this headline"></a>

`sionna.nr.utils.``generate_prng_seq`(<em class="sig-param">`length`</em>, <em class="sig-param">`c_init`</em>)<a class="reference internal" href="../_modules/sionna/nr/utils.html#generate_prng_seq">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.utils.generate_prng_seq" title="Permalink to this definition"></a>
    
Implements pseudo-random sequence generator as defined in Sec. 5.2.1
in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id39">[3GPP38211]</a> based on a length-31 Gold sequence.
Parameters
 
- **length** (<em>int</em>) – Desired output sequence length.
- **c_init** (<em>int</em>) – Initialization sequence of the PRNG. Must be in the range of 0 to
$2^{32}-1$.


Output
    
[`length`], ndarray of 0s and 1s – Containing the scrambling sequence.



**Note**
    
The initialization sequence `c_init` is application specific and is
usually provided be higher layer protocols.

### select_mcs<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#select-mcs" title="Permalink to this headline"></a>

`sionna.nr.utils.``select_mcs`(<em class="sig-param">`mcs_index`</em>, <em class="sig-param">`table_index``=``1`</em>, <em class="sig-param">`channel_type``=``'PUSCH'`</em>, <em class="sig-param">`transform_precoding``=``False`</em>, <em class="sig-param">`pi2bpsk``=``False`</em>, <em class="sig-param">`verbose``=``False`</em>)<a class="reference internal" href="../_modules/sionna/nr/utils.html#select_mcs">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.utils.select_mcs" title="Permalink to this definition"></a>
    
Selects modulation and coding scheme (MCS) as specified in TS 38.214 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id40">[3GPP38214]</a>.
    
Implements MCS tables as defined in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id41">[3GPP38214]</a> for PUSCH and PDSCH.
Parameters
 
- **mcs_index** (<em>int|</em><em> [</em><em>0</em><em>,</em><em>...</em><em>,</em><em>28</em><em>]</em>) – MCS index (denoted as $I_{MCS}$ in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id42">[3GPP38214]</a>).
- **table_index** (<em>int</em><em>, </em><em>1</em><em> (</em><em>default</em><em>) </em><em>| 2 | 3 | 4</em>) – Indicates which MCS table from <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id43">[3GPP38214]</a> to use. Starts with index “1”.
- **channel_type** (<em>str</em><em>, </em><em>"PUSCH"</em><em> (</em><em>default</em><em>) </em><em>| "PDSCH"</em>) – 5G NR physical channel type. Valid choices are “PDSCH” and “PUSCH”.
- **transform_precoding** (<em>bool</em><em>, </em><em>False</em><em> (</em><em>default</em><em>)</em>) – If True, the MCS tables as described in Sec. 6.1.4.1
in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id44">[3GPP38214]</a> are applied. Only relevant for “PUSCH”.
- **pi2bpsk** (<em>bool</em><em>, </em><em>False</em><em> (</em><em>default</em><em>)</em>) – If True, the higher-layer parameter <cite>tp-pi2BPSK</cite> as
described in Sec. 6.1.4.1 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id45">[3GPP38214]</a> is applied. Only relevant
for “PUSCH”.
- **verbose** (<em>bool</em><em>, </em><em>False</em><em> (</em><em>default</em><em>)</em>) – If True, additional information will be printed.


Returns
    
 
- <em>(modulation_order, target_rate)</em> – Tuple:
- **modulation_order** (<em>int</em>) – Modulation order, i.e., number of bits per symbol.
- **target_rate** (<em>float</em>) – Target coderate.





References:
3GPP38211(<a href="https://nvlabs.github.io/sionna/api/nr.html#id1">1</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id4">2</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id5">3</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id6">4</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id7">5</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id8">6</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id9">7</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id10">8</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id11">9</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id12">10</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id13">11</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id14">12</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id15">13</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id17">14</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id18">15</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id19">16</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id20">17</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id30">18</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id32">19</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id35">20</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id39">21</a>)
    
3GPP TS 38.211. “NR; Physical channels and modulation.”

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/nr.html#id2">3GPP38212</a>
    
3GPP TS 38.212. “NR; Multiplexing and channel coding”

3GPP38214(<a href="https://nvlabs.github.io/sionna/api/nr.html#id3">1</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id16">2</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id21">3</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id22">4</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id23">5</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id24">6</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id25">7</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id26">8</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id27">9</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id28">10</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id29">11</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id31">12</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id33">13</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id34">14</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id38">15</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id40">16</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id41">17</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id42">18</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id43">19</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id44">20</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id45">21</a>)
    
3GPP TS 38.214. “NR; Physical layer procedures for data.”



