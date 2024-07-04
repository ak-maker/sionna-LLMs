
# Polar Codes<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polar-codes" title="Permalink to this headline"></a>
    
The Polar code module supports 5G-compliant Polar codes and includes successive cancellation (SC), successive cancellation list (SCL), and belief propagation (BP) decoding.
    
The module supports rate-matching and CRC-aided decoding.
Further, Reed-Muller (RM) code design is available and can be used in combination with the Polar encoding/decoding algorithms.
    
The following code snippets show how to setup and run a rate-matched 5G compliant Polar encoder and a corresponding successive cancellation list (SCL) decoder.
    
First, we need to create instances of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder" title="sionna.fec.polar.encoding.Polar5GEncoder">`Polar5GEncoder`</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder" title="sionna.fec.polar.decoding.Polar5GDecoder">`Polar5GDecoder`</a>:
```python
encoder = Polar5GEncoder(k          = 100, # number of information bits (input)
                         n          = 200) # number of codeword bits (output)

decoder = Polar5GDecoder(encoder    = encoder, # connect the Polar decoder to the encoder
                         dec_type   = "SCL", # can be also "SC" or "BP"
                         list_size  = 8)
```

    
Now, the encoder and decoder can be used by:
```python
# --- encoder ---
# u contains the information bits to be encoded and has shape [...,k].
# c contains the polar encoded codewords and has shape [...,n].
c = encoder(u)
# --- decoder ---
# llr contains the log-likelihood ratios from the demapper and has shape [...,n].
# u_hat contains the estimated information bits and has shape [...,k].
u_hat = decoder(llr)
```
## Polar Encoding<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polar-encoding" title="Permalink to this headline"></a>

### Polar5GEncoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polar5gencoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.polar.encoding.``Polar5GEncoder`(<em class="sig-param">`k`</em>, <em class="sig-param">`n`</em>, <em class="sig-param">`verbose``=``False`</em>, <em class="sig-param">`channel_type``=``'uplink'`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/encoding.html#Polar5GEncoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder" title="Permalink to this definition"></a>
    
5G compliant Polar encoder including rate-matching following <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id1">[3GPPTS38212]</a>
for the uplink scenario (<cite>UCI</cite>) and downlink scenario (<cite>DCI</cite>).
    
This layer performs polar encoding for `k` information bits and
rate-matching such that the codeword lengths is `n`. This includes the CRC
concatenation and the interleaving as defined in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id2">[3GPPTS38212]</a>.
    
Note: <cite>block segmentation</cite> is currently not supported (<cite>I_seq=False</cite>).
    
We follow the basic structure from Fig. 6 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#bioglio-design" id="id3">[Bioglio_Design]</a>.
<img alt="../_images/PolarEncoding5G.png" src="https://nvlabs.github.io/sionna/_images/PolarEncoding5G.png" />
<p class="caption">Fig. 6 Fig. 1: Implemented 5G Polar encoding chain following Fig. 6 in
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#bioglio-design" id="id4">[Bioglio_Design]</a> for the uplink (<cite>I_BIL</cite> = <cite>True</cite>) and the downlink
(<cite>I_IL</cite> = <cite>True</cite>) scenario without <cite>block segmentation</cite>.<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#id35" title="Permalink to this image"></a>
    
For further details, we refer to <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id5">[3GPPTS38212]</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#bioglio-design" id="id6">[Bioglio_Design]</a> and
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#hui-channelcoding" id="id7">[Hui_ChannelCoding]</a>.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model. Further, the class inherits from PolarEncoder.
Parameters
 
- **k** (<em>int</em>) – Defining the number of information bit per codeword.
- **n** (<em>int</em>) – Defining the codeword length.
- **channel_type** (<em>str</em>) – Defaults to “uplink”. Can be “uplink” or “downlink”.
- **verbose** (<em>bool</em>) – Defaults to False. If True, rate-matching parameters will be
printed.
- **dtype** (<em>tf.DType</em>) – Defaults to tf.float32. Defines the output datatype of the layer
(internal precision remains tf.uint8).


Input
    
**inputs** (<em>[…,k], tf.float32</em>) – 2+D tensor containing the information bits to be encoded.

Output
    
<em>[…,n], tf.float32</em> – 2+D tensor containing the codeword bits.

Raises
 
- **AssertionError** – `k` and `n` must be positive integers and `k` must be smaller
    (or equal) than `n`.
- **AssertionError** – If `n` and `k` are invalid code parameters (see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id8">[3GPPTS38212]</a>).
- **AssertionError** – If `verbose` is not <cite>bool</cite>.
- **ValueError** – If `dtype` is not supported.




**Note**
    
The encoder supports the <cite>uplink</cite> Polar coding (<cite>UCI</cite>) scheme from
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id9">[3GPPTS38212]</a> and the <cite>downlink</cite> Polar coding (<cite>DCI</cite>) <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id10">[3GPPTS38212]</a>,
respectively.
    
For <cite>12 <= k <= 19</cite> the 3 additional parity bits as defined in
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id11">[3GPPTS38212]</a> are not implemented as it would also require a
modified decoding procedure to materialize the potential gains.
    
<cite>Code segmentation</cite> is currently not supported and, thus, `n` is
limited to a maximum length of 1088 codeword bits.
    
For the downlink scenario, the input length is limited to <cite>k <= 140</cite>
information bits due to the limited input bit interleaver size
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id12">[3GPPTS38212]</a>.
    
For simplicity, the implementation does not exactly re-implement the
<cite>DCI</cite> scheme from <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id13">[3GPPTS38212]</a>. This implementation neglects the
<cite>all-one</cite> initialization of the CRC shift register and the scrambling of the CRC parity bits with the <cite>RNTI</cite>.

`channel_interleaver`(<em class="sig-param">`c`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/encoding.html#Polar5GEncoder.channel_interleaver">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder.channel_interleaver" title="Permalink to this definition"></a>
    
Triangular interleaver following Sec. 5.4.1.3 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id14">[3GPPTS38212]</a>.
Input
    
**c** (<em>ndarray</em>) – 1D array to be interleaved.

Output
    
<em>ndarray</em> – Interleaved version of `c` with same shape and dtype as `c`.




<em class="property">`property` </em>`enc_crc`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder.enc_crc" title="Permalink to this definition"></a>
    
CRC encoder layer used for CRC concatenation.


`input_interleaver`(<em class="sig-param">`c`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/encoding.html#Polar5GEncoder.input_interleaver">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder.input_interleaver" title="Permalink to this definition"></a>
    
Input interleaver following Sec. 5.4.1.1 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id15">[3GPPTS38212]</a>.
Input
    
**c** (<em>ndarray</em>) – 1D array to be interleaved.

Output
    
<em>ndarray</em> – Interleaved version of `c` with same shape and dtype as `c`.




<em class="property">`property` </em>`k`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder.k" title="Permalink to this definition"></a>
    
Number of information bits including rate-matching.


<em class="property">`property` </em>`k_polar`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder.k_polar" title="Permalink to this definition"></a>
    
Number of information bits of the underlying Polar code.


<em class="property">`property` </em>`k_target`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder.k_target" title="Permalink to this definition"></a>
    
Number of information bits including rate-matching.


<em class="property">`property` </em>`n`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder.n" title="Permalink to this definition"></a>
    
Codeword length including rate-matching.


<em class="property">`property` </em>`n_polar`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder.n_polar" title="Permalink to this definition"></a>
    
Codeword length of the underlying Polar code.


<em class="property">`property` </em>`n_target`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder.n_target" title="Permalink to this definition"></a>
    
Codeword length including rate-matching.


`subblock_interleaving`(<em class="sig-param">`u`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/encoding.html#Polar5GEncoder.subblock_interleaving">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder.subblock_interleaving" title="Permalink to this definition"></a>
    
Input bit interleaving as defined in Sec 5.4.1.1 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id16">[3GPPTS38212]</a>.
Input
    
**u** (<em>ndarray</em>) – 1D array to be interleaved. Length of `u` must be a multiple
of 32.

Output
    
<em>ndarray</em> – Interleaved version of `u` with same shape and dtype as `u`.

Raises
    
**AssertionError** – If length of `u` is not a multiple of 32.




### PolarEncoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polarencoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.polar.encoding.``PolarEncoder`(<em class="sig-param">`frozen_pos`</em>, <em class="sig-param">`n`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/encoding.html#PolarEncoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.PolarEncoder" title="Permalink to this definition"></a>
    
Polar encoder for given code parameters.
    
This layer performs polar encoding for the given `k` information bits and
the <cite>frozen set</cite> (i.e., indices of frozen positions) specified by
`frozen_pos`.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **frozen_pos** (<em>ndarray</em>) – Array of <cite>int</cite> defining the <cite>n-k</cite> frozen indices, i.e., information
bits are mapped onto the <cite>k</cite> complementary positions.
- **n** (<em>int</em>) – Defining the codeword length.
- **dtype** (<em>tf.DType</em>) – Defaults to <cite>tf.float32</cite>. Defines the output datatype of the layer
(internal precision is <cite>tf.uint8</cite>).


Input
    
**inputs** (<em>[…,k], tf.float32</em>) – 2+D tensor containing the information bits to be encoded.

Output
    
<em>[…,n], tf.float32</em> – 2+D tensor containing the codeword bits.

Raises
 
- **AssertionError** – `k` and `n` must be positive integers and `k` must be smaller
    (or equal) than `n`.
- **AssertionError** – If `n` is not a power of 2.
- **AssertionError** – If the number of elements in `frozen_pos` is great than `n`.
- **AssertionError** – If `frozen_pos` does not consists of <cite>int</cite>.
- **ValueError** – If `dtype` is not supported.
- **ValueError** – If `inputs` contains other values than <cite>0</cite> or <cite>1</cite>.
- **TypeError** – If `inputs` is not <cite>tf.float32</cite>.
- **InvalidArgumentError** – When rank(`inputs`)<2.
- **InvalidArgumentError** – When shape of last dim is not `k`.




**Note**
    
As commonly done, we assume frozen bits are set to <cite>0</cite>. Please note
that - although its practical relevance is only little - setting frozen
bits to <cite>1</cite> may result in <cite>affine</cite> codes instead of linear code as the
<cite>all-zero</cite> codeword is not necessarily part of the code any more.

<em class="property">`property` </em>`frozen_pos`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.PolarEncoder.frozen_pos" title="Permalink to this definition"></a>
    
Frozen positions for Polar decoding.


<em class="property">`property` </em>`info_pos`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.PolarEncoder.info_pos" title="Permalink to this definition"></a>
    
Information bit positions for Polar encoding.


<em class="property">`property` </em>`k`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.PolarEncoder.k" title="Permalink to this definition"></a>
    
Number of information bits.


<em class="property">`property` </em>`n`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.PolarEncoder.n" title="Permalink to this definition"></a>
    
Codeword length.


## Polar Decoding<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polar-decoding" title="Permalink to this headline"></a>

### Polar5GDecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polar5gdecoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.polar.decoding.``Polar5GDecoder`(<em class="sig-param">`enc_polar`</em>, <em class="sig-param">`dec_type``=``'SC'`</em>, <em class="sig-param">`list_size``=``8`</em>, <em class="sig-param">`num_iter``=``20`</em>, <em class="sig-param">`return_crc_status``=``False`</em>, <em class="sig-param">`output_dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/decoding.html#Polar5GDecoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder" title="Permalink to this definition"></a>
    
Wrapper for 5G compliant decoding including rate-recovery and CRC removal.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **enc_polar** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder" title="sionna.fec.polar.encoding.Polar5GEncoder"><em>Polar5GEncoder</em></a>) – Instance of the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder" title="sionna.fec.polar.encoding.Polar5GEncoder">`Polar5GEncoder`</a>
used for encoding including rate-matching.
- **dec_type** (<em>str</em>) – Defaults to <cite>“SC”</cite>. Defining the decoder to be used.
Must be one of the following <cite>{“SC”, “SCL”, “hybSCL”, “BP”}</cite>.
- **list_size** (<em>int</em>) – Defaults to 8. Defining the list size <cite>iff</cite> list-decoding is used.
Only required for `dec_types` <cite>{“SCL”, “hybSCL”}</cite>.
- **num_iter** (<em>int</em>) – Defaults to 20. Defining the number of BP iterations. Only required
for `dec_type` <cite>“BP”</cite>.
- **return_crc_status** (<em>bool</em>) – Defaults to False. If True, the decoder additionally returns the
CRC status indicating if a codeword was (most likely) correctly
recovered.
- **output_dtype** (<em>tf.DType</em>) – Defaults to tf.float32. Defines the output datatype of the layer
(internal precision remains tf.float32).


Input
    
**inputs** (<em>[…,n], tf.float32</em>) – 2+D tensor containing the channel logits/llr values.

Output
 
- **b_hat** (<em>[…,k], tf.float32</em>) – 2+D tensor containing hard-decided estimations of all <cite>k</cite>
information bits.
- **crc_status** (<em>[…], tf.bool</em>) – CRC status indicating if a codeword was (most likely) correctly
recovered. This is only returned if `return_crc_status` is True.
Note that false positives are possible.


Raises
 
- **AssertionError** – If `enc_polar` is not <cite>Polar5GEncoder</cite>.
- **ValueError** – If `dec_type` is not <cite>{“SC”, “SCL”, “SCL8”, “SCL32”, “hybSCL”,
    “BP”}</cite>.
- **AssertionError** – If `dec_type` is not <cite>str</cite>.
- **ValueError** – If `inputs` is not of shape <cite>[…, n]</cite> or <cite>dtype</cite> is not
    the same as `output_dtype`.
- **InvalidArgumentError** – When rank(`inputs`)<2.




**Note**
    
This layer supports the uplink and downlink Polar rate-matching scheme
without <cite>codeword segmentation</cite>.
    
Although the decoding <cite>list size</cite> is not provided by 3GPP
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id17">[3GPPTS38212]</a>, the consortium has agreed on a <cite>list size</cite> of 8 for the
5G decoding reference curves <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#bioglio-design" id="id18">[Bioglio_Design]</a>.
    
All list-decoders apply <cite>CRC-aided</cite> decoding, however, the non-list
decoders (<cite>“SC”</cite> and <cite>“BP”</cite>) cannot materialize the CRC leading to an
effective rate-loss.

<em class="property">`property` </em>`dec_type`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder.dec_type" title="Permalink to this definition"></a>
    
Decoder type used for decoding as str.


<em class="property">`property` </em>`frozen_pos`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder.frozen_pos" title="Permalink to this definition"></a>
    
Frozen positions for Polar decoding.


<em class="property">`property` </em>`info_pos`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder.info_pos" title="Permalink to this definition"></a>
    
Information bit positions for Polar encoding.


<em class="property">`property` </em>`k_polar`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder.k_polar" title="Permalink to this definition"></a>
    
Number of information bits of mother Polar code.


<em class="property">`property` </em>`k_target`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder.k_target" title="Permalink to this definition"></a>
    
Number of information bits including rate-matching.


<em class="property">`property` </em>`llr_max`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder.llr_max" title="Permalink to this definition"></a>
    
Maximum LLR value for internal calculations.


<em class="property">`property` </em>`n_polar`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder.n_polar" title="Permalink to this definition"></a>
    
Codeword length of mother Polar code.


<em class="property">`property` </em>`n_target`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder.n_target" title="Permalink to this definition"></a>
    
Codeword length including rate-matching.


<em class="property">`property` </em>`output_dtype`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder.output_dtype" title="Permalink to this definition"></a>
    
Output dtype of decoder.


<em class="property">`property` </em>`polar_dec`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder.polar_dec" title="Permalink to this definition"></a>
    
Decoder instance used for decoding.


### PolarSCDecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polarscdecoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.polar.decoding.``PolarSCDecoder`(<em class="sig-param">`frozen_pos`</em>, <em class="sig-param">`n`</em>, <em class="sig-param">`output_dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/decoding.html#PolarSCDecoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCDecoder" title="Permalink to this definition"></a>
    
Successive cancellation (SC) decoder <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#arikan-polar" id="id19">[Arikan_Polar]</a> for Polar codes and
Polar-like codes.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **frozen_pos** (<em>ndarray</em>) – Array of <cite>int</cite> defining the `n-k` indices of the frozen positions.
- **n** – Defining the codeword length.


Input
    
**inputs** (<em>[…,n], tf.float32</em>) – 2+D tensor containing the channel LLR values (as logits).

Output
    
<em>[…,k], tf.float32</em> – 2+D tensor  containing hard-decided estimations of all `k`
information bits.

Raises
 
- **AssertionError** – If `n` is not <cite>int</cite>.
- **AssertionError** – If `n` is not a power of 2.
- **AssertionError** – If the number of elements in `frozen_pos` is greater than `n`.
- **AssertionError** – If `frozen_pos` does not consists of <cite>int</cite>.
- **ValueError** – If `output_dtype` is not {tf.float16, tf.float32, tf.float64}.




**Note**
    
This layer implements the SC decoder as described in
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#arikan-polar" id="id20">[Arikan_Polar]</a>. However, the implementation follows the <cite>recursive
tree</cite> <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gross-fast-scl" id="id21">[Gross_Fast_SCL]</a> terminology and combines nodes for increased
throughputs without changing the outcome of the algorithm.
    
As commonly done, we assume frozen bits are set to <cite>0</cite>. Please note
that - although its practical relevance is only little - setting frozen
bits to <cite>1</cite> may result in <cite>affine</cite> codes instead of linear code as the
<cite>all-zero</cite> codeword is not necessarily part of the code any more.

<em class="property">`property` </em>`frozen_pos`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCDecoder.frozen_pos" title="Permalink to this definition"></a>
    
Frozen positions for Polar decoding.


<em class="property">`property` </em>`info_pos`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCDecoder.info_pos" title="Permalink to this definition"></a>
    
Information bit positions for Polar encoding.


<em class="property">`property` </em>`k`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCDecoder.k" title="Permalink to this definition"></a>
    
Number of information bits.


<em class="property">`property` </em>`llr_max`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCDecoder.llr_max" title="Permalink to this definition"></a>
    
Maximum LLR value for internal calculations.


<em class="property">`property` </em>`n`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCDecoder.n" title="Permalink to this definition"></a>
    
Codeword length.


<em class="property">`property` </em>`output_dtype`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCDecoder.output_dtype" title="Permalink to this definition"></a>
    
Output dtype of decoder.


### PolarSCLDecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polarscldecoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.polar.decoding.``PolarSCLDecoder`(<em class="sig-param">`frozen_pos`</em>, <em class="sig-param">`n`</em>, <em class="sig-param">`list_size``=``8`</em>, <em class="sig-param">`crc_degree``=``None`</em>, <em class="sig-param">`use_hybrid_sc``=``False`</em>, <em class="sig-param">`use_fast_scl``=``True`</em>, <em class="sig-param">`cpu_only``=``False`</em>, <em class="sig-param">`use_scatter``=``False`</em>, <em class="sig-param">`ind_iil_inv``=``None`</em>, <em class="sig-param">`return_crc_status``=``False`</em>, <em class="sig-param">`output_dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/decoding.html#PolarSCLDecoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCLDecoder" title="Permalink to this definition"></a>
    
Successive cancellation list (SCL) decoder <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#tal-scl" id="id22">[Tal_SCL]</a> for Polar codes
and Polar-like codes.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **frozen_pos** (<em>ndarray</em>) – Array of <cite>int</cite> defining the `n-k` indices of the frozen positions.
- **n** (<em>int</em>) – Defining the codeword length.
- **list_size** (<em>int</em>) – Defaults to 8. Defines the list size of the decoder.
- **crc_degree** (<em>str</em>) – Defining the CRC polynomial to be used. Can be any value from
<cite>{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}</cite>.
- **use_hybrid_sc** (<em>bool</em>) – Defaults to False. If True, SC decoding is applied and only the
codewords with invalid CRC are decoded with SCL. This option
requires an outer CRC specified via `crc_degree`.
Remark: hybrid_sc does not support XLA optimization, i.e.,
<cite>@tf.function(jit_compile=True)</cite>.
- **use_fast_scl** (<em>bool</em>) – Defaults to True. If True, Tree pruning is used to
reduce the decoding complexity. The output is equivalent to the
non-pruned version (besides numerical differences).
- **cpu_only** (<em>bool</em>) – Defaults to False. If True, <cite>tf.py_function</cite> embedding
is used and the decoder runs on the CPU. This option is usually
slower, but also more memory efficient and, in particular,
recommended for larger blocklengths. Remark: cpu_only does not
support XLA optimization <cite>@tf.function(jit_compile=True)</cite>.
- **use_scatter** (<em>bool</em>) – Defaults to False. If True, <cite>tf.tensor_scatter_update</cite> is used for
tensor updates. This option is usually slower, but more memory
efficient.
- **ind_iil_inv** (<em>None</em><em> or </em><em>[</em><em>k+k_crc</em><em>]</em><em>, </em><em>int</em><em> or </em><em>tf.int</em>) – Defaults to None. If not <cite>None</cite>, the sequence is used as inverse
input bit interleaver before evaluating the CRC.
Remark: this only effects the CRC evaluation but the output
sequence is not permuted.
- **return_crc_status** (<em>bool</em>) – Defaults to False. If True, the decoder additionally returns the
CRC status indicating if a codeword was (most likely) correctly
recovered. This is only available if `crc_degree` is not None.
- **output_dtype** (<em>tf.DType</em>) – Defaults to tf.float32. Defines the output datatype of the layer
(internal precision remains tf.float32).


Input
    
**inputs** (<em>[…,n], tf.float32</em>) – 2+D tensor containing the channel LLR values (as logits).

Output
 
- **b_hat** (<em>[…,k], tf.float32</em>) – 2+D tensor containing hard-decided estimations of all <cite>k</cite>
information bits.
- **crc_status** (<em>[…], tf.bool</em>) – CRC status indicating if a codeword was (most likely) correctly
recovered. This is only returned if `return_crc_status` is True.
Note that false positives are possible.


Raises
 
- **AssertionError** – If `n` is not <cite>int</cite>.
- **AssertionError** – If `n` is not a power of 2.
- **AssertionError** – If the number of elements in `frozen_pos` is greater than `n`.
- **AssertionError** – If `frozen_pos` does not consists of <cite>int</cite>.
- **AssertionError** – If `list_size` is not <cite>int</cite>.
- **AssertionError** – If `cpu_only` is not <cite>bool</cite>.
- **AssertionError** – If `use_scatter` is not <cite>bool</cite>.
- **AssertionError** – If `use_fast_scl` is not <cite>bool</cite>.
- **AssertionError** – If `use_hybrid_sc` is not <cite>bool</cite>.
- **AssertionError** – If `list_size` is not a power of 2.
- **ValueError** – If `output_dtype` is not {tf.float16, tf.float32, tf.
    float64}.
- **ValueError** – If `inputs` is not of shape <cite>[…, n]</cite> or <cite>dtype</cite> is not
    correct.
- **InvalidArgumentError** – When rank(`inputs`)<2.




**Note**
    
This layer implements the successive cancellation list (SCL) decoder
as described in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#tal-scl" id="id23">[Tal_SCL]</a> but uses LLR-based message updates
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#stimming-llr" id="id24">[Stimming_LLR]</a>. The implementation follows the notation from
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gross-fast-scl" id="id25">[Gross_Fast_SCL]</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#hashemi-sscl" id="id26">[Hashemi_SSCL]</a>. If option <cite>use_fast_scl</cite> is active
tree pruning is used and tree nodes are combined if possible (see
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#hashemi-sscl" id="id27">[Hashemi_SSCL]</a> for details).
    
Implementing SCL decoding as TensorFlow graph is a difficult task that
requires several design tradeoffs to match the TF constraints while
maintaining a reasonable throughput. Thus, the decoder minimizes
the <cite>control flow</cite> as much as possible, leading to a strong memory
occupation (e.g., due to full path duplication after each decision).
For longer code lengths, the complexity of the decoding graph becomes
large and we recommend to use the <cite>CPU_only</cite> option that uses an
embedded Numpy decoder. Further, this function recursively unrolls the
SCL decoding tree, thus, for larger values of `n` building the
decoding graph can become time consuming. Please consider the
`cpu_only` option if building the graph takes to long.
    
A hybrid SC/SCL decoder as proposed in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#cammerer-hybrid-scl" id="id28">[Cammerer_Hybrid_SCL]</a> (using SC
instead of BP) can be activated with option `use_hybrid_sc` iff an
outer CRC is available. Please note that the results are not exactly
SCL performance caused by the false positive rate of the CRC.
    
As commonly done, we assume frozen bits are set to <cite>0</cite>. Please note
that - although its practical relevance is only little - setting frozen
bits to <cite>1</cite> may result in <cite>affine</cite> codes instead of linear code as the
<cite>all-zero</cite> codeword is not necessarily part of the code any more.

<em class="property">`property` </em>`frozen_pos`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCLDecoder.frozen_pos" title="Permalink to this definition"></a>
    
Frozen positions for Polar decoding.


<em class="property">`property` </em>`info_pos`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCLDecoder.info_pos" title="Permalink to this definition"></a>
    
Information bit positions for Polar encoding.


<em class="property">`property` </em>`k`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCLDecoder.k" title="Permalink to this definition"></a>
    
Number of information bits.


<em class="property">`property` </em>`k_crc`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCLDecoder.k_crc" title="Permalink to this definition"></a>
    
Number of CRC bits.


<em class="property">`property` </em>`list_size`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCLDecoder.list_size" title="Permalink to this definition"></a>
    
List size for SCL decoding.


<em class="property">`property` </em>`llr_max`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCLDecoder.llr_max" title="Permalink to this definition"></a>
    
Maximum LLR value for internal calculations.


<em class="property">`property` </em>`n`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCLDecoder.n" title="Permalink to this definition"></a>
    
Codeword length.


<em class="property">`property` </em>`output_dtype`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCLDecoder.output_dtype" title="Permalink to this definition"></a>
    
Output dtype of decoder.


### PolarBPDecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polarbpdecoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.polar.decoding.``PolarBPDecoder`(<em class="sig-param">`frozen_pos`</em>, <em class="sig-param">`n`</em>, <em class="sig-param">`num_iter``=``20`</em>, <em class="sig-param">`hard_out``=``True`</em>, <em class="sig-param">`output_dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/decoding.html#PolarBPDecoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarBPDecoder" title="Permalink to this definition"></a>
    
Belief propagation (BP) decoder for Polar codes <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#arikan-polar" id="id29">[Arikan_Polar]</a> and
Polar-like codes based on <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#arikan-bp" id="id30">[Arikan_BP]</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#forney-graphs" id="id31">[Forney_Graphs]</a>.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
    
Remark: The PolarBPDecoder does currently not support XLA.
Parameters
 
- **frozen_pos** (<em>ndarray</em>) – Array of <cite>int</cite> defining the `n-k` indices of the frozen positions.
- **n** (<em>int</em>) – Defining the codeword length.
- **num_iter** (<em>int</em>) – Defining the number of decoder iterations (no early stopping used
at the moment).
- **hard_out** (<em>bool</em>) – Defaults to True. If True, the decoder provides hard-decided
information bits instead of soft-values.
- **output_dtype** (<em>tf.DType</em>) – Defaults to tf.float32. Defines the output datatype of the layer
(internal precision remains tf.float32).


Input
    
**inputs** (<em>[…,n], tf.float32</em>) – 2+D tensor containing the channel logits/llr values.

Output
    
<em>[…,k], tf.float32</em> – 2+D tensor containing bit-wise soft-estimates
(or hard-decided bit-values) of all `k` information bits.

Raises
 
- **AssertionError** – If `n` is not <cite>int</cite>.
- **AssertionError** – If `n` is not a power of 2.
- **AssertionError** – If the number of elements in `frozen_pos` is greater than `n`.
- **AssertionError** – If `frozen_pos` does not consists of <cite>int</cite>.
- **AssertionError** – If `hard_out` is not <cite>bool</cite>.
- **ValueError** – If `output_dtype` is not {tf.float16, tf.float32, tf.float64}.
- **AssertionError** – If `num_iter` is not <cite>int</cite>.
- **AssertionError** – If `num_iter` is not a positive value.




**Note**
    
This decoder is fully differentiable and, thus, well-suited for
gradient descent-based learning tasks such as <cite>learned code design</cite>
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#ebada-design" id="id32">[Ebada_Design]</a>.
    
As commonly done, we assume frozen bits are set to <cite>0</cite>. Please note
that - although its practical relevance is only little - setting frozen
bits to <cite>1</cite> may result in <cite>affine</cite> codes instead of linear code as the
<cite>all-zero</cite> codeword is not necessarily part of the code any more.

<em class="property">`property` </em>`frozen_pos`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarBPDecoder.frozen_pos" title="Permalink to this definition"></a>
    
Frozen positions for Polar decoding.


<em class="property">`property` </em>`hard_out`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarBPDecoder.hard_out" title="Permalink to this definition"></a>
    
Indicates if decoder hard-decides outputs.


<em class="property">`property` </em>`info_pos`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarBPDecoder.info_pos" title="Permalink to this definition"></a>
    
Information bit positions for Polar encoding.


<em class="property">`property` </em>`k`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarBPDecoder.k" title="Permalink to this definition"></a>
    
Number of information bits.


<em class="property">`property` </em>`llr_max`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarBPDecoder.llr_max" title="Permalink to this definition"></a>
    
Maximum LLR value for internal calculations.


<em class="property">`property` </em>`n`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarBPDecoder.n" title="Permalink to this definition"></a>
    
Codeword length.


<em class="property">`property` </em>`num_iter`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarBPDecoder.num_iter" title="Permalink to this definition"></a>
    
Number of decoding iterations.


<em class="property">`property` </em>`output_dtype`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarBPDecoder.output_dtype" title="Permalink to this definition"></a>
    
Output dtype of decoder.


## Polar Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polar-utility-functions" title="Permalink to this headline"></a>

### generate_5g_ranking<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#generate-5g-ranking" title="Permalink to this headline"></a>

`sionna.fec.polar.utils.``generate_5g_ranking`(<em class="sig-param">`k`</em>, <em class="sig-param">`n`</em>, <em class="sig-param">`sort``=``True`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/utils.html#generate_5g_ranking">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.utils.generate_5g_ranking" title="Permalink to this definition"></a>
    
Returns information and frozen bit positions of the 5G Polar code
as defined in Tab. 5.3.1.2-1 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id33">[3GPPTS38212]</a> for given values of `k`
and `n`.
Input
 
- **k** (<em>int</em>) – The number of information bit per codeword.
- **n** (<em>int</em>) – The desired codeword length. Must be a power of two.
- **sort** (<em>bool</em>) – Defaults to True. Indicates if the returned indices are
sorted.


Output
 
- **[frozen_pos, info_pos]** – List:
- **frozen_pos** (<em>ndarray</em>) – An array of ints of shape <cite>[n-k]</cite> containing the frozen
position indices.
- **info_pos** (<em>ndarray</em>) – An array of ints of shape <cite>[k]</cite> containing the information
position indices.


Raises
 
- **AssertionError** – If `k` or `n` are not positve ints.
- **AssertionError** – If `sort` is not bool.
- **AssertionError** – If `k` or `n` are larger than 1024
- **AssertionError** – If `n` is less than 32.
- **AssertionError** – If the resulting coderate is invalid (<cite>>1.0</cite>).
- **AssertionError** – If `n` is not a power of 2.




### generate_polar_transform_mat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#generate-polar-transform-mat" title="Permalink to this headline"></a>

`sionna.fec.polar.utils.``generate_polar_transform_mat`(<em class="sig-param">`n_lift`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/utils.html#generate_polar_transform_mat">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.utils.generate_polar_transform_mat" title="Permalink to this definition"></a>
    
Generate the polar transformation matrix (Kronecker product).
Input
    
**n_lift** (<em>int</em>) – Defining the Kronecker power, i.e., how often is the kernel lifted.

Output
    
<em>ndarray</em> – Array of <cite>0s</cite> and <cite>1s</cite> of shape <cite>[2^n_lift , 2^n_lift]</cite> containing
the Polar transformation matrix.



### generate_rm_code<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#generate-rm-code" title="Permalink to this headline"></a>

`sionna.fec.polar.utils.``generate_rm_code`(<em class="sig-param">`r`</em>, <em class="sig-param">`m`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/utils.html#generate_rm_code">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.utils.generate_rm_code" title="Permalink to this definition"></a>
    
Generate frozen positions of the (r, m) Reed Muller (RM) code.
Input
 
- **r** (<em>int</em>) – The order of the RM code.
- **m** (<em>int</em>) – <cite>log2</cite> of the desired codeword length.


Output
 
- **[frozen_pos, info_pos, n, k, d_min]** – List:
- **frozen_pos** (<em>ndarray</em>) – An array of ints of shape <cite>[n-k]</cite> containing the frozen
position indices.
- **info_pos** (<em>ndarray</em>) – An array of ints of shape <cite>[k]</cite> containing the information
position indices.
- **n** (<em>int</em>) – Resulting codeword length
- **k** (<em>int</em>) – Number of information bits
- **d_min** (<em>int</em>) – Minimum distance of the code.


Raises
 
- **AssertionError** – If `r` is larger than `m`.
- **AssertionError** – If `r` or `m` are not positive ints.




### generate_dense_polar<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#generate-dense-polar" title="Permalink to this headline"></a>

`sionna.fec.polar.utils.``generate_dense_polar`(<em class="sig-param">`frozen_pos`</em>, <em class="sig-param">`n`</em>, <em class="sig-param">`verbose``=``True`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/utils.html#generate_dense_polar">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.utils.generate_dense_polar" title="Permalink to this definition"></a>
    
Generate <em>naive</em> (dense) Polar parity-check and generator matrix.
    
This function follows Lemma 1 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#goala-lp" id="id34">[Goala_LP]</a> and returns a parity-check
matrix for Polar codes.

**Note**
    
The resulting matrix can be used for decoding with the
`LDPCBPDecoder` class. However, the resulting
parity-check matrix is (usually) not sparse and, thus, not suitable for
belief propagation decoding as the graph has many short cycles.
Please consider `PolarBPDecoder` for iterative
decoding over the encoding graph.

Input
 
- **frozen_pos** (<em>ndarray</em>) – Array of <cite>int</cite> defining the `n-k` indices of the frozen positions.
- **n** (<em>int</em>) – The codeword length.
- **verbose** (<em>bool</em>) – Defaults to True. If True, the code properties are printed.


Output
 
- **pcm** (ndarray of <cite>zeros</cite> and <cite>ones</cite> of shape [n-k, n]) – The parity-check matrix.
- **gm** (ndarray of <cite>zeros</cite> and <cite>ones</cite> of shape [k, n]) – The generator matrix.





References:
3GPPTS38212(<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id1">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id2">2</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id5">3</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id8">4</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id9">5</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id10">6</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id11">7</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id12">8</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id13">9</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id14">10</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id15">11</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id16">12</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id17">13</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id33">14</a>)
    
ETSI 3GPP TS 38.212 “5G NR Multiplexing and channel
coding”, v.16.5.0, 2021-03.

Bioglio_Design(<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id3">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id4">2</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id6">3</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id18">4</a>)
    
V. Bioglio, C. Condo, I. Land, “Design of
Polar Codes in 5G New Radio,” IEEE Communications Surveys &
Tutorials, 2020. Online availabe <a class="reference external" href="https://arxiv.org/pdf/1804.04389.pdf">https://arxiv.org/pdf/1804.04389.pdf</a>

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/fec.polar.html#id7">Hui_ChannelCoding</a>
    
D. Hui, S. Sandberg, Y. Blankenship, M.
Andersson, L. Grosjean “Channel coding in 5G new radio: A
Tutorial Overview and Performance Comparison with 4G LTE,” IEEE
Vehicular Technology Magazine, 2018.

Arikan_Polar(<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id19">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id20">2</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id29">3</a>)
    
E. Arikan, “Channel polarization: A method for
constructing capacity-achieving codes for symmetric
binary-input memoryless channels,” IEEE Trans. on Information
Theory, 2009.

Gross_Fast_SCL(<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id21">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id25">2</a>)
    
Seyyed Ali Hashemi, Carlo Condo, and Warren J.
Gross, “Fast and Flexible Successive-cancellation List Decoders
for Polar Codes.” IEEE Trans. on Signal Processing, 2017.

Tal_SCL(<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id22">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id23">2</a>)
    
Ido Tal and Alexander Vardy, “List Decoding of Polar
Codes.” IEEE Trans Inf Theory, 2015.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/fec.polar.html#id24">Stimming_LLR</a>
    
Alexios Balatsoukas-Stimming, Mani Bastani Parizi,
Andreas Burg, “LLR-Based Successive Cancellation List Decoding
of Polar Codes.” IEEE Trans Signal Processing, 2015.

Hashemi_SSCL(<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id26">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id27">2</a>)
    
Seyyed Ali Hashemi, Carlo Condo, and Warren J.
Gross, “Simplified Successive-Cancellation List Decoding
of Polar Codes.” IEEE ISIT, 2016.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/fec.polar.html#id28">Cammerer_Hybrid_SCL</a>
    
Sebastian Cammerer, Benedikt Leible, Matthias
Stahl, Jakob Hoydis, and Stephan ten Brink, “Combining Belief
Propagation and Successive Cancellation List Decoding of Polar
Codes on a GPU Platform,” IEEE ICASSP, 2017.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/fec.polar.html#id30">Arikan_BP</a>
    
E. Arikan, “A Performance Comparison of Polar Codes and
Reed-Muller Codes,” IEEE Commun. Lett., vol. 12, no. 6, pp.
447-449, Jun. 2008.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/fec.polar.html#id31">Forney_Graphs</a>
    
G. D. Forney, “Codes on graphs: normal realizations,”
IEEE Trans. Inform. Theory, vol. 47, no. 2, pp. 520-548, Feb. 2001.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/fec.polar.html#id32">Ebada_Design</a>
    
M. Ebada, S. Cammerer, A. Elkelesh and S. ten Brink,
“Deep Learning-based Polar Code Design”, Annual Allerton
Conference on Communication, Control, and Computing, 2019.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/fec.polar.html#id34">Goala_LP</a>
    
N. Goela, S. Korada, M. Gastpar, “On LP decoding of Polar
Codes,” IEEE ITW 2010.



