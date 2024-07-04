# Polar Codes

The Polar code module supports 5G-compliant Polar codes and includes successive cancellation (SC), successive cancellation list (SCL), and belief propagation (BP) decoding.

The module supports rate-matching and CRC-aided decoding.
Further, Reed-Muller (RM) code design is available and can be used in combination with the Polar encoding/decoding algorithms.

The following code snippets show how to setup and run a rate-matched 5G compliant Polar encoder and a corresponding successive cancellation list (SCL) decoder.

First, we need to create instances of [`Polar5GEncoder`](https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder) and [`Polar5GDecoder`](https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder):
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
## Polar Encoding

### Polar5GEncoder

`class` `sionna.fec.polar.encoding.``Polar5GEncoder`(*`k`*, *`n`*, *`verbose``=``False`*, *`channel_type``=``'uplink'`*, *`dtype``=``tf.float32`*)[`[source]`](../_modules/sionna/fec/polar/encoding.html#Polar5GEncoder)

5G compliant Polar encoder including rate-matching following [[3GPPTS38212]](https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212)
for the uplink scenario (<cite>UCI</cite>) and downlink scenario (<cite>DCI</cite>).

This layer performs polar encoding for `k` information bits and
rate-matching such that the codeword lengths is `n`. This includes the CRC
concatenation and the interleaving as defined in [[3GPPTS38212]](https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212).

Note: <cite>block segmentation</cite> is currently not supported (<cite>I_seq=False</cite>).

We follow the basic structure from Fig. 6 in [[Bioglio_Design]](https://nvlabs.github.io/sionna/api/fec.polar.html#bioglio-design).
 ig. 6 Fig. 1: Implemented 5G Polar encoding chain following Fig. 6 in
[[Bioglio_Design]](https://nvlabs.github.io/sionna/api/fec.polar.html#bioglio-design) for the uplink (<cite>I_BIL</cite> = <cite>True</cite>) and the downlink
(<cite>I_IL</cite> = <cite>True</cite>) scenario without <cite>block segmentation</cite>.

For further details, we refer to [[3GPPTS38212]](https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212), [[Bioglio_Design]](https://nvlabs.github.io/sionna/api/fec.polar.html#bioglio-design) and
[[Hui_ChannelCoding]](https://nvlabs.github.io/sionna/api/fec.polar.html#hui-channelcoding).

The class inherits from the Keras layer class and can be used as layer in a
Keras model. Further, the class inherits from PolarEncoder.
Parameters

- **k** (*int*)  Defining the number of information bit per codeword.
- **n** (*int*)  Defining the codeword length.
- **channel_type** (*str*)  Defaults to uplink. Can be uplink or downlink.
- **verbose** (*bool*)  Defaults to False. If True, rate-matching parameters will be
printed.
- **dtype** (*tf.DType*)  Defaults to tf.float32. Defines the output datatype of the layer
(internal precision remains tf.uint8).


Input

**inputs** (*[,k], tf.float32*)  2+D tensor containing the information bits to be encoded.

Output

*[,n], tf.float32*  2+D tensor containing the codeword bits.

Raises

- **AssertionError**  `k` and `n` must be positive integers and `k` must be smaller
    (or equal) than `n`.
- **AssertionError**  If `n` and `k` are invalid code parameters (see [[3GPPTS38212]](https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212)).
- **AssertionError**  If `verbose` is not <cite>bool</cite>.
- **ValueError**  If `dtype` is not supported.


**Note**

The encoder supports the <cite>uplink</cite> Polar coding (<cite>UCI</cite>) scheme from
[[3GPPTS38212]](https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212) and the <cite>downlink</cite> Polar coding (<cite>DCI</cite>) [[3GPPTS38212]](https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212),
respectively.

For <cite>12 <= k <= 19</cite> the 3 additional parity bits as defined in
[[3GPPTS38212]](https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212) are not implemented as it would also require a
modified decoding procedure to materialize the potential gains.

<cite>Code segmentation</cite> is currently not supported and, thus, `n` is
limited to a maximum length of 1088 codeword bits.

For the downlink scenario, the input length is limited to <cite>k <= 140</cite>
information bits due to the limited input bit interleaver size
[[3GPPTS38212]](https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212).

For simplicity, the implementation does not exactly re-implement the
<cite>DCI</cite> scheme from [[3GPPTS38212]](https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212). This implementation neglects the
<cite>all-one</cite> initialization of the CRC shift register and the scrambling of the CRC parity bits with the <cite>RNTI</cite>.

`channel_interleaver`(*`c`*)[`[source]`](../_modules/sionna/fec/polar/encoding.html#Polar5GEncoder.channel_interleaver)

Triangular interleaver following Sec. 5.4.1.3 in [[3GPPTS38212]](https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212).
Input

**c** (*ndarray*)  1D array to be interleaved.

Output

*ndarray*  Interleaved version of `c` with same shape and dtype as `c`.


`property` `enc_crc`

CRC encoder layer used for CRC concatenation.


`input_interleaver`(*`c`*)[`[source]`](../_modules/sionna/fec/polar/encoding.html#Polar5GEncoder.input_interleaver)

Input interleaver following Sec. 5.4.1.1 in [[3GPPTS38212]](https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212).
Input

**c** (*ndarray*)  1D array to be interleaved.

Output

*ndarray*  Interleaved version of `c` with same shape and dtype as `c`.


`property` `k`

Number of information bits including rate-matching.


`property` `k_polar`

Number of information bits of the underlying Polar code.


`property` `k_target`

Number of information bits including rate-matching.


`property` `n`

Codeword length including rate-matching.


`property` `n_polar`

Codeword length of the underlying Polar code.


`property` `n_target`

Codeword length including rate-matching.


`subblock_interleaving`(*`u`*)[`[source]`](../_modules/sionna/fec/polar/encoding.html#Polar5GEncoder.subblock_interleaving)

Input bit interleaving as defined in Sec 5.4.1.1 [[3GPPTS38212]](https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212).
Input

**u** (*ndarray*)  1D array to be interleaved. Length of `u` must be a multiple
of 32.

Output

*ndarray*  Interleaved version of `u` with same shape and dtype as `u`.

Raises

**AssertionError**  If length of `u` is not a multiple of 32.


### PolarEncoder

`class` `sionna.fec.polar.encoding.``PolarEncoder`(*`frozen_pos`*, *`n`*, *`dtype``=``tf.float32`*)[`[source]`](../_modules/sionna/fec/polar/encoding.html#PolarEncoder)

Polar encoder for given code parameters.

This layer performs polar encoding for the given `k` information bits and
the <cite>frozen set</cite> (i.e., indices of frozen positions) specified by
`frozen_pos`.

The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters

- **frozen_pos** (*ndarray*)  Array of <cite>int</cite> defining the <cite>n-k</cite> frozen indices, i.e., information
bits are mapped onto the <cite>k</cite> complementary positions.
- **n** (*int*)  Defining the codeword length.
- **dtype** (*tf.DType*)  Defaults to <cite>tf.float32</cite>. Defines the output datatype of the layer
(internal precision is <cite>tf.uint8</cite>).


Input

**inputs** (*[,k], tf.float32*)  2+D tensor containing the information bits to be encoded.

Output

*[,n], tf.float32*  2+D tensor containing the codeword bits.

Raises

- **AssertionError**  `k` and `n` must be positive integers and `k` must be smaller
    (or equal) than `n`.
- **AssertionError**  If `n` is not a power of 2.
- **AssertionError**  If the number of elements in `frozen_pos` is great than `n`.
- **AssertionError**  If `frozen_pos` does not consists of <cite>int</cite>.
- **ValueError**  If `dtype` is not supported.
- **ValueError**  If `inputs` contains other values than <cite>0</cite> or <cite>1</cite>.
- **TypeError**  If `inputs` is not <cite>tf.float32</cite>.
- **InvalidArgumentError**  When rank(`inputs`)<2.
- **InvalidArgumentError**  When shape of last dim is not `k`.


**Note**

As commonly done, we assume frozen bits are set to <cite>0</cite>. Please note
that - although its practical relevance is only little - setting frozen
bits to <cite>1</cite> may result in <cite>affine</cite> codes instead of linear code as the
<cite>all-zero</cite> codeword is not necessarily part of the code any more.

`property` `frozen_pos`

Frozen positions for Polar decoding.


`property` `info_pos`

Information bit positions for Polar encoding.


`property` `k`

Number of information bits.


`property` `n`

Codeword length.


## Polar Decoding

### Polar5GDecoder

`class` `sionna.fec.polar.decoding.``Polar5GDecoder`(*`enc_polar`*, *`dec_type``=``'SC'`*, *`list_size``=``8`*, *`num_iter``=``20`*, *`return_crc_status``=``False`*, *`output_dtype``=``tf.float32`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/fec/polar/decoding.html#Polar5GDecoder)

Wrapper for 5G compliant decoding including rate-recovery and CRC removal.

The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters

- **enc_polar** ()  Instance of the [`Polar5GEncoder`](https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder)
used for encoding including rate-matching.
- **dec_type** (*str*)  Defaults to <cite>SC</cite>. Defining the decoder to be used.
Must be one of the following <cite>{SC, SCL, hybSCL, BP}</cite>.
- **list_size** (*int*)  Defaults to 8. Defining the list size <cite>iff</cite> list-decoding is used.
Only required for `dec_types` <cite>{SCL, hybSCL}</cite>.
- **num_iter** (*int*)  Defaults to 20. Defining the number of BP iterations. Only required
for `dec_type` <cite>BP</cite>.
- **return_crc_status** (*bool*)  Defaults to False. If True, the decoder additionally returns the
CRC status indicating if a codeword was (most likely) correctly
recovered.
- **output_dtype** (*tf.DType*)  Defaults to tf.float32. Defines the output datatype of the layer
(internal precision remains tf.float32).


Input

**inputs** (*[,n], tf.float32*)  2+D tensor containing the channel logits/llr values.

Output

- **b_hat** (*[,k], tf.float32*)  2+D tensor containing hard-decided estimations of all <cite>k</cite>
information bits.
- **crc_status** (*[], tf.bool*)  CRC status indicating if a codeword was (most likely) correctly
recovered. This is only returned if `return_crc_status` is True.
Note that false positives are possible.


Raises

- **AssertionError**  If `enc_polar` is not <cite>Polar5GEncoder</cite>.
- **ValueError**  If `dec_type` is not <cite>{SC, SCL, SCL8, SCL32, hybSCL,
    BP}</cite>.
- **AssertionError**  If `dec_type` is not <cite>str</cite>.
- **ValueError**  If `inputs` is not of shape <cite>[, n]</cite> or <cite>dtype</cite> is not
    the same as `output_dtype`.
- **InvalidArgumentError**  When rank(`inputs`)<2.


**Note**

This layer supports the uplink and downlink Polar rate-matching scheme
without <cite>codeword segmentation</cite>.

Although the decoding <cite>list size</cite> is not provided by 3GPP
[[3GPPTS38212]](https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212), the consortium has agreed on a <cite>list size</cite> of 8 for the
5G decoding reference curves [[Bioglio_Design]](https://nvlabs.github.io/sionna/api/fec.polar.html#bioglio-design).

All list-decoders apply <cite>CRC-aided</cite> decoding, however, the non-list
decoders (<cite>SC</cite> and <cite>BP</cite>) cannot materialize the CRC leading to an
effective rate-loss.

`property` `dec_type`

Decoder type used for decoding as str.


`property` `frozen_pos`

Frozen positions for Polar decoding.


`property` `info_pos`

Information bit positions for Polar encoding.


`property` `k_polar`

Number of information bits of mother Polar code.


`property` `k_target`

Number of information bits including rate-matching.


`property` `llr_max`

Maximum LLR value for internal calculations.


`property` `n_polar`

Codeword length of mother Polar code.


`property` `n_target`

Codeword length including rate-matching.


`property` `output_dtype`

Output dtype of decoder.


`property` `polar_dec`

Decoder instance used for decoding.


### PolarSCDecoder

`class` `sionna.fec.polar.decoding.``PolarSCDecoder`(*`frozen_pos`*, *`n`*, *`output_dtype``=``tf.float32`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/fec/polar/decoding.html#PolarSCDecoder)

Successive cancellation (SC) decoder [[Arikan_Polar]](https://nvlabs.github.io/sionna/api/fec.polar.html#arikan-polar) for Polar codes and
Polar-like codes.

The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters

- **frozen_pos** (*ndarray*)  Array of <cite>int</cite> defining the `n-k` indices of the frozen positions.
- **n**  Defining the codeword length.


Input

**inputs** (*[,n], tf.float32*)  2+D tensor containing the channel LLR values (as logits).

Output

*[,k], tf.float32*  2+D tensor  containing hard-decided estimations of all `k`
information bits.

Raises

- **AssertionError**  If `n` is not <cite>int</cite>.
- **AssertionError**  If `n` is not a power of 2.
- **AssertionError**  If the number of elements in `frozen_pos` is greater than `n`.
- **AssertionError**  If `frozen_pos` does not consists of <cite>int</cite>.
- **ValueError**  If `output_dtype` is not {tf.float16, tf.float32, tf.float64}.


**Note**

This layer implements the SC decoder as described in
[[Arikan_Polar]](https://nvlabs.github.io/sionna/api/fec.polar.html#arikan-polar). However, the implementation follows the <cite>recursive
tree</cite> [[Gross_Fast_SCL]](https://nvlabs.github.io/sionna/api/fec.polar.html#gross-fast-scl) terminology and combines nodes for increased
throughputs without changing the outcome of the algorithm.

As commonly done, we assume frozen bits are set to <cite>0</cite>. Please note
that - although its practical relevance is only little - setting frozen
bits to <cite>1</cite> may result in <cite>affine</cite> codes instead of linear code as the
<cite>all-zero</cite> codeword is not necessarily part of the code any more.

`property` `frozen_pos`

Frozen positions for Polar decoding.


`property` `info_pos`

Information bit positions for Polar encoding.


`property` `k`

Number of information bits.


`property` `llr_max`

Maximum LLR value for internal calculations.


`property` `n`

Codeword length.


`property` `output_dtype`

Output dtype of decoder.


### PolarSCLDecoder

`class` `sionna.fec.polar.decoding.``PolarSCLDecoder`(*`frozen_pos`*, *`n`*, *`list_size``=``8`*, *`crc_degree``=``None`*, *`use_hybrid_sc``=``False`*, *`use_fast_scl``=``True`*, *`cpu_only``=``False`*, *`use_scatter``=``False`*, *`ind_iil_inv``=``None`*, *`return_crc_status``=``False`*, *`output_dtype``=``tf.float32`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/fec/polar/decoding.html#PolarSCLDecoder)

Successive cancellation list (SCL) decoder [[Tal_SCL]](https://nvlabs.github.io/sionna/api/fec.polar.html#tal-scl) for Polar codes
and Polar-like codes.

The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters

- **frozen_pos** (*ndarray*)  Array of <cite>int</cite> defining the `n-k` indices of the frozen positions.
- **n** (*int*)  Defining the codeword length.
- **list_size** (*int*)  Defaults to 8. Defines the list size of the decoder.
- **crc_degree** (*str*)  Defining the CRC polynomial to be used. Can be any value from
<cite>{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}</cite>.
- **use_hybrid_sc** (*bool*)  Defaults to False. If True, SC decoding is applied and only the
codewords with invalid CRC are decoded with SCL. This option
requires an outer CRC specified via `crc_degree`.
Remark: hybrid_sc does not support XLA optimization, i.e.,
<cite>@tf.function(jit_compile=True)</cite>.
- **use_fast_scl** (*bool*)  Defaults to True. If True, Tree pruning is used to
reduce the decoding complexity. The output is equivalent to the
non-pruned version (besides numerical differences).
- **cpu_only** (*bool*)  Defaults to False. If True, <cite>tf.py_function</cite> embedding
is used and the decoder runs on the CPU. This option is usually
slower, but also more memory efficient and, in particular,
recommended for larger blocklengths. Remark: cpu_only does not
support XLA optimization <cite>@tf.function(jit_compile=True)</cite>.
- **use_scatter** (*bool*)  Defaults to False. If True, <cite>tf.tensor_scatter_update</cite> is used for
tensor updates. This option is usually slower, but more memory
efficient.
- **ind_iil_inv** (*None** or **[**k+k_crc**]**, **int** or **tf.int*)  Defaults to None. If not <cite>None</cite>, the sequence is used as inverse
input bit interleaver before evaluating the CRC.
Remark: this only effects the CRC evaluation but the output
sequence is not permuted.
- **return_crc_status** (*bool*)  Defaults to False. If True, the decoder additionally returns the
CRC status indicating if a codeword was (most likely) correctly
recovered. This is only available if `crc_degree` is not None.
- **output_dtype** (*tf.DType*)  Defaults to tf.float32. Defines the output datatype of the layer
(internal precision remains tf.float32).


Input

**inputs** (*[,n], tf.float32*)  2+D tensor containing the channel LLR values (as logits).

Output

- **b_hat** (*[,k], tf.float32*)  2+D tensor containing hard-decided estimations of all <cite>k</cite>
information bits.
- **crc_status** (*[], tf.bool*)  CRC status indicating if a codeword was (most likely) correctly
recovered. This is only returned if `return_crc_status` is True.
Note that false positives are possible.


Raises

- **AssertionError**  If `n` is not <cite>int</cite>.
- **AssertionError**  If `n` is not a power of 2.
- **AssertionError**  If the number of elements in `frozen_pos` is greater than `n`.
- **AssertionError**  If `frozen_pos` does not consists of <cite>int</cite>.
- **AssertionError**  If `list_size` is not <cite>int</cite>.
- **AssertionError**  If `cpu_only` is not <cite>bool</cite>.
- **AssertionError**  If `use_scatter` is not <cite>bool</cite>.
- **AssertionError**  If `use_fast_scl` is not <cite>bool</cite>.
- **AssertionError**  If `use_hybrid_sc` is not <cite>bool</cite>.
- **AssertionError**  If `list_size` is not a power of 2.
- **ValueError**  If `output_dtype` is not {tf.float16, tf.float32, tf.
    float64}.
- **ValueError**  If `inputs` is not of shape <cite>[, n]</cite> or <cite>dtype</cite> is not
    correct.
- **InvalidArgumentError**  When rank(`inputs`)<2.


**Note**

This layer implements the successive cancellation list (SCL) decoder
as described in [[Tal_SCL]](https://nvlabs.github.io/sionna/api/fec.polar.html#tal-scl) but uses LLR-based message updates
[[Stimming_LLR]](https://nvlabs.github.io/sionna/api/fec.polar.html#stimming-llr). The implementation follows the notation from
[[Gross_Fast_SCL]](https://nvlabs.github.io/sionna/api/fec.polar.html#gross-fast-scl), [[Hashemi_SSCL]](https://nvlabs.github.io/sionna/api/fec.polar.html#hashemi-sscl). If option <cite>use_fast_scl</cite> is active
tree pruning is used and tree nodes are combined if possible (see
[[Hashemi_SSCL]](https://nvlabs.github.io/sionna/api/fec.polar.html#hashemi-sscl) for details).

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

A hybrid SC/SCL decoder as proposed in [[Cammerer_Hybrid_SCL]](https://nvlabs.github.io/sionna/api/fec.polar.html#cammerer-hybrid-scl) (using SC
instead of BP) can be activated with option `use_hybrid_sc` iff an
outer CRC is available. Please note that the results are not exactly
SCL performance caused by the false positive rate of the CRC.

As commonly done, we assume frozen bits are set to <cite>0</cite>. Please note
that - although its practical relevance is only little - setting frozen
bits to <cite>1</cite> may result in <cite>affine</cite> codes instead of linear code as the
<cite>all-zero</cite> codeword is not necessarily part of the code any more.

`property` `frozen_pos`

Frozen positions for Polar decoding.


`property` `info_pos`

Information bit positions for Polar encoding.


`property` `k`

Number of information bits.


`property` `k_crc`

Number of CRC bits.


`property` `list_size`

List size for SCL decoding.


`property` `llr_max`

Maximum LLR value for internal calculations.


`property` `n`

Codeword length.


`property` `output_dtype`

Output dtype of decoder.


### PolarBPDecoder

`class` `sionna.fec.polar.decoding.``PolarBPDecoder`(*`frozen_pos`*, *`n`*, *`num_iter``=``20`*, *`hard_out``=``True`*, *`output_dtype``=``tf.float32`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/fec/polar/decoding.html#PolarBPDecoder)

Belief propagation (BP) decoder for Polar codes [[Arikan_Polar]](https://nvlabs.github.io/sionna/api/fec.polar.html#arikan-polar) and
Polar-like codes based on [[Arikan_BP]](https://nvlabs.github.io/sionna/api/fec.polar.html#arikan-bp) and [[Forney_Graphs]](https://nvlabs.github.io/sionna/api/fec.polar.html#forney-graphs).

The class inherits from the Keras layer class and can be used as layer in a
Keras model.

Remark: The PolarBPDecoder does currently not support XLA.
Parameters

- **frozen_pos** (*ndarray*)  Array of <cite>int</cite> defining the `n-k` indices of the frozen positions.
- **n** (*int*)  Defining the codeword length.
- **num_iter** (*int*)  Defining the number of decoder iterations (no early stopping used
at the moment).
- **hard_out** (*bool*)  Defaults to True. If True, the decoder provides hard-decided
information bits instead of soft-values.
- **output_dtype** (*tf.DType*)  Defaults to tf.float32. Defines the output datatype of the layer
(internal precision remains tf.float32).


Input

**inputs** (*[,n], tf.float32*)  2+D tensor containing the channel logits/llr values.

Output

*[,k], tf.float32*  2+D tensor containing bit-wise soft-estimates
(or hard-decided bit-values) of all `k` information bits.

Raises

- **AssertionError**  If `n` is not <cite>int</cite>.
- **AssertionError**  If `n` is not a power of 2.
- **AssertionError**  If the number of elements in `frozen_pos` is greater than `n`.
- **AssertionError**  If `frozen_pos` does not consists of <cite>int</cite>.
- **AssertionError**  If `hard_out` is not <cite>bool</cite>.
- **ValueError**  If `output_dtype` is not {tf.float16, tf.float32, tf.float64}.
- **AssertionError**  If `num_iter` is not <cite>int</cite>.
- **AssertionError**  If `num_iter` is not a positive value.


**Note**

This decoder is fully differentiable and, thus, well-suited for
gradient descent-based learning tasks such as <cite>learned code design</cite>
[[Ebada_Design]](https://nvlabs.github.io/sionna/api/fec.polar.html#ebada-design).

As commonly done, we assume frozen bits are set to <cite>0</cite>. Please note
that - although its practical relevance is only little - setting frozen
bits to <cite>1</cite> may result in <cite>affine</cite> codes instead of linear code as the
<cite>all-zero</cite> codeword is not necessarily part of the code any more.

`property` `frozen_pos`

Frozen positions for Polar decoding.


`property` `hard_out`

Indicates if decoder hard-decides outputs.


`property` `info_pos`

Information bit positions for Polar encoding.


`property` `k`

Number of information bits.


`property` `llr_max`

Maximum LLR value for internal calculations.


`property` `n`

Codeword length.


`property` `num_iter`

Number of decoding iterations.


`property` `output_dtype`

Output dtype of decoder.


## Polar Utility Functions

### generate_5g_ranking

`sionna.fec.polar.utils.``generate_5g_ranking`(*`k`*, *`n`*, *`sort``=``True`*)[`[source]`](../_modules/sionna/fec/polar/utils.html#generate_5g_ranking)

Returns information and frozen bit positions of the 5G Polar code
as defined in Tab. 5.3.1.2-1 in [[3GPPTS38212]](https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212) for given values of `k`
and `n`.
Input

- **k** (*int*)  The number of information bit per codeword.
- **n** (*int*)  The desired codeword length. Must be a power of two.
- **sort** (*bool*)  Defaults to True. Indicates if the returned indices are
sorted.


Output

- **[frozen_pos, info_pos]**  List:
- **frozen_pos** (*ndarray*)  An array of ints of shape <cite>[n-k]</cite> containing the frozen
position indices.
- **info_pos** (*ndarray*)  An array of ints of shape <cite>[k]</cite> containing the information
position indices.


Raises

- **AssertionError**  If `k` or `n` are not positve ints.
- **AssertionError**  If `sort` is not bool.
- **AssertionError**  If `k` or `n` are larger than 1024
- **AssertionError**  If `n` is less than 32.
- **AssertionError**  If the resulting coderate is invalid (<cite>>1.0</cite>).
- **AssertionError**  If `n` is not a power of 2.


### generate_polar_transform_mat

`sionna.fec.polar.utils.``generate_polar_transform_mat`(*`n_lift`*)[`[source]`](../_modules/sionna/fec/polar/utils.html#generate_polar_transform_mat)

Generate the polar transformation matrix (Kronecker product).
Input

**n_lift** (*int*)  Defining the Kronecker power, i.e., how often is the kernel lifted.

Output

*ndarray*  Array of <cite>0s</cite> and <cite>1s</cite> of shape <cite>[2^n_lift , 2^n_lift]</cite> containing
the Polar transformation matrix.


### generate_rm_code

`sionna.fec.polar.utils.``generate_rm_code`(*`r`*, *`m`*)[`[source]`](../_modules/sionna/fec/polar/utils.html#generate_rm_code)

Generate frozen positions of the (r, m) Reed Muller (RM) code.
Input

- **r** (*int*)  The order of the RM code.
- **m** (*int*)  <cite>log2</cite> of the desired codeword length.


Output

- **[frozen_pos, info_pos, n, k, d_min]**  List:
- **frozen_pos** (*ndarray*)  An array of ints of shape <cite>[n-k]</cite> containing the frozen
position indices.
- **info_pos** (*ndarray*)  An array of ints of shape <cite>[k]</cite> containing the information
position indices.
- **n** (*int*)  Resulting codeword length
- **k** (*int*)  Number of information bits
- **d_min** (*int*)  Minimum distance of the code.


Raises

- **AssertionError**  If `r` is larger than `m`.
- **AssertionError**  If `r` or `m` are not positive ints.


### generate_dense_polar

`sionna.fec.polar.utils.``generate_dense_polar`(*`frozen_pos`*, *`n`*, *`verbose``=``True`*)[`[source]`](../_modules/sionna/fec/polar/utils.html#generate_dense_polar)

Generate *naive* (dense) Polar parity-check and generator matrix.

This function follows Lemma 1 in [[Goala_LP]](https://nvlabs.github.io/sionna/api/fec.polar.html#goala-lp) and returns a parity-check
matrix for Polar codes.

**Note**

The resulting matrix can be used for decoding with the
`LDPCBPDecoder` class. However, the resulting
parity-check matrix is (usually) not sparse and, thus, not suitable for
belief propagation decoding as the graph has many short cycles.
Please consider `PolarBPDecoder` for iterative
decoding over the encoding graph.

Input

- **frozen_pos** (*ndarray*)  Array of <cite>int</cite> defining the `n-k` indices of the frozen positions.
- **n** (*int*)  The codeword length.
- **verbose** (*bool*)  Defaults to True. If True, the code properties are printed.


Output

- **pcm** (ndarray of <cite>zeros</cite> and <cite>ones</cite> of shape [n-k, n])  The parity-check matrix.
- **gm** (ndarray of <cite>zeros</cite> and <cite>ones</cite> of shape [k, n])  The generator matrix.


References:
3GPPTS38212([1](https://nvlabs.github.io/sionna/api/fec.polar.html#id1),[2](https://nvlabs.github.io/sionna/api/fec.polar.html#id2),[3](https://nvlabs.github.io/sionna/api/fec.polar.html#id5),[4](https://nvlabs.github.io/sionna/api/fec.polar.html#id8),[5](https://nvlabs.github.io/sionna/api/fec.polar.html#id9),[6](https://nvlabs.github.io/sionna/api/fec.polar.html#id10),[7](https://nvlabs.github.io/sionna/api/fec.polar.html#id11),[8](https://nvlabs.github.io/sionna/api/fec.polar.html#id12),[9](https://nvlabs.github.io/sionna/api/fec.polar.html#id13),[10](https://nvlabs.github.io/sionna/api/fec.polar.html#id14),[11](https://nvlabs.github.io/sionna/api/fec.polar.html#id15),[12](https://nvlabs.github.io/sionna/api/fec.polar.html#id16),[13](https://nvlabs.github.io/sionna/api/fec.polar.html#id17),[14](https://nvlabs.github.io/sionna/api/fec.polar.html#id33))

ETSI 3GPP TS 38.212 5G NR Multiplexing and channel
coding, v.16.5.0, 2021-03.

Bioglio_Design([1](https://nvlabs.github.io/sionna/api/fec.polar.html#id3),[2](https://nvlabs.github.io/sionna/api/fec.polar.html#id4),[3](https://nvlabs.github.io/sionna/api/fec.polar.html#id6),[4](https://nvlabs.github.io/sionna/api/fec.polar.html#id18))

V. Bioglio, C. Condo, I. Land, Design of
Polar Codes in 5G New Radio, IEEE Communications Surveys &
Tutorials, 2020. Online availabe [https://arxiv.org/pdf/1804.04389.pdf](https://arxiv.org/pdf/1804.04389.pdf)

[Hui_ChannelCoding](https://nvlabs.github.io/sionna/api/fec.polar.html#id7)

D. Hui, S. Sandberg, Y. Blankenship, M.
Andersson, L. Grosjean Channel coding in 5G new radio: A
Tutorial Overview and Performance Comparison with 4G LTE, IEEE
Vehicular Technology Magazine, 2018.

Arikan_Polar([1](https://nvlabs.github.io/sionna/api/fec.polar.html#id19),[2](https://nvlabs.github.io/sionna/api/fec.polar.html#id20),[3](https://nvlabs.github.io/sionna/api/fec.polar.html#id29))

E. Arikan, Channel polarization: A method for
constructing capacity-achieving codes for symmetric
binary-input memoryless channels, IEEE Trans. on Information
Theory, 2009.

Gross_Fast_SCL([1](https://nvlabs.github.io/sionna/api/fec.polar.html#id21),[2](https://nvlabs.github.io/sionna/api/fec.polar.html#id25))

Seyyed Ali Hashemi, Carlo Condo, and Warren J.
Gross, Fast and Flexible Successive-cancellation List Decoders
for Polar Codes. IEEE Trans. on Signal Processing, 2017.

Tal_SCL([1](https://nvlabs.github.io/sionna/api/fec.polar.html#id22),[2](https://nvlabs.github.io/sionna/api/fec.polar.html#id23))

Ido Tal and Alexander Vardy, List Decoding of Polar
Codes. IEEE Trans Inf Theory, 2015.

[Stimming_LLR](https://nvlabs.github.io/sionna/api/fec.polar.html#id24)

Alexios Balatsoukas-Stimming, Mani Bastani Parizi,
Andreas Burg, LLR-Based Successive Cancellation List Decoding
of Polar Codes. IEEE Trans Signal Processing, 2015.

Hashemi_SSCL([1](https://nvlabs.github.io/sionna/api/fec.polar.html#id26),[2](https://nvlabs.github.io/sionna/api/fec.polar.html#id27))

Seyyed Ali Hashemi, Carlo Condo, and Warren J.
Gross, Simplified Successive-Cancellation List Decoding
of Polar Codes. IEEE ISIT, 2016.

[Cammerer_Hybrid_SCL](https://nvlabs.github.io/sionna/api/fec.polar.html#id28)

Sebastian Cammerer, Benedikt Leible, Matthias
Stahl, Jakob Hoydis, and Stephan ten Brink, Combining Belief
Propagation and Successive Cancellation List Decoding of Polar
Codes on a GPU Platform, IEEE ICASSP, 2017.

[Arikan_BP](https://nvlabs.github.io/sionna/api/fec.polar.html#id30)

E. Arikan, A Performance Comparison of Polar Codes and
Reed-Muller Codes, IEEE Commun. Lett., vol. 12, no. 6, pp.
447-449, Jun. 2008.

[Forney_Graphs](https://nvlabs.github.io/sionna/api/fec.polar.html#id31)

G. D. Forney, Codes on graphs: normal realizations,
IEEE Trans. Inform. Theory, vol. 47, no. 2, pp. 520-548, Feb. 2001.

[Ebada_Design](https://nvlabs.github.io/sionna/api/fec.polar.html#id32)

M. Ebada, S. Cammerer, A. Elkelesh and S. ten Brink,
Deep Learning-based Polar Code Design, Annual Allerton
Conference on Communication, Control, and Computing, 2019.

[Goala_LP](https://nvlabs.github.io/sionna/api/fec.polar.html#id34)

N. Goela, S. Korada, M. Gastpar, On LP decoding of Polar
Codes, IEEE ITW 2010.



