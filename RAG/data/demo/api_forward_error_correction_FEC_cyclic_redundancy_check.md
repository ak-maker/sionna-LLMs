
# Cyclic Redundancy Check (CRC)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.crc.html#cyclic-redundancy-check-crc" title="Permalink to this headline"></a>
    
A cyclic redundancy check adds parity bits to detect transmission errors.
The following code snippets show how to add CRC parity bits to a bit sequence
and how to verify that the check is valid.
    
First, we need to create instances of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.crc.html#sionna.fec.crc.CRCEncoder" title="sionna.fec.crc.CRCEncoder">`CRCEncoder`</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.crc.html#sionna.fec.crc.CRCDecoder" title="sionna.fec.crc.CRCDecoder">`CRCDecoder`</a>:
```python
encoder = CRCEncoder(crc_degree="CRC24A") # the crc_degree denotes the number of added parity bits and is taken from the 3GPP 5G NR standard.
decoder = CRCDecoder(crc_encoder=encoder) # the decoder must be associated to a specific encoder
```

    
We can now run the CRC encoder and test if the CRC holds:
```python
# u contains the information bits to be encoded and has shape [...,k].
# c contains u and the CRC parity bits. It has shape [...,k+k_crc].
c = encoder(u)
# u_hat contains the information bits without parity bits and has shape [...,k].
# crc_valid contains a boolean per codeword that indicates if the CRC validation was successful.
# It has shape [...,1].
u_hat, crc_valid = decoder(c)
```
## CRCEncoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.crc.html#crcencoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.crc.``CRCEncoder`(<em class="sig-param">`crc_degree`</em>, <em class="sig-param">`output_dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/fec/crc.html#CRCEncoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.crc.html#sionna.fec.crc.CRCEncoder" title="Permalink to this definition"></a>
    
Adds cyclic redundancy check to input sequence.
    
The CRC polynomials from Sec. 5.1 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.crc.html#gppts38212-crc" id="id1">[3GPPTS38212_CRC]</a> are available:
<cite>{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}</cite>.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **crc_degree** (<em>str</em>) – Defining the CRC polynomial to be used. Can be any value from
<cite>{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}</cite>.
- **dtype** (<em>tf.DType</em>) – Defaults to <cite>tf.float32</cite>. Defines the output dtype.


Input
    
**inputs** (<em>[…,k], tf.float32</em>) – 2+D tensor of arbitrary shape where the last dimension is
<cite>[…,k]</cite>. Must have at least rank two.

Output
    
**x_crc** (<em>[…,k+crc_degree], tf.float32</em>) – 2+D tensor containing CRC encoded bits of same shape as
`inputs` except the last dimension changes to
<cite>[…,k+crc_degree]</cite>.

Raises
 
- **AssertionError** – If `crc_degree` is not <cite>str</cite>.
- **ValueError** – If requested CRC polynomial is not available in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.crc.html#gppts38212-crc" id="id2">[3GPPTS38212_CRC]</a>.
- **InvalidArgumentError** – When rank(`inputs`)<2.




**Note**
    
For performance enhancements, we implement a generator-matrix based
implementation for fixed <cite>k</cite> instead of the more common shift
register-based operations. Thus, the encoder need to trigger an
(internal) rebuild if <cite>k</cite> changes.

<em class="property">`property` </em>`crc_degree`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.crc.html#sionna.fec.crc.CRCEncoder.crc_degree" title="Permalink to this definition"></a>
    
CRC degree as string.


<em class="property">`property` </em>`crc_length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.crc.html#sionna.fec.crc.CRCEncoder.crc_length" title="Permalink to this definition"></a>
    
Length of CRC. Equals number of CRC parity bits.


<em class="property">`property` </em>`crc_pol`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.crc.html#sionna.fec.crc.CRCEncoder.crc_pol" title="Permalink to this definition"></a>
    
CRC polynomial in binary representation.


<em class="property">`property` </em>`k`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.crc.html#sionna.fec.crc.CRCEncoder.k" title="Permalink to this definition"></a>
    
Number of information bits per codeword.


<em class="property">`property` </em>`n`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.crc.html#sionna.fec.crc.CRCEncoder.n" title="Permalink to this definition"></a>
    
Number of codeword bits.


## CRCDecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.crc.html#crcdecoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.crc.``CRCDecoder`(<em class="sig-param">`crc_encoder`</em>, <em class="sig-param">`dtype``=``None`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/fec/crc.html#CRCDecoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.crc.html#sionna.fec.crc.CRCDecoder" title="Permalink to this definition"></a>
    
Allows cyclic redundancy check verification and removes parity-bits.
    
The CRC polynomials from Sec. 5.1 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.crc.html#gppts38212-crc" id="id3">[3GPPTS38212_CRC]</a> are available:
<cite>{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}</cite>.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **crc_encoder** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.crc.html#sionna.fec.crc.CRCEncoder" title="sionna.fec.crc.CRCEncoder"><em>CRCEncoder</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.crc.html#sionna.fec.crc.CRCEncoder" title="sionna.fec.crc.CRCEncoder">`CRCEncoder`</a> to which the
CRCDecoder is associated.
- **dtype** (<em>tf.DType</em>) – Defaults to <cite>None</cite>. Defines the datatype for internal calculations
and the output dtype. If no explicit dtype is provided the dtype
from the associated interleaver is used.


Input
    
**inputs** (<em>[…,k+crc_degree], tf.float32</em>) – 2+D Tensor containing the CRC encoded bits (i.e., the last
<cite>crc_degree</cite> bits are parity bits). Must have at least rank two.

Output
 
- **(x, crc_valid)** – Tuple:
- **x** (<em>[…,k], tf.float32</em>) – 2+D tensor containing the information bit sequence without CRC
parity bits.
- **crc_valid** (<em>[…,1], tf.bool</em>) – 2+D tensor containing the result of the CRC per codeword.


Raises
 
- **AssertionError** – If `crc_encoder` is not <cite>CRCEncoder</cite>.
- **InvalidArgumentError** – When rank(`x`)<2.




<em class="property">`property` </em>`crc_degree`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.crc.html#sionna.fec.crc.CRCDecoder.crc_degree" title="Permalink to this definition"></a>
    
CRC degree as string.


<em class="property">`property` </em>`encoder`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.crc.html#sionna.fec.crc.CRCDecoder.encoder" title="Permalink to this definition"></a>
    
CRC Encoder used for internal validation.



References:
3GPPTS38212_CRC(<a href="https://nvlabs.github.io/sionna/api/fec.crc.html#id1">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.crc.html#id2">2</a>,<a href="https://nvlabs.github.io/sionna/api/fec.crc.html#id3">3</a>)
    
ETSI 3GPP TS 38.212 “5G NR Multiplexing and channel
coding”, v.16.5.0, 2021-03.



