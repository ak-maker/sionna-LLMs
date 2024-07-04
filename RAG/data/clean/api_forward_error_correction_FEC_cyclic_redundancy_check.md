# Cyclic Redundancy Check (CRC)

A cyclic redundancy check adds parity bits to detect transmission errors.
The following code snippets show how to add CRC parity bits to a bit sequence
and how to verify that the check is valid.

First, we need to create instances of [`CRCEncoder`](https://nvlabs.github.io/sionna/api/fec.crc.html#sionna.fec.crc.CRCEncoder) and [`CRCDecoder`](https://nvlabs.github.io/sionna/api/fec.crc.html#sionna.fec.crc.CRCDecoder):
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
## CRCEncoder

`class` `sionna.fec.crc.``CRCEncoder`(*`crc_degree`*, *`output_dtype``=``tf.float32`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/fec/crc.html#CRCEncoder)

Adds cyclic redundancy check to input sequence.

The CRC polynomials from Sec. 5.1 in [[3GPPTS38212_CRC]](https://nvlabs.github.io/sionna/api/fec.crc.html#gppts38212-crc) are available:
<cite>{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}</cite>.

The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters

- **crc_degree** (*str*)  Defining the CRC polynomial to be used. Can be any value from
<cite>{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}</cite>.
- **dtype** (*tf.DType*)  Defaults to <cite>tf.float32</cite>. Defines the output dtype.


Input

**inputs** (*[,k], tf.float32*)  2+D tensor of arbitrary shape where the last dimension is
<cite>[,k]</cite>. Must have at least rank two.

Output

**x_crc** (*[,k+crc_degree], tf.float32*)  2+D tensor containing CRC encoded bits of same shape as
`inputs` except the last dimension changes to
<cite>[,k+crc_degree]</cite>.

Raises

- **AssertionError**  If `crc_degree` is not <cite>str</cite>.
- **ValueError**  If requested CRC polynomial is not available in [[3GPPTS38212_CRC]](https://nvlabs.github.io/sionna/api/fec.crc.html#gppts38212-crc).
- **InvalidArgumentError**  When rank(`inputs`)<2.


**Note**

For performance enhancements, we implement a generator-matrix based
implementation for fixed <cite>k</cite> instead of the more common shift
register-based operations. Thus, the encoder need to trigger an
(internal) rebuild if <cite>k</cite> changes.

`property` `crc_degree`

CRC degree as string.


`property` `crc_length`

Length of CRC. Equals number of CRC parity bits.


`property` `crc_pol`

CRC polynomial in binary representation.


`property` `k`

Number of information bits per codeword.


`property` `n`

Number of codeword bits.


## CRCDecoder

`class` `sionna.fec.crc.``CRCDecoder`(*`crc_encoder`*, *`dtype``=``None`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/fec/crc.html#CRCDecoder)

Allows cyclic redundancy check verification and removes parity-bits.

The CRC polynomials from Sec. 5.1 in [[3GPPTS38212_CRC]](https://nvlabs.github.io/sionna/api/fec.crc.html#gppts38212-crc) are available:
<cite>{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}</cite>.

The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters

- **crc_encoder** ()  An instance of [`CRCEncoder`](https://nvlabs.github.io/sionna/api/fec.crc.html#sionna.fec.crc.CRCEncoder) to which the
CRCDecoder is associated.
- **dtype** (*tf.DType*)  Defaults to <cite>None</cite>. Defines the datatype for internal calculations
and the output dtype. If no explicit dtype is provided the dtype
from the associated interleaver is used.


Input

**inputs** (*[,k+crc_degree], tf.float32*)  2+D Tensor containing the CRC encoded bits (i.e., the last
<cite>crc_degree</cite> bits are parity bits). Must have at least rank two.

Output

- **(x, crc_valid)**  Tuple:
- **x** (*[,k], tf.float32*)  2+D tensor containing the information bit sequence without CRC
parity bits.
- **crc_valid** (*[,1], tf.bool*)  2+D tensor containing the result of the CRC per codeword.


Raises

- **AssertionError**  If `crc_encoder` is not <cite>CRCEncoder</cite>.
- **InvalidArgumentError**  When rank(`x`)<2.


`property` `crc_degree`

CRC degree as string.


`property` `encoder`

CRC Encoder used for internal validation.


References:
3GPPTS38212_CRC([1](https://nvlabs.github.io/sionna/api/fec.crc.html#id1),[2](https://nvlabs.github.io/sionna/api/fec.crc.html#id2),[3](https://nvlabs.github.io/sionna/api/fec.crc.html#id3))

ETSI 3GPP TS 38.212 5G NR Multiplexing and channel
coding, v.16.5.0, 2021-03.



