# Convolutional Codes

This module supports encoding of convolutional codes and provides layers for Viterbi [[Viterbi]](https://nvlabs.github.io/sionna/api/fec.conv.html#viterbi) and BCJR [[BCJR]](https://nvlabs.github.io/sionna/api/fec.conv.html#bcjr) decoding.

While the [`ViterbiDecoder`](https://nvlabs.github.io/sionna/api/fec.conv.html#sionna.fec.conv.ViterbiDecoder) decoding algorithm produces maximum likelihood *sequence* estimates, the [`BCJRDecoder`](https://nvlabs.github.io/sionna/api/fec.conv.html#sionna.fec.conv.BCJRDecoder) produces the maximum a posterior (MAP) bit-estimates.

The following code snippet shows how to set up a rate-1/2, constraint-length-3 [`ConvEncoder`](https://nvlabs.github.io/sionna/api/fec.conv.html#sionna.fec.conv.ConvEncoder) in two alternate ways and a corresponding [`ViterbiDecoder`](https://nvlabs.github.io/sionna/api/fec.conv.html#sionna.fec.conv.ViterbiDecoder) or [`BCJRDecoder`](https://nvlabs.github.io/sionna/api/fec.conv.html#sionna.fec.conv.BCJRDecoder). You can find further examples in the [Channel Coding Tutorial Notebook](../examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html).

Setting-up:
```python
encoder = ConvEncoder(rate=1/2, # rate of the desired code
                      constraint_length=3) # constraint length of the code
# or
encoder = ConvEncoder(gen_poly=['101', '111']) # or polynomial can be used as input directly
# --- Viterbi decoding ---
decoder = ViterbiDecoder(gen_poly=encoder.gen_poly) # polynomial used in encoder
# or just reference to the encoder
decoder = ViterbiDecoder(encoder=encoder) # the code parameters are infered from the encoder
# --- or BCJR decoding ---
decoder = BCJRDecoder(gen_poly=encoder.gen_poly, algorithm="map") # polynomial used in encoder
# or just reference to the encoder
decoder = BCJRDecoder(encoder=encoder, algorithm="map") # the code parameters are infered from the encoder
```


Running the encoder / decoder:
```python
# --- encoder ---
# u contains the information bits to be encoded and has shape [...,k].
# c contains the convolutional encoded codewords and has shape [...,n].
c = encoder(u)
# --- decoder ---
# y contains the de-mapped received codeword from channel and has shape [...,n].
# u_hat contains the estimated information bits and has shape [...,k].
u_hat = decoder(y)
```
## Convolutional Encoding

`class` `sionna.fec.conv.``ConvEncoder`(*`gen_poly``=``None`*, *`rate``=``1` `/` `2`*, *`constraint_length``=``3`*, *`rsc``=``False`*, *`terminate``=``False`*, *`output_dtype``=``tf.float32`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/fec/conv/encoding.html#ConvEncoder)

Encodes an information binary tensor to a convolutional codeword. Currently,
only generator polynomials for codes of rate=1/n for n=2,3,4, are allowed.

The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters

- **gen_poly** (*tuple*)  Sequence of strings with each string being a 0,1 sequence. If
<cite>None</cite>, `rate` and `constraint_length` must be provided.
- **rate** (*float*)  Valid values are 1/3 and 0.5. Only required if `gen_poly` is
<cite>None</cite>.
- **constraint_length** (*int*)  Valid values are between 3 and 8 inclusive. Only required if
`gen_poly` is <cite>None</cite>.
- **rsc** (*boolean*)  Boolean flag indicating whether the Trellis generated is recursive
systematic or not. If <cite>True</cite>, the encoder is recursive-systematic.
In this case first polynomial in `gen_poly` is used as the
feedback polynomial. Defaults to <cite>False</cite>.
- **terminate** (*boolean*)  Encoder is terminated to all zero state if <cite>True</cite>.
If terminated, the <cite>true</cite> rate of the code is slightly lower than
`rate`.
- **output_dtype** (*tf.DType*)  Defaults to <cite>tf.float32</cite>. Defines the output datatype of the layer.


Input

**inputs** (*[,k], tf.float32*)  2+D tensor containing the information bits where <cite>k</cite> is the
information length

Output

*[,k/rate], tf.float32*  2+D tensor containing the encoded codeword for the given input
information tensor where <cite>rate</cite> is
$\frac{1}{\textrm{len}\left(\textrm{gen_poly}\right)}$
(if `gen_poly` is provided).


**Note**

The generator polynomials from [[Moon]](https://nvlabs.github.io/sionna/api/fec.conv.html#moon) are available for various
rate and constraint lengths. To select them, use the `rate` and
`constraint_length` arguments.

In addition, polynomials for any non-recursive convolutional encoder
can be given as input via `gen_poly` argument. Currently, only
polynomials with rate=1/n are supported. When the `gen_poly` argument
is given, the `rate` and `constraint_length` arguments are ignored.

Various notations are used in the literature to represent the generator
polynomials for convolutional codes. In [[Moon]](https://nvlabs.github.io/sionna/api/fec.conv.html#moon), the octal digits
format is primarily used. In the octal format, the generator polynomial
<cite>10011</cite> corresponds to 46. Another widely used format
is decimal notation with MSB. In this notation, polynomial <cite>10011</cite>
corresponds to 19. For simplicity, the
[`ConvEncoder`](https://nvlabs.github.io/sionna/api/fec.conv.html#sionna.fec.conv.ConvEncoder) only accepts the bit
format i.e. <cite>10011</cite> as `gen_poly` argument.

Also note that `constraint_length` and `memory` are two different
terms often used to denote the strength of a convolutional code. In this
sub-package, we use `constraint_length`. For example, the
polynomial <cite>10011</cite> has a `constraint_length` of 5, however its
`memory` is only 4.

When `terminate` is <cite>True</cite>, the true rate of the convolutional
code is slightly lower than `rate`. It equals
$\frac{r*k}{k+\mu}$ where <cite>r</cite> denotes `rate` and
$\mu$ is `constraint_length` - 1. For example when
`terminate` is <cite>True</cite>, `k=100`,
$\mu=4$ and `rate` =0.5, true rate equals
$\frac{0.5*100}{104}=0.481$.

`property` `coderate`

Rate of the code used in the encoder


`property` `gen_poly`

Generator polynomial used by the encoder


`property` `k`

Number of information bits per codeword


`property` `n`

Number of codeword bits


`property` `terminate`

Indicates if the convolutional encoder is terminated


`property` `trellis`

Trellis object used during encoding


## Viterbi Decoding

`class` `sionna.fec.conv.``ViterbiDecoder`(*`encoder``=``None`*, *`gen_poly``=``None`*, *`rate``=``1` `/` `2`*, *`constraint_length``=``3`*, *`rsc``=``False`*, *`terminate``=``False`*, *`method``=``'soft_llr'`*, *`output_dtype``=``tf.float32`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/fec/conv/decoding.html#ViterbiDecoder)

Implements the Viterbi decoding algorithm [[Viterbi]](https://nvlabs.github.io/sionna/api/fec.conv.html#viterbi) that returns an
estimate of the information bits for a noisy convolutional codeword.
Takes as input either LLR values (<cite>method</cite> = <cite>soft_llr</cite>) or hard bit values
(<cite>method</cite> = <cite>hard</cite>) and returns a hard decided estimation of the information
bits.

The class inherits from the Keras layer class and can be used as layer in
a Keras model.
Parameters

- **encoder** ([`ConvEncoder`](https://nvlabs.github.io/sionna/api/fec.conv.html#sionna.fec.conv.ConvEncoder))  If `encoder` is provided as input, the following input parameters
are not required and will be ignored: `gen_poly`, `rate`,
`constraint_length`, `rsc`, `terminate`. They will be inferred
from the `encoder`  object itself. If `encoder` is <cite>None</cite>, the
above parameters must be provided explicitly.
- **gen_poly** (*tuple*)  tuple of strings with each string being a 0, 1 sequence. If <cite>None</cite>,
`rate` and `constraint_length` must be provided.
- **rate** (*float*)  Valid values are 1/3 and 0.5. Only required if `gen_poly` is <cite>None</cite>.
- **constraint_length** (*int*)  Valid values are between 3 and 8 inclusive. Only required if
`gen_poly` is <cite>None</cite>.
- **rsc** (*boolean*)  Boolean flag indicating whether the encoder is recursive-systematic for
given generator polynomials.
<cite>True</cite> indicates encoder is recursive-systematic.
<cite>False</cite> indicates encoder is feed-forward non-systematic.
- **terminate** (*boolean*)  Boolean flag indicating whether the codeword is terminated.
<cite>True</cite> indicates codeword is terminated to all-zero state.
<cite>False</cite> indicates codeword is not terminated.
- **method** (*str*)  Valid values are <cite>soft_llr</cite> or <cite>hard</cite>. In computing path
metrics,
<cite>soft_llr</cite> expects channel LLRs as input
<cite>hard</cite> assumes a <cite>binary symmetric channel</cite> (BSC) with 0/1 values are
inputs. In case of <cite>hard</cite>, <cite>inputs</cite> will be quantized to 0/1 values.
- **output_dtype** (*tf.DType*)  Defaults to tf.float32. Defines the output datatype of the layer.


Input

**inputs** (*[,n], tf.float32*)  2+D tensor containing the (noisy) channel output symbols where <cite>n</cite>
denotes the codeword length

Output

*[,rate*n], tf.float32*  2+D tensor containing the estimates of the information bit tensor


**Note**

A full implementation of the decoder rather than a windowed approach
is used. For a given codeword of duration <cite>T</cite>, the path metric is
computed from time <cite>0</cite> to <cite>T</cite> and the path with optimal metric at time
<cite>T</cite> is selected. The optimal path is then traced back from <cite>T</cite> to <cite>0</cite>
to output the estimate of the information bit vector used to encode.
For larger codewords, note that the current method is sub-optimal
in terms of memory utilization and latency.

`property` `coderate`

Rate of the code used in the encoder


`property` `gen_poly`

Generator polynomial used by the encoder


`property` `k`

Number of information bits per codeword


`property` `n`

Number of codeword bits


`property` `terminate`

Indicates if the encoder is terminated during codeword generation


`property` `trellis`

Trellis object used during encoding


## BCJR Decoding

`class` `sionna.fec.conv.``BCJRDecoder`(*`encoder``=``None`*, *`gen_poly``=``None`*, *`rate``=``1` `/` `2`*, *`constraint_length``=``3`*, *`rsc``=``False`*, *`terminate``=``False`*, *`hard_out``=``True`*, *`algorithm``=``'map'`*, *`output_dtype``=``tf.float32`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/fec/conv/decoding.html#BCJRDecoder)

Implements the BCJR decoding algorithm [[BCJR]](https://nvlabs.github.io/sionna/api/fec.conv.html#bcjr) that returns an
estimate of the information bits for a noisy convolutional codeword.
Takes as input either channel LLRs or a tuple
(channel LLRs, apriori LLRs). Returns an estimate of the information
bits, either output LLRs ( `hard_out` = <cite>False</cite>) or hard decoded
bits ( `hard_out` = <cite>True</cite>), respectively.

The class inherits from the Keras layer class and can be used as layer in
a Keras model.
Parameters

- **encoder** ([`ConvEncoder`](https://nvlabs.github.io/sionna/api/fec.conv.html#sionna.fec.conv.ConvEncoder))  If `encoder` is provided as input, the following input parameters
are not required and will be ignored: `gen_poly`, `rate`,
`constraint_length`, `rsc`, `terminate`. They will be inferred
from the `encoder`  object itself. If `encoder` is <cite>None</cite>, the
above parameters must be provided explicitly.
- **gen_poly** (*tuple*)  tuple of strings with each string being a 0, 1 sequence. If <cite>None</cite>,
`rate` and `constraint_length` must be provided.
- **rate** (*float*)  Valid values are 1/3 and 1/2. Only required if `gen_poly` is <cite>None</cite>.
- **constraint_length** (*int*)  Valid values are between 3 and 8 inclusive. Only required if
`gen_poly` is <cite>None</cite>.
- **rsc** (*boolean*)  Boolean flag indicating whether the encoder is recursive-systematic for
given generator polynomials. <cite>True</cite> indicates encoder is
recursive-systematic. <cite>False</cite> indicates encoder is feed-forward non-systematic.
- **terminate** (*boolean*)  Boolean flag indicating whether the codeword is terminated.
<cite>True</cite> indicates codeword is terminated to all-zero state.
<cite>False</cite> indicates codeword is not terminated.
- **hard_out** (*boolean*)  Boolean flag indicating whether to output hard or soft decisions on
the decoded information vector.
<cite>True</cite> implies a hard-decoded information vector of 0/1s as output.
<cite>False</cite> implies output is decoded LLRs of the information.
- **algorithm** (*str*)  Defaults to <cite>map</cite>. Indicates the implemented BCJR algorithm,
where <cite>map</cite> denotes the exact MAP algorithm, <cite>log</cite> indicates the
exact MAP implementation, but in log-domain, and
<cite>maxlog</cite> indicates the approximated MAP implementation in log-domain,
where $\log(e^{a}+e^{b}) \sim \max(a,b)$.
- **output_dtype** (*tf.DType*)  Defaults to tf.float32. Defines the output datatype of the layer.


Input

- **llr_ch or (llr_ch, llr_a)**  Tensor or Tuple:
- **llr_ch** (*[,n], tf.float32*)  2+D tensor containing the (noisy) channel
LLRs, where <cite>n</cite> denotes the codeword length
- **llr_a** (*[,k], tf.float32*)  2+D tensor containing the a priori information of each information bit.
Implicitly assumed to be 0 if only `llr_ch` is provided.


Output

*tf.float32*  2+D tensor of shape <cite>[,coderate*n]</cite> containing the estimates of the
information bit tensor


`property` `coderate`

Rate of the code used in the encoder


`property` `gen_poly`

Generator polynomial used by the encoder


`property` `k`

Number of information bits per codeword


`property` `n`

Number of codeword bits


`property` `terminate`

Indicates if the encoder is terminated during codeword generation


`property` `trellis`

Trellis object used during encoding


## Convolutional Code Utility Functions

### Trellis

`sionna.fec.conv.utils.``Trellis`(*`gen_poly`*, *`rsc``=``True`*)[`[source]`](../_modules/sionna/fec/conv/utils.html#Trellis)

Trellis structure for a given generator polynomial. Defines
state transitions and output symbols (and bits) for each current
state and input.
Parameters

- **gen_poly** (*tuple*)  Sequence of strings with each string being a 0,1 sequence.
If <cite>None</cite>, `rate` and `constraint_length` must be provided. If
<cite>rsc</cite> is True, then first polynomial will act as denominator for
the remaining generator polynomials. For e.g., `rsc` = <cite>True</cite> and
`gen_poly` = (<cite>111</cite>, <cite>101</cite>, <cite>011</cite>) implies generator matrix equals
$G(D)=[\frac{1+D^2}{1+D+D^2}, \frac{D+D^2}{1+D+D^2}]$.
Currently Trellis is only implemented for generator matrices of
size $\frac{1}{n}$.
- **rsc** (*boolean*)  Boolean flag indicating whether the Trellis is recursive systematic
or not. If <cite>True</cite>, the encoder is recursive systematic in which
case first polynomial in `gen_poly` is used as the feedback
polynomial. Default is <cite>True</cite>.


### polynomial_selector

`sionna.fec.conv.utils.``polynomial_selector`(*`rate`*, *`constraint_length`*)[`[source]`](../_modules/sionna/fec/conv/utils.html#polynomial_selector)

Returns generator polynomials for given code parameters. The
polynomials are chosen from [[Moon]](https://nvlabs.github.io/sionna/api/fec.conv.html#moon) which are tabulated by searching
for polynomials with best free distances for a given rate and
constraint length.
Input

- **rate** (*float*)  Desired rate of the code.
Currently, only r=1/3 and r=1/2 are supported.
- **constraint_length** (*int*)  Desired constraint length of the encoder


Output

*tuple*  Tuple of strings with each string being a 0,1 sequence where
each polynomial is represented in binary form.


References:
Viterbi([1](https://nvlabs.github.io/sionna/api/fec.conv.html#id1),[2](https://nvlabs.github.io/sionna/api/fec.conv.html#id5))

A. Viterbi, Error bounds for convolutional codes and an
asymptotically optimum decoding algorithm, IEEE Trans. Inf. Theory, 1967.

BCJR([1](https://nvlabs.github.io/sionna/api/fec.conv.html#id2),[2](https://nvlabs.github.io/sionna/api/fec.conv.html#id6))

L. Bahl, J. Cocke, F. Jelinek, und J. Raviv, Optimal Decoding
of Linear Codes for Minimizing Symbol Error Rate, IEEE Trans. Inf.
Theory, March 1974.

Moon([1](https://nvlabs.github.io/sionna/api/fec.conv.html#id3),[2](https://nvlabs.github.io/sionna/api/fec.conv.html#id4),[3](https://nvlabs.github.io/sionna/api/fec.conv.html#id7))

Todd. K. Moon, Error Correction Coding: Mathematical
Methods and Algorithms, John Wiley & Sons, 2020.



