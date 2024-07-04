
# Turbo Codes<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#turbo-codes" title="Permalink to this headline"></a>
    
This module supports encoding and decoding of Turbo codes <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#berrou" id="id1">[Berrou]</a>, e.g., as
used in the LTE wireless standard. The convolutional component encoders and
decoders are composed of the <a class="reference internal" href="fec.conv.html#sionna.fec.conv.ConvEncoder" title="sionna.fec.conv.encoding.ConvEncoder">`ConvEncoder`</a> and
<a class="reference internal" href="fec.conv.html#sionna.fec.conv.BCJRDecoder" title="sionna.fec.conv.decoding.BCJRDecoder">`BCJRDecoder`</a> layers, respectively.
    
Please note that various notations are used in literature to represent the
generator polynomials for the underlying convolutional codes. For simplicity,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboEncoder" title="sionna.fec.turbo.encoding.TurboEncoder">`TurboEncoder`</a> only accepts the binary
format, i.e., <cite>10011</cite>, for the generator polynomial which corresponds to the
polynomial $1 + D^3 + D^4$.
    
The following code snippet shows how to set-up a rate-1/3, constraint-length-4 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboEncoder" title="sionna.fec.turbo.encoding.TurboEncoder">`TurboEncoder`</a> and the corresponding <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboDecoder" title="sionna.fec.turbo.decoding.TurboDecoder">`TurboDecoder`</a>.
You can find further examples in the <a class="reference external" href="../examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html">Channel Coding Tutorial Notebook</a>.
    
Setting-up:
```python
encoder = TurboEncoder(constraint_length=4, # Desired constraint length of the polynomials
                       rate=1/3,  # Desired rate of Turbo code
                       terminate=True) # Terminate the constituent convolutional encoders to all-zero state
# or
encoder = TurboEncoder(gen_poly=gen_poly, # Generator polynomials to use in the underlying convolutional encoders
                       rate=1/3, # Rate of the desired Turbo code
                       terminate=False) # Do not terminate the constituent convolutional encoders
# the decoder can be initialized with a reference to the encoder
decoder = TurboDecoder(encoder,
                       num_iter=6, # Number of iterations between component BCJR decoders
                       algorithm="map", # can be also "maxlog"
                       hard_out=True) # hard_decide output
```

    
Running the encoder / decoder:
```python
# --- encoder ---
# u contains the information bits to be encoded and has shape [...,k].
# c contains the turbo encoded codewords and has shape [...,n], where n=k/rate when terminate is False.
c = encoder(u)
# --- decoder ---
# llr contains the log-likelihood ratio values from the de-mapper and has shape [...,n].
# u_hat contains the estimated information bits and has shape [...,k].
u_hat = decoder(llr)
```
## Turbo Encoding<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#turbo-encoding" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.turbo.``TurboEncoder`(<em class="sig-param">`gen_poly``=``None`</em>, <em class="sig-param">`constraint_length``=``3`</em>, <em class="sig-param">`rate``=``1` `/` `3`</em>, <em class="sig-param">`terminate``=``False`</em>, <em class="sig-param">`interleaver_type``=``'3GPP'`</em>, <em class="sig-param">`output_dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/fec/turbo/encoding.html#TurboEncoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboEncoder" title="Permalink to this definition"></a>
    
Performs encoding of information bits to a Turbo code codeword <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#berrou" id="id2">[Berrou]</a>.
Implements the standard Turbo code framework <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#berrou" id="id3">[Berrou]</a>: Two identical
rate-1/2 convolutional encoders <a class="reference internal" href="fec.conv.html#sionna.fec.conv.ConvEncoder" title="sionna.fec.conv.encoding.ConvEncoder">`ConvEncoder`</a>
are combined to produce a rate-1/3 Turbo code. Further,
puncturing to attain a rate-1/2 Turbo code is supported.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **gen_poly** (<em>tuple</em>) – Tuple of strings with each string being a 0,1 sequence. If
<cite>None</cite>, `constraint_length` must be provided.
- **constraint_length** (<em>int</em>) – Valid values are between 3 and 6 inclusive. Only required if
`gen_poly` is <cite>None</cite>.
- **rate** (<em>float</em>) – Valid values are 1/3 and 1/2. Note that `rate` here denotes
the <cite>design</cite> rate of the Turbo code. If `terminate` is <cite>True</cite>, a
small rate-loss occurs.
- **terminate** (<em>boolean</em>) – Underlying convolutional encoders are terminated to all zero state
if <cite>True</cite>. If terminated, the true rate of the code is slightly lower
than `rate`.
- **interleaver_type** (<em>str</em>) – Valid values are <cite>“3GPP”</cite> or <cite>“random”</cite>. Determines the choice of
the interleaver to interleave the message bits before input to the
second convolutional encoder. If <cite>“3GPP”</cite>, the Turbo code interleaver
from the 3GPP LTE standard <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#gppts36212-turbo" id="id4">[3GPPTS36212_Turbo]</a> is used. If <cite>“random”</cite>,
a random interleaver is used.
- **output_dtype** (<em>tf.DType</em>) – Defaults to <cite>tf.float32</cite>. Defines the output datatype of the layer.


Input
    
**inputs** (<em>[…,k], tf.float32</em>) – 2+D tensor of information bits where <cite>k</cite> is the information length

Output
    
<cite>[…,k/rate]</cite>, tf.float32 – 2+D tensor where <cite>rate</cite> is provided as input
parameter. The output is the encoded codeword for the input
information tensor. When `terminate` is <cite>True</cite>, the effective rate
of the Turbo code is slightly less than `rate`.



**Note**
    
Various notations are used in literature to represent the generator
polynomials for convolutional codes. For simplicity
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboEncoder" title="sionna.fec.turbo.encoding.TurboEncoder">`TurboEncoder`</a> only
accepts the binary format, i.e., <cite>10011</cite>, for the `gen_poly` argument
which corresponds to the polynomial $1 + D^3 + D^4$.
    
Note that Turbo codes require the underlying convolutional encoders
to be recursive systematic encoders. Only then the channel output
from the systematic part of the first encoder can be used to decode
the second encoder.
    
Also note that `constraint_length` and `memory` are two different
terms often used to denote the strength of the convolutional code. In
this sub-package we use `constraint_length`. For example, the polynomial
<cite>10011</cite> has a `constraint_length` of 5, however its `memory` is
only 4.
    
When `terminate` is <cite>True</cite>, the true rate of the Turbo code is
slightly lower than `rate`. It can be computed as
$\frac{k}{\frac{k}{r}+\frac{4\mu}{3r}}$ where <cite>r</cite> denotes
`rate` and $\mu$ is the `constraint_length` - 1. For example, in
3GPP, `constraint_length` = 4, `terminate` = <cite>True</cite>, for
`rate` = 1/3, true rate is equal to  $\frac{k}{3k+12}$ .

<em class="property">`property` </em>`coderate`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboEncoder.coderate" title="Permalink to this definition"></a>
    
Rate of the code used in the encoder


<em class="property">`property` </em>`constraint_length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboEncoder.constraint_length" title="Permalink to this definition"></a>
    
Constraint length of the encoder


<em class="property">`property` </em>`gen_poly`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboEncoder.gen_poly" title="Permalink to this definition"></a>
    
Generator polynomial used by the encoder


<em class="property">`property` </em>`k`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboEncoder.k" title="Permalink to this definition"></a>
    
Number of information bits per codeword


<em class="property">`property` </em>`n`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboEncoder.n" title="Permalink to this definition"></a>
    
Number of codeword bits


<em class="property">`property` </em>`punct_pattern`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboEncoder.punct_pattern" title="Permalink to this definition"></a>
    
Puncturing pattern for the Turbo codeword


<em class="property">`property` </em>`terminate`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboEncoder.terminate" title="Permalink to this definition"></a>
    
Indicates if the convolutional encoders are terminated


<em class="property">`property` </em>`trellis`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboEncoder.trellis" title="Permalink to this definition"></a>
    
Trellis object used during encoding


## Turbo Decoding<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#turbo-decoding" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.turbo.``TurboDecoder`(<em class="sig-param">`encoder``=``None`</em>, <em class="sig-param">`gen_poly``=``None`</em>, <em class="sig-param">`rate``=``1` `/` `3`</em>, <em class="sig-param">`constraint_length``=``None`</em>, <em class="sig-param">`interleaver``=``'3GPP'`</em>, <em class="sig-param">`terminate``=``False`</em>, <em class="sig-param">`num_iter``=``6`</em>, <em class="sig-param">`hard_out``=``True`</em>, <em class="sig-param">`algorithm``=``'map'`</em>, <em class="sig-param">`output_dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/fec/turbo/decoding.html#TurboDecoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboDecoder" title="Permalink to this definition"></a>
    
Turbo code decoder based on BCJR component decoders <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#berrou" id="id5">[Berrou]</a>.
Takes as input LLRs and returns LLRs or hard decided bits, i.e., an
estimate of the information tensor.
    
This decoder is based on the <a class="reference internal" href="fec.conv.html#sionna.fec.conv.BCJRDecoder" title="sionna.fec.conv.decoding.BCJRDecoder">`BCJRDecoder`</a>
and, thus, internally instantiates two
<a class="reference internal" href="fec.conv.html#sionna.fec.conv.BCJRDecoder" title="sionna.fec.conv.decoding.BCJRDecoder">`BCJRDecoder`</a> layers.
    
The class inherits from the Keras layer class and can be used as layer in
a Keras model.
Parameters
 
- **encoder** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboEncoder" title="sionna.fec.turbo.encoding.TurboEncoder">`TurboEncoder`</a>) – If `encoder` is provided as input, the following input parameters
are not required and will be ignored: <cite>gen_poly</cite>, <cite>rate</cite>,
<cite>constraint_length</cite>, <cite>terminate</cite>, <cite>interleaver</cite>. They will be inferred
from the `encoder` object itself.
If `encoder` is <cite>None</cite>, the above parameters must be provided
explicitly.
- **gen_poly** (<em>tuple</em>) – Tuple of strings with each string being a 0, 1 sequence. If <cite>None</cite>,
`rate` and `constraint_length` must be provided.
- **rate** (<em>float</em>) – Rate of the Turbo code. Valid values are 1/3 and 1/2. Note that
`gen_poly`, if provided, is used to encode the underlying
convolutional code, which traditionally has rate 1/2.
- **constraint_length** (<em>int</em>) – Valid values are between 3 and 6 inclusive. Only required if
`encoder` and `gen_poly` are <cite>None</cite>.
- **interleaver** (<em>str</em>) – <cite>“3GPP”</cite> or <cite>“Random”</cite>. If <cite>“3GPP”</cite>, the internal interleaver for Turbo
codes as specified in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#gppts36212-turbo" id="id6">[3GPPTS36212_Turbo]</a> will be used. Only required
if `encoder` is <cite>None</cite>.
- **terminate** (<em>bool</em>) – If <cite>True</cite>, the two underlying convolutional encoders are assumed
to have terminated to all zero state.
- **num_iter** (<em>int</em>) – Number of iterations for the Turbo decoding to run. Each iteration of
Turbo decoding entails one BCJR decoder for each of the underlying
convolutional code components.
- **hard_out** (<em>boolean</em>) – Defaults to <cite>True</cite> and indicates whether to output hard or soft
decisions on the decoded information vector. <cite>True</cite> implies a hard-
decoded information vector of 0/1’s is output. <cite>False</cite> implies
decoded LLRs of the information is output.
- **algorithm** (<em>str</em>) – Defaults to <cite>map</cite>. Indicates the implemented BCJR algorithm,
where <cite>map</cite> denotes the exact MAP algorithm, <cite>log</cite> indicates the
exact MAP implementation, but in log-domain, and
<cite>maxlog</cite> indicates the approximated MAP implementation in log-domain,
where $\log(e^{a}+e^{b}) \sim \max(a,b)$.
- **output_dtype** (<em>tf.DType</em>) – Defaults to <cite>tf.float32</cite>. Defines the output datatype of the layer.


Input
    
**inputs** (<em>tf.float32</em>) – 2+D tensor of shape <cite>[…,n]</cite> containing the (noisy) channel
output symbols where <cite>n</cite> is the codeword length

Output
    
<em>tf.float32</em> – 2+D tensor of shape <cite>[…,coderate*n]</cite> containing the estimates of the
information bit tensor



**Note**
    
For decoding, input <cite>logits</cite> defined as
$\operatorname{log} \frac{p(x=1)}{p(x=0)}$ are assumed for
compatibility with the rest of Sionna. Internally,
log-likelihood ratios (LLRs) with definition
$\operatorname{log} \frac{p(x=0)}{p(x=1)}$ are used.

<em class="property">`property` </em>`coderate`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboDecoder.coderate" title="Permalink to this definition"></a>
    
Rate of the code used in the encoder


<em class="property">`property` </em>`constraint_length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboDecoder.constraint_length" title="Permalink to this definition"></a>
    
Constraint length of the encoder


`depuncture`(<em class="sig-param">`y`</em>)<a class="reference internal" href="../_modules/sionna/fec/turbo/decoding.html#TurboDecoder.depuncture">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboDecoder.depuncture" title="Permalink to this definition"></a>
    
Given a tensor <cite>y</cite> of shape <cite>[batch, n]</cite>, depuncture() scatters <cite>y</cite>
elements into shape <cite>[batch, 3*rate*n]</cite> where the
extra elements are filled with 0.
    
For e.g., if input is <cite>y</cite>, rate is 1/2 and
<cite>punct_pattern</cite> is [1, 1, 0, 1, 0, 1], then the
output is [y[0], y[1], 0., y[2], 0., y[3], y[4], y[5], 0., … ,].


<em class="property">`property` </em>`gen_poly`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboDecoder.gen_poly" title="Permalink to this definition"></a>
    
Generator polynomial used by the encoder


<em class="property">`property` </em>`k`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboDecoder.k" title="Permalink to this definition"></a>
    
Number of information bits per codeword


<em class="property">`property` </em>`n`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboDecoder.n" title="Permalink to this definition"></a>
    
Number of codeword bits


<em class="property">`property` </em>`trellis`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboDecoder.trellis" title="Permalink to this definition"></a>
    
Trellis object used during encoding


## Turbo Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#turbo-utility-functions" title="Permalink to this headline"></a>

### TurboTermination<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#turbotermination" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.turbo.``TurboTermination`(<em class="sig-param">`constraint_length`</em>, <em class="sig-param">`conv_n``=``2`</em>, <em class="sig-param">`num_conv_encs``=``2`</em>, <em class="sig-param">`num_bit_streams``=``3`</em>)<a class="reference internal" href="../_modules/sionna/fec/turbo/utils.html#TurboTermination">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboTermination" title="Permalink to this definition"></a>
    
Termination object, handles the transformation of termination bits from
the convolutional encoders to a Turbo codeword. Similarly, it handles the
transformation of channel symbols corresponding to the termination of a
Turbo codeword to the underlying convolutional codewords.
Parameters
 
- **constraint_length** (<em>int</em>) – Constraint length of the convolutional encoder used in the Turbo code.
Note that the memory of the encoder is `constraint_length` - 1.
- **conv_n** (<em>int</em>) – Number of output bits for one state transition in the underlying
convolutional encoder
- **num_conv_encs** (<em>int</em>) – Number of parallel convolutional encoders used in the Turbo code
- **num_bit_streams** (<em>int</em>) – Number of output bit streams from Turbo code




`get_num_term_syms`()<a class="reference internal" href="../_modules/sionna/fec/turbo/utils.html#TurboTermination.get_num_term_syms">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboTermination.get_num_term_syms" title="Permalink to this definition"></a>
    
Computes the number of termination symbols for the Turbo
code based on the underlying convolutional code parameters,
primarily the memory $\mu$.
Note that it is assumed that one Turbo symbol implies
`num_bitstreams` bits.
Input
    
**None**

Output
    
**turbo_term_syms** (<em>int</em>) – Total number of termination symbols for the Turbo Code. One
symbol equals `num_bitstreams` bits.




`term_bits_turbo2conv`(<em class="sig-param">`term_bits`</em>)<a class="reference internal" href="../_modules/sionna/fec/turbo/utils.html#TurboTermination.term_bits_turbo2conv">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboTermination.term_bits_turbo2conv" title="Permalink to this definition"></a>
    
This method splits the termination symbols from a Turbo codeword
to the termination symbols corresponding to the two convolutional
encoders, respectively.
    
Let’s assume $\mu=4$ and the underlying convolutional encoders
are systematic and rate-1/2, for demonstration purposes.
    
Let `term_bits` tensor, corresponding to the termination symbols of
the Turbo codeword be as following:
    
$y = [x_1(K), z_1(K), x_1(K+1), z_1(K+1), x_1(K+2), z_1(K+2)$,
$x_1(K+3), z_1(K+3), x_2(K), z_2(K), x_2(K+1), z_2(K+1),$
$x_2(K+2), z_2(K+2), x_2(K+3), z_2(K+3), 0, 0]$
    
The two termination tensors corresponding to the convolutional encoders
are:
$y[0,..., 2\mu]$, $y[2\mu,..., 4\mu]$. The output from this method is a tuple of two tensors, each of
size $2\mu$ and shape $[\mu,2]$.
    
$[[x_1(K), z_1(K)]$,
    
$[x_1(K+1), z_1(K+1)]$,
    
$[x_1(K+2, z_1(K+2)]$,
    
$[x_1(K+3), z_1(K+3)]]$
    
and
    
$[[x_2(K), z_2(K)],$
    
$[x_2(K+1), z_2(K+1)]$,
    
$[x_2(K+2), z_2(K+2)]$,
    
$[x_2(K+3), z_2(K+3)]]$
Input
    
**term_bits** (<em>tf.float32</em>) – Channel output of the Turbo codeword, corresponding to the
termination part

Output
    
<em>tf.float32</em> – Two tensors of channel outputs, corresponding to encoders 1 and 2,
respectively




`termbits_conv2turbo`(<em class="sig-param">`term_bits1`</em>, <em class="sig-param">`term_bits2`</em>)<a class="reference internal" href="../_modules/sionna/fec/turbo/utils.html#TurboTermination.termbits_conv2turbo">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboTermination.termbits_conv2turbo" title="Permalink to this definition"></a>
    
This method merges `term_bits1` and `term_bits2`, termination
bit streams from the two convolutional encoders, to a bit stream
corresponding to the Turbo codeword.
    
Let `term_bits1` and `term_bits2` be:
    
$[x_1(K), z_1(K), x_1(K+1), z_1(K+1),..., x_1(K+\mu-1),z_1(K+\mu-1)]$
    
$[x_2(K), z_2(K), x_2(K+1), z_2(K+1),..., x_2(K+\mu-1), z_2(K+\mu-1)]$
    
where $x_i, z_i$ are the systematic and parity bit streams
respectively for a rate-1/2 convolutional encoder i, for i = 1, 2.
    
In the example output below, we assume $\mu=4$ to demonstrate zero
padding at the end. Zero padding is done such that the total length is
divisible by `num_bitstreams` (defaults to  3) which is the number of
Turbo bit streams.
    
Assume `num_bitstreams` = 3. Then number of termination symbols for
the TurboEncoder is $\lceil \frac{2*conv\_n*\mu}{3} \rceil$:
    
$[x_1(K), z_1(K), x_1(K+1)]$
    
$[z_1(K+1), x_1(K+2, z_1(K+2)]$
    
$[x_1(K+3), z_1(K+3), x_2(K)]$
    
$[z_2(K), x_2(K+1), z_2(K+1)]$
    
$[x_2(K+2), z_2(K+2), x_2(K+3)]$
    
$[z_2(K+3), 0, 0]$
    
Therefore, the output from this method is a single dimension vector
where all Turbo symbols are concatenated together.
    
$[x_1(K), z_1(K), x_1(K+1), z_1(K+1), x_1(K+2, z_1(K+2), x_1(K+3),$
    
$z_1(K+3), x_2(K),z_2(K), x_2(K+1), z_2(K+1), x_2(K+2), z_2(K+2),$
    
$x_2(K+3), z_2(K+3), 0, 0]$
Input
 
- **term_bits1** (<em>tf.int32</em>) – 2+D Tensor containing termination bits from convolutional encoder 1
- **term_bits2** (<em>tf.int32</em>) – 2+D Tensor containing termination bits from convolutional encoder 2


Output
    
<em>tf.int32</em> – 1+D tensor of termination bits. The output is obtained by
concatenating the inputs and then adding right zero-padding if
needed.




### polynomial_selector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#polynomial-selector" title="Permalink to this headline"></a>

`sionna.fec.turbo.utils.``polynomial_selector`(<em class="sig-param">`constraint_length`</em>)<a class="reference internal" href="../_modules/sionna/fec/turbo/utils.html#polynomial_selector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.utils.polynomial_selector" title="Permalink to this definition"></a>
    
Returns the generator polynomials for rate-1/2 convolutional codes
for a given `constraint_length`.
Input
    
**constraint_length** (<em>int</em>) – An integer defining the desired constraint length of the encoder.
The memory of the encoder is `constraint_length` - 1.

Output
    
**gen_poly** (<em>tuple</em>) – Tuple of strings with each string being a 0,1 sequence where
each polynomial is represented in binary form.



**Note**
    
Please note that the polynomials are optimized for rsc codes and are
not necessarily the same as used in the polynomial selector
<a class="reference internal" href="fec.conv.html#sionna.fec.conv.utils.polynomial_selector" title="sionna.fec.conv.utils.polynomial_selector">`polynomial_selector`</a> of the
convolutional codes.

### puncture_pattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#puncture-pattern" title="Permalink to this headline"></a>

`sionna.fec.turbo.utils.``puncture_pattern`(<em class="sig-param">`turbo_coderate`</em>, <em class="sig-param">`conv_coderate`</em>)<a class="reference internal" href="../_modules/sionna/fec/turbo/utils.html#puncture_pattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.utils.puncture_pattern" title="Permalink to this definition"></a>
    
This method returns puncturing pattern such that the
Turbo code has rate `turbo_coderate` given the underlying
convolutional encoder is of rate `conv_coderate`.
Input
 
- **turbo_coderate** (<em>float</em>) – Desired coderate of the Turbo code
- **conv_coderate** (<em>float</em>) – Coderate of the underlying convolutional encoder


Output
    
<em>tf.bool</em> – 2D tensor indicating the positions to be punctured.




References:
Berrou(<a href="https://nvlabs.github.io/sionna/api/fec.turbo.html#id1">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.turbo.html#id2">2</a>,<a href="https://nvlabs.github.io/sionna/api/fec.turbo.html#id3">3</a>,<a href="https://nvlabs.github.io/sionna/api/fec.turbo.html#id5">4</a>)
<ol class="upperalpha simple" start="3">
- Berrou, A. Glavieux, P. Thitimajshima, “Near Shannon limit error-correcting coding and decoding: Turbo-codes”, IEEE ICC, 1993.
</ol>

3GPPTS36212_Turbo(<a href="https://nvlabs.github.io/sionna/api/fec.turbo.html#id4">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.turbo.html#id6">2</a>)
    
ETSI 3GPP TS 36.212 “Evolved Universal Terrestrial
Radio Access (EUTRA); Multiplexing and channel coding”, v.15.3.0, 2018-09.



