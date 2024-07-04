
# Linear Codes<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.linear.html#linear-codes" title="Permalink to this headline"></a>
    
This package provides generic support for binary linear block codes.
    
For encoding, a universal <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.linear.html#sionna.fec.linear.LinearEncoder" title="sionna.fec.linear.LinearEncoder">`LinearEncoder`</a> is available and can be initialized with either a generator or parity-check matrix. The matrix must be binary and of full rank.
    
For decoding, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.linear.html#sionna.fec.linear.OSDecoder" title="sionna.fec.linear.OSDecoder">`OSDecoder`</a> implements the
ordered-statistics decoding (OSD) algorithm <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.linear.html#fossorier" id="id1">[Fossorier]</a> which provides close to
maximum-likelihood (ML) estimates for a sufficiently large order of the decoder.
Please note that OSD is highly complex and not feasible for all code lengths.
    
<em>Remark:</em> As this package provides support for generic encoding and decoding
(including Polar and LDPC codes), it cannot rely on code specific
optimizations. To benefit from an optimized decoder and keep the complexity as low as possible, please use the code specific enc-/decoders whenever available.
    
The encoder and decoder can be set up as follows:
```python
pcm, k, n, coderate = load_parity_check_examples(pcm_id=1) # load example code
# or directly import an external parity-check matrix in alist format
al = load_alist(path=filename)
pcm, k, n, coderate = alist2mat(al)
# encoder can be directly initialized with the parity-check matrix
encoder = LinearEncoder(enc_mat=pcm, is_pcm=True)
# decoder can be initialized with generator or parity-check matrix
decoder = OSDecoder(pcm, t=4, is_pcm=True) # t is the OSD order
# or instantiated from a specific encoder
decoder = OSDecoder(encoder=encoder, t=4) # t is the OSD order
```

    
We can now run the encoder and decoder:
```python
# u contains the information bits to be encoded and has shape [...,k].
# c contains codeword bits and has shape [...,n]
c = encoder(u)
# after transmission LLRs must be calculated with a demapper
# let's assume the resulting llr_ch has shape [...,n]
c_hat = decoder(llr_ch)
```
## Encoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.linear.html#encoder" title="Permalink to this headline"></a>

### LinearEncoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.linear.html#linearencoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.linear.``LinearEncoder`(<em class="sig-param">`enc_mat`</em>, <em class="sig-param">`is_pcm``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/fec/linear/encoding.html#LinearEncoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.linear.html#sionna.fec.linear.LinearEncoder" title="Permalink to this definition"></a>
    
Linear binary encoder for a given generator or parity-check matrix `enc_mat`.
    
If `is_pcm` is True, `enc_mat` is interpreted as parity-check
matrix and internally converted to a corresponding generator matrix.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **enc_mat** (<em>[</em><em>k</em><em>, </em><em>n</em><em>] or </em><em>[</em><em>n-k</em><em>, </em><em>n</em><em>]</em><em>, </em><em>ndarray</em>) – Binary generator matrix of shape <cite>[k, n]</cite>. If `is_pcm` is
True, `enc_mat` is interpreted as parity-check matrix of shape
<cite>[n-k, n]</cite>.
- **dtype** (<em>tf.DType</em>) – Defaults to <cite>tf.float32</cite>. Defines the datatype for the output dtype.


Input
    
**inputs** (<em>[…,k], tf.float32</em>) – 2+D tensor containing information bits.

Output
    
<em>[…,n], tf.float32</em> – 2+D tensor containing codewords with same shape as inputs, except the
last dimension changes to <cite>[…,n]</cite>.

Raises
    
**AssertionError** – If the encoding matrix is not a valid binary 2-D matrix.



**Note**
    
If `is_pcm` is True, this layer uses
<a class="reference internal" href="fec.utils.html#sionna.fec.utils.pcm2gm" title="sionna.fec.utils.pcm2gm">`pcm2gm`</a> to find the generator matrix for
encoding. Please note that this imposes a few constraints on the
provided parity-check matrix such as full rank and it must be binary.
    
Note that this encoder is generic for all binary linear block codes
and, thus, cannot implement any code specific optimizations. As a
result, the encoding complexity is $O(k^2)$. Please consider code
specific encoders such as the
<a class="reference internal" href="fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder" title="sionna.fec.polar.encoding.Polar5GEncoder">`Polar5GEncoder`</a> or
<a class="reference internal" href="fec.ldpc.html#sionna.fec.ldpc.encoding.LDPC5GEncoder" title="sionna.fec.ldpc.encoding.LDPC5GEncoder">`LDPC5GEncoder`</a> for an improved
encoding performance.

<em class="property">`property` </em>`coderate`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.linear.html#sionna.fec.linear.LinearEncoder.coderate" title="Permalink to this definition"></a>
    
Coderate of the code.


<em class="property">`property` </em>`gm`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.linear.html#sionna.fec.linear.LinearEncoder.gm" title="Permalink to this definition"></a>
    
Generator matrix used for encoding.


<em class="property">`property` </em>`k`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.linear.html#sionna.fec.linear.LinearEncoder.k" title="Permalink to this definition"></a>
    
Number of information bits per codeword.


<em class="property">`property` </em>`n`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.linear.html#sionna.fec.linear.LinearEncoder.n" title="Permalink to this definition"></a>
    
Codeword length.


### AllZeroEncoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.linear.html#allzeroencoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.linear.``AllZeroEncoder`(<em class="sig-param">`k`</em>, <em class="sig-param">`n`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/fec/linear/encoding.html#AllZeroEncoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.linear.html#sionna.fec.linear.AllZeroEncoder" title="Permalink to this definition"></a>
    
Dummy encoder that always outputs the all-zero codeword of length `n`.
    
Note that this encoder is a dummy encoder and does NOT perform real
encoding!
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **k** (<em>int</em>) – Defining the number of information bit per codeword.
- **n** (<em>int</em>) – Defining the desired codeword length.
- **dtype** (<em>tf.DType</em>) – Defaults to <cite>tf.float32</cite>. Defines the datatype for internal
calculations and the output dtype.


Input
    
**inputs** (<em>[…,k], tf.float32</em>) – 2+D tensor containing arbitrary values (not used!).

Output
    
<em>[…,n], tf.float32</em> – 2+D tensor containing all-zero codewords.

Raises
    
**AssertionError** – `k` and `n` must be positive integers and `k` must be smaller
    (or equal) than `n`.



**Note**
    
As the all-zero codeword is part of any linear code, it is often used
to simulate BER curves of arbitrary (LDPC) codes without the need of
having access to the actual generator matrix. However, this <cite>“all-zero
codeword trick”</cite> requires symmetric channels (such as BPSK), otherwise
scrambling is required (cf. <a class="reference internal" href="fec.ldpc.html#pfister" id="id2">[Pfister]</a> for further details).
    
This encoder is a dummy encoder that is needed for some all-zero
codeword simulations independent of the input. It does NOT perform
real encoding although the information bits are taken as input.
This is just to ensure compatibility with other encoding layers.

<em class="property">`property` </em>`coderate`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.linear.html#sionna.fec.linear.AllZeroEncoder.coderate" title="Permalink to this definition"></a>
    
Coderate of the LDPC code.


<em class="property">`property` </em>`k`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.linear.html#sionna.fec.linear.AllZeroEncoder.k" title="Permalink to this definition"></a>
    
Number of information bits per codeword.


<em class="property">`property` </em>`n`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.linear.html#sionna.fec.linear.AllZeroEncoder.n" title="Permalink to this definition"></a>
    
Codeword length.


## Decoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.linear.html#decoder" title="Permalink to this headline"></a>

### OSDecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.linear.html#osdecoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.linear.``OSDecoder`(<em class="sig-param">`enc_mat``=``None`</em>, <em class="sig-param">`t``=``0`</em>, <em class="sig-param">`is_pcm``=``False`</em>, <em class="sig-param">`encoder``=``None`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/fec/linear/decoding.html#OSDecoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.linear.html#sionna.fec.linear.OSDecoder" title="Permalink to this definition"></a>
    
Ordered statistics decoding (OSD) for binary, linear block codes.
    
This layer implements the OSD algorithm as proposed in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.linear.html#fossorier" id="id3">[Fossorier]</a> and,
thereby, approximates maximum likelihood decoding for a sufficiently large
order $t$. The algorithm works for arbitrary linear block codes, but
has a high computational complexity for long codes.
    
The algorithm consists of the following steps:
<blockquote>
<div>    
1. Sort LLRs according to their reliability and apply the same column
permutation to the generator matrix.
    
2. Bring the permuted generator matrix into its systematic form
(so-called <em>most-reliable basis</em>).
    
3. Hard-decide and re-encode the $k$ most reliable bits and
discard the remaining $n-k$ received positions.
    
4. Generate all possible error patterns up to $t$ errors in the
$k$ most reliable positions find the most likely codeword within
these candidates.
</blockquote>
    
This implementation of the OSD algorithm uses the LLR-based distance metric
from <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.linear.html#stimming-llr-osd" id="id4">[Stimming_LLR_OSD]</a> which simplifies the handling of higher-order
modulation schemes.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **enc_mat** (<em>[</em><em>k</em><em>, </em><em>n</em><em>] or </em><em>[</em><em>n-k</em><em>, </em><em>n</em><em>]</em><em>, </em><em>ndarray</em>) – Binary generator matrix of shape <cite>[k, n]</cite>. If `is_pcm` is
True, `enc_mat` is interpreted as parity-check matrix of shape
<cite>[n-k, n]</cite>.
- **t** (<em>int</em>) – Order of the OSD algorithm
- **is_pcm** (<em>bool</em>) – Defaults to False. If True, `enc_mat` is interpreted as parity-check
matrix.
- **encoder** (<em>Layer</em>) – Keras layer that implements a FEC encoder.
If not None, `enc_mat` will be ignored and the code as specified by he
encoder is used to initialize OSD.
- **dtype** (<em>tf.DType</em>) – Defaults to <cite>tf.float32</cite>. Defines the datatype for the output dtype.


Input
    
**llrs_ch** (<em>[…,n], tf.float32</em>) – 2+D tensor containing the channel logits/llr values.

Output
    
<em>[…,n], tf.float32</em> – 2+D Tensor of same shape as `llrs_ch` containing
binary hard-decisions of all codeword bits.



**Note**
    
OS decoding is of high complexity and is only feasible for small values of
$t$ as ${n \choose t}$ patterns must be evaluated. The
advantage of OSD is that it works for arbitrary linear block codes and
provides an estimate of the expected ML performance for sufficiently large
$t$. However, for some code families, more efficient decoding
algorithms with close to ML performance exist which can exploit certain
code specific properties. Examples of such decoders are the
<a class="reference internal" href="fec.conv.html#sionna.fec.conv.ViterbiDecoder" title="sionna.fec.conv.ViterbiDecoder">`ViterbiDecoder`</a> algorithm for  convolutional codes
or the <a class="reference internal" href="fec.polar.html#sionna.fec.polar.decoding.PolarSCLDecoder" title="sionna.fec.polar.decoding.PolarSCLDecoder">`PolarSCLDecoder`</a> for Polar codes
(for a sufficiently large list size).
    
It is recommended to run the decoder in XLA mode as it
significantly reduces the memory complexity.

<em class="property">`property` </em>`gm`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.linear.html#sionna.fec.linear.OSDecoder.gm" title="Permalink to this definition"></a>
    
Generator matrix of the code


<em class="property">`property` </em>`k`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.linear.html#sionna.fec.linear.OSDecoder.k" title="Permalink to this definition"></a>
    
Number of information bits per codeword


<em class="property">`property` </em>`n`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.linear.html#sionna.fec.linear.OSDecoder.n" title="Permalink to this definition"></a>
    
Codeword length


<em class="property">`property` </em>`t`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.linear.html#sionna.fec.linear.OSDecoder.t" title="Permalink to this definition"></a>
    
Order of the OSD algorithm



References:
Fossorier(<a href="https://nvlabs.github.io/sionna/api/fec.linear.html#id1">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.linear.html#id3">2</a>)
    
M. Fossorier, S. Lin, “Soft-Decision Decoding of Linear
Block Codes Based on Ordered Statistics”, IEEE Trans. Inf.
Theory, vol. 41, no.5, 1995.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/fec.linear.html#id4">Stimming_LLR_OSD</a>
    
A.Balatsoukas-Stimming, M. Parizi, A. Burg,
“LLR-Based Successive Cancellation List Decoding
of Polar Codes.” IEEE Trans Signal Processing, 2015.



