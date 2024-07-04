
# Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#utility-functions" title="Permalink to this headline"></a>
    
The utilities sub-package of the Sionna library contains many convenience
functions as well as extensions to existing TensorFlow functions.

## Metrics<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#metrics" title="Permalink to this headline"></a>

### BitErrorRate<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#biterrorrate" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.utils.``BitErrorRate`(<em class="sig-param">`name``=``'bit_error_rate'`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/utils/metrics.html#BitErrorRate">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.BitErrorRate" title="Permalink to this definition"></a>
    
Computes the average bit error rate (BER) between two binary tensors.
    
This class implements a Keras metric for the bit error rate
between two tensors of bits.
Input
 
- **b** (<em>tf.float32</em>) – A tensor of arbitrary shape filled with ones and
zeros.
- **b_hat** (<em>tf.float32</em>) – A tensor of the same shape as `b` filled with
ones and zeros.


Output
    
<em>tf.float32</em> – A scalar, the BER.



### BitwiseMutualInformation<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#bitwisemutualinformation" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.utils.``BitwiseMutualInformation`(<em class="sig-param">`name``=``'bitwise_mutual_information'`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/utils/metrics.html#BitwiseMutualInformation">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.BitwiseMutualInformation" title="Permalink to this definition"></a>
    
Computes the bitwise mutual information between bits and LLRs.
    
This class implements a Keras metric for the bitwise mutual information
between a tensor of bits and LLR (logits).
Input
 
- **bits** (<em>tf.float32</em>) – A tensor of arbitrary shape filled with ones and zeros.
- **llr** (<em>tf.float32</em>) – A tensor of the same shape as `bits` containing logits.


Output
    
<em>tf.float32</em> – A scalar, the bit-wise mutual information.



### compute_ber<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#compute-ber" title="Permalink to this headline"></a>

`sionna.utils.``compute_ber`(<em class="sig-param">`b`</em>, <em class="sig-param">`b_hat`</em>)<a class="reference internal" href="../_modules/sionna/utils/metrics.html#compute_ber">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.compute_ber" title="Permalink to this definition"></a>
    
Computes the bit error rate (BER) between two binary tensors.
Input
 
- **b** (<em>tf.float32</em>) – A tensor of arbitrary shape filled with ones and
zeros.
- **b_hat** (<em>tf.float32</em>) – A tensor of the same shape as `b` filled with
ones and zeros.


Output
    
<em>tf.float64</em> – A scalar, the BER.



### compute_bler<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#compute-bler" title="Permalink to this headline"></a>

`sionna.utils.``compute_bler`(<em class="sig-param">`b`</em>, <em class="sig-param">`b_hat`</em>)<a class="reference internal" href="../_modules/sionna/utils/metrics.html#compute_bler">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.compute_bler" title="Permalink to this definition"></a>
    
Computes the block error rate (BLER) between two binary tensors.
    
A block error happens if at least one element of `b` and `b_hat`
differ in one block. The BLER is evaluated over the last dimension of
the input, i. e., all elements of the last dimension are considered to
define a block.
    
This is also sometimes referred to as <cite>word error rate</cite> or <cite>frame error
rate</cite>.
Input
 
- **b** (<em>tf.float32</em>) – A tensor of arbitrary shape filled with ones and
zeros.
- **b_hat** (<em>tf.float32</em>) – A tensor of the same shape as `b` filled with
ones and zeros.


Output
    
<em>tf.float64</em> – A scalar, the BLER.



### compute_ser<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#compute-ser" title="Permalink to this headline"></a>

`sionna.utils.``compute_ser`(<em class="sig-param">`s`</em>, <em class="sig-param">`s_hat`</em>)<a class="reference internal" href="../_modules/sionna/utils/metrics.html#compute_ser">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.compute_ser" title="Permalink to this definition"></a>
    
Computes the symbol error rate (SER) between two integer tensors.
Input
 
- **s** (<em>tf.int</em>) – A tensor of arbitrary shape filled with integers indicating
the symbol indices.
- **s_hat** (<em>tf.int</em>) – A tensor of the same shape as `s` filled with integers indicating
the estimated symbol indices.


Output
    
<em>tf.float64</em> – A scalar, the SER.



### count_errors<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#count-errors" title="Permalink to this headline"></a>

`sionna.utils.``count_errors`(<em class="sig-param">`b`</em>, <em class="sig-param">`b_hat`</em>)<a class="reference internal" href="../_modules/sionna/utils/metrics.html#count_errors">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.count_errors" title="Permalink to this definition"></a>
    
Counts the number of bit errors between two binary tensors.
Input
 
- **b** (<em>tf.float32</em>) – A tensor of arbitrary shape filled with ones and
zeros.
- **b_hat** (<em>tf.float32</em>) – A tensor of the same shape as `b` filled with
ones and zeros.


Output
    
<em>tf.int64</em> – A scalar, the number of bit errors.



### count_block_errors<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#count-block-errors" title="Permalink to this headline"></a>

`sionna.utils.``count_block_errors`(<em class="sig-param">`b`</em>, <em class="sig-param">`b_hat`</em>)<a class="reference internal" href="../_modules/sionna/utils/metrics.html#count_block_errors">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.count_block_errors" title="Permalink to this definition"></a>
    
Counts the number of block errors between two binary tensors.
    
A block error happens if at least one element of `b` and `b_hat`
differ in one block. The BLER is evaluated over the last dimension of
the input, i. e., all elements of the last dimension are considered to
define a block.
    
This is also sometimes referred to as <cite>word error rate</cite> or <cite>frame error
rate</cite>.
Input
 
- **b** (<em>tf.float32</em>) – A tensor of arbitrary shape filled with ones and
zeros.
- **b_hat** (<em>tf.float32</em>) – A tensor of the same shape as `b` filled with
ones and zeros.


Output
    
<em>tf.int64</em> – A scalar, the number of block errors.



## Tensors<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#tensors" title="Permalink to this headline"></a>

### expand_to_rank<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#expand-to-rank" title="Permalink to this headline"></a>

`sionna.utils.``expand_to_rank`(<em class="sig-param">`tensor`</em>, <em class="sig-param">`target_rank`</em>, <em class="sig-param">`axis``=``-` `1`</em>)<a class="reference internal" href="../_modules/sionna/utils/tensors.html#expand_to_rank">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.expand_to_rank" title="Permalink to this definition"></a>
    
Inserts as many axes to a tensor as needed to achieve a desired rank.
    
This operation inserts additional dimensions to a `tensor` starting at
`axis`, so that so that the rank of the resulting tensor has rank
`target_rank`. The dimension index follows Python indexing rules, i.e.,
zero-based, where a negative index is counted backward from the end.
Parameters
 
- **tensor** – A tensor.
- **target_rank** (<em>int</em>) – The rank of the output tensor.
If `target_rank` is smaller than the rank of `tensor`,
the function does nothing.
- **axis** (<em>int</em>) – The dimension index at which to expand the
shape of `tensor`. Given a `tensor` of <cite>D</cite> dimensions,
`axis` must be within the range <cite>[-(D+1), D]</cite> (inclusive).


Returns
    
A tensor with the same data as `tensor`, with
`target_rank`- rank(`tensor`) additional dimensions inserted at the
index specified by `axis`.
If `target_rank` <= rank(`tensor`), `tensor` is returned.



### flatten_dims<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#flatten-dims" title="Permalink to this headline"></a>

`sionna.utils.``flatten_dims`(<em class="sig-param">`tensor`</em>, <em class="sig-param">`num_dims`</em>, <em class="sig-param">`axis`</em>)<a class="reference internal" href="../_modules/sionna/utils/tensors.html#flatten_dims">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.flatten_dims" title="Permalink to this definition"></a>
    
Flattens a specified set of dimensions of a tensor.
    
This operation flattens `num_dims` dimensions of a `tensor`
starting at a given `axis`.
Parameters
 
- **tensor** – A tensor.
- **num_dims** (<em>int</em>) – The number of dimensions
to combine. Must be larger than two and less or equal than the
rank of `tensor`.
- **axis** (<em>int</em>) – The index of the dimension from which to start.


Returns
    
A tensor of the same type as `tensor` with `num_dims`-1 lesser
dimensions, but the same number of elements.



### flatten_last_dims<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#flatten-last-dims" title="Permalink to this headline"></a>

`sionna.utils.``flatten_last_dims`(<em class="sig-param">`tensor`</em>, <em class="sig-param">`num_dims``=``2`</em>)<a class="reference internal" href="../_modules/sionna/utils/tensors.html#flatten_last_dims">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.flatten_last_dims" title="Permalink to this definition"></a>
    
Flattens the last <cite>n</cite> dimensions of a tensor.
    
This operation flattens the last `num_dims` dimensions of a `tensor`.
It is a simplified version of the function `flatten_dims`.
Parameters
 
- **tensor** – A tensor.
- **num_dims** (<em>int</em>) – The number of dimensions
to combine. Must be greater than or equal to two and less or equal
than the rank of `tensor`.


Returns
    
A tensor of the same type as `tensor` with `num_dims`-1 lesser
dimensions, but the same number of elements.



### insert_dims<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#insert-dims" title="Permalink to this headline"></a>

`sionna.utils.``insert_dims`(<em class="sig-param">`tensor`</em>, <em class="sig-param">`num_dims`</em>, <em class="sig-param">`axis``=``-` `1`</em>)<a class="reference internal" href="../_modules/sionna/utils/tensors.html#insert_dims">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.insert_dims" title="Permalink to this definition"></a>
    
Adds multiple length-one dimensions to a tensor.
    
This operation is an extension to TensorFlow`s `expand_dims` function.
It inserts `num_dims` dimensions of length one starting from the
dimension `axis` of a `tensor`. The dimension
index follows Python indexing rules, i.e., zero-based, where a negative
index is counted backward from the end.
Parameters
 
- **tensor** – A tensor.
- **num_dims** (<em>int</em>) – The number of dimensions to add.
- **axis** – The dimension index at which to expand the
shape of `tensor`. Given a `tensor` of <cite>D</cite> dimensions,
`axis` must be within the range <cite>[-(D+1), D]</cite> (inclusive).


Returns
    
A tensor with the same data as `tensor`, with `num_dims` additional
dimensions inserted at the index specified by `axis`.



### split_dims<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#split-dims" title="Permalink to this headline"></a>

`sionna.utils.``split_dim`(<em class="sig-param">`tensor`</em>, <em class="sig-param">`shape`</em>, <em class="sig-param">`axis`</em>)<a class="reference internal" href="../_modules/sionna/utils/tensors.html#split_dim">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.split_dim" title="Permalink to this definition"></a>
    
Reshapes a dimension of a tensor into multiple dimensions.
    
This operation splits the dimension `axis` of a `tensor` into
multiple dimensions according to `shape`.
Parameters
 
- **tensor** – A tensor.
- **shape** (<em>list</em><em> or </em><em>TensorShape</em>) – The shape to which the dimension should
be reshaped.
- **axis** (<em>int</em>) – The index of the axis to be reshaped.


Returns
    
A tensor of the same type as `tensor` with len(`shape`)-1
additional dimensions, but the same number of elements.



### matrix_sqrt<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#matrix-sqrt" title="Permalink to this headline"></a>

`sionna.utils.``matrix_sqrt`(<em class="sig-param">`tensor`</em>)<a class="reference internal" href="../_modules/sionna/utils/tensors.html#matrix_sqrt">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.matrix_sqrt" title="Permalink to this definition"></a>
    
Computes the square root of a matrix.
    
Given a batch of Hermitian positive semi-definite matrices
$\mathbf{A}$, returns matrices $\mathbf{B}$,
such that $\mathbf{B}\mathbf{B}^H = \mathbf{A}$.
    
The two inner dimensions are assumed to correspond to the matrix rows
and columns, respectively.
Parameters
    
**tensor** (<em>[</em><em>...</em><em>, </em><em>M</em><em>, </em><em>M</em><em>]</em>) – A tensor of rank greater than or equal
to two.

Returns
    
A tensor of the same shape and type as `tensor` containing
the matrix square root of its last two dimensions.



**Note**
    
If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.config.xla_compat=true`.
See `xla_compat`.

### matrix_sqrt_inv<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#matrix-sqrt-inv" title="Permalink to this headline"></a>

`sionna.utils.``matrix_sqrt_inv`(<em class="sig-param">`tensor`</em>)<a class="reference internal" href="../_modules/sionna/utils/tensors.html#matrix_sqrt_inv">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.matrix_sqrt_inv" title="Permalink to this definition"></a>
    
Computes the inverse square root of a Hermitian matrix.
    
Given a batch of Hermitian positive definite matrices
$\mathbf{A}$, with square root matrices $\mathbf{B}$,
such that $\mathbf{B}\mathbf{B}^H = \mathbf{A}$, the function
returns $\mathbf{B}^{-1}$, such that
$\mathbf{B}^{-1}\mathbf{B}=\mathbf{I}$.
    
The two inner dimensions are assumed to correspond to the matrix rows
and columns, respectively.
Parameters
    
**tensor** (<em>[</em><em>...</em><em>, </em><em>M</em><em>, </em><em>M</em><em>]</em>) – A tensor of rank greater than or equal
to two.

Returns
    
A tensor of the same shape and type as `tensor` containing
the inverse matrix square root of its last two dimensions.



**Note**
    
If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

### matrix_inv<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#matrix-inv" title="Permalink to this headline"></a>

`sionna.utils.``matrix_inv`(<em class="sig-param">`tensor`</em>)<a class="reference internal" href="../_modules/sionna/utils/tensors.html#matrix_inv">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.matrix_inv" title="Permalink to this definition"></a>
    
Computes the inverse of a Hermitian matrix.
    
Given a batch of Hermitian positive definite matrices
$\mathbf{A}$, the function
returns $\mathbf{A}^{-1}$, such that
$\mathbf{A}^{-1}\mathbf{A}=\mathbf{I}$.
    
The two inner dimensions are assumed to correspond to the matrix rows
and columns, respectively.
Parameters
    
**tensor** (<em>[</em><em>...</em><em>, </em><em>M</em><em>, </em><em>M</em><em>]</em>) – A tensor of rank greater than or equal
to two.

Returns
    
A tensor of the same shape and type as `tensor`, containing
the inverse of its last two dimensions.



**Note**
    
If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

### matrix_pinv<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#matrix-pinv" title="Permalink to this headline"></a>

`sionna.utils.``matrix_pinv`(<em class="sig-param">`tensor`</em>)<a class="reference internal" href="../_modules/sionna/utils/tensors.html#matrix_pinv">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.matrix_pinv" title="Permalink to this definition"></a>
    
Computes the Moore–Penrose (or pseudo) inverse of a matrix.
    
Given a batch of $M \times K$ matrices $\mathbf{A}$ with rank
$K$ (i.e., linearly independent columns), the function returns
$\mathbf{A}^+$, such that
$\mathbf{A}^{+}\mathbf{A}=\mathbf{I}_K$.
    
The two inner dimensions are assumed to correspond to the matrix rows
and columns, respectively.
Parameters
    
**tensor** (<em>[</em><em>...</em><em>, </em><em>M</em><em>, </em><em>K</em><em>]</em>) – A tensor of rank greater than or equal
to two.

Returns
    
A tensor of shape ([…, K,K]) of the same type as `tensor`,
containing the pseudo inverse of its last two dimensions.



**Note**
    
If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.config.xla_compat=true`.
See `xla_compat`.

## Miscellaneous<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#miscellaneous" title="Permalink to this headline"></a>

### BinarySource<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#binarysource" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.utils.``BinarySource`(<em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`seed``=``None`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/utils/misc.html#BinarySource">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.BinarySource" title="Permalink to this definition"></a>
    
Layer generating random binary tensors.
Parameters
 
- **dtype** (<em>tf.DType</em>) – Defines the output datatype of the layer.
Defaults to <cite>tf.float32</cite>.
- **seed** (<em>int</em><em> or </em><em>None</em>) – Set the seed for the random generator used to generate the bits.
Set to <cite>None</cite> for random initialization of the RNG.


Input
    
**shape** (<em>1D tensor/array/list, int</em>) – The desired shape of the output tensor.

Output
    
`shape`, `dtype` – Tensor filled with random binary values.



### SymbolSource<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#symbolsource" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.utils.``SymbolSource`(<em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`return_indices``=``False`</em>, <em class="sig-param">`return_bits``=``False`</em>, <em class="sig-param">`seed``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/utils/misc.html#SymbolSource">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.SymbolSource" title="Permalink to this definition"></a>
    
Layer generating a tensor of arbitrary shape filled with random constellation symbols.
Optionally, the symbol indices and/or binary representations of the
constellation symbols can be returned.
Parameters
 
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – An instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or
<cite>None</cite>. In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **return_indices** (<em>bool</em>) – If enabled, the function also returns the symbol indices.
Defaults to <cite>False</cite>.
- **return_bits** (<em>bool</em>) – If enabled, the function also returns the binary symbol
representations (i.e., bit labels).
Defaults to <cite>False</cite>.
- **seed** (<em>int</em><em> or </em><em>None</em>) – The seed for the random generator.
<cite>None</cite> leads to a random initialization of the RNG.
Defaults to <cite>None</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>]</em><em>, </em><em>tf.DType</em>) – The output dtype. Defaults to tf.complex64.


Input
    
**shape** (<em>1D tensor/array/list, int</em>) – The desired shape of the output tensor.

Output
 
- **symbols** (`shape`, `dtype`) – Tensor filled with random symbols of the chosen `constellation_type`.
- **symbol_indices** (`shape`, tf.int32) – Tensor filled with the symbol indices.
Only returned if `return_indices` is <cite>True</cite>.
- **bits** ([`shape`, `num_bits_per_symbol`], tf.int32) – Tensor filled with the binary symbol representations (i.e., bit labels).
Only returned if `return_bits` is <cite>True</cite>.




### QAMSource<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#qamsource" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.utils.``QAMSource`(<em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`return_indices``=``False`</em>, <em class="sig-param">`return_bits``=``False`</em>, <em class="sig-param">`seed``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/utils/misc.html#QAMSource">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.QAMSource" title="Permalink to this definition"></a>
    
Layer generating a tensor of arbitrary shape filled with random QAM symbols.
Optionally, the symbol indices and/or binary representations of the
constellation symbols can be returned.
Parameters
 
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
- **return_indices** (<em>bool</em>) – If enabled, the function also returns the symbol indices.
Defaults to <cite>False</cite>.
- **return_bits** (<em>bool</em>) – If enabled, the function also returns the binary symbol
representations (i.e., bit labels).
Defaults to <cite>False</cite>.
- **seed** (<em>int</em><em> or </em><em>None</em>) – The seed for the random generator.
<cite>None</cite> leads to a random initialization of the RNG.
Defaults to <cite>None</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>]</em><em>, </em><em>tf.DType</em>) – The output dtype. Defaults to tf.complex64.


Input
    
**shape** (<em>1D tensor/array/list, int</em>) – The desired shape of the output tensor.

Output
 
- **symbols** (`shape`, `dtype`) – Tensor filled with random QAM symbols.
- **symbol_indices** (`shape`, tf.int32) – Tensor filled with the symbol indices.
Only returned if `return_indices` is <cite>True</cite>.
- **bits** ([`shape`, `num_bits_per_symbol`], tf.int32) – Tensor filled with the binary symbol representations (i.e., bit labels).
Only returned if `return_bits` is <cite>True</cite>.




### PAMSource<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#pamsource" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.utils.``PAMSource`(<em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`return_indices``=``False`</em>, <em class="sig-param">`return_bits``=``False`</em>, <em class="sig-param">`seed``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/utils/misc.html#PAMSource">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.PAMSource" title="Permalink to this definition"></a>
    
Layer generating a tensor of arbitrary shape filled with random PAM symbols.
Optionally, the symbol indices and/or binary representations of the
constellation symbols can be returned.
Parameters
 
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 1 for BPSK.
- **return_indices** (<em>bool</em>) – If enabled, the function also returns the symbol indices.
Defaults to <cite>False</cite>.
- **return_bits** (<em>bool</em>) – If enabled, the function also returns the binary symbol
representations (i.e., bit labels).
Defaults to <cite>False</cite>.
- **seed** (<em>int</em><em> or </em><em>None</em>) – The seed for the random generator.
<cite>None</cite> leads to a random initialization of the RNG.
Defaults to <cite>None</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>]</em><em>, </em><em>tf.DType</em>) – The output dtype. Defaults to tf.complex64.


Input
    
**shape** (<em>1D tensor/array/list, int</em>) – The desired shape of the output tensor.

Output
 
- **symbols** (`shape`, `dtype`) – Tensor filled with random PAM symbols.
- **symbol_indices** (`shape`, tf.int32) – Tensor filled with the symbol indices.
Only returned if `return_indices` is <cite>True</cite>.
- **bits** ([`shape`, `num_bits_per_symbol`], tf.int32) – Tensor filled with the binary symbol representations (i.e., bit labels).
Only returned if `return_bits` is <cite>True</cite>.




### PlotBER<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#plotber" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.utils.plotting.``PlotBER`(<em class="sig-param">`title``=``'Bit/Block` `Error` `Rate'`</em>)<a class="reference internal" href="../_modules/sionna/utils/plotting.html#PlotBER">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.plotting.PlotBER" title="Permalink to this definition"></a>
    
Provides a plotting object to simulate and store BER/BLER curves.
Parameters
    
**title** (<em>str</em>) – A string defining the title of the figure. Defaults to
<cite>“Bit/Block Error Rate”</cite>.

Input
 
- **snr_db** (<em>float</em>) – Python array (or list of Python arrays) of additional SNR values to be
plotted.
- **ber** (<em>float</em>) – Python array (or list of Python arrays) of additional BERs
corresponding to `snr_db`.
- **legend** (<em>str</em>) – String (or list of strings) of legends entries.
- **is_bler** (<em>bool</em>) – A boolean (or list of booleans) defaults to False.
If True, `ber` will be interpreted as BLER.
- **show_ber** (<em>bool</em>) – A boolean defaults to True. If True, BER curves will be plotted.
- **show_bler** (<em>bool</em>) – A boolean defaults to True. If True, BLER curves will be plotted.
- **xlim** (<em>tuple of floats</em>) – Defaults to None. A tuple of two floats defining x-axis limits.
- **ylim** (<em>tuple of floats</em>) – Defaults to None. A tuple of two floats defining y-axis limits.
- **save_fig** (<em>bool</em>) – A boolean defaults to False. If True, the figure
is saved as file.
- **path** (<em>str</em>) – A string defining where to save the figure (if `save_fig`
is True).




`add`(<em class="sig-param">`ebno_db`</em>, <em class="sig-param">`ber`</em>, <em class="sig-param">`is_bler``=``False`</em>, <em class="sig-param">`legend``=``''`</em>)<a class="reference internal" href="../_modules/sionna/utils/plotting.html#PlotBER.add">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.plotting.PlotBER.add" title="Permalink to this definition"></a>
    
Add static reference curves.
Input
 
- **ebno_db** (<em>float</em>) – Python array or list of floats defining the SNR points.
- **ber** (<em>float</em>) – Python array or list of floats defining the BER corresponding
to each SNR point.
- **is_bler** (<em>bool</em>) – A boolean defaults to False. If True, `ber` is interpreted as
BLER.
- **legend** (<em>str</em>) – A string defining the text of the legend entry.





<em class="property">`property` </em>`ber`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.plotting.PlotBER.ber" title="Permalink to this definition"></a>
    
List containing all stored BER curves.


<em class="property">`property` </em>`is_bler`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.plotting.PlotBER.is_bler" title="Permalink to this definition"></a>
    
List of booleans indicating if ber shall be interpreted as BLER.


<em class="property">`property` </em>`legend`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.plotting.PlotBER.legend" title="Permalink to this definition"></a>
    
List containing all stored legend entries curves.


`remove`(<em class="sig-param">`idx``=``-` `1`</em>)<a class="reference internal" href="../_modules/sionna/utils/plotting.html#PlotBER.remove">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.plotting.PlotBER.remove" title="Permalink to this definition"></a>
    
Remove curve with index `idx`.
Input
    
**idx** (<em>int</em>) – An integer defining the index of the dataset that should
be removed. Negative indexing is possible.




`reset`()<a class="reference internal" href="../_modules/sionna/utils/plotting.html#PlotBER.reset">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.plotting.PlotBER.reset" title="Permalink to this definition"></a>
    
Remove all internal data.


`simulate`(<em class="sig-param">`mc_fun`</em>, <em class="sig-param">`ebno_dbs`</em>, <em class="sig-param">`batch_size`</em>, <em class="sig-param">`max_mc_iter`</em>, <em class="sig-param">`legend``=``''`</em>, <em class="sig-param">`add_ber``=``True`</em>, <em class="sig-param">`add_bler``=``False`</em>, <em class="sig-param">`soft_estimates``=``False`</em>, <em class="sig-param">`num_target_bit_errors``=``None`</em>, <em class="sig-param">`num_target_block_errors``=``None`</em>, <em class="sig-param">`target_ber``=``None`</em>, <em class="sig-param">`target_bler``=``None`</em>, <em class="sig-param">`early_stop``=``True`</em>, <em class="sig-param">`graph_mode``=``None`</em>, <em class="sig-param">`distribute``=``None`</em>, <em class="sig-param">`add_results``=``True`</em>, <em class="sig-param">`forward_keyboard_interrupt``=``True`</em>, <em class="sig-param">`show_fig``=``True`</em>, <em class="sig-param">`verbose``=``True`</em>)<a class="reference internal" href="../_modules/sionna/utils/plotting.html#PlotBER.simulate">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.plotting.PlotBER.simulate" title="Permalink to this definition"></a>
    
Simulate BER/BLER curves for given Keras model and saves the results.
    
Internally calls <a class="reference internal" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.sim_ber" title="sionna.utils.sim_ber">`sionna.utils.sim_ber`</a>.
Input
 
- **mc_fun** – Callable that yields the transmitted bits <cite>b</cite> and the
receiver’s estimate <cite>b_hat</cite> for a given `batch_size` and
`ebno_db`. If `soft_estimates` is True, b_hat is interpreted as
logit.
- **ebno_dbs** (<em>ndarray of floats</em>) – SNR points to be evaluated.
- **batch_size** (<em>tf.int32</em>) – Batch-size for evaluation.
- **max_mc_iter** (<em>int</em>) – Max. number of Monte-Carlo iterations per SNR point.
- **legend** (<em>str</em>) – Name to appear in legend.
- **add_ber** (<em>bool</em>) – Defaults to True. Indicate if BER should be added to plot.
- **add_bler** (<em>bool</em>) – Defaults to False. Indicate if BLER should be added
to plot.
- **soft_estimates** (<em>bool</em>) – A boolean, defaults to False. If True, `b_hat`
is interpreted as logit and additional hard-decision is applied
internally.
- **num_target_bit_errors** (<em>int</em>) – Target number of bit errors per SNR point until the simulation
stops.
- **num_target_block_errors** (<em>int</em>) – Target number of block errors per SNR point until the simulation
stops.
- **target_ber** (<em>tf.float32</em>) – Defaults to <cite>None</cite>. The simulation stops after the first SNR point
which achieves a lower bit error rate as specified by
`target_ber`. This requires `early_stop` to be <cite>True</cite>.
- **target_bler** (<em>tf.float32</em>) – Defaults to <cite>None</cite>. The simulation stops after the first SNR point
which achieves a lower block error rate as specified by
`target_bler`.  This requires `early_stop` to be <cite>True</cite>.
- **early_stop** (<em>bool</em>) – A boolean defaults to True. If True, the simulation stops after the
first error-free SNR point (i.e., no error occurred after
`max_mc_iter` Monte-Carlo iterations).
- **graph_mode** (<em>One of [“graph”, “xla”], str</em>) – A string describing the execution mode of `mc_fun`.
Defaults to <cite>None</cite>. In this case, `mc_fun` is executed as is.
- **distribute** (<cite>None</cite> (default) | “all” | list of indices | <cite>tf.distribute.strategy</cite>) – Distributes simulation on multiple parallel devices. If <cite>None</cite>,
multi-device simulations are deactivated. If “all”, the workload
will be automatically distributed across all available GPUs via the
<cite>tf.distribute.MirroredStrategy</cite>.
If an explicit list of indices is provided, only the GPUs with the
given indices will be used. Alternatively, a custom
<cite>tf.distribute.strategy</cite> can be provided. Note that the same
<cite>batch_size</cite> will be used for all GPUs in parallel, but the number
of Monte-Carlo iterations `max_mc_iter` will be scaled by the
number of devices such that the same number of total samples is
simulated. However, all stopping conditions are still in-place
which can cause slight differences in the total number of simulated
samples.
- **add_results** (<em>bool</em>) – Defaults to True. If True, the simulation results will be appended
to the internal list of results.
- **show_fig** (<em>bool</em>) – Defaults to True. If True, a BER figure will be plotted.
- **verbose** (<em>bool</em>) – A boolean defaults to True. If True, the current progress will be
printed.
- **forward_keyboard_interrupt** (<em>bool</em>) – A boolean defaults to True. If False, <cite>KeyboardInterrupts</cite> will be
catched internally and not forwarded (e.g., will not stop outer
loops). If False, the simulation ends and returns the intermediate
simulation results.


Output
 
- **(ber, bler)** – Tuple:
- **ber** (<em>float</em>) – The simulated bit-error rate.
- **bler** (<em>float</em>) – The simulated block-error rate.





<em class="property">`property` </em>`snr`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.plotting.PlotBER.snr" title="Permalink to this definition"></a>
    
List containing all stored SNR curves.


<em class="property">`property` </em>`title`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.plotting.PlotBER.title" title="Permalink to this definition"></a>
    
Title of the plot.


### sim_ber<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sim-ber" title="Permalink to this headline"></a>

`sionna.utils.``sim_ber`(<em class="sig-param">`mc_fun`</em>, <em class="sig-param">`ebno_dbs`</em>, <em class="sig-param">`batch_size`</em>, <em class="sig-param">`max_mc_iter`</em>, <em class="sig-param">`soft_estimates``=``False`</em>, <em class="sig-param">`num_target_bit_errors``=``None`</em>, <em class="sig-param">`num_target_block_errors``=``None`</em>, <em class="sig-param">`target_ber``=``None`</em>, <em class="sig-param">`target_bler``=``None`</em>, <em class="sig-param">`early_stop``=``True`</em>, <em class="sig-param">`graph_mode``=``None`</em>, <em class="sig-param">`distribute``=``None`</em>, <em class="sig-param">`verbose``=``True`</em>, <em class="sig-param">`forward_keyboard_interrupt``=``True`</em>, <em class="sig-param">`callback``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/utils/misc.html#sim_ber">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.sim_ber" title="Permalink to this definition"></a>
    
Simulates until target number of errors is reached and returns BER/BLER.
    
The simulation continues with the next SNR point if either
`num_target_bit_errors` bit errors or `num_target_block_errors` block
errors is achieved. Further, it continues with the next SNR point after
`max_mc_iter` batches of size `batch_size` have been simulated.
Early stopping allows to stop the simulation after the first error-free SNR
point or after reaching a certain `target_ber` or `target_bler`.
Input
 
- **mc_fun** (<em>callable</em>) – Callable that yields the transmitted bits <cite>b</cite> and the
receiver’s estimate <cite>b_hat</cite> for a given `batch_size` and
`ebno_db`. If `soft_estimates` is True, <cite>b_hat</cite> is interpreted as
logit.
- **ebno_dbs** (<em>tf.float32</em>) – A tensor containing SNR points to be evaluated.
- **batch_size** (<em>tf.int32</em>) – Batch-size for evaluation.
- **max_mc_iter** (<em>tf.int32</em>) – Maximum number of Monte-Carlo iterations per SNR point.
- **soft_estimates** (<em>bool</em>) – A boolean, defaults to <cite>False</cite>. If <cite>True</cite>, <cite>b_hat</cite>
is interpreted as logit and an additional hard-decision is applied
internally.
- **num_target_bit_errors** (<em>tf.int32</em>) – Defaults to <cite>None</cite>. Target number of bit errors per SNR point until
the simulation continues to next SNR point.
- **num_target_block_errors** (<em>tf.int32</em>) – Defaults to <cite>None</cite>. Target number of block errors per SNR point
until the simulation continues
- **target_ber** (<em>tf.float32</em>) – Defaults to <cite>None</cite>. The simulation stops after the first SNR point
which achieves a lower bit error rate as specified by `target_ber`.
This requires `early_stop` to be <cite>True</cite>.
- **target_bler** (<em>tf.float32</em>) – Defaults to <cite>None</cite>. The simulation stops after the first SNR point
which achieves a lower block error rate as specified by `target_bler`.
This requires `early_stop` to be <cite>True</cite>.
- **early_stop** (<em>bool</em>) – A boolean defaults to <cite>True</cite>. If <cite>True</cite>, the simulation stops after the
first error-free SNR point (i.e., no error occurred after
`max_mc_iter` Monte-Carlo iterations).
- **graph_mode** (<em>One of [“graph”, “xla”], str</em>) – A string describing the execution mode of `mc_fun`.
Defaults to <cite>None</cite>. In this case, `mc_fun` is executed as is.
- **distribute** (<cite>None</cite> (default) | “all” | list of indices | <cite>tf.distribute.strategy</cite>) – Distributes simulation on multiple parallel devices. If <cite>None</cite>,
multi-device simulations are deactivated. If “all”, the workload will
be automatically distributed across all available GPUs via the
<cite>tf.distribute.MirroredStrategy</cite>.
If an explicit list of indices is provided, only the GPUs with the given
indices will be used. Alternatively, a custom <cite>tf.distribute.strategy</cite>
can be provided. Note that the same <cite>batch_size</cite> will be
used for all GPUs in parallel, but the number of Monte-Carlo iterations
`max_mc_iter` will be scaled by the number of devices such that the
same number of total samples is simulated. However, all stopping
conditions are still in-place which can cause slight differences in the
total number of simulated samples.
- **verbose** (<em>bool</em>) – A boolean defaults to <cite>True</cite>. If <cite>True</cite>, the current progress will be
printed.
- **forward_keyboard_interrupt** (<em>bool</em>) – A boolean defaults to <cite>True</cite>. If <cite>False</cite>, KeyboardInterrupts will be
catched internally and not forwarded (e.g., will not stop outer loops).
If <cite>False</cite>, the simulation ends and returns the intermediate simulation
results.
- **callback** (<cite>None</cite> (default) | callable) – If specified, `callback` will be called after each Monte-Carlo step.
Can be used for logging or advanced early stopping. Input signature of
`callback` must match <cite>callback(mc_iter, snr_idx, ebno_dbs,
bit_errors, block_errors, nb_bits, nb_blocks)</cite> where `mc_iter`
denotes the number of processed batches for the current SNR point,
`snr_idx` is the index of the current SNR point, `ebno_dbs` is the
vector of all SNR points to be evaluated, `bit_errors` the vector of
number of bit errors for each SNR point, `block_errors` the vector of
number of block errors, `nb_bits` the vector of number of simulated
bits, `nb_blocks` the vector of number of simulated blocks,
respectively. If `callable` returns <cite>sim_ber.CALLBACK_NEXT_SNR</cite>, early
stopping is detected and the simulation will continue with the
next SNR point. If `callable` returns
<cite>sim_ber.CALLBACK_STOP</cite>, the simulation is stopped
immediately. For <cite>sim_ber.CALLBACK_CONTINUE</cite> continues with
the simulation.
- **dtype** (<em>tf.complex64</em>) – Datatype of the callable `mc_fun` to be used as input/output.


Output
 
- **(ber, bler)** – Tuple:
- **ber** (<em>tf.float32</em>) – The bit-error rate.
- **bler** (<em>tf.float32</em>) – The block-error rate.


Raises
 
- **AssertionError** – If `soft_estimates` is not bool.
- **AssertionError** – If `dtype` is not <cite>tf.complex</cite>.




**Note**
    
This function is implemented based on tensors to allow
full compatibility with tf.function(). However, to run simulations
in graph mode, the provided `mc_fun` must use the <cite>@tf.function()</cite>
decorator.

### ebnodb2no<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#ebnodb2no" title="Permalink to this headline"></a>

`sionna.utils.``ebnodb2no`(<em class="sig-param">`ebno_db`</em>, <em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`coderate`</em>, <em class="sig-param">`resource_grid``=``None`</em>)<a class="reference internal" href="../_modules/sionna/utils/misc.html#ebnodb2no">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.ebnodb2no" title="Permalink to this definition"></a>
    
Compute the noise variance <cite>No</cite> for a given <cite>Eb/No</cite> in dB.
    
The function takes into account the number of coded bits per constellation
symbol, the coderate, as well as possible additional overheads related to
OFDM transmissions, such as the cyclic prefix and pilots.
    
The value of <cite>No</cite> is computed according to the following expression

$$
N_o = \left(\frac{E_b}{N_o} \frac{r M}{E_s}\right)^{-1}
$$
    
where $2^M$ is the constellation size, i.e., $M$ is the
average number of coded bits per constellation symbol,
$E_s=1$ is the average energy per constellation per symbol,
$r\in(0,1]$ is the coderate,
$E_b$ is the energy per information bit,
and $N_o$ is the noise power spectral density.
For OFDM transmissions, $E_s$ is scaled
according to the ratio between the total number of resource elements in
a resource grid with non-zero energy and the number
of resource elements used for data transmission. Also the additionally
transmitted energy during the cyclic prefix is taken into account, as
well as the number of transmitted streams per transmitter.
Input
 
- **ebno_db** (<em>float</em>) – The <cite>Eb/No</cite> value in dB.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per symbol.
- **coderate** (<em>float</em>) – The coderate used.
- **resource_grid** (<em>ResourceGrid</em>) – An (optional) instance of <a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>
for OFDM transmissions.


Output
    
<em>float</em> – The value of $N_o$ in linear scale.



### hard_decisions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#hard-decisions" title="Permalink to this headline"></a>

`sionna.utils.``hard_decisions`(<em class="sig-param">`llr`</em>)<a class="reference internal" href="../_modules/sionna/utils/misc.html#hard_decisions">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.hard_decisions" title="Permalink to this definition"></a>
    
Transforms LLRs into hard decisions.
    
Positive values are mapped to $1$.
Nonpositive values are mapped to $0$.
Input
    
**llr** (<em>any non-complex tf.DType</em>) – Tensor of LLRs.

Output
    
Same shape and dtype as `llr` – The hard decisions.



### plot_ber<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#plot-ber" title="Permalink to this headline"></a>

`sionna.utils.plotting.``plot_ber`(<em class="sig-param">`snr_db`</em>, <em class="sig-param">`ber`</em>, <em class="sig-param">`legend``=``''`</em>, <em class="sig-param">`ylabel``=``'BER'`</em>, <em class="sig-param">`title``=``'Bit` `Error` `Rate'`</em>, <em class="sig-param">`ebno``=``True`</em>, <em class="sig-param">`is_bler``=``None`</em>, <em class="sig-param">`xlim``=``None`</em>, <em class="sig-param">`ylim``=``None`</em>, <em class="sig-param">`save_fig``=``False`</em>, <em class="sig-param">`path``=``''`</em>)<a class="reference internal" href="../_modules/sionna/utils/plotting.html#plot_ber">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.plotting.plot_ber" title="Permalink to this definition"></a>
    
Plot error-rates.
Input
 
- **snr_db** (<em>ndarray</em>) – Array of floats defining the simulated SNR points.
Can be also a list of multiple arrays.
- **ber** (<em>ndarray</em>) – Array of floats defining the BER/BLER per SNR point.
Can be also a list of multiple arrays.
- **legend** (<em>str</em>) – Defaults to “”. Defining the legend entries. Can be
either a string or a list of strings.
- **ylabel** (<em>str</em>) – Defaults to “BER”. Defining the y-label.
- **title** (<em>str</em>) – Defaults to “Bit Error Rate”. Defining the title of the figure.
- **ebno** (<em>bool</em>) – Defaults to True. If True, the x-label is set to
“EbNo [dB]” instead of “EsNo [dB]”.
- **is_bler** (<em>bool</em>) – Defaults to False. If True, the corresponding curve is dashed.
- **xlim** (<em>tuple of floats</em>) – Defaults to None. A tuple of two floats defining x-axis limits.
- **ylim** (<em>tuple of floats</em>) – Defaults to None. A tuple of two floats defining y-axis limits.
- **save_fig** (<em>bool</em>) – Defaults to False. If True, the figure is saved as <cite>.png</cite>.
- **path** (<em>str</em>) – Defaults to “”. Defining the path to save the figure
(iff `save_fig` is True).


Output
 
- **(fig, ax)** – Tuple:
- **fig** (<em>matplotlib.figure.Figure</em>) – A matplotlib figure handle.
- **ax** (<em>matplotlib.axes.Axes</em>) – A matplotlib axes object.




### complex_normal<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#complex-normal" title="Permalink to this headline"></a>

`sionna.utils.``complex_normal`(<em class="sig-param">`shape`</em>, <em class="sig-param">`var``=``1.0`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/utils/misc.html#complex_normal">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.complex_normal" title="Permalink to this definition"></a>
    
Generates a tensor of complex normal random variables.
Input
 
- **shape** (<em>tf.shape, or list</em>) – The desired shape.
- **var** (<em>float</em>) – The total variance., i.e., each complex dimension has
variance `var/2`.
- **dtype** (<em>tf.complex</em>) – The desired dtype. Defaults to <cite>tf.complex64</cite>.


Output
    
`shape`, `dtype` – Tensor of complex normal random variables.



### log2<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#log2" title="Permalink to this headline"></a>

`sionna.utils.``log2`(<em class="sig-param">`x`</em>)<a class="reference internal" href="../_modules/sionna/utils/misc.html#log2">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.log2" title="Permalink to this definition"></a>
    
TensorFlow implementation of NumPy’s <cite>log2</cite> function.
    
Simple extension to <cite>tf.experimental.numpy.log2</cite>
which casts the result to the <cite>dtype</cite> of the input.
For more details see the <a class="reference external" href="https://www.tensorflow.org/api_docs/python/tf/experimental/numpy/log2">TensorFlow</a> and <a class="reference external" href="https://numpy.org/doc/1.16/reference/generated/numpy.log2.html">NumPy</a> documentation.

### log10<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#log10" title="Permalink to this headline"></a>

`sionna.utils.``log10`(<em class="sig-param">`x`</em>)<a class="reference internal" href="../_modules/sionna/utils/misc.html#log10">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.log10" title="Permalink to this definition"></a>
    
TensorFlow implementation of NumPy’s <cite>log10</cite> function.
    
Simple extension to <cite>tf.experimental.numpy.log10</cite>
which casts the result to the <cite>dtype</cite> of the input.
For more details see the <a class="reference external" href="https://www.tensorflow.org/api_docs/python/tf/experimental/numpy/log10">TensorFlow</a> and <a class="reference external" href="https://numpy.org/doc/1.16/reference/generated/numpy.log10.html">NumPy</a> documentation.
