# Interleaving

The interleaver module allows to permute tensors with either pseudo-random permutations or by row/column swapping.

To simplify distributed graph execution (e.g., by running interleaver and deinterleaver in a different sub-graph/device), the interleavers are implemented stateless. Thus, the internal seed cannot be updated on runtime and does not change after the initialization. However, if required, an explicit random seed can be passed as additional input to the interleaver/deinterleaver pair when calling the layer.

The following code snippet shows how to setup and use an instance of the interleaver:
```python
# set-up system
interleaver = RandomInterleaver(seed=1234, # an explicit seed can be provided
                                keep_batch_constant=False, # if True, all samples in the batch are permuted with the same pattern
                                axis=-1) # axis which shall be permuted
deinterleaver = Deinterleaver(interleaver=interleaver) # connect interleaver and deinterleaver
# --- simplified usage with fixed seed ---
# c has arbitrary shape (rank>=2)
c_int = interleaver(c)
# call deinterleaver to reconstruct the original order
c_deint = deinterleaver(c_int)
# --- advanced usage ---
# provide explicit seed if a new random seed should be used for each call
s = tf.random.uniform((), minval=0, maxval=12345678, dtype=tf.int32)
c_int = interleaver([c, s])
c_deint = deinterleaver([c_int, s])
```
## Interleaver

### RowColumnInterleaver

`class` `sionna.fec.interleaving.``RowColumnInterleaver`(*`row_depth`*, *`axis``=``-` `1`*, *`inverse``=``False`*, *`dtype``=``tf.float32`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/fec/interleaving.html#RowColumnInterleaver)

Interleaves a sequence of inputs via row/column swapping.

The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters

- **row_depth** (*int*)  The row depth, i.e., how many values per row can be stored.
- **axis** (*int*)  The dimension that should be interleaved. First dimension
(<cite>axis=0</cite>) is not allowed.
- **inverse** (*bool*)  A boolean defaults to False. If True, the inverse permutation is
performed.
- **dtype** (*tf.DType*)  Defaults to <cite>tf.float32</cite>. Defines the datatype for internal
calculations and the output dtype.


Input

**inputs** (*tf.DType*)  2+D tensor of arbitrary shape and arbitrary dtype. Must have at
least rank two.

Output

*tf.DType*  2+D tensor of same shape and dtype as `inputs`.

Raises

- **AssertionError**  If `axis` is not an integer.
- **AssertionError**  If `row_depth` is not an integer.
- **AssertionError**  If `axis` > number of input dimensions.


**Note**

If the sequence length is not a multiple of `row_depth`, additional
filler bits are used for the last row that will be removed internally.
However, for the last positions the interleaving distance may be
slightly degraded.

To permute the batch dimension, expand_dims at <cite>axis=0</cite>, interleave and
remove new dimension.

`property` `axis`

Axis to be permuted.


`call_inverse`(*`inputs`*)[`[source]`](../_modules/sionna/fec/interleaving.html#RowColumnInterleaver.call_inverse)

Implements deinterleaver function corresponding to call().
Input

**inputs** (*tf.DType*)  2+D tensor of arbitrary shape and arbitrary dtype. Must have at
least rank two.

Output

*tf.DType*  2+D tensor of same shape and dtype as `inputs`.


`property` `keep_state`

Row-column interleaver always uses same internal state.


`property` `perm_seq`

Permutation sequence.


`property` `perm_seq_inv`

Inverse permutation sequence.


`property` `row_depth`

Row depth of the row-column interleaver.


### RandomInterleaver

`class` `sionna.fec.interleaving.``RandomInterleaver`(*`seed``=``None`*, *`keep_batch_constant``=``True`*, *`inverse``=``False`*, *`keep_state``=``True`*, *`axis``=``-` `1`*, *`dtype``=``tf.float32`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/fec/interleaving.html#RandomInterleaver)

Random interleaver permuting a sequence of input symbols.

The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters

- **seed** (*int*)  Integer defining the random seed used if option `keep_state` is
True.
- **keep_batch_constant** (*bool*)  Defaults to True. If set to True each sample in the batch uses the
same permutation. Otherwise, unique permutations per batch sample
are generate (slower).
- **inverse** (*bool*)  A boolean defaults to False. If True, the inverse permutation is
performed.
- **keep_state** (*bool*)  A boolean defaults to True. If True, the permutation is fixed for
multiple calls (defined by `seed` attribute).
- **axis** (*int*)  Defaults to <cite>-1</cite>. The dimension that should be interleaved.
First dimension (<cite>axis=0</cite>) is not allowed.
- **dtype** (*tf.DType*)  Defaults to <cite>tf.float32</cite>. Defines the datatype for internal
calculations and the output dtype.


Input

- **(x, seed)**  Either Tuple `(x,` `seed)` or `x` only (no tuple) if the internal
seed should be used:
- **x** (*tf.DType*)  2+D tensor of arbitrary shape and dtype.
- **seed** (*int*)  An integer defining the state of the random number
generator. If explicitly given, the global internal seed is
replaced by this seed. Can be used to realize random
interleaver/deinterleaver pairs (call with same random seed).


Output

*tf.DType*  2+D tensor of same shape and dtype as the input `x`.

Raises

- **AssertionError**  If `axis` is not <cite>int</cite>.
- **AssertionError**  If `seed` is not <cite>None</cite> or <cite>int</cite>.
- **AssertionError**  If `axis` > number of input dimensions.
- **AssertionError**  If `inverse` is not bool.
- **AssertionError**  If `keep_state` is not bool.
- **AssertionError**  If `keep_batch_constant` is not bool.
- **InvalidArgumentError**  When rank(`x`)<2.


**Note**

To permute the batch dimension, expand_dims at `axis=0`, interleave
and remove new dimension.

The interleaver layer is stateless, i.e., the seed is either random
during each call or must be explicitly provided during init/call.
This simplifies XLA/graph execution.

This is NOT the 5G interleaver sequence.

`property` `axis`

Axis to be permuted.


`call_inverse`(*`inputs`*)[`[source]`](../_modules/sionna/fec/interleaving.html#RandomInterleaver.call_inverse)

Implements deinterleaver function corresponding to call().
Input

- **(x, seed)**  Either Tuple `(x,` `seed)` or `x` only (no tuple) if the internal
seed should be used:
- **x** (*tf.DType*)  2+D tensor of arbitrary shape and dtype.
- **seed** (*int*)  An integer defining the state of the random number
generator. If explicitly given, the global internal seed is
replaced by this seed. Can be used to realize random
interleaver/deinterleaver pairs (call with same random seed).


Output

*tf.DType*  2+D tensor of same shape and dtype as the input `x`.

Raises

- **InvalidArgumentError**  When rank(`x`)<2.
- **ValueError**  If `keep_state` is False and no explicit seed is provided.


**Note**

In case of inverse interleaving (e.g., at the receiver),
`keep_state` should be True as otherwise a new permutation is
generated and the output is not equal to the original sequence.
Alternatively, an explicit seed must be provided as function
argument.


`find_s_min`(*`seed`*, *`seq_length`*, *`s_min_stop``=``0`*)[`[source]`](../_modules/sionna/fec/interleaving.html#RandomInterleaver.find_s_min)

Find $S$ parameter such that $\pi(i)-\pi(j)>S$ for all
$i-j<S$. This can be used to find optimized interleaver patterns.

`s_min_stop` is an additional stopping condition, i.e., stop if
current $S$ is already smaller than `s_min_stop`.

Please note that this is a Numpy utility function and usually not part
of the graph.
Input

- **seed** (*int*)  seed to draw random permutation that shall be analyzed.
- **seq_length** (*int*)  length of permutation sequence to be analyzed.
- **s_min_stop** (*int*)  Defaults to 0. Enables early stop if already current s_min< `s_min_stop` .


Output

*float*  The S-parameter for the given `seed`.


`property` `keep_state`

Generate new random seed per call.


`property` `seed`

Seed to generate random sequence.


### Turbo3GPPInterleaver

`class` `sionna.fec.interleaving.``Turbo3GPPInterleaver`(*`inverse``=``False`*, *`axis``=``-` `1`*, *`dtype``=``tf.float32`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/fec/interleaving.html#Turbo3GPPInterleaver)

Interleaver as used in the 3GPP Turbo codes [[3GPPTS36212_I]](https://nvlabs.github.io/sionna/api/fec.interleaving.html#gppts36212-i) and, thus,
the maximum length is given as 6144 elements (only for the dimension as
specific by `axis`).

The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters

- **inverse** (*bool*)  A boolean defaults to False. If True, the inverse permutation is
performed.
- **axis** (*int*)  Defaults to <cite>-1</cite>. The dimension that should be interleaved.
First dimension (<cite>axis=0</cite>) is not allowed.
- **dtype** (*tf.DType*)  Defaults to <cite>tf.float32</cite>. Defines the datatype for internal
calculations and the output dtype.


Input

**x** (*tf.DType*)  2+D tensor of arbitrary shape and dtype.

Output

*tf.DType*  2+D tensor of same shape and dtype as the input `x`.

Raises

- **AssertionError**  If `axis` is not <cite>int</cite>.
- **AssertionError**  If `axis` > number of input dimensions.
- **AssertionError**  If `inverse` is not bool.
- **InvalidArgumentError**  When rank(`x`)<2.


**Note**

Note that this implementation slightly deviates from the 3GPP
standard [[3GPPTS36212_I]](https://nvlabs.github.io/sionna/api/fec.interleaving.html#gppts36212-i) in a sense that zero-padding is introduced
for cases when the exact interleaver length is not supported by the
standard.

`property` `axis`

Axis to be permuted.


`call_inverse`(*`inputs`*)[`[source]`](../_modules/sionna/fec/interleaving.html#Turbo3GPPInterleaver.call_inverse)

Implements deinterleaver function corresponding to call().
Input

**x** (*tf.DType*)  2+D tensor of arbitrary shape and dtype.

Output

*tf.DType*  2+D tensor of same shape and dtype as the input `x`.

Raises

**InvalidArgumentError**  When rank(`x`)<2.


`find_s_min`(*`frame_size`*, *`s_min_stop``=``0`*)[`[source]`](../_modules/sionna/fec/interleaving.html#Turbo3GPPInterleaver.find_s_min)

Find $S$ parameter such that $\pi(i)-\pi(j)>S$ for all
$i-j<S$. This can be used to find optimized interleaver patterns.

`s_min_stop` is an additional stopping condition, i.e., stop if
current $S$ is already smaller than `s_min_stop`.

Please note that this is a Numpy utility function and usually not part
of the graph.
Input

- **frame_size** (*int*)  length of interleaver.
- **s_min_stop** (*int*)  Defaults to 0. Enables early stop if already current
s_min<`s_min_stop`.


Output

*float*  The S-parameter for the given `frame_size`.


## Deinterleaver

`class` `sionna.fec.interleaving.``Deinterleaver`(*`interleaver`*, *`dtype``=``None`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/fec/interleaving.html#Deinterleaver)

Deinterleaver that reverts the interleaver for a given input sequence.

The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters

- **interleaver** (*Interleaver*)  Associated Interleaver which shall be deinterleaved by this layer.
Can be either
[`RandomInterleaver`](https://nvlabs.github.io/sionna/api/fec.interleaving.html#sionna.fec.interleaving.RandomInterleaver) or
[`RowColumnInterleaver`](https://nvlabs.github.io/sionna/api/fec.interleaving.html#sionna.fec.interleaving.RowColumnInterleaver).
- **dtype** (*None** or **tf.DType*)  Defaults to <cite>None</cite>. Defines the datatype for internal calculations
and the output dtype. If no explicit dtype is provided the dtype
from the associated interleaver is used.


Input

- **(x, seed)**  Either Tuple `(x,` `seed)` or `x` only (no tuple) if the internal
seed should be used:
- **x** (*tf.DType*)  2+D tensor of arbitrary shape.
- **seed** (*int*)  An integer defining the state of the random number
generator. If explicitly given, the global internal seed is
replaced by this seed. Can be used to realize random
interleaver/deinterleaver pairs (call with same random seed).


Output

*tf.DType*  2+D tensor of same shape and dtype as the input `x`.

Raises

**AssertionError**  If `interleaver` is not a valid instance of Interleaver.


**Note**

This layer provides a wrapper of the inverse interleaver function.

`property` `interleaver`

Associated interleaver instance.


References:
<blockquote>
<div>
3GPPTS36212_I([1](https://nvlabs.github.io/sionna/api/fec.interleaving.html#id1),[2](https://nvlabs.github.io/sionna/api/fec.interleaving.html#id2))

ETSI 3GPP TS 36.212 Evolved Universal Terrestrial
Radio Access (EUTRA); Multiplexing and channel coding, v.15.3.0, 2018-09.


</blockquote></blockquote>