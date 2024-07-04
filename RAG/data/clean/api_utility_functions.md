# Utility Functions

The utilities sub-package of the Sionna library contains many convenience
functions as well as extensions to existing TensorFlow functions.

## Metrics

### BitErrorRate

`class` `sionna.utils.``BitErrorRate`(*`name``=``'bit_error_rate'`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/utils/metrics.html#BitErrorRate)

Computes the average bit error rate (BER) between two binary tensors.

This class implements a Keras metric for the bit error rate
between two tensors of bits.
Input

- **b** (*tf.float32*)  A tensor of arbitrary shape filled with ones and
zeros.
- **b_hat** (*tf.float32*)  A tensor of the same shape as `b` filled with
ones and zeros.


Output

*tf.float32*  A scalar, the BER.


### BitwiseMutualInformation

`class` `sionna.utils.``BitwiseMutualInformation`(*`name``=``'bitwise_mutual_information'`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/utils/metrics.html#BitwiseMutualInformation)

Computes the bitwise mutual information between bits and LLRs.

This class implements a Keras metric for the bitwise mutual information
between a tensor of bits and LLR (logits).
Input

- **bits** (*tf.float32*)  A tensor of arbitrary shape filled with ones and zeros.
- **llr** (*tf.float32*)  A tensor of the same shape as `bits` containing logits.


Output

*tf.float32*  A scalar, the bit-wise mutual information.


### compute_ber

`sionna.utils.``compute_ber`(*`b`*, *`b_hat`*)[`[source]`](../_modules/sionna/utils/metrics.html#compute_ber)

Computes the bit error rate (BER) between two binary tensors.
Input

- **b** (*tf.float32*)  A tensor of arbitrary shape filled with ones and
zeros.
- **b_hat** (*tf.float32*)  A tensor of the same shape as `b` filled with
ones and zeros.


Output

*tf.float64*  A scalar, the BER.


### compute_bler

`sionna.utils.``compute_bler`(*`b`*, *`b_hat`*)[`[source]`](../_modules/sionna/utils/metrics.html#compute_bler)

Computes the block error rate (BLER) between two binary tensors.

A block error happens if at least one element of `b` and `b_hat`
differ in one block. The BLER is evaluated over the last dimension of
the input, i. e., all elements of the last dimension are considered to
define a block.

This is also sometimes referred to as <cite>word error rate</cite> or <cite>frame error
rate</cite>.
Input

- **b** (*tf.float32*)  A tensor of arbitrary shape filled with ones and
zeros.
- **b_hat** (*tf.float32*)  A tensor of the same shape as `b` filled with
ones and zeros.


Output

*tf.float64*  A scalar, the BLER.


### compute_ser

`sionna.utils.``compute_ser`(*`s`*, *`s_hat`*)[`[source]`](../_modules/sionna/utils/metrics.html#compute_ser)

Computes the symbol error rate (SER) between two integer tensors.
Input

- **s** (*tf.int*)  A tensor of arbitrary shape filled with integers indicating
the symbol indices.
- **s_hat** (*tf.int*)  A tensor of the same shape as `s` filled with integers indicating
the estimated symbol indices.


Output

*tf.float64*  A scalar, the SER.


### count_errors

`sionna.utils.``count_errors`(*`b`*, *`b_hat`*)[`[source]`](../_modules/sionna/utils/metrics.html#count_errors)

Counts the number of bit errors between two binary tensors.
Input

- **b** (*tf.float32*)  A tensor of arbitrary shape filled with ones and
zeros.
- **b_hat** (*tf.float32*)  A tensor of the same shape as `b` filled with
ones and zeros.


Output

*tf.int64*  A scalar, the number of bit errors.


### count_block_errors

`sionna.utils.``count_block_errors`(*`b`*, *`b_hat`*)[`[source]`](../_modules/sionna/utils/metrics.html#count_block_errors)

Counts the number of block errors between two binary tensors.

A block error happens if at least one element of `b` and `b_hat`
differ in one block. The BLER is evaluated over the last dimension of
the input, i. e., all elements of the last dimension are considered to
define a block.

This is also sometimes referred to as <cite>word error rate</cite> or <cite>frame error
rate</cite>.
Input

- **b** (*tf.float32*)  A tensor of arbitrary shape filled with ones and
zeros.
- **b_hat** (*tf.float32*)  A tensor of the same shape as `b` filled with
ones and zeros.


Output

*tf.int64*  A scalar, the number of block errors.


## Tensors

### expand_to_rank

`sionna.utils.``expand_to_rank`(*`tensor`*, *`target_rank`*, *`axis``=``-` `1`*)[`[source]`](../_modules/sionna/utils/tensors.html#expand_to_rank)

Inserts as many axes to a tensor as needed to achieve a desired rank.

This operation inserts additional dimensions to a `tensor` starting at
`axis`, so that so that the rank of the resulting tensor has rank
`target_rank`. The dimension index follows Python indexing rules, i.e.,
zero-based, where a negative index is counted backward from the end.
Parameters

- **tensor**  A tensor.
- **target_rank** (*int*)  The rank of the output tensor.
If `target_rank` is smaller than the rank of `tensor`,
the function does nothing.
- **axis** (*int*)  The dimension index at which to expand the
shape of `tensor`. Given a `tensor` of <cite>D</cite> dimensions,
`axis` must be within the range <cite>[-(D+1), D]</cite> (inclusive).


Returns

A tensor with the same data as `tensor`, with
`target_rank`- rank(`tensor`) additional dimensions inserted at the
index specified by `axis`.
If `target_rank` <= rank(`tensor`), `tensor` is returned.


### flatten_dims

`sionna.utils.``flatten_dims`(*`tensor`*, *`num_dims`*, *`axis`*)[`[source]`](../_modules/sionna/utils/tensors.html#flatten_dims)

Flattens a specified set of dimensions of a tensor.

This operation flattens `num_dims` dimensions of a `tensor`
starting at a given `axis`.
Parameters

- **tensor**  A tensor.
- **num_dims** (*int*)  The number of dimensions
to combine. Must be larger than two and less or equal than the
rank of `tensor`.
- **axis** (*int*)  The index of the dimension from which to start.


Returns

A tensor of the same type as `tensor` with `num_dims`-1 lesser
dimensions, but the same number of elements.


### flatten_last_dims

`sionna.utils.``flatten_last_dims`(*`tensor`*, *`num_dims``=``2`*)[`[source]`](../_modules/sionna/utils/tensors.html#flatten_last_dims)

Flattens the last <cite>n</cite> dimensions of a tensor.

This operation flattens the last `num_dims` dimensions of a `tensor`.
It is a simplified version of the function `flatten_dims`.
Parameters

- **tensor**  A tensor.
- **num_dims** (*int*)  The number of dimensions
to combine. Must be greater than or equal to two and less or equal
than the rank of `tensor`.


Returns

A tensor of the same type as `tensor` with `num_dims`-1 lesser
dimensions, but the same number of elements.


### insert_dims

`sionna.utils.``insert_dims`(*`tensor`*, *`num_dims`*, *`axis``=``-` `1`*)[`[source]`](../_modules/sionna/utils/tensors.html#insert_dims)

Adds multiple length-one dimensions to a tensor.

This operation is an extension to TensorFlow`s `expand_dims` function.
It inserts `num_dims` dimensions of length one starting from the
dimension `axis` of a `tensor`. The dimension
index follows Python indexing rules, i.e., zero-based, where a negative
index is counted backward from the end.
Parameters

- **tensor**  A tensor.
- **num_dims** (*int*)  The number of dimensions to add.
- **axis**  The dimension index at which to expand the
shape of `tensor`. Given a `tensor` of <cite>D</cite> dimensions,
`axis` must be within the range <cite>[-(D+1), D]</cite> (inclusive).


Returns

A tensor with the same data as `tensor`, with `num_dims` additional
dimensions inserted at the index specified by `axis`.


### split_dims

`sionna.utils.``split_dim`(*`tensor`*, *`shape`*, *`axis`*)[`[source]`](../_modules/sionna/utils/tensors.html#split_dim)

Reshapes a dimension of a tensor into multiple dimensions.

This operation splits the dimension `axis` of a `tensor` into
multiple dimensions according to `shape`.
Parameters

- **tensor**  A tensor.
- **shape** (*list** or **TensorShape*)  The shape to which the dimension should
be reshaped.
- **axis** (*int*)  The index of the axis to be reshaped.


Returns

A tensor of the same type as `tensor` with len(`shape`)-1
additional dimensions, but the same number of elements.


### matrix_sqrt

`sionna.utils.``matrix_sqrt`(*`tensor`*)[`[source]`](../_modules/sionna/utils/tensors.html#matrix_sqrt)

Computes the square root of a matrix.

Given a batch of Hermitian positive semi-definite matrices
$\mathbf{A}$, returns matrices $\mathbf{B}$,
such that $\mathbf{B}\mathbf{B}^H = \mathbf{A}$.

The two inner dimensions are assumed to correspond to the matrix rows
and columns, respectively.
Parameters

**tensor** (*[**...**, **M**, **M**]*)  A tensor of rank greater than or equal
to two.

Returns

A tensor of the same shape and type as `tensor` containing
the matrix square root of its last two dimensions.


**Note**

If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.config.xla_compat=true`.
See `xla_compat`.

### matrix_sqrt_inv

`sionna.utils.``matrix_sqrt_inv`(*`tensor`*)[`[source]`](../_modules/sionna/utils/tensors.html#matrix_sqrt_inv)

Computes the inverse square root of a Hermitian matrix.

Given a batch of Hermitian positive definite matrices
$\mathbf{A}$, with square root matrices $\mathbf{B}$,
such that $\mathbf{B}\mathbf{B}^H = \mathbf{A}$, the function
returns $\mathbf{B}^{-1}$, such that
$\mathbf{B}^{-1}\mathbf{B}=\mathbf{I}$.

The two inner dimensions are assumed to correspond to the matrix rows
and columns, respectively.
Parameters

**tensor** (*[**...**, **M**, **M**]*)  A tensor of rank greater than or equal
to two.

Returns

A tensor of the same shape and type as `tensor` containing
the inverse matrix square root of its last two dimensions.


**Note**

If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See [`xla_compat`](config.html#sionna.Config.xla_compat).

### matrix_inv

`sionna.utils.``matrix_inv`(*`tensor`*)[`[source]`](../_modules/sionna/utils/tensors.html#matrix_inv)

Computes the inverse of a Hermitian matrix.

Given a batch of Hermitian positive definite matrices
$\mathbf{A}$, the function
returns $\mathbf{A}^{-1}$, such that
$\mathbf{A}^{-1}\mathbf{A}=\mathbf{I}$.

The two inner dimensions are assumed to correspond to the matrix rows
and columns, respectively.
Parameters

**tensor** (*[**...**, **M**, **M**]*)  A tensor of rank greater than or equal
to two.

Returns

A tensor of the same shape and type as `tensor`, containing
the inverse of its last two dimensions.


**Note**

If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See [`xla_compat`](config.html#sionna.Config.xla_compat).

### matrix_pinv

`sionna.utils.``matrix_pinv`(*`tensor`*)[`[source]`](../_modules/sionna/utils/tensors.html#matrix_pinv)

Computes the MoorePenrose (or pseudo) inverse of a matrix.

Given a batch of $M \times K$ matrices $\mathbf{A}$ with rank
$K$ (i.e., linearly independent columns), the function returns
$\mathbf{A}^+$, such that
$\mathbf{A}^{+}\mathbf{A}=\mathbf{I}_K$.

The two inner dimensions are assumed to correspond to the matrix rows
and columns, respectively.
Parameters

**tensor** (*[**...**, **M**, **K**]*)  A tensor of rank greater than or equal
to two.

Returns

A tensor of shape ([, K,K]) of the same type as `tensor`,
containing the pseudo inverse of its last two dimensions.


**Note**

If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.config.xla_compat=true`.
See `xla_compat`.

## Miscellaneous

### BinarySource

`class` `sionna.utils.``BinarySource`(*`dtype``=``tf.float32`*, *`seed``=``None`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/utils/misc.html#BinarySource)

Layer generating random binary tensors.
Parameters

- **dtype** (*tf.DType*)  Defines the output datatype of the layer.
Defaults to <cite>tf.float32</cite>.
- **seed** (*int** or **None*)  Set the seed for the random generator used to generate the bits.
Set to <cite>None</cite> for random initialization of the RNG.


Input

**shape** (*1D tensor/array/list, int*)  The desired shape of the output tensor.

Output

`shape`, `dtype`  Tensor filled with random binary values.


### SymbolSource

`class` `sionna.utils.``SymbolSource`(*`constellation_type``=``None`*, *`num_bits_per_symbol``=``None`*, *`constellation``=``None`*, *`return_indices``=``False`*, *`return_bits``=``False`*, *`seed``=``None`*, *`dtype``=``tf.complex64`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/utils/misc.html#SymbolSource)

Layer generating a tensor of arbitrary shape filled with random constellation symbols.
Optionally, the symbol indices and/or binary representations of the
constellation symbols can be returned.
Parameters

- **constellation_type** (*One of** [**"qam"**, **"pam"**, **"custom"**]**, **str*)  For custom, an instance of [`Constellation`](mapping.html#sionna.mapping.Constellation)
must be provided.
- **num_bits_per_symbol** (*int*)  The number of bits per constellation symbol.
Only required for `constellation_type` in [qam, pam].
- **constellation** ()  An instance of [`Constellation`](mapping.html#sionna.mapping.Constellation) or
<cite>None</cite>. In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **return_indices** (*bool*)  If enabled, the function also returns the symbol indices.
Defaults to <cite>False</cite>.
- **return_bits** (*bool*)  If enabled, the function also returns the binary symbol
representations (i.e., bit labels).
Defaults to <cite>False</cite>.
- **seed** (*int** or **None*)  The seed for the random generator.
<cite>None</cite> leads to a random initialization of the RNG.
Defaults to <cite>None</cite>.
- **dtype** (*One of** [**tf.complex64**, **tf.complex128**]**, **tf.DType*)  The output dtype. Defaults to tf.complex64.


Input

**shape** (*1D tensor/array/list, int*)  The desired shape of the output tensor.

Output

- **symbols** (`shape`, `dtype`)  Tensor filled with random symbols of the chosen `constellation_type`.
- **symbol_indices** (`shape`, tf.int32)  Tensor filled with the symbol indices.
Only returned if `return_indices` is <cite>True</cite>.
- **bits** ([`shape`, `num_bits_per_symbol`], tf.int32)  Tensor filled with the binary symbol representations (i.e., bit labels).
Only returned if `return_bits` is <cite>True</cite>.


### QAMSource

`class` `sionna.utils.``QAMSource`(*`num_bits_per_symbol``=``None`*, *`return_indices``=``False`*, *`return_bits``=``False`*, *`seed``=``None`*, *`dtype``=``tf.complex64`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/utils/misc.html#QAMSource)

Layer generating a tensor of arbitrary shape filled with random QAM symbols.
Optionally, the symbol indices and/or binary representations of the
constellation symbols can be returned.
Parameters

- **num_bits_per_symbol** (*int*)  The number of bits per constellation symbol, e.g., 4 for QAM16.
- **return_indices** (*bool*)  If enabled, the function also returns the symbol indices.
Defaults to <cite>False</cite>.
- **return_bits** (*bool*)  If enabled, the function also returns the binary symbol
representations (i.e., bit labels).
Defaults to <cite>False</cite>.
- **seed** (*int** or **None*)  The seed for the random generator.
<cite>None</cite> leads to a random initialization of the RNG.
Defaults to <cite>None</cite>.
- **dtype** (*One of** [**tf.complex64**, **tf.complex128**]**, **tf.DType*)  The output dtype. Defaults to tf.complex64.


Input

**shape** (*1D tensor/array/list, int*)  The desired shape of the output tensor.

Output

- **symbols** (`shape`, `dtype`)  Tensor filled with random QAM symbols.
- **symbol_indices** (`shape`, tf.int32)  Tensor filled with the symbol indices.
Only returned if `return_indices` is <cite>True</cite>.
- **bits** ([`shape`, `num_bits_per_symbol`], tf.int32)  Tensor filled with the binary symbol representations (i.e., bit labels).
Only returned if `return_bits` is <cite>True</cite>.


### PAMSource

`class` `sionna.utils.``PAMSource`(*`num_bits_per_symbol``=``None`*, *`return_indices``=``False`*, *`return_bits``=``False`*, *`seed``=``None`*, *`dtype``=``tf.complex64`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/utils/misc.html#PAMSource)

Layer generating a tensor of arbitrary shape filled with random PAM symbols.
Optionally, the symbol indices and/or binary representations of the
constellation symbols can be returned.
Parameters

- **num_bits_per_symbol** (*int*)  The number of bits per constellation symbol, e.g., 1 for BPSK.
- **return_indices** (*bool*)  If enabled, the function also returns the symbol indices.
Defaults to <cite>False</cite>.
- **return_bits** (*bool*)  If enabled, the function also returns the binary symbol
representations (i.e., bit labels).
Defaults to <cite>False</cite>.
- **seed** (*int** or **None*)  The seed for the random generator.
<cite>None</cite> leads to a random initialization of the RNG.
Defaults to <cite>None</cite>.
- **dtype** (*One of** [**tf.complex64**, **tf.complex128**]**, **tf.DType*)  The output dtype. Defaults to tf.complex64.


Input

**shape** (*1D tensor/array/list, int*)  The desired shape of the output tensor.

Output

- **symbols** (`shape`, `dtype`)  Tensor filled with random PAM symbols.
- **symbol_indices** (`shape`, tf.int32)  Tensor filled with the symbol indices.
Only returned if `return_indices` is <cite>True</cite>.
- **bits** ([`shape`, `num_bits_per_symbol`], tf.int32)  Tensor filled with the binary symbol representations (i.e., bit labels).
Only returned if `return_bits` is <cite>True</cite>.


### PlotBER

`class` `sionna.utils.plotting.``PlotBER`(*`title``=``'Bit/Block` `Error` `Rate'`*)[`[source]`](../_modules/sionna/utils/plotting.html#PlotBER)

Provides a plotting object to simulate and store BER/BLER curves.
Parameters

**title** (*str*)  A string defining the title of the figure. Defaults to
<cite>Bit/Block Error Rate</cite>.

Input

- **snr_db** (*float*)  Python array (or list of Python arrays) of additional SNR values to be
plotted.
- **ber** (*float*)  Python array (or list of Python arrays) of additional BERs
corresponding to `snr_db`.
- **legend** (*str*)  String (or list of strings) of legends entries.
- **is_bler** (*bool*)  A boolean (or list of booleans) defaults to False.
If True, `ber` will be interpreted as BLER.
- **show_ber** (*bool*)  A boolean defaults to True. If True, BER curves will be plotted.
- **show_bler** (*bool*)  A boolean defaults to True. If True, BLER curves will be plotted.
- **xlim** (*tuple of floats*)  Defaults to None. A tuple of two floats defining x-axis limits.
- **ylim** (*tuple of floats*)  Defaults to None. A tuple of two floats defining y-axis limits.
- **save_fig** (*bool*)  A boolean defaults to False. If True, the figure
is saved as file.
- **path** (*str*)  A string defining where to save the figure (if `save_fig`
is True).


`add`(*`ebno_db`*, *`ber`*, *`is_bler``=``False`*, *`legend``=``''`*)[`[source]`](../_modules/sionna/utils/plotting.html#PlotBER.add)

Add static reference curves.
Input

- **ebno_db** (*float*)  Python array or list of floats defining the SNR points.
- **ber** (*float*)  Python array or list of floats defining the BER corresponding
to each SNR point.
- **is_bler** (*bool*)  A boolean defaults to False. If True, `ber` is interpreted as
BLER.
- **legend** (*str*)  A string defining the text of the legend entry.


`property` `ber`

List containing all stored BER curves.


`property` `is_bler`

List of booleans indicating if ber shall be interpreted as BLER.


`property` `legend`

List containing all stored legend entries curves.


`remove`(*`idx``=``-` `1`*)[`[source]`](../_modules/sionna/utils/plotting.html#PlotBER.remove)

Remove curve with index `idx`.
Input

**idx** (*int*)  An integer defining the index of the dataset that should
be removed. Negative indexing is possible.


`reset`()[`[source]`](../_modules/sionna/utils/plotting.html#PlotBER.reset)

Remove all internal data.


`simulate`(*`mc_fun`*, *`ebno_dbs`*, *`batch_size`*, *`max_mc_iter`*, *`legend``=``''`*, *`add_ber``=``True`*, *`add_bler``=``False`*, *`soft_estimates``=``False`*, *`num_target_bit_errors``=``None`*, *`num_target_block_errors``=``None`*, *`target_ber``=``None`*, *`target_bler``=``None`*, *`early_stop``=``True`*, *`graph_mode``=``None`*, *`distribute``=``None`*, *`add_results``=``True`*, *`forward_keyboard_interrupt``=``True`*, *`show_fig``=``True`*, *`verbose``=``True`*)[`[source]`](../_modules/sionna/utils/plotting.html#PlotBER.simulate)

Simulate BER/BLER curves for given Keras model and saves the results.

Internally calls [`sionna.utils.sim_ber`](https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.sim_ber).
Input

- **mc_fun**  Callable that yields the transmitted bits <cite>b</cite> and the
receivers estimate <cite>b_hat</cite> for a given `batch_size` and
`ebno_db`. If `soft_estimates` is True, b_hat is interpreted as
logit.
- **ebno_dbs** (*ndarray of floats*)  SNR points to be evaluated.
- **batch_size** (*tf.int32*)  Batch-size for evaluation.
- **max_mc_iter** (*int*)  Max. number of Monte-Carlo iterations per SNR point.
- **legend** (*str*)  Name to appear in legend.
- **add_ber** (*bool*)  Defaults to True. Indicate if BER should be added to plot.
- **add_bler** (*bool*)  Defaults to False. Indicate if BLER should be added
to plot.
- **soft_estimates** (*bool*)  A boolean, defaults to False. If True, `b_hat`
is interpreted as logit and additional hard-decision is applied
internally.
- **num_target_bit_errors** (*int*)  Target number of bit errors per SNR point until the simulation
stops.
- **num_target_block_errors** (*int*)  Target number of block errors per SNR point until the simulation
stops.
- **target_ber** (*tf.float32*)  Defaults to <cite>None</cite>. The simulation stops after the first SNR point
which achieves a lower bit error rate as specified by
`target_ber`. This requires `early_stop` to be <cite>True</cite>.
- **target_bler** (*tf.float32*)  Defaults to <cite>None</cite>. The simulation stops after the first SNR point
which achieves a lower block error rate as specified by
`target_bler`.  This requires `early_stop` to be <cite>True</cite>.
- **early_stop** (*bool*)  A boolean defaults to True. If True, the simulation stops after the
first error-free SNR point (i.e., no error occurred after
`max_mc_iter` Monte-Carlo iterations).
- **graph_mode** (*One of [graph, xla], str*)  A string describing the execution mode of `mc_fun`.
Defaults to <cite>None</cite>. In this case, `mc_fun` is executed as is.
- **distribute** (<cite>None</cite> (default) | all | list of indices | <cite>tf.distribute.strategy</cite>)  Distributes simulation on multiple parallel devices. If <cite>None</cite>,
multi-device simulations are deactivated. If all, the workload
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
- **add_results** (*bool*)  Defaults to True. If True, the simulation results will be appended
to the internal list of results.
- **show_fig** (*bool*)  Defaults to True. If True, a BER figure will be plotted.
- **verbose** (*bool*)  A boolean defaults to True. If True, the current progress will be
printed.
- **forward_keyboard_interrupt** (*bool*)  A boolean defaults to True. If False, <cite>KeyboardInterrupts</cite> will be
catched internally and not forwarded (e.g., will not stop outer
loops). If False, the simulation ends and returns the intermediate
simulation results.


Output

- **(ber, bler)**  Tuple:
- **ber** (*float*)  The simulated bit-error rate.
- **bler** (*float*)  The simulated block-error rate.


`property` `snr`

List containing all stored SNR curves.


`property` `title`

Title of the plot.


### sim_ber

`sionna.utils.``sim_ber`(*`mc_fun`*, *`ebno_dbs`*, *`batch_size`*, *`max_mc_iter`*, *`soft_estimates``=``False`*, *`num_target_bit_errors``=``None`*, *`num_target_block_errors``=``None`*, *`target_ber``=``None`*, *`target_bler``=``None`*, *`early_stop``=``True`*, *`graph_mode``=``None`*, *`distribute``=``None`*, *`verbose``=``True`*, *`forward_keyboard_interrupt``=``True`*, *`callback``=``None`*, *`dtype``=``tf.complex64`*)[`[source]`](../_modules/sionna/utils/misc.html#sim_ber)

Simulates until target number of errors is reached and returns BER/BLER.

The simulation continues with the next SNR point if either
`num_target_bit_errors` bit errors or `num_target_block_errors` block
errors is achieved. Further, it continues with the next SNR point after
`max_mc_iter` batches of size `batch_size` have been simulated.
Early stopping allows to stop the simulation after the first error-free SNR
point or after reaching a certain `target_ber` or `target_bler`.
Input

- **mc_fun** (*callable*)  Callable that yields the transmitted bits <cite>b</cite> and the
receivers estimate <cite>b_hat</cite> for a given `batch_size` and
`ebno_db`. If `soft_estimates` is True, <cite>b_hat</cite> is interpreted as
logit.
- **ebno_dbs** (*tf.float32*)  A tensor containing SNR points to be evaluated.
- **batch_size** (*tf.int32*)  Batch-size for evaluation.
- **max_mc_iter** (*tf.int32*)  Maximum number of Monte-Carlo iterations per SNR point.
- **soft_estimates** (*bool*)  A boolean, defaults to <cite>False</cite>. If <cite>True</cite>, <cite>b_hat</cite>
is interpreted as logit and an additional hard-decision is applied
internally.
- **num_target_bit_errors** (*tf.int32*)  Defaults to <cite>None</cite>. Target number of bit errors per SNR point until
the simulation continues to next SNR point.
- **num_target_block_errors** (*tf.int32*)  Defaults to <cite>None</cite>. Target number of block errors per SNR point
until the simulation continues
- **target_ber** (*tf.float32*)  Defaults to <cite>None</cite>. The simulation stops after the first SNR point
which achieves a lower bit error rate as specified by `target_ber`.
This requires `early_stop` to be <cite>True</cite>.
- **target_bler** (*tf.float32*)  Defaults to <cite>None</cite>. The simulation stops after the first SNR point
which achieves a lower block error rate as specified by `target_bler`.
This requires `early_stop` to be <cite>True</cite>.
- **early_stop** (*bool*)  A boolean defaults to <cite>True</cite>. If <cite>True</cite>, the simulation stops after the
first error-free SNR point (i.e., no error occurred after
`max_mc_iter` Monte-Carlo iterations).
- **graph_mode** (*One of [graph, xla], str*)  A string describing the execution mode of `mc_fun`.
Defaults to <cite>None</cite>. In this case, `mc_fun` is executed as is.
- **distribute** (<cite>None</cite> (default) | all | list of indices | <cite>tf.distribute.strategy</cite>)  Distributes simulation on multiple parallel devices. If <cite>None</cite>,
multi-device simulations are deactivated. If all, the workload will
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
- **verbose** (*bool*)  A boolean defaults to <cite>True</cite>. If <cite>True</cite>, the current progress will be
printed.
- **forward_keyboard_interrupt** (*bool*)  A boolean defaults to <cite>True</cite>. If <cite>False</cite>, KeyboardInterrupts will be
catched internally and not forwarded (e.g., will not stop outer loops).
If <cite>False</cite>, the simulation ends and returns the intermediate simulation
results.
- **callback** (<cite>None</cite> (default) | callable)  If specified, `callback` will be called after each Monte-Carlo step.
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
- **dtype** (*tf.complex64*)  Datatype of the callable `mc_fun` to be used as input/output.


Output

- **(ber, bler)**  Tuple:
- **ber** (*tf.float32*)  The bit-error rate.
- **bler** (*tf.float32*)  The block-error rate.


Raises

- **AssertionError**  If `soft_estimates` is not bool.
- **AssertionError**  If `dtype` is not <cite>tf.complex</cite>.


**Note**

This function is implemented based on tensors to allow
full compatibility with tf.function(). However, to run simulations
in graph mode, the provided `mc_fun` must use the <cite>@tf.function()</cite>
decorator.

### ebnodb2no

`sionna.utils.``ebnodb2no`(*`ebno_db`*, *`num_bits_per_symbol`*, *`coderate`*, *`resource_grid``=``None`*)[`[source]`](../_modules/sionna/utils/misc.html#ebnodb2no)

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

- **ebno_db** (*float*)  The <cite>Eb/No</cite> value in dB.
- **num_bits_per_symbol** (*int*)  The number of bits per symbol.
- **coderate** (*float*)  The coderate used.
- **resource_grid** (*ResourceGrid*)  An (optional) instance of [`ResourceGrid`](ofdm.html#sionna.ofdm.ResourceGrid)
for OFDM transmissions.


Output

*float*  The value of $N_o$ in linear scale.


### hard_decisions

`sionna.utils.``hard_decisions`(*`llr`*)[`[source]`](../_modules/sionna/utils/misc.html#hard_decisions)

Transforms LLRs into hard decisions.

Positive values are mapped to $1$.
Nonpositive values are mapped to $0$.
Input

**llr** (*any non-complex tf.DType*)  Tensor of LLRs.

Output

Same shape and dtype as `llr`  The hard decisions.


### plot_ber

`sionna.utils.plotting.``plot_ber`(*`snr_db`*, *`ber`*, *`legend``=``''`*, *`ylabel``=``'BER'`*, *`title``=``'Bit` `Error` `Rate'`*, *`ebno``=``True`*, *`is_bler``=``None`*, *`xlim``=``None`*, *`ylim``=``None`*, *`save_fig``=``False`*, *`path``=``''`*)[`[source]`](../_modules/sionna/utils/plotting.html#plot_ber)

Plot error-rates.
Input

- **snr_db** (*ndarray*)  Array of floats defining the simulated SNR points.
Can be also a list of multiple arrays.
- **ber** (*ndarray*)  Array of floats defining the BER/BLER per SNR point.
Can be also a list of multiple arrays.
- **legend** (*str*)  Defaults to . Defining the legend entries. Can be
either a string or a list of strings.
- **ylabel** (*str*)  Defaults to BER. Defining the y-label.
- **title** (*str*)  Defaults to Bit Error Rate. Defining the title of the figure.
- **ebno** (*bool*)  Defaults to True. If True, the x-label is set to
EbNo [dB] instead of EsNo [dB].
- **is_bler** (*bool*)  Defaults to False. If True, the corresponding curve is dashed.
- **xlim** (*tuple of floats*)  Defaults to None. A tuple of two floats defining x-axis limits.
- **ylim** (*tuple of floats*)  Defaults to None. A tuple of two floats defining y-axis limits.
- **save_fig** (*bool*)  Defaults to False. If True, the figure is saved as <cite>.png</cite>.
- **path** (*str*)  Defaults to . Defining the path to save the figure
(iff `save_fig` is True).


Output

- **(fig, ax)**  Tuple:
- **fig** (*matplotlib.figure.Figure*)  A matplotlib figure handle.
- **ax** (*matplotlib.axes.Axes*)  A matplotlib axes object.


### complex_normal

`sionna.utils.``complex_normal`(*`shape`*, *`var``=``1.0`*, *`dtype``=``tf.complex64`*)[`[source]`](../_modules/sionna/utils/misc.html#complex_normal)

Generates a tensor of complex normal random variables.
Input

- **shape** (*tf.shape, or list*)  The desired shape.
- **var** (*float*)  The total variance., i.e., each complex dimension has
variance `var/2`.
- **dtype** (*tf.complex*)  The desired dtype. Defaults to <cite>tf.complex64</cite>.


Output

`shape`, `dtype`  Tensor of complex normal random variables.


### log2

`sionna.utils.``log2`(*`x`*)[`[source]`](../_modules/sionna/utils/misc.html#log2)

TensorFlow implementation of NumPys <cite>log2</cite> function.

Simple extension to <cite>tf.experimental.numpy.log2</cite>
which casts the result to the <cite>dtype</cite> of the input.
For more details see the [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/experimental/numpy/log2) and [NumPy](https://numpy.org/doc/1.16/reference/generated/numpy.log2.html) documentation.

### log10

`sionna.utils.``log10`(*`x`*)[`[source]`](../_modules/sionna/utils/misc.html#log10)

TensorFlow implementation of NumPys <cite>log10</cite> function.

Simple extension to <cite>tf.experimental.numpy.log10</cite>
which casts the result to the <cite>dtype</cite> of the input.
For more details see the [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/experimental/numpy/log10) and [NumPy](https://numpy.org/doc/1.16/reference/generated/numpy.log10.html) documentation.
For more details see the [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/experimental/numpy/log10) and [NumPy](https://numpy.org/doc/1.16/reference/generated/numpy.log10.html) documentation.
