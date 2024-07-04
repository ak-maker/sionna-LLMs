# Mapping

This module contains classes and functions related to mapping
of bits to constellation symbols and demapping of soft-symbols
to log-likelihood ratios (LLRs). The key components are the
[`Constellation`](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation), [`Mapper`](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Mapper),
and [`Demapper`](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Demapper). A [`Constellation`](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation)
can be made trainable to enable learning of geometric shaping.

## Constellations

### Constellation

`class` `sionna.mapping.``Constellation`(*`constellation_type`*, *`num_bits_per_symbol`*, *`initial_value``=``None`*, *`normalize``=``True`*, *`center``=``False`*, *`trainable``=``False`*, *`dtype``=``tf.complex64`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/mapping.html#Constellation)

Constellation that can be used by a (de)mapper.

This class defines a constellation, i.e., a complex-valued vector of
constellation points. A constellation can be trainable. The binary
representation of the index of an element of this vector corresponds
to the bit label of the constellation point. This implicit bit
labeling is used by the `Mapper` and `Demapper` classes.
Parameters

- **constellation_type** (*One of** [**"qam"**, **"pam"**, **"custom"**]**, **str*)  For custom, the constellation points are randomly initialized
if no `initial_value` is provided.
- **num_bits_per_symbol** (*int*)  The number of bits per constellation symbol, e.g., 4 for QAM16.
- **initial_value** ($[2^\text{num_bits_per_symbol}]$, NumPy array or Tensor)  Initial values of the constellation points. If `normalize` or
`center` are <cite>True</cite>, the initial constellation might be changed.
- **normalize** (*bool*)  If <cite>True</cite>, the constellation is normalized to have unit power.
Defaults to <cite>True</cite>.
- **center** (*bool*)  If <cite>True</cite>, the constellation is ensured to have zero mean.
Defaults to <cite>False</cite>.
- **trainable** (*bool*)  If <cite>True</cite>, the constellation points are trainable variables.
Defaults to <cite>False</cite>.
- **dtype** (*[**tf.complex64**, **tf.complex128**]**, **tf.DType*)  The dtype of the constellation.


Output

$[2^\text{num_bits_per_symbol}]$, `dtype`  The constellation.


**Note**

One can create a trainable PAM/QAM constellation. This is
equivalent to creating a custom trainable constellation which is
initialized with PAM/QAM constellation points.

`property` `center`

Indicates if the constellation is centered.


`create_or_check_constellation`(*`num_bits_per_symbol``=``None`*, *`constellation``=``None`*, *`dtype``=``tf.complex64`*)[`[source]`](../_modules/sionna/mapping.html#Constellation.create_or_check_constellation)

Static method for conviently creating a constellation object or checking that an existing one
is consistent with requested settings.

If `constellation` is <cite>None</cite>, then this method creates a [`Constellation`](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation)
object of type `constellation_type` and with `num_bits_per_symbol` bits per symbol.
Otherwise, this method checks that <cite>constellation</cite> is consistent with `constellation_type` and
`num_bits_per_symbol`. If it is, `constellation` is returned. Otherwise, an assertion is raised.
Input

- **constellation_type** (*One of [qam, pam, custom], str*)  For custom, an instance of [`Constellation`](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation)
must be provided.
- **num_bits_per_symbol** (*int*)  The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [qam, pam].
- **constellation** (*Constellation*)  An instance of [`Constellation`](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation) or
<cite>None</cite>. In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.


Output

[`Constellation`](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation)  A constellation object.


`property` `normalize`

Indicates if the constellation is normalized or not.


`property` `num_bits_per_symbol`

The number of bits per constellation symbol.


`property` `points`

The (possibly) centered and normalized constellation points.


`show`(*`labels``=``True`*, *`figsize``=``(7,` `7)`*)[`[source]`](../_modules/sionna/mapping.html#Constellation.show)

Generate a scatter-plot of the constellation.
Input

- **labels** (*bool*)  If <cite>True</cite>, the bit labels will be drawn next to each constellation
point. Defaults to <cite>True</cite>.
- **figsize** (*Two-element Tuple, float*)  Width and height in inches. Defaults to <cite>(7,7)</cite>.


Output

*matplotlib.figure.Figure*  A handle to a matplot figure object.


### qam

`sionna.mapping.``qam`(*`num_bits_per_symbol`*, *`normalize``=``True`*)[`[source]`](../_modules/sionna/mapping.html#qam)

Generates a QAM constellation.

This function generates a complex-valued vector, where each element is
a constellation point of an M-ary QAM constellation. The bit
label of the `n` th point is given by the length-`num_bits_per_symbol`
binary represenation of `n`.
Input

- **num_bits_per_symbol** (*int*)  The number of bits per constellation point.
Must be a multiple of two, e.g., 2, 4, 6, 8, etc.
- **normalize** (*bool*)  If <cite>True</cite>, the constellation is normalized to have unit power.
Defaults to <cite>True</cite>.


Output

$[2^{\text{num_bits_per_symbol}}]$, np.complex64  The QAM constellation.


**Note**

The bit label of the nth constellation point is given by the binary
representation of its position within the array and can be obtained
through `np.binary_repr(n,` `num_bits_per_symbol)`.

The normalization factor of a QAM constellation is given in
closed-form as:

$$
\sqrt{\frac{1}{2^{n-2}}\sum_{i=1}^{2^{n-1}}(2i-1)^2}
$$

where $n= \text{num_bits_per_symbol}/2$ is the number of bits
per dimension.

This algorithm is a recursive implementation of the expressions found in
Section 5.1 of [[3GPPTS38211]](https://nvlabs.github.io/sionna/api/mapping.html#gppts38211). It is used in the 5G standard.

### pam

`sionna.mapping.``pam`(*`num_bits_per_symbol`*, *`normalize``=``True`*)[`[source]`](../_modules/sionna/mapping.html#pam)

Generates a PAM constellation.

This function generates a real-valued vector, where each element is
a constellation point of an M-ary PAM constellation. The bit
label of the `n` th point is given by the length-`num_bits_per_symbol`
binary represenation of `n`.
Input

- **num_bits_per_symbol** (*int*)  The number of bits per constellation point.
Must be positive.
- **normalize** (*bool*)  If <cite>True</cite>, the constellation is normalized to have unit power.
Defaults to <cite>True</cite>.


Output

$[2^{\text{num_bits_per_symbol}}]$, np.float32  The PAM constellation.


**Note**

The bit label of the nth constellation point is given by the binary
representation of its position within the array and can be obtained
through `np.binary_repr(n,` `num_bits_per_symbol)`.

The normalization factor of a PAM constellation is given in
closed-form as:

$$
\sqrt{\frac{1}{2^{n-1}}\sum_{i=1}^{2^{n-1}}(2i-1)^2}
$$

where $n= \text{num_bits_per_symbol}$ is the number of bits
per symbol.

This algorithm is a recursive implementation of the expressions found in
Section 5.1 of [[3GPPTS38211]](https://nvlabs.github.io/sionna/api/mapping.html#gppts38211). It is used in the 5G standard.

### pam_gray

`sionna.mapping.``pam_gray`(*`b`*)[`[source]`](../_modules/sionna/mapping.html#pam_gray)

Maps a vector of bits to a PAM constellation points with Gray labeling.

This recursive function maps a binary vector to Gray-labelled PAM
constellation points. It can be used to generated QAM constellations.
The constellation is not normalized.
Input

**b** (*[n], NumPy array*)  Tensor with with binary entries.

Output

*signed int*  The PAM constellation point taking values in
$\{\pm 1,\pm 3,\dots,\pm (2^n-1)\}$.


**Note**

This algorithm is a recursive implementation of the expressions found in
Section 5.1 of [[3GPPTS38211]](https://nvlabs.github.io/sionna/api/mapping.html#gppts38211). It is used in the 5G standard.

## Mapper

`class` `sionna.mapping.``Mapper`(*`constellation_type``=``None`*, *`num_bits_per_symbol``=``None`*, *`constellation``=``None`*, *`return_indices``=``False`*, *`dtype``=``tf.complex64`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/mapping.html#Mapper)

Maps binary tensors to points of a constellation.

This class defines a layer that maps a tensor of binary values
to a tensor of points from a provided constellation.
Parameters

- **constellation_type** (*One of** [**"qam"**, **"pam"**, **"custom"**]**, **str*)  For custom, an instance of [`Constellation`](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation)
must be provided.
- **num_bits_per_symbol** (*int*)  The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [qam, pam].
- **constellation** ()  An instance of [`Constellation`](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation) or
<cite>None</cite>. In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **return_indices** (*bool*)  If enabled, symbol indices are additionally returned.
Defaults to <cite>False</cite>.
- **dtype** (*One of** [**tf.complex64**, **tf.complex128**]**, **tf.DType*)  The output dtype. Defaults to tf.complex64.


Input

*[, n], tf.float or tf.int*  Tensor with with binary entries.

Output

- *[,n/Constellation.num_bits_per_symbol], tf.complex*  The mapped constellation symbols.
- *[,n/Constellation.num_bits_per_symbol], tf.int32*  The symbol indices corresponding to the constellation symbols.
Only returned if `return_indices` is set to True.


**Note**

The last input dimension must be an integer multiple of the
number of bits per constellation symbol.

`property` `constellation`

The Constellation used by the Mapper.


## Demapping

### Demapper

`class` `sionna.mapping.``Demapper`(*`demapping_method`*, *`constellation_type``=``None`*, *`num_bits_per_symbol``=``None`*, *`constellation``=``None`*, *`hard_out``=``False`*, *`with_prior``=``False`*, *`dtype``=``tf.complex64`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/mapping.html#Demapper)

Computes log-likelihood ratios (LLRs) or hard-decisions on bits
for a tensor of received symbols.
If the flag `with_prior` is set, prior knowledge on the bits is assumed to be available.

This class defines a layer implementing different demapping
functions. All demapping functions are fully differentiable when soft-decisions
are computed.
Parameters

- **demapping_method** (*One of** [**"app"**, **"maxlog"**]**, **str*)  The demapping method used.
- **constellation_type** (*One of** [**"qam"**, **"pam"**, **"custom"**]**, **str*)  For custom, an instance of [`Constellation`](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation)
must be provided.
- **num_bits_per_symbol** (*int*)  The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [qam, pam].
- **constellation** ()  An instance of [`Constellation`](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation) or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (*bool*)  If <cite>True</cite>, the demapper provides hard-decided bits instead of soft-values.
Defaults to <cite>False</cite>.
- **with_prior** (*bool*)  If <cite>True</cite>, it is assumed that prior knowledge on the bits is available.
This prior information is given as LLRs as an additional input to the layer.
Defaults to <cite>False</cite>.
- **dtype** (*One of** [**tf.complex64**, **tf.complex128**] **tf.DType** (**dtype**)*)  The dtype of <cite>y</cite>. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input

- **(y,no) or (y, prior, no)**  Tuple:
- **y** (*[,n], tf.complex*)  The received symbols.
- **prior** (*[num_bits_per_symbol] or [,num_bits_per_symbol], tf.float*)  Prior for every bit as LLRs.
It can be provided either as a tensor of shape <cite>[num_bits_per_symbol]</cite> for the
entire input batch, or as a tensor that is broadcastable
to <cite>[, n, num_bits_per_symbol]</cite>.
Only required if the `with_prior` flag is set.
- **no** (*Scalar or [,n], tf.float*)  The noise variance estimate. It can be provided either as scalar
for the entire input batch or as a tensor that is broadcastable to
`y`.


Output

*[,n*num_bits_per_symbol], tf.float*  LLRs or hard-decisions for every bit.


**Note**

With the app demapping method, the LLR for the $i\text{th}$ bit
is computed according to

$$
LLR(i) = \ln\left(\frac{\Pr\left(b_i=1\lvert y,\mathbf{p}\right)}{\Pr\left(b_i=0\lvert y,\mathbf{p}\right)}\right) =\ln\left(\frac{
        \sum_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
        \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
        }{
        \sum_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
        \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
        }\right)
$$

where $\mathcal{C}_{i,1}$ and $\mathcal{C}_{i,0}$ are the
sets of constellation points for which the $i\text{th}$ bit is
equal to 1 and 0, respectively. $\mathbf{p} = \left[p_0,\dots,p_{K-1}\right]$
is the vector of LLRs that serves as prior knowledge on the $K$ bits that are mapped to
a constellation point and is set to $\mathbf{0}$ if no prior knowledge is assumed to be available,
and $\Pr(c\lvert\mathbf{p})$ is the prior probability on the constellation symbol $c$:

$$
\Pr\left(c\lvert\mathbf{p}\right) = \prod_{k=0}^{K-1} \text{sigmoid}\left(p_k \ell(c)_k\right)
$$

where $\ell(c)_k$ is the $k^{th}$ bit label of $c$, where 0 is
replaced by -1.
The definition of the LLR has been
chosen such that it is equivalent with that of logits. This is
different from many textbooks in communications, where the LLR is
defined as $LLR(i) = \ln\left(\frac{\Pr\left(b_i=0\lvert y\right)}{\Pr\left(b_i=1\lvert y\right)}\right)$.

With the maxlog demapping method, LLRs for the $i\text{th}$ bit
are approximated like

$$
\begin{split}\begin{align}
    LLR(i) &\approx\ln\left(\frac{
        \max_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
            \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
        }{
        \max_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
            \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
        }\right)\\
        &= \max_{c\in\mathcal{C}_{i,0}}
            \left(\ln\left(\Pr\left(c\lvert\mathbf{p}\right)\right)-\frac{|y-c|^2}{N_o}\right) -
         \max_{c\in\mathcal{C}_{i,1}}\left( \ln\left(\Pr\left(c\lvert\mathbf{p}\right)\right) - \frac{|y-c|^2}{N_o}\right)
        .
\end{align}\end{split}
$$


### DemapperWithPrior

`class` `sionna.mapping.``DemapperWithPrior`(*`demapping_method`*, *`constellation_type``=``None`*, *`num_bits_per_symbol``=``None`*, *`constellation``=``None`*, *`hard_out``=``False`*, *`dtype``=``tf.complex64`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/mapping.html#DemapperWithPrior)

Computes log-likelihood ratios (LLRs) or hard-decisions on bits
for a tensor of received symbols, assuming that prior knowledge on the bits is available.

This class defines a layer implementing different demapping
functions. All demapping functions are fully differentiable when soft-decisions
are computed.

This class is deprecated as the functionality has been integrated
into [`Demapper`](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Demapper).
Parameters

- **demapping_method** (*One of** [**"app"**, **"maxlog"**]**, **str*)  The demapping method used.
- **constellation_type** (*One of** [**"qam"**, **"pam"**, **"custom"**]**, **str*)  For custom, an instance of [`Constellation`](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation)
must be provided.
- **num_bits_per_symbol** (*int*)  The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [qam, pam].
- **constellation** ()  An instance of [`Constellation`](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation) or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (*bool*)  If <cite>True</cite>, the demapper provides hard-decided bits instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (*One of** [**tf.complex64**, **tf.complex128**] **tf.DType** (**dtype**)*)  The dtype of <cite>y</cite>. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input

- **(y, prior, no)**  Tuple:
- **y** (*[,n], tf.complex*)  The received symbols.
- **prior** (*[num_bits_per_symbol] or [,num_bits_per_symbol], tf.float*)  Prior for every bit as LLRs.
It can be provided either as a tensor of shape <cite>[num_bits_per_symbol]</cite> for the
entire input batch, or as a tensor that is broadcastable
to <cite>[, n, num_bits_per_symbol]</cite>.
- **no** (*Scalar or [,n], tf.float*)  The noise variance estimate. It can be provided either as scalar
for the entire input batch or as a tensor that is broadcastable to
`y`.


Output

*[,n*num_bits_per_symbol], tf.float*  LLRs or hard-decisions for every bit.


**Note**

With the app demapping method, the LLR for the $i\text{th}$ bit
is computed according to

$$
LLR(i) = \ln\left(\frac{\Pr\left(b_i=1\lvert y,\mathbf{p}\right)}{\Pr\left(b_i=0\lvert y,\mathbf{p}\right)}\right) =\ln\left(\frac{
        \sum_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
        \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
        }{
        \sum_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
        \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
        }\right)
$$

where $\mathcal{C}_{i,1}$ and $\mathcal{C}_{i,0}$ are the
sets of constellation points for which the $i\text{th}$ bit is
equal to 1 and 0, respectively. $\mathbf{p} = \left[p_0,\dots,p_{K-1}\right]$
is the vector of LLRs that serves as prior knowledge on the $K$ bits that are mapped to
a constellation point,
and $\Pr(c\lvert\mathbf{p})$ is the prior probability on the constellation symbol $c$:

$$
\Pr\left(c\lvert\mathbf{p}\right) = \prod_{k=0}^{K-1} \text{sigmoid}\left(p_k \ell(c)_k\right)
$$

where $\ell(c)_k$ is the $k^{th}$ bit label of $c$, where 0 is
replaced by -1.
The definition of the LLR has been
chosen such that it is equivalent with that of logits. This is
different from many textbooks in communications, where the LLR is
defined as $LLR(i) = \ln\left(\frac{\Pr\left(b_i=0\lvert y\right)}{\Pr\left(b_i=1\lvert y\right)}\right)$.

With the maxlog demapping method, LLRs for the $i\text{th}$ bit
are approximated like

$$
\begin{split}\begin{align}
    LLR(i) &\approx\ln\left(\frac{
        \max_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
            \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
        }{
        \max_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
            \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
        }\right)\\
        &= \max_{c\in\mathcal{C}_{i,0}}
            \left(\ln\left(\Pr\left(c\lvert\mathbf{p}\right)\right)-\frac{|y-c|^2}{N_o}\right) -
         \max_{c\in\mathcal{C}_{i,1}}\left( \ln\left(\Pr\left(c\lvert\mathbf{p}\right)\right) - \frac{|y-c|^2}{N_o}\right)
        .
\end{align}\end{split}
$$


### SymbolDemapper

`class` `sionna.mapping.``SymbolDemapper`(*`constellation_type``=``None`*, *`num_bits_per_symbol``=``None`*, *`constellation``=``None`*, *`hard_out``=``False`*, *`with_prior``=``False`*, *`dtype``=``tf.complex64`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/mapping.html#SymbolDemapper)

Computes normalized log-probabilities (logits) or hard-decisions on symbols
for a tensor of received symbols.
If the `with_prior` flag is set, prior knowldge on the transmitted constellation points is assumed to be available.
The demapping function is fully differentiable when soft-values are
computed.
Parameters

- **constellation_type** (*One of** [**"qam"**, **"pam"**, **"custom"**]**, **str*)  For custom, an instance of [`Constellation`](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation)
must be provided.
- **num_bits_per_symbol** (*int*)  The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [qam, pam].
- **constellation** ()  An instance of [`Constellation`](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation) or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (*bool*)  If <cite>True</cite>, the demapper provides hard-decided symbols instead of soft-values.
Defaults to <cite>False</cite>.
- **with_prior** (*bool*)  If <cite>True</cite>, it is assumed that prior knowledge on the constellation points is available.
This prior information is given as log-probabilities (logits) as an additional input to the layer.
Defaults to <cite>False</cite>.
- **dtype** (*One of** [**tf.complex64**, **tf.complex128**] **tf.DType** (**dtype**)*)  The dtype of <cite>y</cite>. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input

- **(y, no) or (y, prior, no)**  Tuple:
- **y** (*[,n], tf.complex*)  The received symbols.
- **prior** (*[num_points] or [,num_points], tf.float*)  Prior for every symbol as log-probabilities (logits).
It can be provided either as a tensor of shape <cite>[num_points]</cite> for the
entire input batch, or as a tensor that is broadcastable
to <cite>[, n, num_points]</cite>.
Only required if the `with_prior` flag is set.
- **no** (*Scalar or [,n], tf.float*)  The noise variance estimate. It can be provided either as scalar
for the entire input batch or as a tensor that is broadcastable to
`y`.


Output

*[,n, num_points] or [,n], tf.float*  A tensor of shape <cite>[,n, num_points]</cite> of logits for every constellation
point if <cite>hard_out</cite> is set to <cite>False</cite>.
Otherwise, a tensor of shape <cite>[,n]</cite> of hard-decisions on the symbols.


**Note**

The normalized log-probability for the constellation point $c$ is computed according to

$$
\ln\left(\Pr\left(c \lvert y,\mathbf{p}\right)\right) = \ln\left( \frac{\exp\left(-\frac{|y-c|^2}{N_0} + p_c \right)}{\sum_{c'\in\mathcal{C}} \exp\left(-\frac{|y-c'|^2}{N_0} + p_{c'} \right)} \right)
$$

where $\mathcal{C}$ is the set of constellation points used for modulation,
and $\mathbf{p} = \left\{p_c \lvert c \in \mathcal{C}\right\}$ the prior information on constellation points given as log-probabilities
and which is set to $\mathbf{0}$ if no prior information on the constellation points is assumed to be available.

### SymbolDemapperWithPrior

`class` `sionna.mapping.``SymbolDemapperWithPrior`(*`constellation_type``=``None`*, *`num_bits_per_symbol``=``None`*, *`constellation``=``None`*, *`hard_out``=``False`*, *`dtype``=``tf.complex64`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/mapping.html#SymbolDemapperWithPrior)

Computes normalized log-probabilities (logits) or hard-decisions on symbols
for a tensor of received symbols, assuming that prior knowledge on the constellation points is available.
The demapping function is fully differentiable when soft-values are
computed.

This class is deprecated as the functionality has been integrated
into [`SymbolDemapper`](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.SymbolDemapper).
Parameters

- **constellation_type** (*One of** [**"qam"**, **"pam"**, **"custom"**]**, **str*)  For custom, an instance of [`Constellation`](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation)
must be provided.
- **num_bits_per_symbol** (*int*)  The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [qam, pam].
- **constellation** ()  An instance of [`Constellation`](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation) or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (*bool*)  If <cite>True</cite>, the demapper provides hard-decided symbols instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (*One of** [**tf.complex64**, **tf.complex128**] **tf.DType** (**dtype**)*)  The dtype of <cite>y</cite>. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input

- **(y, prior, no)**  Tuple:
- **y** (*[,n], tf.complex*)  The received symbols.
- **prior** (*[num_points] or [,num_points], tf.float*)  Prior for every symbol as log-probabilities (logits).
It can be provided either as a tensor of shape <cite>[num_points]</cite> for the
entire input batch, or as a tensor that is broadcastable
to <cite>[, n, num_points]</cite>.
- **no** (*Scalar or [,n], tf.float*)  The noise variance estimate. It can be provided either as scalar
for the entire input batch or as a tensor that is broadcastable to
`y`.


Output

*[,n, num_points] or [,n], tf.float*  A tensor of shape <cite>[,n, num_points]</cite> of logits for every constellation
point if <cite>hard_out</cite> is set to <cite>False</cite>.
Otherwise, a tensor of shape <cite>[,n]</cite> of hard-decisions on the symbols.


**Note**

The normalized log-probability for the constellation point $c$ is computed according to

$$
\ln\left(\Pr\left(c \lvert y,\mathbf{p}\right)\right) = \ln\left( \frac{\exp\left(-\frac{|y-c|^2}{N_0} + p_c \right)}{\sum_{c'\in\mathcal{C}} \exp\left(-\frac{|y-c'|^2}{N_0} + p_{c'} \right)} \right)
$$

where $\mathcal{C}$ is the set of constellation points used for modulation,
and $\mathbf{p} = \left\{p_c \lvert c \in \mathcal{C}\right\}$ the prior information on constellation points given as log-probabilities.

## Utility Functions

### SymbolLogits2LLRs

`class` `sionna.mapping.``SymbolLogits2LLRs`(*`method`*, *`num_bits_per_symbol`*, *`hard_out``=``False`*, *`with_prior``=``False`*, *`dtype``=``tf.float32`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/mapping.html#SymbolLogits2LLRs)

Computes log-likelihood ratios (LLRs) or hard-decisions on bits
from a tensor of logits (i.e., unnormalized log-probabilities) on constellation points.
If the flag `with_prior` is set, prior knowledge on the bits is assumed to be available.
Parameters

- **method** (*One of** [**"app"**, **"maxlog"**]**, **str*)  The method used for computing the LLRs.
- **num_bits_per_symbol** (*int*)  The number of bits per constellation symbol, e.g., 4 for QAM16.
- **hard_out** (*bool*)  If <cite>True</cite>, the layer provides hard-decided bits instead of soft-values.
Defaults to <cite>False</cite>.
- **with_prior** (*bool*)  If <cite>True</cite>, it is assumed that prior knowledge on the bits is available.
This prior information is given as LLRs as an additional input to the layer.
Defaults to <cite>False</cite>.
- **dtype** (*One of** [**tf.float32**, **tf.float64**] **tf.DType** (**dtype**)*)  The dtype for the input and output.
Defaults to <cite>tf.float32</cite>.


Input

- **logits or (logits, prior)**  Tuple:
- **logits** (*[,n, num_points], tf.float*)  Logits on constellation points.
- **prior** (*[num_bits_per_symbol] or [n, num_bits_per_symbol], tf.float*)  Prior for every bit as LLRs.
It can be provided either as a tensor of shape <cite>[num_bits_per_symbol]</cite>
for the entire input batch, or as a tensor that is broadcastable
to <cite>[, n, num_bits_per_symbol]</cite>.
Only required if the `with_prior` flag is set.


Output

*[,n, num_bits_per_symbol], tf.float*  LLRs or hard-decisions for every bit.


**Note**

With the app method, the LLR for the $i\text{th}$ bit
is computed according to

$$
LLR(i) = \ln\left(\frac{\Pr\left(b_i=1\lvert \mathbf{z},\mathbf{p}\right)}{\Pr\left(b_i=0\lvert \mathbf{z},\mathbf{p}\right)}\right) =\ln\left(\frac{
        \sum_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
        e^{z_c}
        }{
        \sum_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
        e^{z_c}
        }\right)
$$

where $\mathcal{C}_{i,1}$ and $\mathcal{C}_{i,0}$ are the
sets of $2^K$ constellation points for which the $i\text{th}$ bit is
equal to 1 and 0, respectively. $\mathbf{z} = \left[z_{c_0},\dots,z_{c_{2^K-1}}\right]$ is the vector of logits on the constellation points, $\mathbf{p} = \left[p_0,\dots,p_{K-1}\right]$
is the vector of LLRs that serves as prior knowledge on the $K$ bits that are mapped to
a constellation point and is set to $\mathbf{0}$ if no prior knowledge is assumed to be available,
and $\Pr(c\lvert\mathbf{p})$ is the prior probability on the constellation symbol $c$:

$$
\Pr\left(c\lvert\mathbf{p}\right) = \prod_{k=0}^{K-1} \Pr\left(b_k = \ell(c)_k \lvert\mathbf{p} \right)
= \prod_{k=0}^{K-1} \text{sigmoid}\left(p_k \ell(c)_k\right)
$$

where $\ell(c)_k$ is the $k^{th}$ bit label of $c$, where 0 is
replaced by -1.
The definition of the LLR has been
chosen such that it is equivalent with that of logits. This is
different from many textbooks in communications, where the LLR is
defined as $LLR(i) = \ln\left(\frac{\Pr\left(b_i=0\lvert y\right)}{\Pr\left(b_i=1\lvert y\right)}\right)$.

With the maxlog method, LLRs for the $i\text{th}$ bit
are approximated like

$$
\begin{align}
    LLR(i) &\approx\ln\left(\frac{
        \max_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
            e^{z_c}
        }{
        \max_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
            e^{z_c}
        }\right)
        .
\end{align}
$$


### LLRs2SymbolLogits

`class` `sionna.mapping.``LLRs2SymbolLogits`(*`num_bits_per_symbol`*, *`hard_out``=``False`*, *`dtype``=``tf.float32`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/mapping.html#LLRs2SymbolLogits)

Computes logits (i.e., unnormalized log-probabilities) or hard decisions
on constellation points from a tensor of log-likelihood ratios (LLRs) on bits.
Parameters

- **num_bits_per_symbol** (*int*)  The number of bits per constellation symbol, e.g., 4 for QAM16.
- **hard_out** (*bool*)  If <cite>True</cite>, the layer provides hard-decided constellation points instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (*One of** [**tf.float32**, **tf.float64**] **tf.DType** (**dtype**)*)  The dtype for the input and output.
Defaults to <cite>tf.float32</cite>.


Input

**llrs** (*[, n, num_bits_per_symbol], tf.float*)  LLRs for every bit.

Output

*[,n, num_points], tf.float or [, n], tf.int32*  Logits or hard-decisions on constellation points.


**Note**

The logit for the constellation $c$ point
is computed according to

$$
\begin{split}\begin{align}
    \log{\left(\Pr\left(c\lvert LLRs \right)\right)}
        &= \log{\left(\prod_{k=0}^{K-1} \Pr\left(b_k = \ell(c)_k \lvert LLRs \right)\right)}\\
        &= \log{\left(\prod_{k=0}^{K-1} \text{sigmoid}\left(LLR(k) \ell(c)_k\right)\right)}\\
        &= \sum_{k=0}^{K-1} \log{\left(\text{sigmoid}\left(LLR(k) \ell(c)_k\right)\right)}
\end{align}\end{split}
$$

where $\ell(c)_k$ is the $k^{th}$ bit label of $c$, where 0 is
replaced by -1.
The definition of the LLR has been
chosen such that it is equivalent with that of logits. This is
different from many textbooks in communications, where the LLR is
defined as $LLR(i) = \ln\left(\frac{\Pr\left(b_i=0\lvert y\right)}{\Pr\left(b_i=1\lvert y\right)}\right)$.

### SymbolLogits2LLRsWithPrior

`class` `sionna.mapping.``SymbolLogits2LLRsWithPrior`(*`method`*, *`num_bits_per_symbol`*, *`hard_out``=``False`*, *`dtype``=``tf.float32`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/mapping.html#SymbolLogits2LLRsWithPrior)

Computes log-likelihood ratios (LLRs) or hard-decisions on bits
from a tensor of logits (i.e., unnormalized log-probabilities) on constellation points,
assuming that prior knowledge on the bits is available.

This class is deprecated as the functionality has been integrated
into [`SymbolLogits2LLRs`](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.SymbolLogits2LLRs).
Parameters

- **method** (*One of** [**"app"**, **"maxlog"**]**, **str*)  The method used for computing the LLRs.
- **num_bits_per_symbol** (*int*)  The number of bits per constellation symbol, e.g., 4 for QAM16.
- **hard_out** (*bool*)  If <cite>True</cite>, the layer provides hard-decided bits instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (*One of** [**tf.float32**, **tf.float64**] **tf.DType** (**dtype**)*)  The dtype for the input and output.
Defaults to <cite>tf.float32</cite>.


Input

- **(logits, prior)**  Tuple:
- **logits** (*[,n, num_points], tf.float*)  Logits on constellation points.
- **prior** (*[num_bits_per_symbol] or [n, num_bits_per_symbol], tf.float*)  Prior for every bit as LLRs.
It can be provided either as a tensor of shape <cite>[num_bits_per_symbol]</cite> for the
entire input batch, or as a tensor that is broadcastable
to <cite>[, n, num_bits_per_symbol]</cite>.


Output

*[,n, num_bits_per_symbol], tf.float*  LLRs or hard-decisions for every bit.


**Note**

With the app method, the LLR for the $i\text{th}$ bit
is computed according to

$$
LLR(i) = \ln\left(\frac{\Pr\left(b_i=1\lvert \mathbf{z},\mathbf{p}\right)}{\Pr\left(b_i=0\lvert \mathbf{z},\mathbf{p}\right)}\right) =\ln\left(\frac{
        \sum_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
        e^{z_c}
        }{
        \sum_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
        e^{z_c}
        }\right)
$$

where $\mathcal{C}_{i,1}$ and $\mathcal{C}_{i,0}$ are the
sets of $2^K$ constellation points for which the $i\text{th}$ bit is
equal to 1 and 0, respectively. $\mathbf{z} = \left[z_{c_0},\dots,z_{c_{2^K-1}}\right]$ is the vector of logits on the constellation points, $\mathbf{p} = \left[p_0,\dots,p_{K-1}\right]$
is the vector of LLRs that serves as prior knowledge on the $K$ bits that are mapped to
a constellation point,
and $\Pr(c\lvert\mathbf{p})$ is the prior probability on the constellation symbol $c$:

$$
\Pr\left(c\lvert\mathbf{p}\right) = \prod_{k=0}^{K-1} \Pr\left(b_k = \ell(c)_k \lvert\mathbf{p} \right)
= \prod_{k=0}^{K-1} \text{sigmoid}\left(p_k \ell(c)_k\right)
$$

where $\ell(c)_k$ is the $k^{th}$ bit label of $c$, where 0 is
replaced by -1.
The definition of the LLR has been
chosen such that it is equivalent with that of logits. This is
different from many textbooks in communications, where the LLR is
defined as $LLR(i) = \ln\left(\frac{\Pr\left(b_i=0\lvert y\right)}{\Pr\left(b_i=1\lvert y\right)}\right)$.

With the maxlog method, LLRs for the $i\text{th}$ bit
are approximated like

$$
\begin{align}
    LLR(i) &\approx\ln\left(\frac{
        \max_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
            e^{z_c}
        }{
        \max_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
            e^{z_c}
        }\right)
        .
\end{align}
$$


### SymbolLogits2Moments

`class` `sionna.mapping.``SymbolLogits2Moments`(*`constellation_type``=``None`*, *`num_bits_per_symbol``=``None`*, *`constellation``=``None`*, *`dtype``=``tf.float32`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/mapping.html#SymbolLogits2Moments)

Computes the mean and variance of a constellation from logits (unnormalized log-probabilities) on the
constellation points.

More precisely, given a constellation $\mathcal{C} = \left[ c_0,\dots,c_{N-1} \right]$ of size $N$, this layer computes the mean and variance
according to

$$
\begin{split}\begin{align}
    \mu &= \sum_{n = 0}^{N-1} c_n \Pr \left(c_n \lvert \mathbf{\ell} \right)\\
    \nu &= \sum_{n = 0}^{N-1} \left( c_n - \mu \right)^2 \Pr \left(c_n \lvert \mathbf{\ell} \right)
\end{align}\end{split}
$$

where $\mathbf{\ell} = \left[ \ell_0, \dots, \ell_{N-1} \right]$ are the logits, and

$$
\Pr \left(c_n \lvert \mathbf{\ell} \right) = \frac{\exp \left( \ell_n \right)}{\sum_{i=0}^{N-1} \exp \left( \ell_i \right) }.
$$

Parameters

- **constellation_type** (*One of** [**"qam"**, **"pam"**, **"custom"**]**, **str*)  For custom, an instance of [`Constellation`](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation)
must be provided.
- **num_bits_per_symbol** (*int*)  The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [qam, pam].
- **constellation** ()  An instance of [`Constellation`](https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation) or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **dtype** (*One of** [**tf.float32**, **tf.float64**] **tf.DType** (**dtype**)*)  The dtype for the input and output.
Defaults to <cite>tf.float32</cite>.


Input

**logits** (*[,n, num_points], tf.float*)  Logits on constellation points.

Output

- **mean** (*[,n], tf.float*)  Mean of the constellation.
- **var** (*[,n], tf.float*)  Variance of the constellation


### SymbolInds2Bits

`class` `sionna.mapping.``SymbolInds2Bits`(*`num_bits_per_symbol`*, *`dtype``=``tf.float32`*, *`**``kwargs`*)[`[source]`](../_modules/sionna/mapping.html#SymbolInds2Bits)

Transforms symbol indices to their binary representations.
Parameters

- **num_bits_per_symbol** (*int*)  Number of bits per constellation symbol
- **dtype** (*tf.DType*)  Output dtype. Defaults to <cite>tf.float32</cite>.


Input

*Tensor, tf.int*  Symbol indices

Output

*input.shape + [num_bits_per_symbol], dtype*  Binary representation of symbol indices


### PAM2QAM

`class` `sionna.mapping.``PAM2QAM`(*`num_bits_per_symbol`*, *`hard_in_out``=``True`*)[`[source]`](../_modules/sionna/mapping.html#PAM2QAM)

Transforms PAM symbol indices/logits to QAM symbol indices/logits.

For two PAM constellation symbol indices or logits, corresponding to
the real and imaginary components of a QAM constellation,
compute the QAM symbol index or logits.
Parameters

- **num_bits_per_symbol** (*int*)  Number of bits per QAM constellation symbol, e.g., 4 for QAM16
- **hard_in_out** (*bool*)  Determines if inputs and outputs are indices or logits over
constellation symbols.
Defaults to <cite>True</cite>.


Input

- **pam1** (*Tensor, tf.int, or [,2**(num_bits_per_symbol/2)], tf.float*)  Indices or logits for the first PAM constellation
- **pam2** (*Tensor, tf.int, or [,2**(num_bits_per_symbol/2)], tf.float*)  Indices or logits for the second PAM constellation


Output

**qam** (*Tensor, tf.int, or [,2**num_bits_per_symbol], tf.float*)  Indices or logits for the corresponding QAM constellation


### QAM2PAM

`class` `sionna.mapping.``QAM2PAM`(*`num_bits_per_symbol`*)[`[source]`](../_modules/sionna/mapping.html#QAM2PAM)

Transforms QAM symbol indices to PAM symbol indices.

For indices in a QAM constellation, computes the corresponding indices
for the two PAM constellations corresponding the real and imaginary
components of the QAM constellation.
Parameters

**num_bits_per_symbol** (*int*)  The number of bits per QAM constellation symbol, e.g., 4 for QAM16.

Input

**ind_qam** (*Tensor, tf.int*)  Indices in the QAM constellation

Output

- **ind_pam1** (*Tensor, tf.int*)  Indices for the first component of the corresponding PAM modulation
- **ind_pam2** (*Tensor, tf.int*)  Indices for the first component of the corresponding PAM modulation


References:
3GPPTS38211([1](https://nvlabs.github.io/sionna/api/mapping.html#id1),[2](https://nvlabs.github.io/sionna/api/mapping.html#id2),[3](https://nvlabs.github.io/sionna/api/mapping.html#id3))

ETSI TS 38.211 5G NR Physical channels and modulation, V16.2.0, Jul. 2020
[https://www.3gpp.org/ftp/Specs/archive/38_series/38.211/38211-h00.zip](https://www.3gpp.org/ftp/Specs/archive/38_series/38.211/38211-h00.zip)



