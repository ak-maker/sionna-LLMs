
# Mapping<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#mapping" title="Permalink to this headline"></a>
    
This module contains classes and functions related to mapping
of bits to constellation symbols and demapping of soft-symbols
to log-likelihood ratios (LLRs). The key components are the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Mapper" title="sionna.mapping.Mapper">`Mapper`</a>,
and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Demapper" title="sionna.mapping.Demapper">`Demapper`</a>. A <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
can be made trainable to enable learning of geometric shaping.

## Constellations<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#constellations" title="Permalink to this headline"></a>

### Constellation<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#constellation" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``Constellation`(<em class="sig-param">`constellation_type`</em>, <em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`initial_value``=``None`</em>, <em class="sig-param">`normalize``=``True`</em>, <em class="sig-param">`center``=``False`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#Constellation">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="Permalink to this definition"></a>
    
Constellation that can be used by a (de)mapper.
    
This class defines a constellation, i.e., a complex-valued vector of
constellation points. A constellation can be trainable. The binary
representation of the index of an element of this vector corresponds
to the bit label of the constellation point. This implicit bit
labeling is used by the `Mapper` and `Demapper` classes.
Parameters
 
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, the constellation points are randomly initialized
if no `initial_value` is provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
- **initial_value** ($[2^\text{num_bits_per_symbol}]$, NumPy array or Tensor) – Initial values of the constellation points. If `normalize` or
`center` are <cite>True</cite>, the initial constellation might be changed.
- **normalize** (<em>bool</em>) – If <cite>True</cite>, the constellation is normalized to have unit power.
Defaults to <cite>True</cite>.
- **center** (<em>bool</em>) – If <cite>True</cite>, the constellation is ensured to have zero mean.
Defaults to <cite>False</cite>.
- **trainable** (<em>bool</em>) – If <cite>True</cite>, the constellation points are trainable variables.
Defaults to <cite>False</cite>.
- **dtype** (<em>[</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>]</em><em>, </em><em>tf.DType</em>) – The dtype of the constellation.


Output
    
$[2^\text{num_bits_per_symbol}]$, `dtype` – The constellation.



**Note**
    
One can create a trainable PAM/QAM constellation. This is
equivalent to creating a custom trainable constellation which is
initialized with PAM/QAM constellation points.

<em class="property">`property` </em>`center`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation.center" title="Permalink to this definition"></a>
    
Indicates if the constellation is centered.


`create_or_check_constellation`(<em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#Constellation.create_or_check_constellation">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation.create_or_check_constellation" title="Permalink to this definition"></a>
    
Static method for conviently creating a constellation object or checking that an existing one
is consistent with requested settings.
    
If `constellation` is <cite>None</cite>, then this method creates a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
object of type `constellation_type` and with `num_bits_per_symbol` bits per symbol.
Otherwise, this method checks that <cite>constellation</cite> is consistent with `constellation_type` and
`num_bits_per_symbol`. If it is, `constellation` is returned. Otherwise, an assertion is raised.
Input
 
- **constellation_type** (<em>One of [“qam”, “pam”, “custom”], str</em>) – For “custom”, an instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<em>Constellation</em>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or
<cite>None</cite>. In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.


Output
    
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> – A constellation object.




<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation.normalize" title="Permalink to this definition"></a>
    
Indicates if the constellation is normalized or not.


<em class="property">`property` </em>`num_bits_per_symbol`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation.num_bits_per_symbol" title="Permalink to this definition"></a>
    
The number of bits per constellation symbol.


<em class="property">`property` </em>`points`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation.points" title="Permalink to this definition"></a>
    
The (possibly) centered and normalized constellation points.


`show`(<em class="sig-param">`labels``=``True`</em>, <em class="sig-param">`figsize``=``(7,` `7)`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#Constellation.show">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation.show" title="Permalink to this definition"></a>
    
Generate a scatter-plot of the constellation.
Input
 
- **labels** (<em>bool</em>) – If <cite>True</cite>, the bit labels will be drawn next to each constellation
point. Defaults to <cite>True</cite>.
- **figsize** (<em>Two-element Tuple, float</em>) – Width and height in inches. Defaults to <cite>(7,7)</cite>.


Output
    
<em>matplotlib.figure.Figure</em> – A handle to a matplot figure object.




### qam<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#qam" title="Permalink to this headline"></a>

`sionna.mapping.``qam`(<em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`normalize``=``True`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#qam">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.qam" title="Permalink to this definition"></a>
    
Generates a QAM constellation.
    
This function generates a complex-valued vector, where each element is
a constellation point of an M-ary QAM constellation. The bit
label of the `n` th point is given by the length-`num_bits_per_symbol`
binary represenation of `n`.
Input
 
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation point.
Must be a multiple of two, e.g., 2, 4, 6, 8, etc.
- **normalize** (<em>bool</em>) – If <cite>True</cite>, the constellation is normalized to have unit power.
Defaults to <cite>True</cite>.


Output
    
$[2^{\text{num_bits_per_symbol}}]$, np.complex64 – The QAM constellation.



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
Section 5.1 of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#gppts38211" id="id1">[3GPPTS38211]</a>. It is used in the 5G standard.

### pam<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#pam" title="Permalink to this headline"></a>

`sionna.mapping.``pam`(<em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`normalize``=``True`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#pam">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.pam" title="Permalink to this definition"></a>
    
Generates a PAM constellation.
    
This function generates a real-valued vector, where each element is
a constellation point of an M-ary PAM constellation. The bit
label of the `n` th point is given by the length-`num_bits_per_symbol`
binary represenation of `n`.
Input
 
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation point.
Must be positive.
- **normalize** (<em>bool</em>) – If <cite>True</cite>, the constellation is normalized to have unit power.
Defaults to <cite>True</cite>.


Output
    
$[2^{\text{num_bits_per_symbol}}]$, np.float32 – The PAM constellation.



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
Section 5.1 of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#gppts38211" id="id2">[3GPPTS38211]</a>. It is used in the 5G standard.

### pam_gray<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#pam-gray" title="Permalink to this headline"></a>

`sionna.mapping.``pam_gray`(<em class="sig-param">`b`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#pam_gray">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.pam_gray" title="Permalink to this definition"></a>
    
Maps a vector of bits to a PAM constellation points with Gray labeling.
    
This recursive function maps a binary vector to Gray-labelled PAM
constellation points. It can be used to generated QAM constellations.
The constellation is not normalized.
Input
    
**b** (<em>[n], NumPy array</em>) – Tensor with with binary entries.

Output
    
<em>signed int</em> – The PAM constellation point taking values in
$\{\pm 1,\pm 3,\dots,\pm (2^n-1)\}$.



**Note**
    
This algorithm is a recursive implementation of the expressions found in
Section 5.1 of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#gppts38211" id="id3">[3GPPTS38211]</a>. It is used in the 5G standard.

## Mapper<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#mapper" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``Mapper`(<em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`return_indices``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#Mapper">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Mapper" title="Permalink to this definition"></a>
    
Maps binary tensors to points of a constellation.
    
This class defines a layer that maps a tensor of binary values
to a tensor of points from a provided constellation.
Parameters
 
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or
<cite>None</cite>. In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **return_indices** (<em>bool</em>) – If enabled, symbol indices are additionally returned.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>]</em><em>, </em><em>tf.DType</em>) – The output dtype. Defaults to tf.complex64.


Input
    
<em>[…, n], tf.float or tf.int</em> – Tensor with with binary entries.

Output
 
- <em>[…,n/Constellation.num_bits_per_symbol], tf.complex</em> – The mapped constellation symbols.
- <em>[…,n/Constellation.num_bits_per_symbol], tf.int32</em> – The symbol indices corresponding to the constellation symbols.
Only returned if `return_indices` is set to True.




**Note**
    
The last input dimension must be an integer multiple of the
number of bits per constellation symbol.

<em class="property">`property` </em>`constellation`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Mapper.constellation" title="Permalink to this definition"></a>
    
The Constellation used by the Mapper.


## Demapping<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#demapping" title="Permalink to this headline"></a>

### Demapper<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#demapper" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``Demapper`(<em class="sig-param">`demapping_method`</em>, <em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`with_prior``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#Demapper">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Demapper" title="Permalink to this definition"></a>
    
Computes log-likelihood ratios (LLRs) or hard-decisions on bits
for a tensor of received symbols.
If the flag `with_prior` is set, prior knowledge on the bits is assumed to be available.
    
This class defines a layer implementing different demapping
functions. All demapping functions are fully differentiable when soft-decisions
are computed.
Parameters
 
- **demapping_method** (<em>One of</em><em> [</em><em>"app"</em><em>, </em><em>"maxlog"</em><em>]</em><em>, </em><em>str</em>) – The demapping method used.
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the demapper provides hard-decided bits instead of soft-values.
Defaults to <cite>False</cite>.
- **with_prior** (<em>bool</em>) – If <cite>True</cite>, it is assumed that prior knowledge on the bits is available.
This prior information is given as LLRs as an additional input to the layer.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of <cite>y</cite>. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y,no) or (y, prior, no)** – Tuple:
- **y** (<em>[…,n], tf.complex</em>) – The received symbols.
- **prior** (<em>[num_bits_per_symbol] or […,num_bits_per_symbol], tf.float</em>) – Prior for every bit as LLRs.
It can be provided either as a tensor of shape <cite>[num_bits_per_symbol]</cite> for the
entire input batch, or as a tensor that is “broadcastable”
to <cite>[…, n, num_bits_per_symbol]</cite>.
Only required if the `with_prior` flag is set.
- **no** (<em>Scalar or […,n], tf.float</em>) – The noise variance estimate. It can be provided either as scalar
for the entire input batch or as a tensor that is “broadcastable” to
`y`.


Output
    
<em>[…,n*num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit.



**Note**
    
With the “app” demapping method, the LLR for the $i\text{th}$ bit
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
    
With the “maxlog” demapping method, LLRs for the $i\text{th}$ bit
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


### DemapperWithPrior<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#demapperwithprior" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``DemapperWithPrior`(<em class="sig-param">`demapping_method`</em>, <em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#DemapperWithPrior">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.DemapperWithPrior" title="Permalink to this definition"></a>
    
Computes log-likelihood ratios (LLRs) or hard-decisions on bits
for a tensor of received symbols, assuming that prior knowledge on the bits is available.
    
This class defines a layer implementing different demapping
functions. All demapping functions are fully differentiable when soft-decisions
are computed.
    
This class is deprecated as the functionality has been integrated
into <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Demapper" title="sionna.mapping.Demapper">`Demapper`</a>.
Parameters
 
- **demapping_method** (<em>One of</em><em> [</em><em>"app"</em><em>, </em><em>"maxlog"</em><em>]</em><em>, </em><em>str</em>) – The demapping method used.
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the demapper provides hard-decided bits instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of <cite>y</cite>. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y, prior, no)** – Tuple:
- **y** (<em>[…,n], tf.complex</em>) – The received symbols.
- **prior** (<em>[num_bits_per_symbol] or […,num_bits_per_symbol], tf.float</em>) – Prior for every bit as LLRs.
It can be provided either as a tensor of shape <cite>[num_bits_per_symbol]</cite> for the
entire input batch, or as a tensor that is “broadcastable”
to <cite>[…, n, num_bits_per_symbol]</cite>.
- **no** (<em>Scalar or […,n], tf.float</em>) – The noise variance estimate. It can be provided either as scalar
for the entire input batch or as a tensor that is “broadcastable” to
`y`.


Output
    
<em>[…,n*num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit.



**Note**
    
With the “app” demapping method, the LLR for the $i\text{th}$ bit
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
    
With the “maxlog” demapping method, LLRs for the $i\text{th}$ bit
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


### SymbolDemapper<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#symboldemapper" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``SymbolDemapper`(<em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`with_prior``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#SymbolDemapper">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.SymbolDemapper" title="Permalink to this definition"></a>
    
Computes normalized log-probabilities (logits) or hard-decisions on symbols
for a tensor of received symbols.
If the `with_prior` flag is set, prior knowldge on the transmitted constellation points is assumed to be available.
The demapping function is fully differentiable when soft-values are
computed.
Parameters
 
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the demapper provides hard-decided symbols instead of soft-values.
Defaults to <cite>False</cite>.
- **with_prior** (<em>bool</em>) – If <cite>True</cite>, it is assumed that prior knowledge on the constellation points is available.
This prior information is given as log-probabilities (logits) as an additional input to the layer.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of <cite>y</cite>. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y, no) or (y, prior, no)** – Tuple:
- **y** (<em>[…,n], tf.complex</em>) – The received symbols.
- **prior** (<em>[num_points] or […,num_points], tf.float</em>) – Prior for every symbol as log-probabilities (logits).
It can be provided either as a tensor of shape <cite>[num_points]</cite> for the
entire input batch, or as a tensor that is “broadcastable”
to <cite>[…, n, num_points]</cite>.
Only required if the `with_prior` flag is set.
- **no** (<em>Scalar or […,n], tf.float</em>) – The noise variance estimate. It can be provided either as scalar
for the entire input batch or as a tensor that is “broadcastable” to
`y`.


Output
    
<em>[…,n, num_points] or […,n], tf.float</em> – A tensor of shape <cite>[…,n, num_points]</cite> of logits for every constellation
point if <cite>hard_out</cite> is set to <cite>False</cite>.
Otherwise, a tensor of shape <cite>[…,n]</cite> of hard-decisions on the symbols.



**Note**
    
The normalized log-probability for the constellation point $c$ is computed according to

$$
\ln\left(\Pr\left(c \lvert y,\mathbf{p}\right)\right) = \ln\left( \frac{\exp\left(-\frac{|y-c|^2}{N_0} + p_c \right)}{\sum_{c'\in\mathcal{C}} \exp\left(-\frac{|y-c'|^2}{N_0} + p_{c'} \right)} \right)
$$
    
where $\mathcal{C}$ is the set of constellation points used for modulation,
and $\mathbf{p} = \left\{p_c \lvert c \in \mathcal{C}\right\}$ the prior information on constellation points given as log-probabilities
and which is set to $\mathbf{0}$ if no prior information on the constellation points is assumed to be available.

### SymbolDemapperWithPrior<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#symboldemapperwithprior" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``SymbolDemapperWithPrior`(<em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#SymbolDemapperWithPrior">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.SymbolDemapperWithPrior" title="Permalink to this definition"></a>
    
Computes normalized log-probabilities (logits) or hard-decisions on symbols
for a tensor of received symbols, assuming that prior knowledge on the constellation points is available.
The demapping function is fully differentiable when soft-values are
computed.
    
This class is deprecated as the functionality has been integrated
into <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.SymbolDemapper" title="sionna.mapping.SymbolDemapper">`SymbolDemapper`</a>.
Parameters
 
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the demapper provides hard-decided symbols instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of <cite>y</cite>. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y, prior, no)** – Tuple:
- **y** (<em>[…,n], tf.complex</em>) – The received symbols.
- **prior** (<em>[num_points] or […,num_points], tf.float</em>) – Prior for every symbol as log-probabilities (logits).
It can be provided either as a tensor of shape <cite>[num_points]</cite> for the
entire input batch, or as a tensor that is “broadcastable”
to <cite>[…, n, num_points]</cite>.
- **no** (<em>Scalar or […,n], tf.float</em>) – The noise variance estimate. It can be provided either as scalar
for the entire input batch or as a tensor that is “broadcastable” to
`y`.


Output
    
<em>[…,n, num_points] or […,n], tf.float</em> – A tensor of shape <cite>[…,n, num_points]</cite> of logits for every constellation
point if <cite>hard_out</cite> is set to <cite>False</cite>.
Otherwise, a tensor of shape <cite>[…,n]</cite> of hard-decisions on the symbols.



**Note**
    
The normalized log-probability for the constellation point $c$ is computed according to

$$
\ln\left(\Pr\left(c \lvert y,\mathbf{p}\right)\right) = \ln\left( \frac{\exp\left(-\frac{|y-c|^2}{N_0} + p_c \right)}{\sum_{c'\in\mathcal{C}} \exp\left(-\frac{|y-c'|^2}{N_0} + p_{c'} \right)} \right)
$$
    
where $\mathcal{C}$ is the set of constellation points used for modulation,
and $\mathbf{p} = \left\{p_c \lvert c \in \mathcal{C}\right\}$ the prior information on constellation points given as log-probabilities.

## Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#utility-functions" title="Permalink to this headline"></a>

### SymbolLogits2LLRs<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#symbollogits2llrs" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``SymbolLogits2LLRs`(<em class="sig-param">`method`</em>, <em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`with_prior``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#SymbolLogits2LLRs">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.SymbolLogits2LLRs" title="Permalink to this definition"></a>
    
Computes log-likelihood ratios (LLRs) or hard-decisions on bits
from a tensor of logits (i.e., unnormalized log-probabilities) on constellation points.
If the flag `with_prior` is set, prior knowledge on the bits is assumed to be available.
Parameters
 
- **method** (<em>One of</em><em> [</em><em>"app"</em><em>, </em><em>"maxlog"</em><em>]</em><em>, </em><em>str</em>) – The method used for computing the LLRs.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the layer provides hard-decided bits instead of soft-values.
Defaults to <cite>False</cite>.
- **with_prior** (<em>bool</em>) – If <cite>True</cite>, it is assumed that prior knowledge on the bits is available.
This prior information is given as LLRs as an additional input to the layer.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.float32</em><em>, </em><em>tf.float64</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype for the input and output.
Defaults to <cite>tf.float32</cite>.


Input
 
- **logits or (logits, prior)** – Tuple:
- **logits** (<em>[…,n, num_points], tf.float</em>) – Logits on constellation points.
- **prior** (<em>[num_bits_per_symbol] or […n, num_bits_per_symbol], tf.float</em>) – Prior for every bit as LLRs.
It can be provided either as a tensor of shape <cite>[num_bits_per_symbol]</cite>
for the entire input batch, or as a tensor that is “broadcastable”
to <cite>[…, n, num_bits_per_symbol]</cite>.
Only required if the `with_prior` flag is set.


Output
    
<em>[…,n, num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit.



**Note**
    
With the “app” method, the LLR for the $i\text{th}$ bit
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
    
With the “maxlog” method, LLRs for the $i\text{th}$ bit
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


### LLRs2SymbolLogits<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#llrs2symbollogits" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``LLRs2SymbolLogits`(<em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#LLRs2SymbolLogits">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.LLRs2SymbolLogits" title="Permalink to this definition"></a>
    
Computes logits (i.e., unnormalized log-probabilities) or hard decisions
on constellation points from a tensor of log-likelihood ratios (LLRs) on bits.
Parameters
 
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the layer provides hard-decided constellation points instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.float32</em><em>, </em><em>tf.float64</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype for the input and output.
Defaults to <cite>tf.float32</cite>.


Input
    
**llrs** (<em>[…, n, num_bits_per_symbol], tf.float</em>) – LLRs for every bit.

Output
    
<em>[…,n, num_points], tf.float or […, n], tf.int32</em> – Logits or hard-decisions on constellation points.



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

### SymbolLogits2LLRsWithPrior<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#symbollogits2llrswithprior" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``SymbolLogits2LLRsWithPrior`(<em class="sig-param">`method`</em>, <em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#SymbolLogits2LLRsWithPrior">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.SymbolLogits2LLRsWithPrior" title="Permalink to this definition"></a>
    
Computes log-likelihood ratios (LLRs) or hard-decisions on bits
from a tensor of logits (i.e., unnormalized log-probabilities) on constellation points,
assuming that prior knowledge on the bits is available.
    
This class is deprecated as the functionality has been integrated
into <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.SymbolLogits2LLRs" title="sionna.mapping.SymbolLogits2LLRs">`SymbolLogits2LLRs`</a>.
Parameters
 
- **method** (<em>One of</em><em> [</em><em>"app"</em><em>, </em><em>"maxlog"</em><em>]</em><em>, </em><em>str</em>) – The method used for computing the LLRs.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the layer provides hard-decided bits instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.float32</em><em>, </em><em>tf.float64</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype for the input and output.
Defaults to <cite>tf.float32</cite>.


Input
 
- **(logits, prior)** – Tuple:
- **logits** (<em>[…,n, num_points], tf.float</em>) – Logits on constellation points.
- **prior** (<em>[num_bits_per_symbol] or […n, num_bits_per_symbol], tf.float</em>) – Prior for every bit as LLRs.
It can be provided either as a tensor of shape <cite>[num_bits_per_symbol]</cite> for the
entire input batch, or as a tensor that is “broadcastable”
to <cite>[…, n, num_bits_per_symbol]</cite>.


Output
    
<em>[…,n, num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit.



**Note**
    
With the “app” method, the LLR for the $i\text{th}$ bit
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
    
With the “maxlog” method, LLRs for the $i\text{th}$ bit
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


### SymbolLogits2Moments<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#symbollogits2moments" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``SymbolLogits2Moments`(<em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#SymbolLogits2Moments">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.SymbolLogits2Moments" title="Permalink to this definition"></a>
    
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
 
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **dtype** (<em>One of</em><em> [</em><em>tf.float32</em><em>, </em><em>tf.float64</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype for the input and output.
Defaults to <cite>tf.float32</cite>.


Input
    
**logits** (<em>[…,n, num_points], tf.float</em>) – Logits on constellation points.

Output
 
- **mean** (<em>[…,n], tf.float</em>) – Mean of the constellation.
- **var** (<em>[…,n], tf.float</em>) – Variance of the constellation




### SymbolInds2Bits<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#symbolinds2bits" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``SymbolInds2Bits`(<em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#SymbolInds2Bits">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.SymbolInds2Bits" title="Permalink to this definition"></a>
    
Transforms symbol indices to their binary representations.
Parameters
 
- **num_bits_per_symbol** (<em>int</em>) – Number of bits per constellation symbol
- **dtype** (<em>tf.DType</em>) – Output dtype. Defaults to <cite>tf.float32</cite>.


Input
    
<em>Tensor, tf.int</em> – Symbol indices

Output
    
<em>input.shape + [num_bits_per_symbol], dtype</em> – Binary representation of symbol indices



### PAM2QAM<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#pam2qam" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``PAM2QAM`(<em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`hard_in_out``=``True`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#PAM2QAM">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.PAM2QAM" title="Permalink to this definition"></a>
    
Transforms PAM symbol indices/logits to QAM symbol indices/logits.
    
For two PAM constellation symbol indices or logits, corresponding to
the real and imaginary components of a QAM constellation,
compute the QAM symbol index or logits.
Parameters
 
- **num_bits_per_symbol** (<em>int</em>) – Number of bits per QAM constellation symbol, e.g., 4 for QAM16
- **hard_in_out** (<em>bool</em>) – Determines if inputs and outputs are indices or logits over
constellation symbols.
Defaults to <cite>True</cite>.


Input
 
- **pam1** (<em>Tensor, tf.int, or […,2**(num_bits_per_symbol/2)], tf.float</em>) – Indices or logits for the first PAM constellation
- **pam2** (<em>Tensor, tf.int, or […,2**(num_bits_per_symbol/2)], tf.float</em>) – Indices or logits for the second PAM constellation


Output
    
**qam** (<em>Tensor, tf.int, or […,2**num_bits_per_symbol], tf.float</em>) – Indices or logits for the corresponding QAM constellation



### QAM2PAM<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#qam2pam" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``QAM2PAM`(<em class="sig-param">`num_bits_per_symbol`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#QAM2PAM">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.QAM2PAM" title="Permalink to this definition"></a>
    
Transforms QAM symbol indices to PAM symbol indices.
    
For indices in a QAM constellation, computes the corresponding indices
for the two PAM constellations corresponding the real and imaginary
components of the QAM constellation.
Parameters
    
**num_bits_per_symbol** (<em>int</em>) – The number of bits per QAM constellation symbol, e.g., 4 for QAM16.

Input
    
**ind_qam** (<em>Tensor, tf.int</em>) – Indices in the QAM constellation

Output
 
- **ind_pam1** (<em>Tensor, tf.int</em>) – Indices for the first component of the corresponding PAM modulation
- **ind_pam2** (<em>Tensor, tf.int</em>) – Indices for the first component of the corresponding PAM modulation





References:
3GPPTS38211(<a href="https://nvlabs.github.io/sionna/api/mapping.html#id1">1</a>,<a href="https://nvlabs.github.io/sionna/api/mapping.html#id2">2</a>,<a href="https://nvlabs.github.io/sionna/api/mapping.html#id3">3</a>)
    
ETSI TS 38.211 “5G NR Physical channels and modulation”, V16.2.0, Jul. 2020
<a class="reference external" href="https://www.3gpp.org/ftp/Specs/archive/38_series/38.211/38211-h00.zip">https://www.3gpp.org/ftp/Specs/archive/38_series/38.211/38211-h00.zip</a>



