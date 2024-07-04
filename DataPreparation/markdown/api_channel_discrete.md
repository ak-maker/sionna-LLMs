
# Discrete<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.discrete.html#discrete" title="Permalink to this headline"></a>
    
This module provides layers and functions that implement channel
models with discrete input/output alphabets.
    
All channel models support binary inputs $x \in \{0, 1\}$ and <cite>bipolar</cite>
inputs $x \in \{-1, 1\}$, respectively. In the later case, it is assumed
that each <cite>0</cite> is mapped to <cite>-1</cite>.
    
The channels can either return discrete values or log-likelihood ratios (LLRs).
These LLRs describe the channel transition probabilities
$L(y|X=1)=L(X=1|y)+L_a(X=1)$ where $L_a(X=1)=\operatorname{log} \frac{P(X=1)}{P(X=0)}$ depends only on the <cite>a priori</cite> probability of $X=1$. These LLRs equal the <cite>a posteriori</cite> probability if $P(X=1)=P(X=0)=0.5$.
    
Further, the channel reliability parameter $p_b$ can be either a scalar
value or a tensor of any shape that can be broadcasted to the input. This
allows for the efficient implementation of
channels with non-uniform error probabilities.
    
The channel models are based on the <cite>Gumble-softmax trick</cite> <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.discrete.html#gumblesoftmax" id="id1">[GumbleSoftmax]</a> to
ensure differentiability of the channel w.r.t. to the channel reliability
parameter. Please see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.discrete.html#learningshaping" id="id2">[LearningShaping]</a> for further details.
    
Setting-up:
```python
>>> bsc = BinarySymmetricChannel(return_llrs=False, bipolar_input=False)
```

    
Running:
```python
>>> x = tf.zeros((128,)) # x is the channel input
>>> pb = 0.1 # pb is the bit flipping probability
>>> y = bsc((x, pb))
```
## BinaryMemorylessChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.discrete.html#binarymemorylesschannel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``BinaryMemorylessChannel`(<em class="sig-param">`return_llrs``=``False`</em>, <em class="sig-param">`bipolar_input``=``False`</em>, <em class="sig-param">`llr_max``=``100.`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/channel/discrete_channel.html#BinaryMemorylessChannel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.discrete.html#sionna.channel.BinaryMemorylessChannel" title="Permalink to this definition"></a>
    
Discrete binary memory less channel with (possibly) asymmetric bit flipping
probabilities.
    
Inputs bits are flipped with probability $p_\text{b,0}$ and
$p_\text{b,1}$, respectively.
<img alt="../_images/BMC_channel.png" src="https://nvlabs.github.io/sionna/_images/BMC_channel.png" />
    
This layer supports binary inputs ($x \in \{0, 1\}$) and <cite>bipolar</cite>
inputs ($x \in \{-1, 1\}$).
    
If activated, the channel directly returns log-likelihood ratios (LLRs)
defined as

$$
\begin{split}\ell =
\begin{cases}
    \operatorname{log} \frac{p_{b,1}}{1-p_{b,0}}, \qquad \text{if} \, y=0 \\
    \operatorname{log} \frac{1-p_{b,1}}{p_{b,0}}, \qquad \text{if} \, y=1 \\
\end{cases}\end{split}
$$
    
The error probability $p_\text{b}$ can be either scalar or a
tensor (broadcastable to the shape of the input). This allows
different erasure probabilities per bit position. In any case, its last
dimension must be of length 2 and is interpreted as $p_\text{b,0}$ and
$p_\text{b,1}$.
    
This class inherits from the Keras <cite>Layer</cite> class and can be used as layer in
a Keras model.
Parameters
 
- **return_llrs** (<em>bool</em>) – Defaults to <cite>False</cite>. If <cite>True</cite>, the layer returns log-likelihood ratios
instead of binary values based on `pb`.
- **bipolar_input** (<em>bool</em><em>, </em><em>False</em>) – Defaults to <cite>False</cite>. If <cite>True</cite>, the expected input is given as
$\{-1,1\}$ instead of $\{0,1\}$.
- **llr_max** (<em>tf.float</em>) – Defaults to 100. Defines the clipping value of the LLRs.
- **dtype** (<em>tf.DType</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.float32</cite>.


Input
 
- **(x, pb)** – Tuple:
- **x** (<em>[…,n], tf.float32</em>) – Input sequence to the channel consisting of binary values $\{0,1\}
${-1,1}`, respectively.
- **pb** (<em>[…,2], tf.float32</em>) – Error probability. Can be a tuple of two scalars or of any
shape that can be broadcasted to the shape of `x`. It has an
additional last dimension which is interpreted as $p_\text{b,0}$
and $p_\text{b,1}$.


Output
    
<em>[…,n], tf.float32</em> – Output sequence of same length as the input `x`. If
`return_llrs` is <cite>False</cite>, the output is ternary where a <cite>-1</cite> and
<cite>0</cite> indicate an erasure for the binary and bipolar input,
respectively.



<em class="property">`property` </em>`llr_max`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.discrete.html#sionna.channel.BinaryMemorylessChannel.llr_max" title="Permalink to this definition"></a>
    
Maximum value used for LLR calculations.


<em class="property">`property` </em>`temperature`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.discrete.html#sionna.channel.BinaryMemorylessChannel.temperature" title="Permalink to this definition"></a>
    
Temperature for Gumble-softmax trick.


## BinarySymmetricChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.discrete.html#binarysymmetricchannel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``BinarySymmetricChannel`(<em class="sig-param">`return_llrs``=``False`</em>, <em class="sig-param">`bipolar_input``=``False`</em>, <em class="sig-param">`llr_max``=``100.`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/channel/discrete_channel.html#BinarySymmetricChannel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.discrete.html#sionna.channel.BinarySymmetricChannel" title="Permalink to this definition"></a>
    
Discrete binary symmetric channel which randomly flips bits with probability
$p_\text{b}$.
<img alt="../_images/BSC_channel.png" src="https://nvlabs.github.io/sionna/_images/BSC_channel.png" />
    
This layer supports binary inputs ($x \in \{0, 1\}$) and <cite>bipolar</cite>
inputs ($x \in \{-1, 1\}$).
    
If activated, the channel directly returns log-likelihood ratios (LLRs)
defined as

$$
\begin{split}\ell =
\begin{cases}
    \operatorname{log} \frac{p_{b}}{1-p_{b}}, \qquad \text{if}\, y=0 \\
    \operatorname{log} \frac{1-p_{b}}{p_{b}}, \qquad \text{if}\, y=1 \\
\end{cases}\end{split}
$$
    
where $y$ denotes the binary output of the channel.
    
The bit flipping probability $p_\text{b}$ can be either a scalar or  a
tensor (broadcastable to the shape of the input). This allows
different bit flipping probabilities per bit position.
    
This class inherits from the Keras <cite>Layer</cite> class and can be used as layer in
a Keras model.
Parameters
 
- **return_llrs** (<em>bool</em>) – Defaults to <cite>False</cite>. If <cite>True</cite>, the layer returns log-likelihood ratios
instead of binary values based on `pb`.
- **bipolar_input** (<em>bool</em><em>, </em><em>False</em>) – Defaults to <cite>False</cite>. If <cite>True</cite>, the expected input is given as {-1,1}
instead of {0,1}.
- **llr_max** (<em>tf.float</em>) – Defaults to 100. Defines the clipping value of the LLRs.
- **dtype** (<em>tf.DType</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.float32</cite>.


Input
 
- **(x, pb)** – Tuple:
- **x** (<em>[…,n], tf.float32</em>) – Input sequence to the channel.
- **pb** (<em>tf.float32</em>) – Bit flipping probability. Can be a scalar or of any shape that
can be broadcasted to the shape of `x`.


Output
    
<em>[…,n], tf.float32</em> – Output sequence of same length as the input `x`. If
`return_llrs` is <cite>False</cite>, the output is binary and otherwise
soft-values are returned.



## BinaryErasureChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.discrete.html#binaryerasurechannel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``BinaryErasureChannel`(<em class="sig-param">`return_llrs``=``False`</em>, <em class="sig-param">`bipolar_input``=``False`</em>, <em class="sig-param">`llr_max``=``100.`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/channel/discrete_channel.html#BinaryErasureChannel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.discrete.html#sionna.channel.BinaryErasureChannel" title="Permalink to this definition"></a>
    
Binary erasure channel (BEC) where a bit is either correctly received
or erased.
    
In the binary erasure channel, bits are always correctly received or erased
with erasure probability $p_\text{b}$.
<img alt="../_images/BEC_channel.png" src="https://nvlabs.github.io/sionna/_images/BEC_channel.png" />
    
This layer supports binary inputs ($x \in \{0, 1\}$) and <cite>bipolar</cite>
inputs ($x \in \{-1, 1\}$).
    
If activated, the channel directly returns log-likelihood ratios (LLRs)
defined as

$$
\begin{split}\ell =
\begin{cases}
    -\infty, \qquad \text{if} \, y=0 \\
    0, \qquad \quad \,\, \text{if} \, y=? \\
    \infty, \qquad \quad \text{if} \, y=1 \\
\end{cases}\end{split}
$$
    
The erasure probability $p_\text{b}$ can be either a scalar or a
tensor (broadcastable to the shape of the input). This allows
different erasure probabilities per bit position.
    
Please note that the output of the BEC is ternary. Hereby, <cite>-1</cite> indicates an
erasure for the binary configuration and <cite>0</cite> for the bipolar mode,
respectively.
    
This class inherits from the Keras <cite>Layer</cite> class and can be used as layer in
a Keras model.
Parameters
 
- **return_llrs** (<em>bool</em>) – Defaults to <cite>False</cite>. If <cite>True</cite>, the layer returns log-likelihood ratios
instead of binary values based on `pb`.
- **bipolar_input** (<em>bool</em><em>, </em><em>False</em>) – Defaults to <cite>False</cite>. If <cite>True</cite>, the expected input is given as {-1,1}
instead of {0,1}.
- **llr_max** (<em>tf.float</em>) – Defaults to 100. Defines the clipping value of the LLRs.
- **dtype** (<em>tf.DType</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.float32</cite>.


Input
 
- **(x, pb)** – Tuple:
- **x** (<em>[…,n], tf.float32</em>) – Input sequence to the channel.
- **pb** (<em>tf.float32</em>) – Erasure probability. Can be a scalar or of any shape that can be
broadcasted to the shape of `x`.


Output
    
<em>[…,n], tf.float32</em> – Output sequence of same length as the input `x`. If
`return_llrs` is <cite>False</cite>, the output is ternary where each <cite>-1</cite>
and each <cite>0</cite> indicate an erasure for the binary and bipolar input,
respectively.



## BinaryZChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.discrete.html#binaryzchannel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``BinaryZChannel`(<em class="sig-param">`return_llrs``=``False`</em>, <em class="sig-param">`bipolar_input``=``False`</em>, <em class="sig-param">`llr_max``=``100.`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/channel/discrete_channel.html#BinaryZChannel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.discrete.html#sionna.channel.BinaryZChannel" title="Permalink to this definition"></a>
    
Layer that implements the binary Z-channel.
    
In the Z-channel, transmission errors only occur for the transmission of
second input element (i.e., if a <cite>1</cite> is transmitted) with error probability
probability $p_\text{b}$ but the first element is always correctly
received.
<img alt="../_images/Z_channel.png" src="https://nvlabs.github.io/sionna/_images/Z_channel.png" />
    
This layer supports binary inputs ($x \in \{0, 1\}$) and <cite>bipolar</cite>
inputs ($x \in \{-1, 1\}$).
    
If activated, the channel directly returns log-likelihood ratios (LLRs)
defined as

$$
\begin{split}\ell =
\begin{cases}
    \operatorname{log} \left( p_b \right), \qquad \text{if} \, y=0 \\
    \infty, \qquad \qquad \text{if} \, y=1 \\
\end{cases}\end{split}
$$
    
assuming equal probable inputs $P(X=0) = P(X=1) = 0.5$.
    
The error probability $p_\text{b}$ can be either a scalar or a
tensor (broadcastable to the shape of the input). This allows
different error probabilities per bit position.
    
This class inherits from the Keras <cite>Layer</cite> class and can be used as layer in
a Keras model.
Parameters
 
- **return_llrs** (<em>bool</em>) – Defaults to <cite>False</cite>. If <cite>True</cite>, the layer returns log-likelihood ratios
instead of binary values based on `pb`.
- **bipolar_input** (<em>bool</em><em>, </em><em>False</em>) – Defaults to <cite>False</cite>. If True, the expected input is given as {-1,1}
instead of {0,1}.
- **llr_max** (<em>tf.float</em>) – Defaults to 100. Defines the clipping value of the LLRs.
- **dtype** (<em>tf.DType</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.float32</cite>.


Input
 
- **(x, pb)** – Tuple:
- **x** (<em>[…,n], tf.float32</em>) – Input sequence to the channel.
- **pb** (<em>tf.float32</em>) – Error probability. Can be a scalar or of any shape that can be
broadcasted to the shape of `x`.


Output
    
<em>[…,n], tf.float32</em> – Output sequence of same length as the input `x`. If
`return_llrs` is <cite>False</cite>, the output is binary and otherwise
soft-values are returned.




References:
<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/channel.discrete.html#id1">GumbleSoftmax</a>
<ol class="upperalpha simple" start="5">
- Jang, G. Shixiang, and Ben Poole. <cite>“Categorical reparameterization with gumbel-softmax,”</cite> arXiv preprint arXiv:1611.01144 (2016).
</ol>

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/channel.discrete.html#id2">LearningShaping</a>
<ol class="upperalpha simple" start="13">
- Stark, F. Ait Aoudia, and J. Hoydis. <cite>“Joint learning of geometric and probabilistic constellation shaping,”</cite> 2019 IEEE Globecom Workshops (GC Wkshps). IEEE, 2019.
</ol>



