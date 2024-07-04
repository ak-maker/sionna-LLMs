
# Signal<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#signal" title="Permalink to this headline"></a>
    
This module contains classes and functions for <a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#filter">filtering</a> (pulse shaping), <a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#window">windowing</a>, and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#upsampling">up-</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#downsampling">downsampling</a>.
The following figure shows the different components that can be implemented using this module.
<a class="reference internal image-reference" href="../_images/signal_module.png"><img alt="../_images/signal_module.png" src="https://nvlabs.github.io/sionna/_images/signal_module.png" style="width: 75%;" /></a>
    
This module also contains <a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#utility">utility functions</a> for computing the (inverse) discrete Fourier transform (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#fft">FFT</a>/<a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#ifft">IFFT</a>), and for empirically computing the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#empirical-psd">power spectral density (PSD)</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#empirical-aclr">adjacent channel leakage ratio (ACLR)</a> of a signal.
    
The following code snippet shows how to filter a sequence of QAM baseband symbols using a root-raised-cosine filter with a Hann window:
```python
# Create batch of QAM-16 sequences
batch_size = 128
num_symbols = 1000
num_bits_per_symbol = 4
x = QAMSource(num_bits_per_symbol)([batch_size, num_symbols])
# Create a root-raised-cosine filter with Hann windowing
beta = 0.22 # Roll-off factor
span_in_symbols = 32 # Filter span in symbols
samples_per_symbol = 4 # Number of samples per symbol, i.e., the oversampling factor
rrcf_hann = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta, window="hann")
# Create instance of the Upsampling layer
us = Upsampling(samples_per_symbol)
# Upsample the baseband x
x_us = us(x)
# Filter the upsampled sequence
x_rrcf = rrcf_hann(x_us)
```

    
On the receiver side, one would recover the baseband symbols as follows:
```python
# Instantiate a downsampling layer
ds = Downsampling(samples_per_symbol, rrcf_hann.length-1, num_symbols)
# Apply the matched filter
x_mf = rrcf_hann(x_rrcf)
# Recover the transmitted symbol sequence
x_hat = ds(x_mf)
```
## Filters<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#filters" title="Permalink to this headline"></a>

### SincFilter<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sincfilter" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.signal.``SincFilter`(<em class="sig-param">`span_in_symbols`</em>, <em class="sig-param">`samples_per_symbol`</em>, <em class="sig-param">`window``=``None`</em>, <em class="sig-param">`normalize``=``True`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/signal/filter.html#SincFilter">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.SincFilter" title="Permalink to this definition"></a>
    
Layer for applying a sinc filter of `length` K
to an input `x` of length N.
    
The sinc filter is defined by

$$
h(t) = \frac{1}{T}\text{sinc}\left(\frac{t}{T}\right)
$$
    
where $T$ the symbol duration.
    
The filter length K is equal to the filter span in symbols (`span_in_symbols`)
multiplied by the oversampling factor (`samples_per_symbol`).
If this product is even, a value of one will be added.
    
The filter is applied through discrete convolution.
    
An optional windowing function `window` can be applied to the filter.
    
The <cite>dtype</cite> of the output is <cite>tf.float</cite> if both `x` and the filter coefficients have dtype <cite>tf.float</cite>.
Otherwise, the dtype of the output is <cite>tf.complex</cite>.
    
Three padding modes are available for applying the filter:
 
- “full” (default): Returns the convolution at each point of overlap between `x` and the filter.
The length of the output is N + K - 1. Zero-padding of the input `x` is performed to
compute the convolution at the borders.
- “same”: Returns an output of the same length as the input `x`. The convolution is computed such
that the coefficients of the input `x` are centered on the coefficient of the filter with index
(K-1)/2. Zero-padding of the input signal is performed to compute the convolution at the borders.
- “valid”: Returns the convolution only at points where `x` and the filter completely overlap.
The length of the output is N - K + 1.

Parameters
 
- **span_in_symbols** (<em>int</em>) – Filter span as measured by the number of symbols.
- **samples_per_symbol** (<em>int</em>) – Number of samples per symbol, i.e., the oversampling factor.
- **window** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window" title="sionna.signal.Window"><em>Window</em></a><em> or </em><em>string</em><em> (</em><em>[</em><em>"hann"</em><em>, </em><em>"hamming"</em><em>, </em><em>"blackman"</em><em>]</em><em>)</em>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window" title="sionna.signal.Window">`Window`</a> that is applied to the filter coefficients.
Alternatively, a string indicating the window name can be provided. In this case,
the chosen window will be instantiated with the default parameters. Custom windows
must be provided as instance.
- **normalize** (<em>bool</em>) – If <cite>True</cite>, the filter is normalized to have unit power.
Defaults to <cite>True</cite>.
- **trainable** (<em>bool</em>) – If <cite>True</cite>, the filter coefficients are trainable variables.
Defaults to <cite>False</cite>.
- **dtype** (<em>tf.DType</em>) – The <cite>dtype</cite> of the filter coefficients.
Defaults to <cite>tf.float32</cite>.


Input
 
- **x** (<em>[…, N], tf.complex or tf.float</em>) – The input to which the filter is applied.
The filter is applied along the last dimension.
- **padding** (<em>string ([“full”, “valid”, “same”])</em>) – Padding mode for convolving `x` and the filter.
Must be one of “full”, “valid”, or “same”. Case insensitive.
Defaults to “full”.
- **conjugate** (<em>bool</em>) – If <cite>True</cite>, the complex conjugate of the filter is applied.
Defaults to <cite>False</cite>.


Output
    
**y** (<em>[…,M], tf.complex or tf.float</em>) – Filtered input.
It is <cite>tf.float</cite> only if both `x` and the filter are <cite>tf.float</cite>.
It is <cite>tf.complex</cite> otherwise.
The length M depends on the `padding`.



<em class="property">`property` </em>`aclr`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.SincFilter.aclr" title="Permalink to this definition"></a>
    
ACLR of the filter
    
This ACLR corresponds to what one would obtain from using
this filter as pulse shaping filter on an i.i.d. sequence of symbols.
The in-band is assumed to range from [-0.5, 0.5] in normalized
frequency.


<em class="property">`property` </em>`coefficients`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.SincFilter.coefficients" title="Permalink to this definition"></a>
    
The filter coefficients (after normalization)


<em class="property">`property` </em>`length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.SincFilter.length" title="Permalink to this definition"></a>
    
The filter length in samples


<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.SincFilter.normalize" title="Permalink to this definition"></a>
    
<cite>True</cite> if the filter is normalized to have unit power. <cite>False</cite> otherwise.


<em class="property">`property` </em>`sampling_times`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.SincFilter.sampling_times" title="Permalink to this definition"></a>
    
Sampling times in multiples of the symbol duration


`show`(<em class="sig-param">`response``=``'impulse'`</em>, <em class="sig-param">`scale``=``'lin'`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.SincFilter.show" title="Permalink to this definition"></a>
    
Plot the impulse or magnitude response
    
Plots the impulse response (time domain) or magnitude response
(frequency domain) of the filter.
    
For the computation of the magnitude response, a minimum DFT size
of 1024 is assumed which is obtained through zero padding of
the filter coefficients in the time domain.
Input
 
- **response** (<em>str, one of [“impulse”, “magnitude”]</em>) – The desired response type.
Defaults to “impulse”
- **scale** (<em>str, one of [“lin”, “db”]</em>) – The y-scale of the magnitude response.
Can be “lin” (i.e., linear) or “db” (, i.e., Decibel).
Defaults to “lin”.





<em class="property">`property` </em>`trainable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.SincFilter.trainable" title="Permalink to this definition"></a>
    
<cite>True</cite> if the filter coefficients are trainable. <cite>False</cite> otherwise.


<em class="property">`property` </em>`window`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.SincFilter.window" title="Permalink to this definition"></a>
    
The window function that is applied to the filter coefficients. <cite>None</cite> if no window is applied.


### RaisedCosineFilter<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#raisedcosinefilter" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.signal.``RaisedCosineFilter`(<em class="sig-param">`span_in_symbols`</em>, <em class="sig-param">`samples_per_symbol`</em>, <em class="sig-param">`beta`</em>, <em class="sig-param">`window``=``None`</em>, <em class="sig-param">`normalize``=``True`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/signal/filter.html#RaisedCosineFilter">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RaisedCosineFilter" title="Permalink to this definition"></a>
    
Layer for applying a raised-cosine filter of `length` K
to an input `x` of length N.
    
The raised-cosine filter is defined by

$$
\begin{split}h(t) =
\begin{cases}
\frac{\pi}{4T} \text{sinc}\left(\frac{1}{2\beta}\right), & \text { if }t = \pm \frac{T}{2\beta}\\
\frac{1}{T}\text{sinc}\left(\frac{t}{T}\right)\frac{\cos\left(\frac{\pi\beta t}{T}\right)}{1-\left(\frac{2\beta t}{T}\right)^2}, & \text{otherwise}
\end{cases}\end{split}
$$
    
where $\beta$ is the roll-off factor and $T$ the symbol duration.
    
The filter length K is equal to the filter span in symbols (`span_in_symbols`)
multiplied by the oversampling factor (`samples_per_symbol`).
If this product is even, a value of one will be added.
    
The filter is applied through discrete convolution.
    
An optional windowing function `window` can be applied to the filter.
    
The <cite>dtype</cite> of the output is <cite>tf.float</cite> if both `x` and the filter coefficients have dtype <cite>tf.float</cite>.
Otherwise, the dtype of the output is <cite>tf.complex</cite>.
    
Three padding modes are available for applying the filter:
 
- “full” (default): Returns the convolution at each point of overlap between `x` and the filter.
The length of the output is N + K - 1. Zero-padding of the input `x` is performed to
compute the convolution at the borders.
- “same”: Returns an output of the same length as the input `x`. The convolution is computed such
that the coefficients of the input `x` are centered on the coefficient of the filter with index
(K-1)/2. Zero-padding of the input signal is performed to compute the convolution at the borders.
- “valid”: Returns the convolution only at points where `x` and the filter completely overlap.
The length of the output is N - K + 1.

Parameters
 
- **span_in_symbols** (<em>int</em>) – Filter span as measured by the number of symbols.
- **samples_per_symbol** (<em>int</em>) – Number of samples per symbol, i.e., the oversampling factor.
- **beta** (<em>float</em>) – Roll-off factor.
Must be in the range $[0,1]$.
- **window** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window" title="sionna.signal.Window"><em>Window</em></a><em> or </em><em>string</em><em> (</em><em>[</em><em>"hann"</em><em>, </em><em>"hamming"</em><em>, </em><em>"blackman"</em><em>]</em><em>)</em>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window" title="sionna.signal.Window">`Window`</a> that is applied to the filter coefficients.
Alternatively, a string indicating the window name can be provided. In this case,
the chosen window will be instantiated with the default parameters. Custom windows
must be provided as instance.
- **normalize** (<em>bool</em>) – If <cite>True</cite>, the filter is normalized to have unit power.
Defaults to <cite>True</cite>.
- **trainable** (<em>bool</em>) – If <cite>True</cite>, the filter coefficients are trainable variables.
Defaults to <cite>False</cite>.
- **dtype** (<em>tf.DType</em>) – The <cite>dtype</cite> of the filter coefficients.
Defaults to <cite>tf.float32</cite>.


Input
 
- **x** (<em>[…, N], tf.complex or tf.float</em>) – The input to which the filter is applied.
The filter is applied along the last dimension.
- **padding** (<em>string ([“full”, “valid”, “same”])</em>) – Padding mode for convolving `x` and the filter.
Must be one of “full”, “valid”, or “same”.
Defaults to “full”.
- **conjugate** (<em>bool</em>) – If <cite>True</cite>, the complex conjugate of the filter is applied.
Defaults to <cite>False</cite>.


Output
    
**y** (<em>[…,M], tf.complex or tf.float</em>) – Filtered input.
It is <cite>tf.float</cite> only if both `x` and the filter are <cite>tf.float</cite>.
It is <cite>tf.complex</cite> otherwise.
The length M depends on the `padding`.



<em class="property">`property` </em>`aclr`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RaisedCosineFilter.aclr" title="Permalink to this definition"></a>
    
ACLR of the filter
    
This ACLR corresponds to what one would obtain from using
this filter as pulse shaping filter on an i.i.d. sequence of symbols.
The in-band is assumed to range from [-0.5, 0.5] in normalized
frequency.


<em class="property">`property` </em>`beta`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RaisedCosineFilter.beta" title="Permalink to this definition"></a>
    
Roll-off factor


<em class="property">`property` </em>`coefficients`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RaisedCosineFilter.coefficients" title="Permalink to this definition"></a>
    
The filter coefficients (after normalization)


<em class="property">`property` </em>`length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RaisedCosineFilter.length" title="Permalink to this definition"></a>
    
The filter length in samples


<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RaisedCosineFilter.normalize" title="Permalink to this definition"></a>
    
<cite>True</cite> if the filter is normalized to have unit power. <cite>False</cite> otherwise.


<em class="property">`property` </em>`sampling_times`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RaisedCosineFilter.sampling_times" title="Permalink to this definition"></a>
    
Sampling times in multiples of the symbol duration


`show`(<em class="sig-param">`response``=``'impulse'`</em>, <em class="sig-param">`scale``=``'lin'`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RaisedCosineFilter.show" title="Permalink to this definition"></a>
    
Plot the impulse or magnitude response
    
Plots the impulse response (time domain) or magnitude response
(frequency domain) of the filter.
    
For the computation of the magnitude response, a minimum DFT size
of 1024 is assumed which is obtained through zero padding of
the filter coefficients in the time domain.
Input
 
- **response** (<em>str, one of [“impulse”, “magnitude”]</em>) – The desired response type.
Defaults to “impulse”
- **scale** (<em>str, one of [“lin”, “db”]</em>) – The y-scale of the magnitude response.
Can be “lin” (i.e., linear) or “db” (, i.e., Decibel).
Defaults to “lin”.





<em class="property">`property` </em>`trainable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RaisedCosineFilter.trainable" title="Permalink to this definition"></a>
    
<cite>True</cite> if the filter coefficients are trainable. <cite>False</cite> otherwise.


<em class="property">`property` </em>`window`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RaisedCosineFilter.window" title="Permalink to this definition"></a>
    
The window function that is applied to the filter coefficients. <cite>None</cite> if no window is applied.


### RootRaisedCosineFilter<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#rootraisedcosinefilter" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.signal.``RootRaisedCosineFilter`(<em class="sig-param">`span_in_symbols`</em>, <em class="sig-param">`samples_per_symbol`</em>, <em class="sig-param">`beta`</em>, <em class="sig-param">`window``=``None`</em>, <em class="sig-param">`normalize``=``True`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/signal/filter.html#RootRaisedCosineFilter">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RootRaisedCosineFilter" title="Permalink to this definition"></a>
    
Layer for applying a root-raised-cosine filter of `length` K
to an input `x` of length N.
    
The root-raised-cosine filter is defined by

$$
\begin{split}h(t) =
\begin{cases}
\frac{1}{T} \left(1 + \beta\left(\frac{4}{\pi}-1\right) \right), & \text { if }t = 0\\
\frac{\beta}{T\sqrt{2}} \left[ \left(1+\frac{2}{\pi}\right)\sin\left(\frac{\pi}{4\beta}\right) + \left(1-\frac{2}{\pi}\right)\cos\left(\frac{\pi}{4\beta}\right) \right], & \text { if }t = \pm\frac{T}{4\beta} \\
\frac{1}{T} \frac{\sin\left(\pi\frac{t}{T}(1-\beta)\right) + 4\beta\frac{t}{T}\cos\left(\pi\frac{t}{T}(1+\beta)\right)}{\pi\frac{t}{T}\left(1-\left(4\beta\frac{t}{T}\right)^2\right)}, & \text { otherwise}
\end{cases}\end{split}
$$
    
where $\beta$ is the roll-off factor and $T$ the symbol duration.
    
The filter length K is equal to the filter span in symbols (`span_in_symbols`)
multiplied by the oversampling factor (`samples_per_symbol`).
If this product is even, a value of one will be added.
    
The filter is applied through discrete convolution.
    
An optional windowing function `window` can be applied to the filter.
    
The <cite>dtype</cite> of the output is <cite>tf.float</cite> if both `x` and the filter coefficients have dtype <cite>tf.float</cite>.
Otherwise, the dtype of the output is <cite>tf.complex</cite>.
    
Three padding modes are available for applying the filter:
 
- “full” (default): Returns the convolution at each point of overlap between `x` and the filter.
The length of the output is N + K - 1. Zero-padding of the input `x` is performed to
compute the convolution at the borders.
- “same”: Returns an output of the same length as the input `x`. The convolution is computed such
that the coefficients of the input `x` are centered on the coefficient of the filter with index
(K-1)/2. Zero-padding of the input signal is performed to compute the convolution at the borders.
- “valid”: Returns the convolution only at points where `x` and the filter completely overlap.
The length of the output is N - K + 1.

Parameters
 
- **span_in_symbols** (<em>int</em>) – Filter span as measured by the number of symbols.
- **samples_per_symbol** (<em>int</em>) – Number of samples per symbol, i.e., the oversampling factor.
- **beta** (<em>float</em>) – Roll-off factor.
Must be in the range $[0,1]$.
- **window** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window" title="sionna.signal.Window"><em>Window</em></a><em> or </em><em>string</em><em> (</em><em>[</em><em>"hann"</em><em>, </em><em>"hamming"</em><em>, </em><em>"blackman"</em><em>]</em><em>)</em>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window" title="sionna.signal.Window">`Window`</a> that is applied to the filter coefficients.
Alternatively, a string indicating the window name can be provided. In this case,
the chosen window will be instantiated with the default parameters. Custom windows
must be provided as instance.
- **normalize** (<em>bool</em>) – If <cite>True</cite>, the filter is normalized to have unit power.
Defaults to <cite>True</cite>.
- **trainable** (<em>bool</em>) – If <cite>True</cite>, the filter coefficients are trainable variables.
Defaults to <cite>False</cite>.
- **dtype** (<em>tf.DType</em>) – The <cite>dtype</cite> of the filter coefficients.
Defaults to <cite>tf.float32</cite>.


Input
 
- **x** (<em>[…, N], tf.complex or tf.float</em>) – The input to which the filter is applied.
The filter is applied along the last dimension.
- **padding** (<em>string ([“full”, “valid”, “same”])</em>) – Padding mode for convolving `x` and the filter.
Must be one of “full”, “valid”, or “same”. Case insensitive.
Defaults to “full”.
- **conjugate** (<em>bool</em>) – If <cite>True</cite>, the complex conjugate of the filter is applied.
Defaults to <cite>False</cite>.


Output
    
**y** (<em>[…,M], tf.complex or tf.float</em>) – Filtered input.
It is <cite>tf.float</cite> only if both `x` and the filter are <cite>tf.float</cite>.
It is <cite>tf.complex</cite> otherwise.
The length M depends on the `padding`.



<em class="property">`property` </em>`aclr`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RootRaisedCosineFilter.aclr" title="Permalink to this definition"></a>
    
ACLR of the filter
    
This ACLR corresponds to what one would obtain from using
this filter as pulse shaping filter on an i.i.d. sequence of symbols.
The in-band is assumed to range from [-0.5, 0.5] in normalized
frequency.


<em class="property">`property` </em>`beta`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RootRaisedCosineFilter.beta" title="Permalink to this definition"></a>
    
Roll-off factor


<em class="property">`property` </em>`coefficients`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RootRaisedCosineFilter.coefficients" title="Permalink to this definition"></a>
    
The filter coefficients (after normalization)


<em class="property">`property` </em>`length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RootRaisedCosineFilter.length" title="Permalink to this definition"></a>
    
The filter length in samples


<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RootRaisedCosineFilter.normalize" title="Permalink to this definition"></a>
    
<cite>True</cite> if the filter is normalized to have unit power. <cite>False</cite> otherwise.


<em class="property">`property` </em>`sampling_times`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RootRaisedCosineFilter.sampling_times" title="Permalink to this definition"></a>
    
Sampling times in multiples of the symbol duration


`show`(<em class="sig-param">`response``=``'impulse'`</em>, <em class="sig-param">`scale``=``'lin'`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RootRaisedCosineFilter.show" title="Permalink to this definition"></a>
    
Plot the impulse or magnitude response
    
Plots the impulse response (time domain) or magnitude response
(frequency domain) of the filter.
    
For the computation of the magnitude response, a minimum DFT size
of 1024 is assumed which is obtained through zero padding of
the filter coefficients in the time domain.
Input
 
- **response** (<em>str, one of [“impulse”, “magnitude”]</em>) – The desired response type.
Defaults to “impulse”
- **scale** (<em>str, one of [“lin”, “db”]</em>) – The y-scale of the magnitude response.
Can be “lin” (i.e., linear) or “db” (, i.e., Decibel).
Defaults to “lin”.





<em class="property">`property` </em>`trainable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RootRaisedCosineFilter.trainable" title="Permalink to this definition"></a>
    
<cite>True</cite> if the filter coefficients are trainable. <cite>False</cite> otherwise.


<em class="property">`property` </em>`window`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RootRaisedCosineFilter.window" title="Permalink to this definition"></a>
    
The window function that is applied to the filter coefficients. <cite>None</cite> if no window is applied.


### CustomFilter<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#customfilter" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.signal.``CustomFilter`(<em class="sig-param">`span_in_symbols``=``None`</em>, <em class="sig-param">`samples_per_symbol``=``None`</em>, <em class="sig-param">`coefficients``=``None`</em>, <em class="sig-param">`window``=``None`</em>, <em class="sig-param">`normalize``=``True`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/signal/filter.html#CustomFilter">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomFilter" title="Permalink to this definition"></a>
    
Layer for applying a custom filter of `length` K
to an input `x` of length N.
    
The filter length K is equal to the filter span in symbols (`span_in_symbols`)
multiplied by the oversampling factor (`samples_per_symbol`).
If this product is even, a value of one will be added.
    
The filter is applied through discrete convolution.
    
An optional windowing function `window` can be applied to the filter.
    
The <cite>dtype</cite> of the output is <cite>tf.float</cite> if both `x` and the filter coefficients have dtype <cite>tf.float</cite>.
Otherwise, the dtype of the output is <cite>tf.complex</cite>.
    
Three padding modes are available for applying the filter:
 
- “full” (default): Returns the convolution at each point of overlap between `x` and the filter.
The length of the output is N + K - 1. Zero-padding of the input `x` is performed to
compute the convolution at the borders.
- “same”: Returns an output of the same length as the input `x`. The convolution is computed such
that the coefficients of the input `x` are centered on the coefficient of the filter with index
(K-1)/2. Zero-padding of the input signal is performed to compute the convolution at the borders.
- “valid”: Returns the convolution only at points where `x` and the filter completely overlap.
The length of the output is N - K + 1.

Parameters
 
- **span_in_symbols** (<em>int</em>) – Filter span as measured by the number of symbols.
Only needs to be provided if `coefficients` is None.
- **samples_per_symbol** (<em>int</em>) – Number of samples per symbol, i.e., the oversampling factor.
Must always be provided.
- **coefficients** (<em>[</em><em>K</em><em>]</em><em>, </em><em>tf.float</em><em> or </em><em>tf.complex</em>) – Optional filter coefficients.
If set to <cite>None</cite>, then a random filter of K is generated
by sampling a Gaussian distribution. Defaults to <cite>None</cite>.
- **window** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window" title="sionna.signal.Window"><em>Window</em></a><em> or </em><em>string</em><em> (</em><em>[</em><em>"hann"</em><em>, </em><em>"hamming"</em><em>, </em><em>"blackman"</em><em>]</em><em>)</em>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window" title="sionna.signal.Window">`Window`</a> that is applied to the filter coefficients.
Alternatively, a string indicating the window name can be provided. In this case,
the chosen window will be instantiated with the default parameters. Custom windows
must be provided as instance.
- **normalize** (<em>bool</em>) – If <cite>True</cite>, the filter is normalized to have unit power.
Defaults to <cite>True</cite>.
- **trainable** (<em>bool</em>) – If <cite>True</cite>, the filter coefficients are trainable variables.
Defaults to <cite>False</cite>.
- **dtype** (<em>tf.DType</em>) – The <cite>dtype</cite> of the filter coefficients.
Defaults to <cite>tf.float32</cite>.


Input
 
- **x** (<em>[…, N], tf.complex or tf.float</em>) – The input to which the filter is applied.
The filter is applied along the last dimension.
- **padding** (<em>string ([“full”, “valid”, “same”])</em>) – Padding mode for convolving `x` and the filter.
Must be one of “full”, “valid”, or “same”. Case insensitive.
Defaults to “full”.
- **conjugate** (<em>bool</em>) – If <cite>True</cite>, the complex conjugate of the filter is applied.
Defaults to <cite>False</cite>.


Output
    
**y** (<em>[…,M], tf.complex or tf.float</em>) – Filtered input.
It is <cite>tf.float</cite> only if both `x` and the filter are <cite>tf.float</cite>.
It is <cite>tf.complex</cite> otherwise.
The length M depends on the `padding`.



<em class="property">`property` </em>`aclr`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomFilter.aclr" title="Permalink to this definition"></a>
    
ACLR of the filter
    
This ACLR corresponds to what one would obtain from using
this filter as pulse shaping filter on an i.i.d. sequence of symbols.
The in-band is assumed to range from [-0.5, 0.5] in normalized
frequency.


<em class="property">`property` </em>`coefficients`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomFilter.coefficients" title="Permalink to this definition"></a>
    
The filter coefficients (after normalization)


<em class="property">`property` </em>`length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomFilter.length" title="Permalink to this definition"></a>
    
The filter length in samples


<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomFilter.normalize" title="Permalink to this definition"></a>
    
<cite>True</cite> if the filter is normalized to have unit power. <cite>False</cite> otherwise.


<em class="property">`property` </em>`sampling_times`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomFilter.sampling_times" title="Permalink to this definition"></a>
    
Sampling times in multiples of the symbol duration


`show`(<em class="sig-param">`response``=``'impulse'`</em>, <em class="sig-param">`scale``=``'lin'`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomFilter.show" title="Permalink to this definition"></a>
    
Plot the impulse or magnitude response
    
Plots the impulse response (time domain) or magnitude response
(frequency domain) of the filter.
    
For the computation of the magnitude response, a minimum DFT size
of 1024 is assumed which is obtained through zero padding of
the filter coefficients in the time domain.
Input
 
- **response** (<em>str, one of [“impulse”, “magnitude”]</em>) – The desired response type.
Defaults to “impulse”
- **scale** (<em>str, one of [“lin”, “db”]</em>) – The y-scale of the magnitude response.
Can be “lin” (i.e., linear) or “db” (, i.e., Decibel).
Defaults to “lin”.





<em class="property">`property` </em>`trainable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomFilter.trainable" title="Permalink to this definition"></a>
    
<cite>True</cite> if the filter coefficients are trainable. <cite>False</cite> otherwise.


<em class="property">`property` </em>`window`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomFilter.window" title="Permalink to this definition"></a>
    
The window function that is applied to the filter coefficients. <cite>None</cite> if no window is applied.


### Filter<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#id1" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.signal.``Filter`(<em class="sig-param">`span_in_symbols`</em>, <em class="sig-param">`samples_per_symbol`</em>, <em class="sig-param">`window``=``None`</em>, <em class="sig-param">`normalize``=``True`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/signal/filter.html#Filter">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Filter" title="Permalink to this definition"></a>
    
This is an abtract class for defining a filter of `length` K which can be
applied to an input `x` of length N.
    
The filter length K is equal to the filter span in symbols (`span_in_symbols`)
multiplied by the oversampling factor (`samples_per_symbol`).
If this product is even, a value of one will be added.
    
The filter is applied through discrete convolution.
    
An optional windowing function `window` can be applied to the filter.
    
The <cite>dtype</cite> of the output is <cite>tf.float</cite> if both `x` and the filter coefficients have dtype <cite>tf.float</cite>.
Otherwise, the dtype of the output is <cite>tf.complex</cite>.
    
Three padding modes are available for applying the filter:
 
- “full” (default): Returns the convolution at each point of overlap between `x` and the filter.
The length of the output is N + K - 1. Zero-padding of the input `x` is performed to
compute the convolution at the borders.
- “same”: Returns an output of the same length as the input `x`. The convolution is computed such
that the coefficients of the input `x` are centered on the coefficient of the filter with index
(K-1)/2. Zero-padding of the input signal is performed to compute the convolution at the borders.
- “valid”: Returns the convolution only at points where `x` and the filter completely overlap.
The length of the output is N - K + 1.

Parameters
 
- **span_in_symbols** (<em>int</em>) – Filter span as measured by the number of symbols.
- **samples_per_symbol** (<em>int</em>) – Number of samples per symbol, i.e., the oversampling factor.
- **window** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window" title="sionna.signal.Window"><em>Window</em></a><em> or </em><em>string</em><em> (</em><em>[</em><em>"hann"</em><em>, </em><em>"hamming"</em><em>, </em><em>"blackman"</em><em>]</em><em>)</em>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window" title="sionna.signal.Window">`Window`</a> that is applied to the filter coefficients.
Alternatively, a string indicating the window name can be provided. In this case,
the chosen window will be instantiated with the default parameters. Custom windows
must be provided as instance.
- **normalize** (<em>bool</em>) – If <cite>True</cite>, the filter is normalized to have unit power.
Defaults to <cite>True</cite>.
- **trainable** (<em>bool</em>) – If <cite>True</cite>, the filter coefficients are trainable.
Defaults to <cite>False</cite>.
- **dtype** (<em>tf.DType</em>) – The <cite>dtype</cite> of the filter coefficients.
Defaults to <cite>tf.float32</cite>.


Input
 
- **x** (<em>[…, N], tf.complex or tf.float</em>) – The input to which the filter is applied.
The filter is applied along the last dimension.
- **padding** (<em>string ([“full”, “valid”, “same”])</em>) – Padding mode for convolving `x` and the filter.
Must be one of “full”, “valid”, or “same”. Case insensitive.
Defaults to “full”.
- **conjugate** (<em>bool</em>) – If <cite>True</cite>, the complex conjugate of the filter is applied.
Defaults to <cite>False</cite>.


Output
    
**y** (<em>[…,M], tf.complex or tf.float</em>) – Filtered input.
It is <cite>tf.float</cite> only if both `x` and the filter are <cite>tf.float</cite>.
It is <cite>tf.complex</cite> otherwise.
The length M depends on the `padding`.



<em class="property">`property` </em>`aclr`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Filter.aclr" title="Permalink to this definition"></a>
    
ACLR of the filter
    
This ACLR corresponds to what one would obtain from using
this filter as pulse shaping filter on an i.i.d. sequence of symbols.
The in-band is assumed to range from [-0.5, 0.5] in normalized
frequency.


<em class="property">`property` </em>`coefficients`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Filter.coefficients" title="Permalink to this definition"></a>
    
The filter coefficients (after normalization)


<em class="property">`property` </em>`length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Filter.length" title="Permalink to this definition"></a>
    
The filter length in samples


<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Filter.normalize" title="Permalink to this definition"></a>
    
<cite>True</cite> if the filter is normalized to have unit power. <cite>False</cite> otherwise.


<em class="property">`property` </em>`sampling_times`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Filter.sampling_times" title="Permalink to this definition"></a>
    
Sampling times in multiples of the symbol duration


`show`(<em class="sig-param">`response``=``'impulse'`</em>, <em class="sig-param">`scale``=``'lin'`</em>)<a class="reference internal" href="../_modules/sionna/signal/filter.html#Filter.show">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Filter.show" title="Permalink to this definition"></a>
    
Plot the impulse or magnitude response
    
Plots the impulse response (time domain) or magnitude response
(frequency domain) of the filter.
    
For the computation of the magnitude response, a minimum DFT size
of 1024 is assumed which is obtained through zero padding of
the filter coefficients in the time domain.
Input
 
- **response** (<em>str, one of [“impulse”, “magnitude”]</em>) – The desired response type.
Defaults to “impulse”
- **scale** (<em>str, one of [“lin”, “db”]</em>) – The y-scale of the magnitude response.
Can be “lin” (i.e., linear) or “db” (, i.e., Decibel).
Defaults to “lin”.





<em class="property">`property` </em>`trainable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Filter.trainable" title="Permalink to this definition"></a>
    
<cite>True</cite> if the filter coefficients are trainable. <cite>False</cite> otherwise.


<em class="property">`property` </em>`window`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Filter.window" title="Permalink to this definition"></a>
    
The window function that is applied to the filter coefficients. <cite>None</cite> if no window is applied.


## Window functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#window-functions" title="Permalink to this headline"></a>

### HannWindow<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#hannwindow" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.signal.``HannWindow`(<em class="sig-param">`length`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`normalize``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/signal/window.html#HannWindow">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.HannWindow" title="Permalink to this definition"></a>
    
Layer for applying a Hann window function of length `length` to an input `x` of the same length.
    
The window function is applied through element-wise multiplication.
    
The window function is real-valued, i.e., has <cite>tf.float</cite> as <cite>dtype</cite>.
The <cite>dtype</cite> of the output is the same as the <cite>dtype</cite> of the input `x` to which the window function is applied.
The window function and the input must have the same precision.
    
The Hann window is defined by

$$
w_n = \sin^2 \left( \frac{\pi n}{N} \right), 0 \leq n \leq N-1
$$
    
where $N$ is the window length.
Parameters
 
- **length** (<em>int</em>) – Window length (number of samples).
- **trainable** (<em>bool</em>) – If <cite>True</cite>, the window coefficients are trainable variables.
Defaults to <cite>False</cite>.
- **normalize** (<em>bool</em>) – If <cite>True</cite>, the window is normalized to have unit average power
per coefficient.
Defaults to <cite>False</cite>.
- **dtype** (<em>tf.DType</em>) – The <cite>dtype</cite> of the filter coefficients.
Must be either <cite>tf.float32</cite> or <cite>tf.float64</cite>.
Defaults to <cite>tf.float32</cite>.


Input
    
**x** (<em>[…, N], tf.complex or tf.float</em>) – The input to which the window function is applied.
The window function is applied along the last dimension.
The length of the last dimension `N` must be the same as the `length` of the window function.

Output
    
**y** (<em>[…,N], tf.complex or tf.float</em>) – Output of the windowing operation.
The output has the same shape and <cite>dtype</cite> as the input `x`.



<em class="property">`property` </em>`coefficients`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.HannWindow.coefficients" title="Permalink to this definition"></a>
    
The window coefficients (after normalization)


<em class="property">`property` </em>`length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.HannWindow.length" title="Permalink to this definition"></a>
    
Window length in number of samples


<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.HannWindow.normalize" title="Permalink to this definition"></a>
    
<cite>True</cite> if the window is normalized to have unit average power per coefficient. <cite>False</cite>
otherwise.


`show`(<em class="sig-param">`samples_per_symbol`</em>, <em class="sig-param">`domain``=``'time'`</em>, <em class="sig-param">`scale``=``'lin'`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.HannWindow.show" title="Permalink to this definition"></a>
    
Plot the window in time or frequency domain
    
For the computation of the Fourier transform, a minimum DFT size
of 1024 is assumed which is obtained through zero padding of
the window coefficients in the time domain.
Input
 
- **samples_per_symbol** (<em>int</em>) – Number of samples per symbol, i.e., the oversampling factor.
- **domain** (<em>str, one of [“time”, “frequency”]</em>) – The desired domain.
Defaults to “time”
- **scale** (<em>str, one of [“lin”, “db”]</em>) – The y-scale of the magnitude in the frequency domain.
Can be “lin” (i.e., linear) or “db” (, i.e., Decibel).
Defaults to “lin”.





<em class="property">`property` </em>`trainable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.HannWindow.trainable" title="Permalink to this definition"></a>
    
<cite>True</cite> if the window coefficients are trainable. <cite>False</cite> otherwise.


### HammingWindow<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#hammingwindow" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.signal.``HammingWindow`(<em class="sig-param">`length`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`normalize``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/signal/window.html#HammingWindow">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.HammingWindow" title="Permalink to this definition"></a>
    
Layer for applying a Hamming window function of length `length` to an input `x` of the same length.
    
The window function is applied through element-wise multiplication.
    
The window function is real-valued, i.e., has <cite>tf.float</cite> as <cite>dtype</cite>.
The <cite>dtype</cite> of the output is the same as the <cite>dtype</cite> of the input `x` to which the window function is applied.
The window function and the input must have the same precision.
    
The Hamming window is defined by

$$
w_n = a_0 - (1-a_0) \cos \left( \frac{2 \pi n}{N} \right), 0 \leq n \leq N-1
$$
    
where $N$ is the window length and $a_0 = \frac{25}{46}$.
Parameters
 
- **length** (<em>int</em>) – Window length (number of samples).
- **trainable** (<em>bool</em>) – If <cite>True</cite>, the window coefficients are trainable variables.
Defaults to <cite>False</cite>.
- **normalize** (<em>bool</em>) – If <cite>True</cite>, the window is normalized to have unit average power
per coefficient.
Defaults to <cite>False</cite>.
- **dtype** (<em>tf.DType</em>) – The <cite>dtype</cite> of the filter coefficients.
Must be either <cite>tf.float32</cite> or <cite>tf.float64</cite>.
Defaults to <cite>tf.float32</cite>.


Input
    
**x** (<em>[…, N], tf.complex or tf.float</em>) – The input to which the window function is applied.
The window function is applied along the last dimension.
The length of the last dimension `N` must be the same as the `length` of the window function.

Output
    
**y** (<em>[…,N], tf.complex or tf.float</em>) – Output of the windowing operation.
The output has the same shape and <cite>dtype</cite> as the input `x`.



<em class="property">`property` </em>`coefficients`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.HammingWindow.coefficients" title="Permalink to this definition"></a>
    
The window coefficients (after normalization)


<em class="property">`property` </em>`length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.HammingWindow.length" title="Permalink to this definition"></a>
    
Window length in number of samples


<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.HammingWindow.normalize" title="Permalink to this definition"></a>
    
<cite>True</cite> if the window is normalized to have unit average power per coefficient. <cite>False</cite>
otherwise.


`show`(<em class="sig-param">`samples_per_symbol`</em>, <em class="sig-param">`domain``=``'time'`</em>, <em class="sig-param">`scale``=``'lin'`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.HammingWindow.show" title="Permalink to this definition"></a>
    
Plot the window in time or frequency domain
    
For the computation of the Fourier transform, a minimum DFT size
of 1024 is assumed which is obtained through zero padding of
the window coefficients in the time domain.
Input
 
- **samples_per_symbol** (<em>int</em>) – Number of samples per symbol, i.e., the oversampling factor.
- **domain** (<em>str, one of [“time”, “frequency”]</em>) – The desired domain.
Defaults to “time”
- **scale** (<em>str, one of [“lin”, “db”]</em>) – The y-scale of the magnitude in the frequency domain.
Can be “lin” (i.e., linear) or “db” (, i.e., Decibel).
Defaults to “lin”.





<em class="property">`property` </em>`trainable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.HammingWindow.trainable" title="Permalink to this definition"></a>
    
<cite>True</cite> if the window coefficients are trainable. <cite>False</cite> otherwise.


### BlackmanWindow<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#blackmanwindow" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.signal.``BlackmanWindow`(<em class="sig-param">`length`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`normalize``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/signal/window.html#BlackmanWindow">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.BlackmanWindow" title="Permalink to this definition"></a>
    
Layer for applying a Blackman window function of length `length` to an input `x` of the same length.
    
The window function is applied through element-wise multiplication.
    
The window function is real-valued, i.e., has <cite>tf.float</cite> as <cite>dtype</cite>.
The <cite>dtype</cite> of the output is the same as the <cite>dtype</cite> of the input `x` to which the window function is applied.
The window function and the input must have the same precision.
    
The Blackman window is defined by

$$
w_n = a_0 - a_1 \cos \left( \frac{2 \pi n}{N} \right) + a_2 \cos \left( \frac{4 \pi n}{N} \right), 0 \leq n \leq N-1
$$
    
where $N$ is the window length, $a_0 = \frac{7938}{18608}$, $a_1 = \frac{9240}{18608}$, and $a_2 = \frac{1430}{18608}$.
Parameters
 
- **length** (<em>int</em>) – Window length (number of samples).
- **trainable** (<em>bool</em>) – If <cite>True</cite>, the window coefficients are trainable variables.
Defaults to <cite>False</cite>.
- **normalize** (<em>bool</em>) – If <cite>True</cite>, the window is normalized to have unit average power
per coefficient.
Defaults to <cite>False</cite>.
- **dtype** (<em>tf.DType</em>) – The <cite>dtype</cite> of the filter coefficients.
Must be either <cite>tf.float32</cite> or <cite>tf.float64</cite>.
Defaults to <cite>tf.float32</cite>.


Input
    
**x** (<em>[…, N], tf.complex or tf.float</em>) – The input to which the window function is applied.
The window function is applied along the last dimension.
The length of the last dimension `N` must be the same as the `length` of the window function.

Output
    
**y** (<em>[…,N], tf.complex or tf.float</em>) – Output of the windowing operation.
The output has the same shape and <cite>dtype</cite> as the input `x`.



<em class="property">`property` </em>`coefficients`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.BlackmanWindow.coefficients" title="Permalink to this definition"></a>
    
The window coefficients (after normalization)


<em class="property">`property` </em>`length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.BlackmanWindow.length" title="Permalink to this definition"></a>
    
Window length in number of samples


<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.BlackmanWindow.normalize" title="Permalink to this definition"></a>
    
<cite>True</cite> if the window is normalized to have unit average power per coefficient. <cite>False</cite>
otherwise.


`show`(<em class="sig-param">`samples_per_symbol`</em>, <em class="sig-param">`domain``=``'time'`</em>, <em class="sig-param">`scale``=``'lin'`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.BlackmanWindow.show" title="Permalink to this definition"></a>
    
Plot the window in time or frequency domain
    
For the computation of the Fourier transform, a minimum DFT size
of 1024 is assumed which is obtained through zero padding of
the window coefficients in the time domain.
Input
 
- **samples_per_symbol** (<em>int</em>) – Number of samples per symbol, i.e., the oversampling factor.
- **domain** (<em>str, one of [“time”, “frequency”]</em>) – The desired domain.
Defaults to “time”
- **scale** (<em>str, one of [“lin”, “db”]</em>) – The y-scale of the magnitude in the frequency domain.
Can be “lin” (i.e., linear) or “db” (, i.e., Decibel).
Defaults to “lin”.





<em class="property">`property` </em>`trainable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.BlackmanWindow.trainable" title="Permalink to this definition"></a>
    
<cite>True</cite> if the window coefficients are trainable. <cite>False</cite> otherwise.


### CustomWindow<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#customwindow" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.signal.``CustomWindow`(<em class="sig-param">`length`</em>, <em class="sig-param">`coefficients``=``None`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`normalize``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/signal/window.html#CustomWindow">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomWindow" title="Permalink to this definition"></a>
    
Layer for defining and applying a custom window function of length `length` to an input `x` of the same length.
    
The window function is applied through element-wise multiplication.
    
The window function is real-valued, i.e., has <cite>tf.float</cite> as <cite>dtype</cite>.
The <cite>dtype</cite> of the output is the same as the <cite>dtype</cite> of the input `x` to which the window function is applied.
The window function and the input must have the same precision.
    
The window coefficients can be set through the `coefficients` parameter.
If not provided, random window coefficients are generated by sampling a Gaussian distribution.
Parameters
 
- **length** (<em>int</em>) – Window length (number of samples).
- **coefficients** (<em>[</em><em>N</em><em>]</em><em>, </em><em>tf.float</em>) – Optional window coefficients.
If set to <cite>None</cite>, then a random window of length `length` is generated by sampling a Gaussian distribution.
Defaults to <cite>None</cite>.
- **trainable** (<em>bool</em>) – If <cite>True</cite>, the window coefficients are trainable variables.
Defaults to <cite>False</cite>.
- **normalize** (<em>bool</em>) – If <cite>True</cite>, the window is normalized to have unit average power
per coefficient.
Defaults to <cite>False</cite>.
- **dtype** (<em>tf.DType</em>) – The <cite>dtype</cite> of the filter coefficients.
Must be either <cite>tf.float32</cite> or <cite>tf.float64</cite>.
Defaults to <cite>tf.float32</cite>.


Input
    
**x** (<em>[…, N], tf.complex or tf.float</em>) – The input to which the window function is applied.
The window function is applied along the last dimension.
The length of the last dimension `N` must be the same as the `length` of the window function.

Output
    
**y** (<em>[…,N], tf.complex or tf.float</em>) – Output of the windowing operation.
The output has the same shape and <cite>dtype</cite> as the input `x`.



<em class="property">`property` </em>`coefficients`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomWindow.coefficients" title="Permalink to this definition"></a>
    
The window coefficients (after normalization)


<em class="property">`property` </em>`length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomWindow.length" title="Permalink to this definition"></a>
    
Window length in number of samples


<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomWindow.normalize" title="Permalink to this definition"></a>
    
<cite>True</cite> if the window is normalized to have unit average power per coefficient. <cite>False</cite>
otherwise.


`show`(<em class="sig-param">`samples_per_symbol`</em>, <em class="sig-param">`domain``=``'time'`</em>, <em class="sig-param">`scale``=``'lin'`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomWindow.show" title="Permalink to this definition"></a>
    
Plot the window in time or frequency domain
    
For the computation of the Fourier transform, a minimum DFT size
of 1024 is assumed which is obtained through zero padding of
the window coefficients in the time domain.
Input
 
- **samples_per_symbol** (<em>int</em>) – Number of samples per symbol, i.e., the oversampling factor.
- **domain** (<em>str, one of [“time”, “frequency”]</em>) – The desired domain.
Defaults to “time”
- **scale** (<em>str, one of [“lin”, “db”]</em>) – The y-scale of the magnitude in the frequency domain.
Can be “lin” (i.e., linear) or “db” (, i.e., Decibel).
Defaults to “lin”.





<em class="property">`property` </em>`trainable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomWindow.trainable" title="Permalink to this definition"></a>
    
<cite>True</cite> if the window coefficients are trainable. <cite>False</cite> otherwise.


### Window<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#id2" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.signal.``Window`(<em class="sig-param">`length`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`normalize``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/signal/window.html#Window">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window" title="Permalink to this definition"></a>
    
This is an abtract class for defining and applying a window function of length `length` to an input `x` of the same length.
    
The window function is applied through element-wise multiplication.
    
The window function is real-valued, i.e., has <cite>tf.float</cite> as <cite>dtype</cite>.
The <cite>dtype</cite> of the output is the same as the <cite>dtype</cite> of the input `x` to which the window function is applied.
The window function and the input must have the same precision.
Parameters
 
- **length** (<em>int</em>) – Window length (number of samples).
- **trainable** (<em>bool</em>) – If <cite>True</cite>, the window coefficients are trainable variables.
Defaults to <cite>False</cite>.
- **normalize** (<em>bool</em>) – If <cite>True</cite>, the window is normalized to have unit average power
per coefficient.
Defaults to <cite>False</cite>.
- **dtype** (<em>tf.DType</em>) – The <cite>dtype</cite> of the filter coefficients.
Must be either <cite>tf.float32</cite> or <cite>tf.float64</cite>.
Defaults to <cite>tf.float32</cite>.


Input
    
**x** (<em>[…, N], tf.complex or tf.float</em>) – The input to which the window function is applied.
The window function is applied along the last dimension.
The length of the last dimension `N` must be the same as the `length` of the window function.

Output
    
**y** (<em>[…,N], tf.complex or tf.float</em>) – Output of the windowing operation.
The output has the same shape and <cite>dtype</cite> as the input `x`.



<em class="property">`property` </em>`coefficients`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window.coefficients" title="Permalink to this definition"></a>
    
The window coefficients (after normalization)


<em class="property">`property` </em>`length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window.length" title="Permalink to this definition"></a>
    
Window length in number of samples


<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window.normalize" title="Permalink to this definition"></a>
    
<cite>True</cite> if the window is normalized to have unit average power per coefficient. <cite>False</cite>
otherwise.


`show`(<em class="sig-param">`samples_per_symbol`</em>, <em class="sig-param">`domain``=``'time'`</em>, <em class="sig-param">`scale``=``'lin'`</em>)<a class="reference internal" href="../_modules/sionna/signal/window.html#Window.show">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window.show" title="Permalink to this definition"></a>
    
Plot the window in time or frequency domain
    
For the computation of the Fourier transform, a minimum DFT size
of 1024 is assumed which is obtained through zero padding of
the window coefficients in the time domain.
Input
 
- **samples_per_symbol** (<em>int</em>) – Number of samples per symbol, i.e., the oversampling factor.
- **domain** (<em>str, one of [“time”, “frequency”]</em>) – The desired domain.
Defaults to “time”
- **scale** (<em>str, one of [“lin”, “db”]</em>) – The y-scale of the magnitude in the frequency domain.
Can be “lin” (i.e., linear) or “db” (, i.e., Decibel).
Defaults to “lin”.





<em class="property">`property` </em>`trainable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window.trainable" title="Permalink to this definition"></a>
    
<cite>True</cite> if the window coefficients are trainable. <cite>False</cite> otherwise.


## Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#utility-functions" title="Permalink to this headline"></a>

### convolve<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#convolve" title="Permalink to this headline"></a>

`sionna.signal.``convolve`(<em class="sig-param">`inp`</em>, <em class="sig-param">`ker`</em>, <em class="sig-param">`padding``=``'full'`</em>, <em class="sig-param">`axis``=``-` `1`</em>)<a class="reference internal" href="../_modules/sionna/signal/utils.html#convolve">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.convolve" title="Permalink to this definition"></a>
    
Filters an input `inp` of length <cite>N</cite> by convolving it with a kernel `ker` of length <cite>K</cite>.
    
The length of the kernel `ker` must not be greater than the one of the input sequence `inp`.
    
The <cite>dtype</cite> of the output is <cite>tf.float</cite> only if both `inp` and `ker` are <cite>tf.float</cite>. It is <cite>tf.complex</cite> otherwise.
`inp` and `ker` must have the same precision.
    
Three padding modes are available:
 
- “full” (default): Returns the convolution at each point of overlap between `ker` and `inp`.
The length of the output is <cite>N + K - 1</cite>. Zero-padding of the input `inp` is performed to
compute the convolution at the border points.
- “same”: Returns an output of the same length as the input `inp`. The convolution is computed such
that the coefficients of the input `inp` are centered on the coefficient of the kernel `ker` with index
`(K-1)/2` for kernels of odd length, and `K/2` `-` `1` for kernels of even length.
Zero-padding of the input signal is performed to compute the convolution at the border points.
- “valid”: Returns the convolution only at points where `inp` and `ker` completely overlap.
The length of the output is <cite>N - K + 1</cite>.

Input
 
- **inp** (<em>[…,N], tf.complex or tf.real</em>) – Input to filter.
- **ker** (<em>[K], tf.complex or tf.real</em>) – Kernel of the convolution.
- **padding** (<em>string</em>) – Padding mode. Must be one of “full”, “valid”, or “same”. Case insensitive.
Defaults to “full”.
- **axis** (<em>int</em>) – Axis along which to perform the convolution.
Defaults to <cite>-1</cite>.


Output
    
**out** (<em>[…,M], tf.complex or tf.float</em>) – Convolution output.
It is <cite>tf.float</cite> only if both `inp` and `ker` are <cite>tf.float</cite>. It is <cite>tf.complex</cite> otherwise.
The length <cite>M</cite> of the output depends on the `padding`.



### fft<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#fft" title="Permalink to this headline"></a>

`sionna.signal.``fft`(<em class="sig-param">`tensor`</em>, <em class="sig-param">`axis``=``-` `1`</em>)<a class="reference internal" href="../_modules/sionna/signal/utils.html#fft">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.fft" title="Permalink to this definition"></a>
    
Computes the normalized DFT along a specified axis.
    
This operation computes the normalized one-dimensional discrete Fourier
transform (DFT) along the `axis` dimension of a `tensor`.
For a vector $\mathbf{x}\in\mathbb{C}^N$, the DFT
$\mathbf{X}\in\mathbb{C}^N$ is computed as

$$
X_m = \frac{1}{\sqrt{N}}\sum_{n=0}^{N-1} x_n \exp \left\{
    -j2\pi\frac{mn}{N}\right\},\quad m=0,\dots,N-1.
$$

Input
 
- **tensor** (<em>tf.complex</em>) – Tensor of arbitrary shape.
- **axis** (<em>int</em>) – Indicates the dimension along which the DFT is taken.


Output
    
<em>tf.complex</em> – Tensor of the same dtype and shape as `tensor`.



### ifft<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#ifft" title="Permalink to this headline"></a>

`sionna.signal.``ifft`(<em class="sig-param">`tensor`</em>, <em class="sig-param">`axis``=``-` `1`</em>)<a class="reference internal" href="../_modules/sionna/signal/utils.html#ifft">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.ifft" title="Permalink to this definition"></a>
    
Computes the normalized IDFT along a specified axis.
    
This operation computes the normalized one-dimensional discrete inverse
Fourier transform (IDFT) along the `axis` dimension of a `tensor`.
For a vector $\mathbf{X}\in\mathbb{C}^N$, the IDFT
$\mathbf{x}\in\mathbb{C}^N$ is computed as

$$
x_n = \frac{1}{\sqrt{N}}\sum_{m=0}^{N-1} X_m \exp \left\{
    j2\pi\frac{mn}{N}\right\},\quad n=0,\dots,N-1.
$$

Input
 
- **tensor** (<em>tf.complex</em>) – Tensor of arbitrary shape.
- **axis** (<em>int</em>) – Indicates the dimension along which the IDFT is taken.


Output
    
<em>tf.complex</em> – Tensor of the same dtype and shape as `tensor`.



### Upsampling<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#upsampling" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.signal.``Upsampling`(<em class="sig-param">`samples_per_symbol`</em>, <em class="sig-param">`axis``=``-` `1`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/signal/upsampling.html#Upsampling">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Upsampling" title="Permalink to this definition"></a>
    
Upsamples a tensor along a specified axis by inserting zeros
between samples.
Parameters
 
- **samples_per_symbol** (<em>int</em>) – The upsampling factor. If `samples_per_symbol` is equal to <cite>n</cite>,
then the upsampled axis will be <cite>n</cite>-times longer.
- **axis** (<em>int</em>) – The dimension to be up-sampled. Must not be the first dimension.


Input
    
**x** (<em>[…,n,…], tf.DType</em>) – The tensor to be upsampled. <cite>n</cite> is the size of the <cite>axis</cite> dimension.

Output
    
**y** ([…,n*samples_per_symbol,…], same dtype as `x`) – The upsampled tensor.



### Downsampling<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#downsampling" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.signal.``Downsampling`(<em class="sig-param">`samples_per_symbol`</em>, <em class="sig-param">`offset``=``0`</em>, <em class="sig-param">`num_symbols``=``None`</em>, <em class="sig-param">`axis``=``-` `1`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/signal/downsampling.html#Downsampling">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Downsampling" title="Permalink to this definition"></a>
    
Downsamples a tensor along a specified axis by retaining one out of
`samples_per_symbol` elements.
Parameters
 
- **samples_per_symbol** (<em>int</em>) – The downsampling factor. If `samples_per_symbol` is equal to <cite>n</cite>, then the
downsampled axis will be <cite>n</cite>-times shorter.
- **offset** (<em>int</em>) – Defines the index of the first element to be retained.
Defaults to zero.
- **num_symbols** (<em>int</em>) – Defines the total number of symbols to be retained after
downsampling.
Defaults to None (i.e., the maximum possible number).
- **axis** (<em>int</em>) – The dimension to be downsampled. Must not be the first dimension.


Input
    
**x** (<em>[…,n,…], tf.DType</em>) – The tensor to be downsampled. <cite>n</cite> is the size of the <cite>axis</cite> dimension.

Output
    
**y** ([…,k,…], same dtype as `x`) – The downsampled tensor, where `k`
is min((`n`-`offset`)//`samples_per_symbol`, `num_symbols`).



### empirical_psd<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#empirical-psd" title="Permalink to this headline"></a>

`sionna.signal.``empirical_psd`(<em class="sig-param">`x`</em>, <em class="sig-param">`show``=``True`</em>, <em class="sig-param">`oversampling``=``1.0`</em>, <em class="sig-param">`ylim``=``(-` `30,` `3)`</em>)<a class="reference internal" href="../_modules/sionna/signal/utils.html#empirical_psd">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.empirical_psd" title="Permalink to this definition"></a>
    
Computes the empirical power spectral density.
    
Computes the empirical power spectral density (PSD) of tensor `x`
along the last dimension by averaging over all other dimensions.
Note that this function
simply returns the averaged absolute squared discrete Fourier
spectrum of `x`.
Input
 
- **x** (<em>[…,N], tf.complex</em>) – The signal of which to compute the PSD.
- **show** (<em>bool</em>) – Indicates if a plot of the PSD should be generated.
Defaults to True,
- **oversampling** (<em>float</em>) – The oversampling factor. Defaults to 1.
- **ylim** (<em>tuple of floats</em>) – The limits of the y axis. Defaults to [-30, 3].
Only relevant if `show` is True.


Output
 
- **freqs** (<em>[N], float</em>) – The normalized frequencies at which the PSD was evaluated.
- **psd** (<em>[N], float</em>) – The PSD.




### empirical_aclr<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#empirical-aclr" title="Permalink to this headline"></a>

`sionna.signal.``empirical_aclr`(<em class="sig-param">`x`</em>, <em class="sig-param">`oversampling``=``1.0`</em>, <em class="sig-param">`f_min``=``-` `0.5`</em>, <em class="sig-param">`f_max``=``0.5`</em>)<a class="reference internal" href="../_modules/sionna/signal/utils.html#empirical_aclr">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.empirical_aclr" title="Permalink to this definition"></a>
    
Computes the empirical ACLR.
    
Computes the empirical adjacent channel leakgae ration (ACLR)
of tensor `x` based on its empirical power spectral density (PSD)
which is computed along the last dimension by averaging over
all other dimensions.
    
It is assumed that the in-band ranges from [`f_min`, `f_max`] in
normalized frequency. The ACLR is then defined as

$$
\text{ACLR} = \frac{P_\text{out}}{P_\text{in}}
$$
    
where $P_\text{in}$ and $P_\text{out}$ are the in-band
and out-of-band power, respectively.
Input
 
- **x** (<em>[…,N],  complex</em>) – The signal for which to compute the ACLR.
- **oversampling** (<em>float</em>) – The oversampling factor. Defaults to 1.
- **f_min** (<em>float</em>) – The lower border of the in-band in normalized frequency.
Defaults to -0.5.
- **f_max** (<em>float</em>) – The upper border of the in-band in normalized frequency.
Defaults to 0.5.


Output
    
**aclr** (<em>float</em>) – The ACLR in linear scale.


