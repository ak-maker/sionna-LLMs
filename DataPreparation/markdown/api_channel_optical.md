
# Optical<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.optical.html#optical" title="Permalink to this headline"></a>
    
This module provides layers and functions that implement channel models for (fiber) optical communications.
The currently only available model is the split-step Fourier method (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.optical.html#sionna.channel.SSFM" title="sionna.channel.SSFM">`SSFM`</a>, for dual- and
single-polarization) that can be combined with an Erbium-doped amplifier (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.optical.html#sionna.channel.EDFA" title="sionna.channel.EDFA">`EDFA`</a>).
    
The following code snippets show how to setup and simulate the transmission
over a single-mode fiber (SMF) by using the split-step Fourier method.
```python
# init fiber
span = sionna.channel.optical.SSFM(
                              alpha=0.046,
                              beta_2=-21.67,
                              f_c=193.55e12,
                              gamma=1.27,
                              length=80,
                              n_ssfm=200,
                              n_sp=1.0,
                              t_norm=1e-12,
                              with_amplification=False,
                              with_attenuation=True,
                              with_dispersion=True,
                              with_nonlinearity=True,
                              dtype=tf.complex64)
# init amplifier
amplifier = sionna.channel.optical.EDFA(
                              g=4.0,
                              f=2.0,
                              f_c=193.55e12,
                              dt=1.0e-12)
@tf.function
def simulate_transmission(x, n_span):
      y = x
      # simulate n_span fiber spans
      for _ in range(n_span):
            # simulate single span
            y = span(y)
            # simulate amplifier
            y = amplifier(y)
      return y
```

    
Running the channel model is done as follows:
```python
# x is the optical input signal, n_span the number of spans
y = simulate_transmission(x, n_span)
```

    
For further details, the tutorial <a class="reference external" href="../examples/Optical_Lumped_Amplification_Channel.html">“Optical Channel with Lumped Amplification”</a>  provides more sophisticated examples of how to use this module.
    
For the purpose of the present document, the following symbols apply:
<table class="docutils align-default">
<colgroup>
<col style="width: 30%" />
<col style="width: 70%" />
</colgroup>
<tbody>
<tr class="row-odd"><td>    
$T_\text{norm}$</td>
<td>    
Time normalization for the SSFM in $(\text{s})$</td>
</tr>
<tr class="row-even"><td>    
$L_\text{norm}$</td>
<td>    
Distance normalization the for SSFM in $(\text{m})$</td>
</tr>
<tr class="row-odd"><td>    
$W$</td>
<td>    
Bandwidth</td>
</tr>
<tr class="row-even"><td>    
$\alpha$</td>
<td>    
Attenuation coefficient in $(1/L_\text{norm})$</td>
</tr>
<tr class="row-odd"><td>    
$\beta_2$</td>
<td>    
Group velocity dispersion coeff. in $(T_\text{norm}^2/L_\text{norm})$</td>
</tr>
<tr class="row-even"><td>    
$f_\mathrm{c}$</td>
<td>    
Carrier frequency in  $\text{(Hz)}$</td>
</tr>
<tr class="row-odd"><td>    
$\gamma$</td>
<td>    
Nonlinearity coefficient in $(1/L_\text{norm}/\text{W})$</td>
</tr>
<tr class="row-even"><td>    
$\ell$</td>
<td>    
Fiber length in $(L_\text{norm})$</td>
</tr>
<tr class="row-odd"><td>    
$h$</td>
<td>    
Planck constant</td>
</tr>
<tr class="row-even"><td>    
$N_\mathrm{SSFM}$</td>
<td>    
Number of SSFM simulation steps</td>
</tr>
<tr class="row-odd"><td>    
$n_\mathrm{sp}$</td>
<td>    
Spontaneous emission factor of Raman amplification</td>
</tr>
<tr class="row-even"><td>    
$\Delta_t$</td>
<td>    
Normalized simulation time step in $(T_\text{norm})$</td>
</tr>
<tr class="row-odd"><td>    
$\Delta_z$</td>
<td>    
Normalized simulation step size in $(L_\text{norm})$</td>
</tr>
<tr class="row-even"><td>    
$G$</td>
<td>    
Amplifier gain</td>
</tr>
<tr class="row-odd"><td>    
$F$</td>
<td>    
Amplifier’s noise figure</td>
</tr>
<tr class="row-even"><td>    
$\rho_\text{ASE}$</td>
<td>    
Noise spectral density</td>
</tr>
<tr class="row-odd"><td>    
$P$</td>
<td>    
Signal power</td>
</tr>
<tr class="row-even"><td>    
$\hat{D}$</td>
<td>    
Linear SSFM operator <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.optical.html#a2012" id="id1">[A2012]</a></td>
</tr>
<tr class="row-odd"><td>    
$\hat{N}$</td>
<td>    
Non-linear SSFM operator <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.optical.html#a2012" id="id2">[A2012]</a></td>
</tr>
<tr class="row-even"><td>    
$f_\textrm{sim}$</td>
<td>    
Simulation bandwidth</td>
</tr>
</tbody>
</table>
    
**Remark:** Depending on the exact simulation parameters, the SSFM algorithm may require `dtype=tf.complex128` for accurate simulation results. However, this may increase the simulation complexity significantly.

## Split-step Fourier method<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.optical.html#split-step-fourier-method" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``SSFM`(<em class="sig-param">`alpha``=``0.046`</em>, <em class="sig-param">`beta_2``=``-` `21.67`</em>, <em class="sig-param">`f_c``=``193.55e12`</em>, <em class="sig-param">`gamma``=``1.27`</em>, <em class="sig-param">`half_window_length``=``0`</em>, <em class="sig-param">`length``=``80`</em>, <em class="sig-param">`n_ssfm``=``1`</em>, <em class="sig-param">`n_sp``=``1.0`</em>, <em class="sig-param">`sample_duration``=``1.0`</em>, <em class="sig-param">`t_norm``=``1e-12`</em>, <em class="sig-param">`with_amplification``=``False`</em>, <em class="sig-param">`with_attenuation``=``True`</em>, <em class="sig-param">`with_dispersion``=``True`</em>, <em class="sig-param">`with_manakov``=``False`</em>, <em class="sig-param">`with_nonlinearity``=``True`</em>, <em class="sig-param">`swap_memory``=``True`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/channel/optical/fiber.html#SSFM">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.optical.html#sionna.channel.SSFM" title="Permalink to this definition"></a>
    
Layer implementing the split-step Fourier method (SSFM)
    
The SSFM (first mentioned in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.optical.html#ht1973" id="id3">[HT1973]</a>) numerically solves the generalized
nonlinear Schrödinger equation (NLSE)

$$
\frac{\partial E(t,z)}{\partial z}=-\frac{\alpha}{2} E(t,z)+j\frac{\beta_2}{2}\frac{\partial^2 E(t,z)}{\partial t^2}-j\gamma |E(t,z)|^2 E(t,z) + n(n_{\text{sp}};\,t,\,z)
$$
    
for an unpolarized (or single polarized) optical signal;
or the Manakov equation (according to <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.optical.html#wmc1991" id="id4">[WMC1991]</a>)

$$
\frac{\partial \mathbf{E}(t,z)}{\partial z}=-\frac{\alpha}{2} \mathbf{E}(t,z)+j\frac{\beta_2}{2}\frac{\partial^2 \mathbf{E}(t,z)}{\partial t^2}-j\gamma \frac{8}{9}||\mathbf{E}(t,z)||_2^2 \mathbf{E}(t,z) + \mathbf{n}(n_{\text{sp}};\,t,\,z)
$$
    
for dual polarization, with attenuation coefficient $\alpha$, group
velocity dispersion parameters $\beta_2$, and nonlinearity
coefficient $\gamma$. The noise terms $n(n_{\text{sp}};\,t,\,z)$
and $\mathbf{n}(n_{\text{sp}};\,t,\,z)$, respectively, stem from
an (optional) ideally distributed Raman amplification with
spontaneous emission factor $n_\text{sp}$. The optical signal
$E(t,\,z)$ has the unit $\sqrt{\text{W}}$. For the dual
polarized case, $\mathbf{E}(t,\,z)=(E_x(t,\,z), E_y(t,\,z))$
is a vector consisting of the signal components of both polarizations.
    
The symmetrized SSFM is applied according to Eq. (7) of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.optical.html#fmf1976" id="id5">[FMF1976]</a> that
can be written as

$$
E(z+\Delta_z,t) \approx \exp\left(\frac{\Delta_z}{2}\hat{D}\right)\exp\left(\int^{z+\Delta_z}_z \hat{N}(z')dz'\right)\exp\left(\frac{\Delta_z}{2}\hat{D}\right)E(z,\,t)
$$
    
where only the single-polarized case is shown. The integral is
approximated by $\Delta_z\hat{N}$ with $\hat{D}$ and
$\hat{N}$ denoting the linear and nonlinear SSFM operator,
respectively <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.optical.html#a2012" id="id6">[A2012]</a>.
    
Additionally, ideally distributed Raman amplification may be applied, which
is implemented as in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.optical.html#mffp2009" id="id7">[MFFP2009]</a>. Please note that the implemented
Raman amplification currently results in a transparent fiber link. Hence,
the introduced gain cannot be parametrized.
    
The SSFM operates on normalized time $T_\text{norm}$
(e.g., $T_\text{norm}=1\,\text{ps}=1\cdot 10^{-12}\,\text{s}$) and
distance units $L_\text{norm}$
(e.g., $L_\text{norm}=1\,\text{km}=1\cdot 10^{3}\,\text{m}$).
Hence, all parameters as well as the signal itself have to be given with the
same unit prefix for the
same unit (e.g., always pico for time, or kilo for distance). Despite the normalization,
the SSFM is implemented with physical
units, which is different from the normalization, e.g., used for the
nonlinear Fourier transform. For simulations, only $T_\text{norm}$ has to be
provided.
    
To avoid reflections at the signal boundaries during simulation, a Hamming
window can be applied in each SSFM-step, whose length can be
defined by `half_window_length`.
<p class="rubric">Example
    
Setting-up:
```python
>>> ssfm = SSFM(
>>>     alpha=0.046,
>>>     beta_2=-21.67,
>>>     f_c=193.55e12,
>>>     gamma=1.27,
>>>     half_window_length=100,
>>>     length=80,
>>>     n_ssfm=200,
>>>     n_sp=1.0,
>>>     t_norm=1e-12,
>>>     with_amplification=False,
>>>     with_attenuation=True,
>>>     with_dispersion=True,
>>>     with_manakov=False,
>>>     with_nonlinearity=True)
```

    
Running:
```python
>>> # x is the optical input signal
>>> y = ssfm(x)
```

Parameters
 
- **alpha** (<em>float</em>) – Attenuation coefficient $\alpha$ in $(1/L_\text{norm})$.
Defaults to 0.046.
- **beta_2** (<em>float</em>) – Group velocity dispersion coefficient $\beta_2$ in $(T_\text{norm}^2/L_\text{norm})$.
Defaults to -21.67
- **f_c** (<em>float</em>) – Carrier frequency $f_\mathrm{c}$ in $(\text{Hz})$.
Defaults to 193.55e12.
- **gamma** (<em>float</em>) – Nonlinearity coefficient $\gamma$ in $(1/L_\text{norm}/\text{W})$.
Defaults to 1.27.
- **half_window_length** (<em>int</em>) – Half of the Hamming window length. Defaults to 0.
- **length** (<em>float</em>) – Fiber length $\ell$ in $(L_\text{norm})$.
Defaults to 80.0.
- **n_ssfm** (<em>int | "adaptive"</em>) – Number of steps $N_\mathrm{SSFM}$.
Set to “adaptive” to use nonlinear-phase rotation to calculate
the step widths adaptively (maxmimum rotation can be set in phase_inc).
Defaults to 1.
- **n_sp** (<em>float</em>) – Spontaneous emission factor $n_\mathrm{sp}$ of Raman amplification.
Defaults to 1.0.
- **sample_duration** (<em>float</em>) – Normalized time step $\Delta_t$ in $(T_\text{norm})$.
Defaults to 1.0.
- **t_norm** (<em>float</em>) – Time normalization $T_\text{norm}$ in $(\text{s})$.
Defaults to 1e-12.
- **with_amplification** (<em>bool</em>) – Enable ideal inline amplification and corresponding
noise. Defaults to <cite>False</cite>.
- **with_attenuation** (<em>bool</em>) – Enable attenuation. Defaults to <cite>True</cite>.
- **with_dispersion** (<em>bool</em>) – Apply chromatic dispersion. Defaults to <cite>True</cite>.
- **with_manakov** (<em>bool</em>) – Considers axis [-2] as x- and y-polarization and calculates the
nonlinear step as given by the Manakov equation. Defaults to <cite>False.</cite>
- **with_nonlinearity** (<em>bool</em>) – Apply Kerr nonlinearity. Defaults to <cite>True</cite>.
- **phase_inc** (<em>float</em>) – Maximum nonlinear-phase rotation in rad allowed during simulation.
To be used with `n_ssfm` = “adaptive”.
Defaults to 1e-4.
- **swap_memory** (<em>bool</em>) – Use CPU memory for while loop. Defaults to <cite>True</cite>.
- **dtype** (<em>tf.complex</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.


Input
    
**x** (<em>[…,n] or […,2,n], tf.complex</em>) – Input signal in $(\sqrt{\text{W}})$. If `with_manakov` is <cite>True</cite>,
the second last dimension is interpreted as x- and y-polarization,
respectively.

Output
    
**y** (Tensor with same shape as `x`, <cite>tf.complex</cite>) – Channel output



## Erbium-doped fiber amplifier<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.optical.html#erbium-doped-fiber-amplifier" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``EDFA`(<em class="sig-param">`g``=``4.0`</em>, <em class="sig-param">`f``=``7.0`</em>, <em class="sig-param">`f_c``=``193.55e12`</em>, <em class="sig-param">`dt``=``1e-12`</em>, <em class="sig-param">`with_dual_polarization``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/channel/optical/edfa.html#EDFA">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.optical.html#sionna.channel.EDFA" title="Permalink to this definition"></a>
    
Layer implementing a model of an Erbium-Doped Fiber Amplifier
    
Amplifies the optical input signal by a given gain and adds
amplified spontaneous emission (ASE) noise.
    
The noise figure including the noise due to beating of signal and
spontaneous emission is $F_\mathrm{ASE,shot} =\frac{\mathrm{SNR}
_\mathrm{in}}{\mathrm{SNR}_\mathrm{out}}$,
where ideally the detector is limited by shot noise only, and
$\text{SNR}$ is the signal-to-noise-ratio. Shot noise is
neglected here but is required to derive the noise power of the amplifier, as
otherwise the input SNR is infinitely large. Hence, for the input SNR,
it follows <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.optical.html#a2012" id="id8">[A2012]</a> that
$\mathrm{SNR}_\mathrm{in}=\frac{P}{2hf_cW}$, where $h$ denotes
Planck’s constant, $P$ is the signal power, and $W$ the
considered bandwidth.
The output SNR is decreased by ASE noise induced by the amplification.
Note that shot noise is applied after the amplifier and is hence not
amplified. It results that $\mathrm{SNR}_\mathrm{out}=\frac{GP}{\left
(4\rho_\mathrm{ASE}+2hf_c\right)W}$, where $G$ is the
parametrized gain.
Hence, one can write the former equation as $F_\mathrm{ASE,shot} = 2
n_\mathrm{sp} \left(1-G^{-1}\right) + G^{-1}$.
Dropping shot noise again results in $F = 2 n_\mathrm{sp} \left(1-G^
{-1}\right)=2 n_\mathrm{sp} \frac{G-1}{G}$.
    
For a transparent link, e.g., the required gain per span is $G =
\exp\left(\alpha \ell \right)$.
The spontaneous emission factor is $n_\mathrm{sp}=\frac{F}
{2}\frac{G}{G-1}$.
According to <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.optical.html#a2012" id="id9">[A2012]</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.optical.html#ekwfg2010" id="id10">[EKWFG2010]</a> combined with <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.optical.html#bgt2000" id="id11">[BGT2000]</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.optical.html#gd1991" id="id12">[GD1991]</a>,
the noise power spectral density of the EDFA per state of
polarization is obtained as $\rho_\mathrm{ASE}^{(1)} = n_\mathrm{sp}\left
(G-1\right) h f_c=\frac{1}{2}G F h f_c$.
At simulation frequency $f_\mathrm{sim}$, the noise has a power of
$P_\mathrm{ASE}^{(1)}=\sigma_\mathrm{n,ASE}^2=2\rho_\mathrm{ASE}^{(1)}
\cdot f_\mathrm{sim}$,
where the factor $2$ accounts for the unpolarized noise (for dual
polarization the factor is $1$ per polarization).
Here, the notation $()^{(1)}$ means that this is the noise introduced by a
single EDFA.
    
This class inherits from the Keras <cite>Layer</cite> class and can be used as layer in
a Keras model.
<p class="rubric">Example
    
Setting-up:
```python
>>> edfa = EDFA(
>>>     g=4.0,
>>>     f=2.0,
>>>     f_c=193.55e12,
>>>     dt=1.0e-12,
>>>     with_dual_polarization=False)
```

    
Running:
```python
>>> # x is the optical input signal
>>> y = EDFA(x)
```

Parameters
 
- **g** (<em>float</em>) – Amplifier gain (linear domain). Defaults to 4.0.
- **f** (<em>float</em>) – Noise figure (linear domain). Defaults to 7.0.
- **f_c** (<em>float</em>) – Carrier frequency $f_\mathrm{c}$ in $(\text{Hz})$.
Defaults to 193.55e12.
- **dt** (<em>float</em>) – Time step $\Delta_t$ in $(\text{s})$.
Defaults to 1e-12.
- **with_dual_polarization** (<em>bool</em>) – Considers axis [-2] as x- and y-polarization and applies the noise
per polarization. Defaults to <cite>False</cite>.
- **dtype** (<em>tf.complex</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.


Input
    
**x** (<em>Tensor, tf.complex</em>) – Optical input signal

Output
    
**y** (Tensor with same shape as `x`, `dtype`) – Amplifier output



## Utility functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.optical.html#utility-functions" title="Permalink to this headline"></a>

### time_frequency_vector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.optical.html#time-frequency-vector" title="Permalink to this headline"></a>

`sionna.channel.utils.``time_frequency_vector`(<em class="sig-param">`num_samples`</em>, <em class="sig-param">`sample_duration`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#time_frequency_vector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.optical.html#sionna.channel.utils.time_frequency_vector" title="Permalink to this definition"></a>
    
Compute the time and frequency vector for a given number of samples
and duration per sample in normalized time unit.
```python
>>> t = tf.cast(tf.linspace(-n_min, n_max, num_samples), dtype) * sample_duration
>>> f = tf.cast(tf.linspace(-n_min, n_max, num_samples), dtype) * 1/(sample_duration*num_samples)
```

Input
 
- **num_samples** (<em>int</em>) – Number of samples
- **sample_duration** (<em>float</em>) – Sample duration in normalized time
- **dtype** (<em>tf.DType</em>) – Datatype to use for internal processing and output.
Defaults to <cite>tf.float32</cite>.


Output
 
- **t** ([`num_samples`], `dtype`) – Time vector
- **f** ([`num_samples`], `dtype`) – Frequency vector





References:
<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/channel.optical.html#id3">HT1973</a>
    
R. H. Hardin and F. D. Tappert,
“Applications of the Split-Step Fourier Method to the Numerical Solution of Nonlinear and Variable Coefficient Wave Equations.”,
SIAM Review Chronicles, Vol. 15, No. 2, Part 1, p 423, 1973.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/channel.optical.html#id5">FMF1976</a>
    
J. A. Fleck, J. R. Morris, and M. D. Feit,
“Time-dependent Propagation of High Energy Laser Beams Through the Atmosphere”,
Appl. Phys., Vol. 10, pp 129–160, 1976.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/channel.optical.html#id7">MFFP2009</a>
    
N. J. Muga, M. C. Fugihara, M. F. S. Ferreira, and A. N. Pinto,
“ASE Noise Simulation in Raman Amplification Systems”,
Conftele, 2009.

A2012(<a href="https://nvlabs.github.io/sionna/api/channel.optical.html#id1">1</a>,<a href="https://nvlabs.github.io/sionna/api/channel.optical.html#id2">2</a>,<a href="https://nvlabs.github.io/sionna/api/channel.optical.html#id6">3</a>,<a href="https://nvlabs.github.io/sionna/api/channel.optical.html#id8">4</a>,<a href="https://nvlabs.github.io/sionna/api/channel.optical.html#id9">5</a>)
    
G. P. Agrawal,
“Fiber-optic Communication Systems”,
4th ed. Wiley series in microwave and optical engineering 222. New York: Wiley, 2010.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/channel.optical.html#id10">EKWFG2010</a>
    
R. J. Essiambre, G. Kramer, P. J. Winzer, G. J. Foschini, and B. Goebel,
“Capacity Limits of Optical Fiber Networks”,
Journal of Lightwave Technology 28, No. 4, 2010.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/channel.optical.html#id11">BGT2000</a>
    
D. M. Baney, P. Gallion, and R. S. Tucker,
“Theory and Measurement Techniques for the Noise Figure of Optical Amplifiers”,
Optical Fiber Technology 6, No. 2, 2000.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/channel.optical.html#id12">GD1991</a>
    
C.R. Giles, and E. Desurvire,
“Modeling Erbium-Doped Fiber Amplifiers”,
Journal of Lightwave Technology 9, No. 2, 1991.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/channel.optical.html#id4">WMC1991</a>
    
P. K. A. Wai, C. R. Menyuk, and H. H. Chen,
“Stability of Solitons in Randomly Varying Birefringent Fibers”,
Optics Letters, No. 16, 1991.



