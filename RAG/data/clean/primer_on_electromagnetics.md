# Primer on Electromagnetics

This section provides useful background for the general understanding of ray tracing for wireless propagation modelling. In particular, our goal is to provide a concise definition of a <cite>channel impulse response</cite> between a transmitting and receiving antenna, as done in (Ch. 2 & 3) [[Wiesbeck]](https://nvlabs.github.io/sionna/em_primer.html#wiesbeck). The notations and definitions will be used in the API documentation of Sionnas [Ray Tracing module](api/rt.html).

## Coordinate system, rotations, and vector fields

We consider a global coordinate system (GCS) with Cartesian standard basis $\hat{\mathbf{x}}$, $\hat{\mathbf{y}}$, $\hat{\mathbf{z}}$.
The spherical unit vectors are defined as

$$
\begin{split}\begin{align}
    \hat{\mathbf{r}}          (\theta, \varphi) &= \sin(\theta)\cos(\varphi) \hat{\mathbf{x}} + \sin(\theta)\sin(\varphi) \hat{\mathbf{y}} + \cos(\theta)\hat{\mathbf{z}}\\
    \hat{\boldsymbol{\theta}} (\theta, \varphi) &= \cos(\theta)\cos(\varphi) \hat{\mathbf{x}} + \cos(\theta)\sin(\varphi) \hat{\mathbf{y}} - \sin(\theta)\hat{\mathbf{z}}\\
    \hat{\boldsymbol{\varphi}}(\theta, \varphi) &=            -\sin(\varphi) \hat{\mathbf{x}} +             \cos(\varphi) \hat{\mathbf{y}}.
\end{align}\end{split}
$$

For an arbitrary unit norm vector $\hat{\mathbf{v}} = (x, y, z)$, the elevation and azimuth angles $\theta$ and $\varphi$ can be computed as

$$
\begin{split}\theta  &= \cos^{-1}(z) \\
\varphi &= \mathop{\text{atan2}}(y, x)\end{split}
$$

where $\mathop{\text{atan2}}(y, x)$ is the two-argument inverse tangent function [[atan2]](https://nvlabs.github.io/sionna/em_primer.html#atan2). As any vector uniquely determines $\theta$ and $\varphi$, we sometimes also
write $\hat{\boldsymbol{\theta}}(\hat{\mathbf{v}})$ and $\hat{\boldsymbol{\varphi}}(\hat{\mathbf{v}})$ instead of $\hat{\boldsymbol{\theta}} (\theta, \varphi)$ and $\hat{\boldsymbol{\varphi}}(\theta, \varphi)$.

A 3D rotation with yaw, pitch, and roll angles $\alpha$, $\beta$, and $\gamma$, respectively, is expressed by the matrix

$$
\begin{align}
    \mathbf{R}(\alpha, \beta, \gamma) = \mathbf{R}_z(\alpha)\mathbf{R}_y(\beta)\mathbf{R}_x(\gamma)
\end{align}
$$

where $\mathbf{R}_z(\alpha)$, $\mathbf{R}_y(\beta)$, and $\mathbf{R}_x(\gamma)$ are rotation matrices around the $z$, $y$, and $x$ axes, respectively, which are defined as

$$
\begin{split}\begin{align}
    \mathbf{R}_z(\alpha) &= \begin{pmatrix}
                    \cos(\alpha) & -\sin(\alpha) & 0\\
                    \sin(\alpha) & \cos(\alpha) & 0\\
                    0 & 0 & 1
                  \end{pmatrix}\\
    \mathbf{R}_y(\beta) &= \begin{pmatrix}
                    \cos(\beta) & 0 & \sin(\beta)\\
                    0 & 1 & 0\\
                    -\sin(\beta) & 0 & \cos(\beta)
                  \end{pmatrix}\\
    \mathbf{R}_x(\gamma) &= \begin{pmatrix}
                        1 & 0 & 0\\
                        0 & \cos(\gamma) & -\sin(\gamma)\\
                        0 & \sin(\gamma) & \cos(\gamma)
                  \end{pmatrix}.
\end{align}\end{split}
$$

A closed-form expression for $\mathbf{R}(\alpha, \beta, \gamma)$ can be found in (7.1-4) [[TR38901]](api/channel.wireless.html#tr38901).
The reverse rotation is simply defined by $\mathbf{R}^{-1}(\alpha, \beta, \gamma)=\mathbf{R}^\mathsf{T}(\alpha, \beta, \gamma)$.
A vector $\mathbf{x}$ defined in a first coordinate system is represented in a second coordinate system rotated by $\mathbf{R}(\alpha, \beta, \gamma)$ with respect to the first one as $\mathbf{x}'=\mathbf{R}^\mathsf{T}(\alpha, \beta, \gamma)\mathbf{x}$.
If a point in the first coordinate system has spherical angles $(\theta, \varphi)$, the corresponding angles $(\theta', \varphi')$ in the second coordinate system can be found to be

$$
\begin{split}\begin{align}
    \theta' &= \cos^{-1}\left( \mathbf{z}^\mathsf{T} \mathbf{R}^\mathsf{T}(\alpha, \beta, \gamma)\hat{\mathbf{r}}(\theta, \varphi)          \right)\\
    \varphi' &= \arg\left( \left( \mathbf{x} + j\mathbf{y}\right)^\mathsf{T} \mathbf{R}^\mathsf{T}(\alpha, \beta, \gamma)\hat{\mathbf{r}}(\theta, \varphi) \right).
\end{align}\end{split}
$$

For a vector field $\mathbf{F}'(\theta',\varphi')$ expressed in local spherical coordinates

$$
\mathbf{F}'(\theta',\varphi') = F_{\theta'}(\theta',\varphi')\hat{\boldsymbol{\theta}}'(\theta',\varphi') + F_{\varphi'}(\theta',\varphi')\hat{\boldsymbol{\varphi}}'(\theta',\varphi')
$$

that are rotated by $\mathbf{R}=\mathbf{R}(\alpha, \beta, \gamma)$ with respect to the GCS, the spherical field components in the GCS can be expressed as

$$
\begin{split}\begin{bmatrix}
    F_\theta(\theta, \varphi) \\
    F_\varphi(\theta, \varphi)
\end{bmatrix} =
\begin{bmatrix}
    \hat{\boldsymbol{\theta}}(\theta,\varphi)^\mathsf{T}\mathbf{R}\hat{\boldsymbol{\theta}}'(\theta',\varphi') & \hat{\boldsymbol{\theta}}(\theta,\varphi)^\mathsf{T}\mathbf{R}\hat{\boldsymbol{\varphi}}'(\theta',\varphi') \\
    \hat{\boldsymbol{\varphi}}(\theta,\varphi)^\mathsf{T}\mathbf{R}\hat{\boldsymbol{\theta}}'(\theta',\varphi') & \hat{\boldsymbol{\varphi}}(\theta,\varphi)^\mathsf{T}\mathbf{R}\hat{\boldsymbol{\varphi}}'(\theta',\varphi')
\end{bmatrix}
\begin{bmatrix}
    F_{\theta'}(\theta', \varphi') \\
    F_{\varphi'}(\theta', \varphi')
\end{bmatrix}\end{split}
$$

so that

$$
\mathbf{F}(\theta,\varphi) = F_{\theta}(\theta,\varphi)\hat{\boldsymbol{\theta}}(\theta,\varphi) + F_{\varphi}(\theta,\varphi)\hat{\boldsymbol{\varphi}}(\theta,\varphi).
$$

It sometimes also useful to find the rotation matrix that maps a unit vector $\hat{\mathbf{a}}$ to $\hat{\mathbf{b}}$. This can be achieved with the help of Rodrigues rotation formula [[Wikipedia_Rodrigues]](https://nvlabs.github.io/sionna/em_primer.html#wikipedia-rodrigues) which defines the matrix

$$
\mathbf{R}(\hat{\mathbf{a}}, \hat{\mathbf{b}}) = \mathbf{I} + \sin(\theta)\mathbf{K} + (1-\cos(\theta)) \mathbf{K}^2
$$

where

$$
\begin{split}\mathbf{K} &= \begin{bmatrix}
                        0 & -\hat{k}_z &  \hat{k}_y \\
                \hat{k}_z &          0 & -\hat{k}_x \\
               -\hat{k}_y &  \hat{k}_x &          0
             \end{bmatrix}\\
\hat{\mathbf{k}} &= \frac{\hat{\mathbf{a}} \times \hat{\mathbf{b}}}{\lVert \hat{\mathbf{a}} \times \hat{\mathbf{b}} \rVert}\\
\theta &=\hat{\mathbf{a}}^\mathsf{T}\hat{\mathbf{b}}\end{split}
$$

such that $\mathbf{R}(\hat{\mathbf{a}}, \hat{\mathbf{b}})\hat{\mathbf{a}}=\hat{\mathbf{b}}$.

## Planar Time-Harmonic Waves

A time-harmonic planar electric wave $\mathbf{E}(\mathbf{x}, t)\in\mathbb{C}^3$ travelling in a homogeneous medium with wave vector $\mathbf{k}\in\mathbb{C}^3$ can be described at position $\mathbf{x}\in\mathbb{R}^3$ and time $t$ as

$$
\begin{split}\begin{align}
    \mathbf{E}(\mathbf{x}, t) &= \mathbf{E}_0 e^{j(\omega t -\mathbf{k}^{\mathsf{H}}\mathbf{x})}\\
                              &= \mathbf{E}(\mathbf{x}) e^{j\omega t}
\end{align}\end{split}
$$

where $\mathbf{E}_0\in\mathbb{C}^3$ is the field phasor. The wave vector can be decomposed as $\mathbf{k}=k \hat{\mathbf{k}}$, where $\hat{\mathbf{k}}$ is a unit norm vector, $k=\omega\sqrt{\varepsilon\mu}$ is the wave number, and $\omega=2\pi f$ is the angular frequency. The permittivity $\varepsilon$ and permeability $\mu$ are defined as

$$
\varepsilon = \eta \varepsilon_0
$$

$$
\mu = \mu_r \mu_0
$$

where $\eta$ and $\varepsilon_0$ are the complex relative and vacuum permittivities, $\mu_r$ and $\mu_0$ are the relative and vacuum permeabilities, and $\sigma$ is the conductivity.
The complex relative permittivity $\eta$ is given as

$$
\eta = \varepsilon_r - j\frac{\sigma}{\varepsilon_0\omega}
$$

where $\varepsilon_r$ is the real relative permittivity of a non-conducting dielectric.

With these definitions, the speed of light is given as (Eq. 4-28d) [[Balanis]](https://nvlabs.github.io/sionna/em_primer.html#balanis)

$$
c=\frac{1}{\sqrt{\varepsilon_0\varepsilon_r\mu}}\left\{\frac12\left(\sqrt{1+\left(\frac{\sigma}{\omega\varepsilon_0\varepsilon_r}\right)^2}+1\right)\right\}^{-\frac{1}{2}}
$$

where the factor in curly brackets vanishes for non-conducting materials. The speed of light in vacuum is denoted $c_0=\frac{1}{\sqrt{\varepsilon_0 \mu_0}}$ and the vacuum wave number $k_0=\frac{\omega}{c_0}$. In conducting materials, the wave number is complex which translates to propagation losses.

The associated magnetic field $\mathbf{H}(\mathbf{x}, t)\in\mathbb{C}^3$ is

$$
\mathbf{H}(\mathbf{x}, t) = \frac{\hat{\mathbf{k}}\times  \mathbf{E}(\mathbf{x}, t)}{Z} = \mathbf{H}(\mathbf{x})e^{j\omega t}
$$

where $Z=\sqrt{\mu/\varepsilon}$ is the wave impedance. The vacuum impedance is denoted by $Z_0=\sqrt{\mu_0/\varepsilon_0}\approx 376.73\,\Omega$.

The time-averaged Poynting vector is defined as

$$
\mathbf{S}(\mathbf{x}) = \frac{1}{2} \Re\left\{\mathbf{E}(\mathbf{x})\times  \mathbf{H}(\mathbf{x})\right\}
                       = \frac{1}{2} \Re\left\{\frac{1}{Z} \right\} \lVert \mathbf{E}(\mathbf{x})  \rVert^2 \hat{\mathbf{k}}
$$

which describes the directional energy flux (W/m), i.e., energy transfer per unit area per unit time.

Note that the actual electromagnetic waves are the real parts of $\mathbf{E}(\mathbf{x}, t)$ and $\mathbf{H}(\mathbf{x}, t)$.

## Far Field of a Transmitting Antenna

We assume that the electric far field of an antenna in free space can be described by a spherical wave originating from the center of the antenna:

$$
\mathbf{E}(r, \theta, \varphi, t) = \mathbf{E}(r,\theta, \varphi) e^{j\omega t} = \mathbf{E}_0(\theta, \varphi) \frac{e^{-jk_0r}}{r} e^{j\omega t}
$$

where $\mathbf{E}_0(\theta, \varphi)$ is the electric field phasor, $r$ is the distance (or radius), $\theta$ the zenith angle, and $\varphi$ the azimuth angle.
In contrast to a planar wave, the field strength decays as $1/r$.

The complex antenna field pattern $\mathbf{F}(\theta, \varphi)$ is defined as

$$
\begin{align}
    \mathbf{F}(\theta, \varphi) = \frac{ \mathbf{E}_0(\theta, \varphi)}{\max_{\theta,\varphi}\lVert  \mathbf{E}_0(\theta, \varphi) \rVert}.
\end{align}
$$

The time-averaged Poynting vector for such a spherical wave is

$$
\mathbf{S}(r, \theta, \varphi) = \frac{1}{2Z_0}\lVert \mathbf{E}(r, \theta, \varphi) \rVert^2 \hat{\mathbf{r}}
$$

where $\hat{\mathbf{r}}$ is the radial unit vector. It simplifies for an ideal isotropic antenna with input power $P_\text{T}$ to

$$
\mathbf{S}_\text{iso}(r, \theta, \varphi) = \frac{P_\text{T}}{4\pi r^2} \hat{\mathbf{r}}.
$$

The antenna gain $G$ is the ratio of the maximum radiation power density of the antenna in radial direction and that of an ideal isotropic radiating antenna:

$$
    G = \frac{\max_{\theta,\varphi}\lVert \mathbf{S}(r, \theta, \varphi)\rVert}{ \lVert\mathbf{S}_\text{iso}(r, \theta, \varphi)\rVert}
      = \frac{2\pi}{Z_0 P_\text{T}} \max_{\theta,\varphi}\lVert \mathbf{E}_0(\theta, \varphi) \rVert^2.
$$

One can similarly define a gain with directional dependency by ignoring the computation of the maximum the last equation:

$$
    G(\theta, \varphi) = \frac{2\pi}{Z_0 P_\text{T}} \lVert \mathbf{E}_0(\theta, \varphi) \rVert^2 = G \lVert \mathbf{F}(\theta, \varphi) \rVert^2.
$$

If one uses in the last equation the radiated power $P=\eta_\text{rad} P_\text{T}$, where $\eta_\text{rad}$ is the radiation efficiency, instead of the input power $P_\text{T}$, one obtains the directivity $D(\theta,\varphi)$. Both are related through $G(\theta, \varphi)=\eta_\text{rad} D(\theta, \varphi)$.

**Antenna pattern**

Since $\mathbf{F}(\theta, \varphi)$ contains no information about the maximum gain $G$ and $G(\theta, \varphi)$ does not carry any phase information, we define the <cite>antenna pattern</cite> $\mathbf{C}(\theta, \varphi)$ as

$$
\mathbf{C}(\theta, \varphi) = \sqrt{G}\mathbf{F}(\theta, \varphi)
$$

such that $G(\theta, \varphi)= \lVert\mathbf{C}(\theta, \varphi) \rVert^2$.

Using the spherical unit vectors $\hat{\boldsymbol{\theta}}\in\mathbb{R}^3$
and $\hat{\boldsymbol{\varphi}}\in\mathbb{R}^3$,
we can rewrite $\mathbf{C}(\theta, \varphi)$ as

$$
\mathbf{C}(\theta, \varphi) = C_\theta(\theta,\varphi) \hat{\boldsymbol{\theta}} + C_\varphi(\theta,\varphi) \hat{\boldsymbol{\varphi}}
$$

where $C_\theta(\theta,\varphi)\in\mathbb{C}$ and $C_\varphi(\theta,\varphi)\in\mathbb{C}$ are the
<cite>zenith pattern</cite> and <cite>azimuth pattern</cite>, respectively.

Combining [(10)](https://nvlabs.github.io/sionna/em_primer.html#equation-f) and [(12)](https://nvlabs.github.io/sionna/em_primer.html#equation-g), we can obtain the following expression of the electric far field

$$
\mathbf{E}_\text{T}(r,\theta_\text{T},\varphi_\text{T}) = \sqrt{ \frac{P_\text{T} G_\text{T} Z_0}{2\pi}} \frac{e^{-jk_0 r}}{r} \mathbf{F}_\text{T}(\theta_\text{T}, \varphi_\text{T})
$$

where we have added the subscript $\text{T}$ to all quantities that are specific to the transmitting antenna.

The input power $P_\text{T}$ of an antenna with (conjugate matched) impedance $Z_\text{T}$, fed by a voltage source with complex amplitude $V_\text{T}$, is given by (see, e.g., [[Wikipedia]](https://nvlabs.github.io/sionna/em_primer.html#wikipedia))

$$
P_\text{T} = \frac{|V_\text{T}|^2}{8\Re\{Z_\text{T}\}}.
$$

**Normalization of antenna patterns**

The radiated power $\eta_\text{rad} P_\text{T}$ of an antenna can be obtained by integrating the Poynting vector over the surface of a closed sphere of radius $r$ around the antenna:

$$
\begin{split}\begin{align}
    \eta_\text{rad} P_\text{T} &=  \int_0^{2\pi}\int_0^{\pi} \mathbf{S}(r, \theta, \varphi)^\mathsf{T} \hat{\mathbf{r}} r^2 \sin(\theta)d\theta d\varphi \\
                    &= \int_0^{2\pi}\int_0^{\pi} \frac{1}{2Z_0} \lVert \mathbf{E}(r, \theta, \varphi) \rVert^2 r^2\sin(\theta)d\theta d\varphi \\
                    &= \frac{P_\text{T}}{4 \pi} \int_0^{2\pi}\int_0^{\pi} G(\theta, \varphi) \sin(\theta)d\theta d\varphi.
\end{align}\end{split}
$$

We can see from the last equation that the directional gain of any antenna must satisfy

$$
\int_0^{2\pi}\int_0^{\pi} G(\theta, \varphi) \sin(\theta)d\theta d\varphi = 4 \pi \eta_\text{rad}.
$$
## Modelling of a Receiving Antenna

Although the transmitting antenna radiates a spherical wave $\mathbf{E}_\text{T}(r,\theta_\text{T},\varphi_\text{T})$,
we assume that the receiving antenna observes a planar incoming wave $\mathbf{E}_\text{R}$ that arrives from the angles $\theta_\text{R}$ and $\varphi_\text{R}$
which are defined in the local spherical coordinates of the receiving antenna. The Poynting vector of the incoming wave $\mathbf{S}_\text{R}$ is hence [(11)](https://nvlabs.github.io/sionna/em_primer.html#equation-s-spherical)

$$
\mathbf{S}_\text{R} = -\frac{1}{2Z_0} \lVert \mathbf{E}_\text{R} \rVert^2 \hat{\mathbf{r}}(\theta_\text{R}, \varphi_\text{R})
$$

where $\hat{\mathbf{r}}(\theta_\text{R}, \varphi_\text{R})$ is the radial unit vector in the spherical coordinate system of the receiver.

The aperture or effective area $A_\text{R}$ of an antenna with gain $G_\text{R}$ is defined as the ratio of the available received power $P_\text{R}$ at the output of the antenna and the absolute value of the Poynting vector, i.e., the power density:

$$
A_\text{R} = \frac{P_\text{R}}{\lVert \mathbf{S}_\text{R}\rVert} = G_\text{R}\frac{\lambda^2}{4\pi}
$$

where $\frac{\lambda^2}{4\pi}$ is the aperture of an isotropic antenna. In the definition above, it is assumed that the antenna is ideally directed towards and polarization matched to the incoming wave.
For an arbitrary orientation of the antenna (but still assuming polarization matching), we can define a direction dependent effective area

$$
A_\text{R}(\theta_\text{R}, \varphi_\text{R}) = G_\text{R}(\theta_\text{R}, \varphi_\text{R})\frac{\lambda^2}{4\pi}.
$$

The available received power at the output of the antenna can be expressed as

$$
P_\text{R} = \frac{|V_\text{R}|^2}{8\Re\{Z_\text{R}\}}
$$

where $Z_\text{R}$ is the impedance of the receiving antenna and $V_\text{R}$ the open circuit voltage.

We can now combine [(20)](https://nvlabs.github.io/sionna/em_primer.html#equation-p-r), [(19)](https://nvlabs.github.io/sionna/em_primer.html#equation-a-dir), and [(18)](https://nvlabs.github.io/sionna/em_primer.html#equation-a-r) to obtain the following expression for the absolute value of the voltage $|V_\text{R}|$
assuming matched polarization:

$$
\begin{split}\begin{align}
    |V_\text{R}| &= \sqrt{P_\text{R} 8\Re\{Z_\text{R}\}}\\
                 &= \sqrt{\frac{\lambda^2}{4\pi} G_\text{R}(\theta_\text{R}, \varphi_\text{R}) \frac{8\Re\{Z_\text{R}\}}{2 Z_0} \lVert \mathbf{E}_\text{R} \rVert^2}\\
                 &= \sqrt{\frac{\lambda^2}{4\pi} G_\text{R} \frac{4\Re\{Z_\text{R}\}}{Z_0}} \lVert \mathbf{F}_\text{R}(\theta_\text{R}, \varphi_\text{R})\rVert\lVert\mathbf{E}_\text{R}\rVert.
\end{align}\end{split}
$$

By extension of the previous equation, we can obtain an expression for $V_\text{R}$ which is valid for
arbitrary polarizations of the incoming wave and the receiving antenna:

$$
V_\text{R} = \sqrt{\frac{\lambda^2}{4\pi} G_\text{R} \frac{4\Re\{Z_\text{R}\}}{Z_0}} \mathbf{F}_\text{R}(\theta_\text{R}, \varphi_\text{R})^{\mathsf{H}}\mathbf{E}_\text{R}.
$$

**Example: Recovering Friis equation**

In the case of free space propagation, we have $\mathbf{E}_\text{R}=\mathbf{E}_\text{T}(r,\theta_\text{T},\varphi_\text{T})$.
Combining [(21)](https://nvlabs.github.io/sionna/em_primer.html#equation-v-r), [(20)](https://nvlabs.github.io/sionna/em_primer.html#equation-p-r), and [(15)](https://nvlabs.github.io/sionna/em_primer.html#equation-e-t), we obtain the following expression for the received power:

$$
P_\text{R} = \left(\frac{\lambda}{4\pi r}\right)^2 G_\text{R} G_\text{T} P_\text{T} \left|\mathbf{F}_\text{R}(\theta_\text{R}, \varphi_\text{R})^{\mathsf{H}} \mathbf{F}_\text{T}(\theta_\text{T}, \varphi_\text{T})\right|^2.
$$

It is important that $\mathbf{F}_\text{R}$ and $\mathbf{F}_\text{T}$ are expressed in the same coordinate system for the last equation to make sense.
For perfect orientation and polarization matching, we can recover the well-known Friis transmission equation:

$$
\frac{P_\text{R}}{P_\text{T}} = \left(\frac{\lambda}{4\pi r}\right)^2 G_\text{R} G_\text{T}.
$$
## General Propagation Path

A single propagation path consists of a cascade of multiple scattering processes, where a scattering process can be anything that prevents the wave from propagating as in free space. This includes reflection, refraction, diffraction, and diffuse scattering. For each scattering process, one needs to compute a relationship between the incoming field at the scatter center and the created far field at the next scatter center or the receiving antenna.
We can represent this cascade of scattering processes by a single matrix $\widetilde{\mathbf{T}}$
that describes the transformation that the radiated field $\mathbf{E}_\text{T}(r, \theta_\text{T}, \varphi_\text{T})$ undergoes until it reaches the receiving antenna:

$$
\mathbf{E}_\text{R} = \sqrt{ \frac{P_\text{T} G_\text{T} Z_0}{2\pi}} \widetilde{\mathbf{T}} \mathbf{F}_\text{T}(\theta_\text{T}, \varphi_\text{T}).
$$

Note that we have obtained this expression by replacing the free space propagation term $\frac{e^{-jk_0r}}{r}$ in [(15)](https://nvlabs.github.io/sionna/em_primer.html#equation-e-t) by the matrix $\widetilde{\mathbf{T}}$. This requires that all quantities are expressed in the same coordinate system which is also assumed in the following expressions. Further, it is assumed that the matrix $\widetilde{\mathbf{T}}$ includes the necessary coordinate transformations. In some cases, e.g., for diffuse scattering (see [(38)](https://nvlabs.github.io/sionna/em_primer.html#equation-scattered-field) in [Scattering](https://nvlabs.github.io/sionna/em_primer.html#scattering)), the matrix $\widetilde{\mathbf{T}}$ depends on the incoming field and is not a linear transformation.

Plugging [(22)](https://nvlabs.github.io/sionna/em_primer.html#equation-e-r) into [(21)](https://nvlabs.github.io/sionna/em_primer.html#equation-v-r), we can obtain a general expression for the received voltage of a propagation path:

$$
V_\text{R} = \sqrt{\left(\frac{\lambda}{4\pi}\right)^2 G_\text{R}G_\text{T}P_\text{T} 8\Re\{Z_\text{R}\}} \,\mathbf{F}_\text{R}(\theta_\text{R}, \varphi_\text{R})^{\mathsf{H}}\widetilde{\mathbf{T}} \mathbf{F}_\text{T}(\theta_\text{T}, \varphi_\text{T}).
$$

If the electromagnetic wave arrives at the receiving antenna over $N$ propagation paths, we can simply add the received voltages
from all paths to obtain

$$
\begin{split}\begin{align}
V_\text{R} &= \sqrt{\left(\frac{\lambda}{4\pi}\right)^2 G_\text{R}G_\text{T}P_\text{T} 8\Re\{Z_\text{R}\}} \sum_{n=1}^N\mathbf{F}_\text{R}(\theta_{\text{R},i}, \varphi_{\text{R},i})^{\mathsf{H}}\widetilde{\mathbf{T}}_i \mathbf{F}_\text{T}(\theta_{\text{T},i}, \varphi_{\text{T},i})\\
&= \sqrt{\left(\frac{\lambda}{4\pi}\right)^2 P_\text{T} 8\Re\{Z_\text{R}\}} \sum_{n=1}^N\mathbf{C}_\text{R}(\theta_{\text{R},i}, \varphi_{\text{R},i})^{\mathsf{H}}\widetilde{\mathbf{T}}_i \mathbf{C}_\text{T}(\theta_{\text{T},i}, \varphi_{\text{T},i})
\end{align}\end{split}
$$

where all path-dependent quantities carry the subscript $i$. Note that the matrices $\widetilde{\mathbf{T}}_i$ also ensure appropriate scaling so that the total received power can never be larger than the transmit power.

## Frequency & Impulse Response

The channel frequency response $H(f)$ at frequency $f=\frac{c}{\lambda}$ is defined as the ratio between the received voltage and the voltage at the input to the transmitting antenna:

$$
H(f) = \frac{V_\text{R}}{V_\text{T}} = \frac{V_\text{R}}{|V_\text{T}|}
$$

where it is assumed that the input voltage has zero phase.

It is useful to separate phase shifts due to wave propagation from the transfer matrices $\widetilde{\mathbf{T}}_i$. If we denote by $r_i$ the total length of path $i$ with average propagation speed $c_i$, the path delay is $\tau_i=r_i/c_i$. We can now define the new transfer matrix

$$
\mathbf{T}_i=\widetilde{\mathbf{T}}_ie^{j2\pi f \tau_i}.
$$

Using [(16)](https://nvlabs.github.io/sionna/em_primer.html#equation-p-t) and [(25)](https://nvlabs.github.io/sionna/em_primer.html#equation-t-tilde) in [(23)](https://nvlabs.github.io/sionna/em_primer.html#equation-v-rmulti) while assuming equal real parts of both antenna impedances, i.e., $\Re\{Z_\text{T}\}=\Re\{Z_\text{R}\}$ (which is typically the case), we obtain the final expression for the channel frequency response:

$$
\boxed{H(f) = \sum_{i=1}^N \underbrace{\frac{\lambda}{4\pi} \mathbf{C}_\text{R}(\theta_{\text{R},i}, \varphi_{\text{R},i})^{\mathsf{H}}\mathbf{T}_i \mathbf{C}_\text{T}(\theta_{\text{T},i}, \varphi_{\text{T},i})}_{\triangleq a_i} e^{-j2\pi f\tau_i}}
$$

Taking the inverse Fourier transform, we finally obtain the channel impulse response

$$
\boxed{h(\tau) = \int_{-\infty}^{\infty} H(f) e^{j2\pi f \tau} df = \sum_{i=1}^N a_i \delta(\tau-\tau_i)}
$$

The baseband equivalent channel impulse reponse is then defined as (Eq. 2.28) [[Tse]](https://nvlabs.github.io/sionna/em_primer.html#tse):

$$
h_\text{b}(\tau) = \sum_{i=1}^N \underbrace{a_i e^{-j2\pi f \tau_i}}_{\triangleq a^\text{b}_i} \delta(\tau-\tau_i).
$$
## Reflection and Refraction

When a plane wave hits a plane interface which separates two materials, e.g., air and concrete, a part of the wave gets reflected and the other transmitted (or *refracted*), i.e., it propagates into the other material.  We assume in the following description that both materials are uniform non-magnetic dielectrics, i.e., $\mu_r=1$, and follow the definitions as in [[ITURP20402]](https://nvlabs.github.io/sionna/em_primer.html#iturp20402). The incoming wave phasor $\mathbf{E}_\text{i}$ is expressed by two arbitrary orthogonal polarization components, i.e.,

$$
\mathbf{E}_\text{i} = E_{\text{i},s} \hat{\mathbf{e}}_{\text{i},s} + E_{\text{i},p} \hat{\mathbf{e}}_{\text{i},p}
$$

which are both orthogonal to the incident wave vector, i.e., $\hat{\mathbf{e}}_{\text{i},s}^{\mathsf{T}} \hat{\mathbf{e}}_{\text{i},p}=\hat{\mathbf{e}}_{\text{i},s}^{\mathsf{T}} \hat{\mathbf{k}}_\text{i}=\hat{\mathbf{e}}_{\text{i},p}^{\mathsf{T}} \hat{\mathbf{k}}_\text{i} =0$.

 ig. 1 Reflection and refraction of a plane wave at a plane interface between two materials.

[Fig. 1](https://nvlabs.github.io/sionna/em_primer.html#fig-reflection) shows reflection and refraction of the incoming wave at the plane interface between two materials with relative permittivities $\eta_1$ and $\eta_2$. The coordinate system is chosen such that the wave vectors of the incoming, reflected, and transmitted waves lie within the plane of incidence, which is chosen to be the x-z plane. The normal vector of the interface $\hat{\mathbf{n}}$ is pointing toward the negative z axis.
The incoming wave is must be represented in a different basis, i.e., in the form two different orthogonal polarization components $E_{\text{i}, \perp}$ and $E_{\text{i}, \parallel}$, i.e.,

$$
\mathbf{E}_\text{i} = E_{\text{i},\perp} \hat{\mathbf{e}}_{\text{i},\perp} + E_{\text{i},\parallel} \hat{\mathbf{e}}_{\text{i},\parallel}
$$

where the former is orthogonal to the plane of incidence and called transverse electric (TE) polarization (left), and the latter is parallel to the plane of incidence and called transverse magnetic (TM) polarization (right). We adopt in the following the convention that all transverse components are coming out of the figure (indicated by the $\odot$ symbol). One can easily verify that the following relationships must hold:

$$
\begin{split}\begin{align}
    \hat{\mathbf{e}}_{\text{i},\perp} &= \frac{\hat{\mathbf{k}}_\text{i} \times \hat{\mathbf{n}}}{\lVert \hat{\mathbf{k}}_\text{i} \times \hat{\mathbf{n}} \rVert} \\
    \hat{\mathbf{e}}_{\text{i},\parallel} &= \hat{\mathbf{e}}_{\text{i},\perp} \times \hat{\mathbf{k}}_\text{i}
\end{align}\end{split}
$$

$$
\begin{split}\begin{align}
\begin{bmatrix}E_{\text{i},\perp} \\ E_{\text{i},\parallel} \end{bmatrix} &=
    \begin{bmatrix}
        \hat{\mathbf{e}}_{\text{i},\perp}^\mathsf{T}\hat{\mathbf{e}}_{\text{i},s} & \hat{\mathbf{e}}_{\text{i},\perp}^\mathsf{T}\hat{\mathbf{e}}_{\text{i},p}\\
        \hat{\mathbf{e}}_{\text{i},\parallel}^\mathsf{T}\hat{\mathbf{e}}_{\text{i},s} & \hat{\mathbf{e}}_{\text{i},\parallel}^\mathsf{T}\hat{\mathbf{e}}_{\text{i},p}
    \end{bmatrix}
 \begin{bmatrix}E_{\text{i},s} \\ E_{\text{i},p}\end{bmatrix} =
 \mathbf{W}\left(\hat{\mathbf{e}}_{\text{i},\perp}, \hat{\mathbf{e}}_{\text{i},\parallel}, \hat{\mathbf{e}}_{\text{i},s}, \hat{\mathbf{e}}_{\text{i},p}\right)
\end{align}\end{split}
$$

where we have defined the following matrix-valued function

$$
\begin{split}\begin{align}
\mathbf{W}\left(\hat{\mathbf{a}}, \hat{\mathbf{b}}, \hat{\mathbf{q}}, \hat{\mathbf{r}} \right) =
    \begin{bmatrix}
        \hat{\mathbf{a}}^\textsf{T} \hat{\mathbf{q}} & \hat{\mathbf{a}}^\textsf{T} \hat{\mathbf{r}} \\
        \hat{\mathbf{b}}^\textsf{T} \hat{\mathbf{q}} & \hat{\mathbf{b}}^\textsf{T} \hat{\mathbf{r}}
    \end{bmatrix}.
\end{align}\end{split}
$$

While the angles of incidence and reflection are both equal to $\theta_1$, the angle of the refracted wave $\theta_2$ is given by Snells law:

$$
\sin(\theta_2) = \sqrt{\frac{\eta_1}{\eta_2}} \sin(\theta_1)
$$

or, equivalently,

$$
\cos(\theta_2) = \sqrt{1 - \frac{\eta_1}{\eta_2} \sin^2(\theta_1)}.
$$

The reflected and transmitted wave phasors $\mathbf{E}_\text{r}$ and $\mathbf{E}_\text{t}$ are similarly represented as

$$
\begin{split}\begin{align}
    \mathbf{E}_\text{r} &= E_{\text{r},\perp} \hat{\mathbf{e}}_{\text{r},\perp} + E_{\text{r},\parallel} \hat{\mathbf{e}}_{\text{r},\parallel}\\
    \mathbf{E}_\text{t} &= E_{\text{t},\perp} \hat{\mathbf{e}}_{\text{t},\perp} + E_{\text{t},\parallel} \hat{\mathbf{e}}_{\text{t},\parallel}
\end{align}\end{split}
$$

where

$$
\begin{split}\begin{align}
    \hat{\mathbf{e}}_{\text{r},\perp} &= \hat{\mathbf{e}}_{\text{i},\perp}\\
    \hat{\mathbf{e}}_{\text{r},\parallel} &= \frac{\hat{\mathbf{e}}_{\text{r},\perp}\times\hat{\mathbf{k}}_\text{r}}{\lVert \hat{\mathbf{e}}_{\text{r},\perp}\times\hat{\mathbf{k}}_\text{r} \rVert}\\
    \hat{\mathbf{e}}_{\text{t},\perp} &= \hat{\mathbf{e}}_{\text{i},\perp}\\
    \hat{\mathbf{e}}_{\text{t},\parallel} &= \frac{\hat{\mathbf{e}}_{\text{t},\perp}\times\hat{\mathbf{k}}_\text{t}}{ \Vert \hat{\mathbf{e}}_{\text{t},\perp}\times\hat{\mathbf{k}}_\text{t} \rVert}
\end{align}\end{split}
$$

and

$$
\begin{split}\begin{align}
    \hat{\mathbf{k}}_\text{r} &= \hat{\mathbf{k}}_\text{i} - 2\left( \hat{\mathbf{k}}_\text{i}^\mathsf{T}\hat{\mathbf{n}} \right)\hat{\mathbf{n}}\\
    \hat{\mathbf{k}}_\text{t} &= \sqrt{\frac{\eta_1}{\eta_2}} \hat{\mathbf{k}}_\text{i} + \left(\sqrt{\frac{\eta_1}{\eta_2}}\cos(\theta_1) - \cos(\theta_2) \right)\hat{\mathbf{n}}.
\end{align}\end{split}
$$

The *Fresnel* equations provide relationships between the incident, reflected, and refracted field components for $\sqrt{\left| \eta_1/\eta_2 \right|}\sin(\theta_1)<1$:

$$
\begin{split}\begin{align}
    r_{\perp}     &= \frac{E_{\text{r}, \perp    }}{E_{\text{i}, \perp    }} = \frac{ \sqrt{\eta_1}\cos(\theta_1) - \sqrt{\eta_2}\cos(\theta_2) }{ \sqrt{\eta_1}\cos(\theta_1) + \sqrt{\eta_2}\cos(\theta_2) } \\
    r_{\parallel} &= \frac{E_{\text{r}, \parallel}}{E_{\text{i}, \parallel}} = \frac{ \sqrt{\eta_2}\cos(\theta_1) - \sqrt{\eta_1}\cos(\theta_2) }{ \sqrt{\eta_2}\cos(\theta_1) + \sqrt{\eta_1}\cos(\theta_2) } \\
    t_{\perp}     &= \frac{E_{\text{t}, \perp    }}{E_{\text{t}, \perp    }} = \frac{ 2\sqrt{\eta_1}\cos(\theta_1) }{ \sqrt{\eta_1}\cos(\theta_1) + \sqrt{\eta_2}\cos(\theta_2) } \\
    t_{\parallel} &= \frac{E_{\text{t}, \parallel}}{E_{\text{t}, \parallel}} = \frac{ 2\sqrt{\eta_1}\cos(\theta_1) }{ \sqrt{\eta_2}\cos(\theta_1) + \sqrt{\eta_1}\cos(\theta_2) }.
\end{align}\end{split}
$$

If $\sqrt{\left| \eta_1/\eta_2 \right|}\sin(\theta_1)\ge 1$, we have $r_{\perp}=r_{\parallel}=1$ and $t_{\perp}=t_{\parallel}=0$, i.e., total reflection.

For the case of an incident wave in vacuum, i.e., $\eta_1=1$, the Fresnel equations [(33)](https://nvlabs.github.io/sionna/em_primer.html#equation-fresnel) simplify to

$$
\begin{split}\begin{align}
    r_{\perp}     &= \frac{\cos(\theta_1) -\sqrt{\eta_2 -\sin^2(\theta_1)}}{\cos(\theta_1) +\sqrt{\eta_2 -\sin^2(\theta_1)}} \\
    r_{\parallel} &= \frac{\eta_2\cos(\theta_1) -\sqrt{\eta_2 -\sin^2(\theta_1)}}{\eta_2\cos(\theta_1) +\sqrt{\eta_2 -\sin^2(\theta_1)}} \\
    t_{\perp}     &= \frac{2\cos(\theta_1)}{\cos(\theta_1) + \sqrt{\eta_2-\sin^2(\theta_1)}}\\
    t_{\parallel} &= \frac{2\sqrt{\eta_2}\cos(\theta_1)}{\eta_2 \cos(\theta_1) + \sqrt{\eta_2-\sin^2(\theta_1)}}.
\end{align}\end{split}
$$

Putting everything together, we obtain the following relationships between incident, reflected, and transmitted waves:

$$
\begin{split}\begin{align}
    \begin{bmatrix}E_{\text{r},\perp} \\ E_{\text{r},\parallel} \end{bmatrix} &=
    \begin{bmatrix}
        r_{\perp} & 0 \\
        0         & r_{\parallel}
    \end{bmatrix}
    \mathbf{W}\left(\hat{\mathbf{e}}_{\text{i},\perp}, \hat{\mathbf{e}}_{\text{i},\parallel}, \hat{\mathbf{e}}_{\text{i},s}, \hat{\mathbf{e}}_{\text{i},p}\right)
 \begin{bmatrix}E_{\text{i},s} \\ E_{\text{i},p}\end{bmatrix} \\
 \begin{bmatrix}E_{\text{t},\perp} \\ E_{\text{t},\parallel} \end{bmatrix} &=
    \begin{bmatrix}
        t_{\perp} & 0 \\
        0         & t_{\parallel}
    \end{bmatrix}
    \mathbf{W}\left(\hat{\mathbf{e}}_{\text{i},\perp}, \hat{\mathbf{e}}_{\text{i},\parallel}, \hat{\mathbf{e}}_{\text{i},s}, \hat{\mathbf{e}}_{\text{i},p}\right)
 \begin{bmatrix}E_{\text{i},s} \\ E_{\text{i},p}\end{bmatrix}.
\end{align}\end{split}
$$
## Diffraction

While modern geometrical optics (GO) [[Kline]](https://nvlabs.github.io/sionna/em_primer.html#kline), [[Luneberg]](https://nvlabs.github.io/sionna/em_primer.html#luneberg) can accurately describe phase and polarization properties of electromagnetic fields undergoing reflection and refraction (transmission) as described above, they fail to account for the phenomenon of diffraction, e.g., bending of waves around corners. This leads to the undesired and physically incorrect effect that the field abruptly falls to zero at geometrical shadow boundaries (for incident and reflected fields).

Joseph Keller presented in [[Keller62]](https://nvlabs.github.io/sionna/em_primer.html#keller62) a method which allowed the incorporation of diffraction into GO which is known as the geometrical theory of diffraction (GTD). He introduced the notion of diffracted rays that follow the law of edge diffraction, i.e., the diffracted and incident rays make the same angle with the edge at the point of diffraction and lie on opposite sides of the plane normal to the edge. The GTD suffers, however from several shortcomings, most importantly the fact that the diffracted field is infinite at shadow boundaries.

The uniform theory of diffraction (UTD) [[Kouyoumjian74]](https://nvlabs.github.io/sionna/em_primer.html#kouyoumjian74) alleviates this problem and provides solutions that are uniformly valid, even at shadow boundaries. For a great introduction to the UTD, we refer to [[McNamara90]](https://nvlabs.github.io/sionna/em_primer.html#mcnamara90). While [[Kouyoumjian74]](https://nvlabs.github.io/sionna/em_primer.html#kouyoumjian74) deals with diffraction at edges of perfectly conducting surfaces, it was heuristically extended to finitely conducting wedges in [[Luebbers84]](https://nvlabs.github.io/sionna/em_primer.html#luebbers84). This solution, which is also recomended by the ITU [[ITURP52615]](https://nvlabs.github.io/sionna/em_primer.html#iturp52615), is implemented in Sionna. However, both [[Luebbers84]](https://nvlabs.github.io/sionna/em_primer.html#luebbers84) and [[ITURP52615]](https://nvlabs.github.io/sionna/em_primer.html#iturp52615) only deal with two-dimensional scenes where source and observation lie in the same plane, orthogonal to the edge. We will provide below the three-dimensional version of [[Luebbers84]](https://nvlabs.github.io/sionna/em_primer.html#luebbers84), following the defintitions of (Ch. 6) [[McNamara90]](https://nvlabs.github.io/sionna/em_primer.html#mcnamara90). A similar result can be found, e.g., in (Eq. 6-296-39) [[METIS]](https://nvlabs.github.io/sionna/em_primer.html#metis).

 ig. 2 Incident and diffracted rays for an infinitely long wedge in an edge-fixed coordinate system.

We consider an infinitely long wedge with unit norm edge vector $\hat{\mathbf{e}}$, as shown in [Fig. 2](https://nvlabs.github.io/sionna/em_primer.html#fig-kellers-cone). An incident ray of a spherical wave with field phasor $\mathbf{E}_i(S')$ at point $S'$ propagates in the direction $\hat{\mathbf{s}}'$ and is diffracted at point $Q_d$ on the edge. The diffracted ray of interest (there are infinitely many on Kellers cone) propagates
in the direction $\hat{\mathbf{s}}$ towards the point of observation $S$. We denote by $s'=\lVert S'-Q_d \rVert$ and $s=\lVert Q_d - S\rVert$ the lengths of the incident and diffracted path segments, respectively. By the law of edge diffraction, the angles $\beta_0'$ and $\beta_0$ between the edge and the incident and diffracted rays, respectively, satisfy:

$$
\begin{equation}
    \cos(\beta_0') = |\hat{\mathbf{s}}'^\textsf{T}\hat{\mathbf{e}}| = |\hat{\mathbf{s}}^\textsf{T}\hat{\mathbf{e}}| = \cos(\beta_0).
\end{equation}
$$

To be able to express the diffraction coefficients as a 2x2 matrixsimilar to what is done for reflection and refractionthe incident field must be resolved into two components $E_{i,\phi'}$ and $E_{i,\beta_0'}$, the former orthogonal and the latter parallel to the edge-fixed plane of incidence, i.e., the plane containing $\hat{\mathbf{e}}$ and $\hat{\mathbf{s}}'$. The diffracted field is then represented by two components $E_{d,\phi}$ and $E_{d,\beta_0}$ that are respectively orthogonal and parallel to the edge-fixed plane of diffraction, i.e., the plane containing $\hat{\mathbf{e}}$ and $\hat{\mathbf{s}}$.
The corresponding component unit vectors are defined as

$$
\begin{split}\begin{align}
    \hat{\boldsymbol{\phi}}' &= \frac{\hat{\mathbf{s}}' \times \hat{\mathbf{e}}}{\lVert \hat{\mathbf{s}}' \times \hat{\mathbf{e}} \rVert }\\
    \hat{\boldsymbol{\beta}}_0' &=  \hat{\boldsymbol{\phi}}' \times \hat{\mathbf{s}}' \\
    \hat{\boldsymbol{\phi}} &= -\frac{\hat{\mathbf{s}} \times \hat{\mathbf{e}}}{\lVert \hat{\mathbf{s}} \times \hat{\mathbf{e}} \rVert }\\
    \hat{\boldsymbol{\beta}}_0 &=  \hat{\boldsymbol{\phi}} \times \hat{\mathbf{s}}.
\end{align}\end{split}
$$

[Fig. 3](https://nvlabs.github.io/sionna/em_primer.html#fig-diffraction) below shows the top view on the wedge that we need for some additional definitions.

 ig. 3 Top view on the wedge with edge vector pointing upwards.

The wedge has two faces called *0-face* and *n-face*, respectively, with surface normal vectors $\hat{\mathbf{n}}_0$ and $\hat{\mathbf{n}}_n$. The exterior wedge angle is $n\pi$, with $1\le n \le 2$. Note that the surfaces are chosen such that $\hat{\mathbf{e}} = \hat{\mathbf{n}}_0 \times \hat{\mathbf{n}}_n$. For $n=2$, the wedge reduces to a screen and the choice of the *0-face* and *n-face* is arbitrary as they point in opposite directions.

The incident and diffracted rays have angles $\phi'$ and $\phi$ measured with respect to the *0-face* in the plane perpendicular to the edge.
They can be computed as follows:

$$
\begin{split}\begin{align}
    \phi' & = \pi - \left[\pi - \cos^{-1}\left( -\hat{\mathbf{s}}_t'^\textsf{T} \hat{\mathbf{t}}_0\right) \right] \mathop{\text{sgn}}\left(-\hat{\mathbf{s}}_t'^\textsf{T} \hat{\mathbf{n}}_0\right)\\
    \phi & = \pi - \left[\pi - \cos^{-1}\left( \hat{\mathbf{s}}_t^\textsf{T} \hat{\mathbf{t}}_0\right) \right] \mathop{\text{sgn}}\left(\hat{\mathbf{s}}_t^\textsf{T} \hat{\mathbf{n}}_0\right)
\end{align}\end{split}
$$

where

$$
\begin{split}\begin{align}
    \hat{\mathbf{t}}_0 &= \hat{\mathbf{n}}_0 \times \hat{\mathbf{e}}\\
    \hat{\mathbf{s}}_t' &= \frac{ \hat{\mathbf{s}}' - \left( \hat{\mathbf{s}}'^\textsf{T}\hat{\mathbf{e}} \right)\hat{\mathbf{e}} }{\lVert \hat{\mathbf{s}}' - \left( \hat{\mathbf{s}}'^\textsf{T}\hat{\mathbf{e}} \right)\hat{\mathbf{e}}  \rVert}\\
    \hat{\mathbf{s}}_t  &= \frac{ \hat{\mathbf{s}} - \left( \hat{\mathbf{s}}^\textsf{T}\hat{\mathbf{e}} \right)\hat{\mathbf{e}} }{\lVert \hat{\mathbf{s}} - \left( \hat{\mathbf{s}}^\textsf{T}\hat{\mathbf{e}} \right)\hat{\mathbf{e}}  \rVert}
\end{align}\end{split}
$$

are the unit vector tangential to the *0-face*, as well as the unit vectors pointing in the directions of $\hat{\mathbf{s}}'$ and $\hat{\mathbf{s}}$, projected on the plane perpendicular to the edge, respectively. The function $\mathop{\text{sgn}}(x)$ is defined in this context as

$$
\begin{split}\mathop{\text{sgn}}(x) = \begin{cases}
                         1  &, x \ge 0\\
                         -1 &, x< 0.
                         \end{cases}\end{split}
$$

With these definitions, the diffracted field at point $S$ can be computed from the incoming field at point $S'$ as follows:

$$
\begin{split}\begin{align}
    \begin{bmatrix}
        E_{d,\phi} \\
        E_{d,\beta_0}
    \end{bmatrix} (S) = - \left( \left(D_1 + D_2\right)\mathbf{I} - D_3 \mathbf{R}_n - D_4\mathbf{R}_0 \right)\begin{bmatrix}
        E_{i,\phi'} \\
        E_{i,\beta_0'}
    \end{bmatrix}(S') \sqrt{\frac{1}{s's(s'+s)}} e^{-jk(s'+s)}
\end{align}\end{split}
$$

where $k=2\pi/\lambda$ is the wave number and the matrices $\mathbf{R}_\nu,\, \nu \in [0,n]$, are given as

$$
\begin{split}\begin{align}
    \mathbf{R}_\nu = \mathbf{W}\left(\hat{\boldsymbol{\phi}}, \hat{\boldsymbol{\beta}}_0, \hat{\mathbf{e}}_{r, \perp, \nu}, \hat{\mathbf{e}}_{r, \parallel, \nu}  \right)
                    \begin{bmatrix}
                        r_{\perp}(\theta_{r,\nu}, \eta_{\nu}) & 0\\
                        0 & r_{\parallel}(\theta_{r,\nu}, \eta_{nu})
                    \end{bmatrix}
                     \mathbf{W}\left( \hat{\mathbf{e}}_{i, \perp, \nu}, \hat{\mathbf{e}}_{i, \parallel, \nu}, \hat{\boldsymbol{\phi}}', \hat{\boldsymbol{\beta}}_0' \right)
\end{align}\end{split}
$$

with $\mathbf{W}(\cdot)$ as defined in [(30)](https://nvlabs.github.io/sionna/em_primer.html#equation-w), where $r_{\perp}(\theta_{r,\nu}, \eta_{\nu})$ and $r_{\parallel}(\theta_{r,\nu}, \eta_{\nu})$ are the Fresnel reflection coefficents from [(34)](https://nvlabs.github.io/sionna/em_primer.html#equation-fresnel-vac), evaluated for the complex relative permittivities $\eta_{\nu}$ and angles $\theta_{r_,\nu}$ with cosines

$$
\begin{split}\begin{align}
    \cos\left(\theta_{r,0}\right) &= \left|\sin(\phi') \right|\\
    \cos\left(\theta_{r,n}\right) &= \left|\sin(n\pi -\phi) \right|.
\end{align}\end{split}
$$

and where

$$
\begin{split}\begin{align}
    \hat{\mathbf{e}}_{i,\perp,\nu} &= \frac{ \hat{\mathbf{s}}' \times \hat{\mathbf{n}}_{\nu} }{\lVert \hat{\mathbf{s}}' \times \hat{\mathbf{n}}_{\nu} \rVert}\\
    \hat{\mathbf{e}}_{i,\parallel,\nu} &=  \hat{\mathbf{e}}_{i,\perp,\nu} \times \hat{\mathbf{s}}'\\
    \hat{\mathbf{e}}_{r,\perp,\nu} &=  \hat{\mathbf{e}}_{i,\perp,\nu}\\
    \hat{\mathbf{e}}_{r,\parallel,\nu} &=  \hat{\mathbf{e}}_{i,\perp,\nu} \times \hat{\mathbf{s}}
\end{align}\end{split}
$$

as already defined in [(29)](https://nvlabs.github.io/sionna/em_primer.html#equation-fresnel-in-vectors) and [(31)](https://nvlabs.github.io/sionna/em_primer.html#equation-fresnel-out-vectors), but made explicit here for the case of diffraction. The matrices $\mathbf{R}_\nu$ simply describe the reflected field from both surfaces in the basis used for the description of the diffraction process. Note that the absolute value is used in [(36)](https://nvlabs.github.io/sionna/em_primer.html#equation-diffraction-cos) to account for virtual reflections from shadowed surfaces, see the discussion in (p.185) [[McNamara90]](https://nvlabs.github.io/sionna/em_primer.html#mcnamara90).
The diffraction coefficients $D_1,\dots,D_4$ are computed as

$$
\begin{split}\begin{align}
    D_1 &= \frac{-e^{-\frac{j\pi}{4}}}{2n\sqrt{2\pi k} \sin(\beta_0)} \mathop{\text{cot}}\left( \frac{\pi+(\phi-\phi')}{2n}\right) F\left( k L a^+(\phi-\phi')\right)\\
    D_2 &= \frac{-e^{-\frac{j\pi}{4}}}{2n\sqrt{2\pi k} \sin(\beta_0)} \mathop{\text{cot}}\left( \frac{\pi-(\phi-\phi')}{2n}\right) F\left( k L a^-(\phi-\phi')\right)\\
    D_3 &= \frac{-e^{-\frac{j\pi}{4}}}{2n\sqrt{2\pi k} \sin(\beta_0)} \mathop{\text{cot}}\left( \frac{\pi+(\phi+\phi')}{2n}\right) F\left( k L a^+(\phi+\phi')\right)\\
    D_4 &= \frac{-e^{-\frac{j\pi}{4}}}{2n\sqrt{2\pi k} \sin(\beta_0)} \mathop{\text{cot}}\left( \frac{\pi-(\phi+\phi')}{2n}\right) F\left( k L a^-(\phi+\phi')\right)
\end{align}\end{split}
$$

where

$$
\begin{split}\begin{align}
    L &= \frac{ss'}{s+s'}\sin^2(\beta_0)\\
    a^{\pm}(\beta) &= 2\cos^2\left(\frac{2n\pi N^{\pm}-\beta}{2}\right)\\
    N^{\pm} &= \mathop{\text{round}}\left(\frac{\beta\pm\pi}{2n\pi}\right)\\
    F(x) &= 2j\sqrt{x}e^{jx}\int_{\sqrt{x}}^\infty e^{-jt^2}dt
\end{align}\end{split}
$$

and $\mathop{\text{round}}()$ is the function that rounds to the closest integer. The function $F(x)$ can be expressed with the help of the standard Fresnel integrals [[Fresnel]](https://nvlabs.github.io/sionna/em_primer.html#fresnel)

$$
\begin{split}\begin{align}
    S(x) &= \int_0^x \sin\left( \pi t^2/2 \right)dt \\
    C(x) &= \int_0^x \cos\left( \pi t^2/2 \right)dt
\end{align}\end{split}
$$

as

$$
\begin{align}
    F(x) = \sqrt{\frac{\pi x}{2}} e^{jx} \left[1+j-2\left( S\left(\sqrt{2x/\pi}\right) +jC\left(\sqrt{2x/\pi}\right) \right) \right].
\end{align}
$$
## Scattering

When an electromagnetic wave impinges on a surface, one part of the energy gets reflected while the other part gets refracted, i.e., it propagates into the surface.
We distinguish between two types of reflection, specular and diffuse. The former type is discussed in [Reflection and Refraction](https://nvlabs.github.io/sionna/em_primer.html#reflection-and-refraction) and we will focus now on the latter type which is also called diffuse scattering. When a rays hits a diffuse reflection surface, it is not reflected into a single (specular) direction but rather scattered toward many different directions. Since most surfaces give both specular and diffuse reflections, we denote by $S^2$ the fraction of the reflected energy that is diffusely scattered, where $S\in[0,1]$ is the so-called *scattering coefficient* [[Degli-Esposti07]](https://nvlabs.github.io/sionna/em_primer.html#degli-esposti07). Similarly, $R^2$ is the specularly reflected fraction of the reflected energy, where $R\in[0,1]$ is the *reflection reduction factor*. The following relationship between $R$ and $S$ holds:

$$
R = \sqrt{1-S^2}.
$$

Whenever a material has a scattering coefficient $S>0$, the Fresnel reflection coefficents in [(33)](https://nvlabs.github.io/sionna/em_primer.html#equation-fresnel) must be multiplied by $R$. These *reduced* coefficients must then be also used in the compuation of the diffraction coefficients [(35)](https://nvlabs.github.io/sionna/em_primer.html#equation-diff-mat).

 ig. 4 Diffuse and specular reflection of an incoming wave.

Let us consider an incoming locally planar linearly polarized wave with field phasor $\mathbf{E}_\text{i}(\mathbf{q})$ at the scattering point $\mathbf{q}$ on the surface, as shown in [Fig. 4](https://nvlabs.github.io/sionna/em_primer.html#fig-scattering). We focus on the scattered field of and infinitesimally small surface element $dA$ in the direction $\hat{\mathbf{k}}_\text{s}$. Note that the surface normal $\hat{\mathbf{n}}$ has an arbitrary orientation with respect to the global coordinate system, whose $(x,y,z)$ axes are shown in green dotted lines.
The incoming field phasor can be represented by two arbitrary orthogonal polarization components (both orthogonal to the incoming wave vector $\hat{\mathbf{k}}_i$):

$$
\begin{split}\begin{align}
\mathbf{E}_\text{i} &= E_{\text{i},s} \hat{\mathbf{e}}_{\text{i},s} + E_{\text{i},p} \hat{\mathbf{e}}_{\text{i},p} \\
                    &= E_{\text{i},\perp} \hat{\mathbf{e}}_{\text{i},\perp} + E_{\text{i},\parallel} \hat{\mathbf{e}}_{\text{i},\parallel} \\
                    &= E_{\text{i},\text{pol}} \hat{\mathbf{e}}_{\text{i},\text{pol}} + E_{\text{i},\text{xpol}} \hat{\mathbf{e}}_{\text{i},\text{xpol}}
\end{align}\end{split}
$$

where me have omitted the dependence of the field strength on the position $\mathbf{q}$ for brevity.
The second representation via $(E_{\text{i},\perp}, E_{\text{i},\parallel})$ is used for the computation of the specularly reflected field as explained in [Reflection and refraction](https://nvlabs.github.io/sionna/em_primer.html#reflection-and-refraction). The third representation via $(E_{\text{i},\text{pol}}, E_{\text{i},\text{xpol}})$ will be used to express the scattered field, where

$$
\begin{split}\begin{align}
\hat{\mathbf{e}}_{\text{i},\text{pol}} &= = \frac{\Re\left\{\mathbf{E}_\text{i}\right\}}{\lVert \Re\left\{\mathbf{E}_\text{i}\right\} \rVert} =  \frac{\Re\left\{E_{\text{i},s}\right\}}{ \lVert\Re\left\{\mathbf{E}_\text{i} \right\} \rVert} \hat{\mathbf{e}}_{\text{i},s} + \frac{\Re\left\{E_{\text{i},p}\right\}}{\lVert\Re\left\{\mathbf{E}_\text{i} \right\} \rVert} \hat{\mathbf{e}}_{\text{i},p}\\
\hat{\mathbf{e}}_{\text{i},\text{xpol}} &= \hat{\mathbf{e}}_\text{pol} \times \hat{\mathbf{k}}_\text{i}
\end{align}\end{split}
$$

such that $|E_{\text{i},\text{pol}}|=\lVert \mathbf{E}_\text{i} \rVert$ and $E_{\text{i},\text{xpol}}=0$. That means that $\hat{\mathbf{e}}_{\text{i},\text{pol}}$ points toward the polarization direction which carries all of the energy.

According to (Eq. 9) [[Degli-Esposti11]](https://nvlabs.github.io/sionna/em_primer.html#degli-esposti11), the diffusely scattered field $\mathbf{E}_\text{s}(\mathbf{r})$ at the observation point $\mathbf{r}$ can be modeled as
$\mathbf{E}_\text{s}(\mathbf{r})=E_{\text{s}, \theta}\hat{\boldsymbol{\theta}}(\hat{\mathbf{k}}_\text{s}) + E_{\text{s}, \varphi}\hat{\boldsymbol{\varphi}}(\hat{\mathbf{k}}_\text{s})$, where
$\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\varphi}}$ are defined in [(1)](https://nvlabs.github.io/sionna/em_primer.html#equation-spherical-vecs) and the orthogonal field components are computed as

$$
\begin{split}\begin{bmatrix}E_{\text{s}, \theta} \\ E_{\text{s}, \varphi} \end{bmatrix}(\mathbf{r}) &= \frac{\lVert \mathbf{E}_\text{s}(\mathbf{q}) \rVert}{\lVert \mathbf{r} - \mathbf{q} \rVert}
\mathbf{W}\left( \hat{\boldsymbol{\theta}}(-\hat{\mathbf{k}}_\text{i}), \hat{\boldsymbol{\varphi}}(-\hat{\mathbf{k}}_\text{i}), \hat{\mathbf{e}}_{\text{i},\text{pol}}, \hat{\mathbf{e}}_{\text{i},\text{xpol}} \right)
 \begin{bmatrix} \sqrt{1-K_x}e^{j\chi_1} \\ \sqrt{K_x}e^{j\chi_2}  \end{bmatrix}\end{split}
$$

where $\mathbf{W}(\cdot)$ as defined in [(30)](https://nvlabs.github.io/sionna/em_primer.html#equation-w), $\chi_1, \chi_2 \in [0,2\pi]$ are independent random phase shifts, and the quantity $K_x\in[0,1]$ is defined by the scattering cross-polarization discrimination

$$
\text{XPD}_\text{s} = 10\log_{10}\left(\frac{|E_{\text{s}, \text{pol}}|^2}{|E_{\text{s}, \text{xpol}}|^2} \right) = 10\log_{10}\left(\frac{1-K_x}{K_x} \right).
$$

This quantity determines how much energy gets transfered from $\hat{\mathbf{e}}_{\text{i},\text{pol}}$ into the orthogonal polarization direction $\hat{\mathbf{e}}_{\text{i},\text{xpol}}$ through the scattering process. The matrix $\mathbf{W}$ is used to represent the scattered electric field in the vertical ($\hat{\boldsymbol{\theta}}$) and horizontal ($\hat{\boldsymbol{\varphi}}$) polarization components according to the incoming ray direction $-\hat{\mathbf{k}}_\text{i}$. It is then assumed that the same polarization is kept for the outgoing ray in the $\hat{\mathbf{k}}_\text{s}$ direction.

The squared amplitude of the diffusely scattered field in [(38)](https://nvlabs.github.io/sionna/em_primer.html#equation-scattered-field) can be expressed as (Eq. 8) [[Degli-Esposti07]](https://nvlabs.github.io/sionna/em_primer.html#degli-esposti07):

$$
\lVert \mathbf{E}_\text{s}(\mathbf{q})) \rVert^2 = \underbrace{\lVert \mathbf{E}_\text{i}(\mathbf{q}) \rVert^2 \cos(\theta_i) dA}_{\sim \text{incoming power} } \cdot \underbrace{\left(S\Gamma\right)^2}_{\text{fraction of diffusely reflected power}} \cdot \underbrace{f_\text{s}\left(\hat{\mathbf{k}}_\text{i}, \hat{\mathbf{k}}_\text{s}, \hat{\mathbf{n}}\right)}_{\text{scattering pattern}}
$$

where $\Gamma^2$ is the percentage of the incoming power that is reflected (specularly and diffuse), which can be computed as

$$
\Gamma = \frac{\sqrt{ |r_{\perp} E_{\text{i},\perp} |^2 + |r_{\parallel} E_{\text{i},\parallel} |^2}}
          {\lVert \mathbf{E}_\text{i}(\mathbf{q}) \rVert}
$$

where $r_{\perp}, r_{\parallel}$ are defined in [(33)](https://nvlabs.github.io/sionna/em_primer.html#equation-fresnel), $dA$ is the size of the small area element on the reflecting surface under consideration, and $f_\text{s}\left(\hat{\mathbf{k}}_i, \hat{\mathbf{k}}_s, \hat{\mathbf{n}}\right)$ is the *scattering pattern*, which has similarities with the bidirectional reflectance distribution function (BRDF) in computer graphics (Ch. 5.6.1) [[Pharr]](https://nvlabs.github.io/sionna/em_primer.html#pharr).
The scattering pattern must be normalized to satisfy the condition

$$
\int_{0}^{\pi/2}\int_0^{2\pi} f_\text{s}\left(\hat{\mathbf{k}}_\text{i}, \hat{\mathbf{k}}_\text{s}, \hat{\mathbf{n}}\right) \sin(\theta_s) d\phi_s d\theta_s = 1
$$

which ensures the power balance between the incoming, reflected, and refracted fields.

**Example scattering patterns**

The authors of [[Degli-Esposti07]](https://nvlabs.github.io/sionna/em_primer.html#degli-esposti07) derived several simple scattering patterns that were shown to achieve good agreement with measurements when correctly parametrized.

**Lambertian Model** ([`LambertianPattern`](api/rt.html#sionna.rt.LambertianPattern)):
This model describes a perfectly diffuse scattering surface whose *scattering radiation lobe* has its maximum in the direction of the surface normal:

$$
f^\text{Lambert}_\text{s}\left(\hat{\mathbf{k}}_\text{i}, \hat{\mathbf{k}}_\text{s}, \hat{\mathbf{n}}\right) = \frac{\hat{\mathbf{n}}^\mathsf{T} \hat{\mathbf{k}}_\text{s} }{\pi} = \frac{\cos(\theta_s)}{\pi}
$$

**Directive Model** ([`DirectivePattern`](api/rt.html#sionna.rt.DirectivePattern)):
This model assumes that the scattered field is concentrated around the direction of the specular reflection $\hat{\mathbf{k}}_\text{r}$ (defined in [(32)](https://nvlabs.github.io/sionna/em_primer.html#equation-reflected-refracted-vectors)). The width of the scattering lobe
can be controlled via the integer parameter $\alpha_\text{R}=1,2,\dots$:

$$
f^\text{directive}_\text{s}\left(\hat{\mathbf{k}}_\text{i}, \hat{\mathbf{k}}_\text{s}, \hat{\mathbf{n}}\right) = F_{\alpha_\text{R}}(\theta_i)^{-1} \left(\frac{ 1 + \hat{\mathbf{k}}_\text{r}^\mathsf{T} \hat{\mathbf{k}}_\text{s}}{2}\right)^{\alpha_\text{R}}
$$

$$
F_{\alpha}(\theta_i) = \frac{1}{2^\alpha} \sum_{k=0}^\alpha \binom{\alpha}{k} I_k,\qquad \theta_i =\cos^{-1}(-\hat{\mathbf{k}}_\text{i}^\mathsf{T}\hat{\mathbf{n}})
$$

$$
\begin{split}I_k = \frac{2\pi}{k+1} \begin{cases}
        1 & ,\quad k \text{ even} \\
        \cos(\theta_i) \sum_{w=0}^{(k-1)/2} \binom{2w}{w} \frac{\sin^{2w}(\theta_i)}{2^{2w}}  &,\quad k \text{ odd}
      \end{cases}\end{split}
$$

**Backscattering Lobe Model** ([`BackscatteringPattern`](api/rt.html#sionna.rt.BackscatteringPattern)):
This model adds a scattering lobe to the directive model described above which points toward the direction from which the incident wave arrives (i.e., $-\hat{\mathbf{k}}_\text{i}$). The width of this lobe is controlled by the parameter $\alpha_\text{I}=1,2,\dots$. The parameter $\Lambda\in[0,1]$ determines the distribution of energy between both lobes. For $\Lambda=1$, this models reduces to the directive model.

$$
f^\text{bs}_\text{s}\left(\hat{\mathbf{k}}_\text{i}, \hat{\mathbf{k}}_\text{s}, \hat{\mathbf{n}}\right) = F_{\alpha_\text{R}, \alpha_\text{I}}(\theta_i)^{-1} \left[ \Lambda \left(\frac{ 1 + \hat{\mathbf{k}}_\text{r}^\mathsf{T} \hat{\mathbf{k}}_\text{s}}{2}\right)^{\alpha_\text{R}} + (1-\Lambda) \left(\frac{ 1 - \hat{\mathbf{k}}_\text{i}^\mathsf{T} \hat{\mathbf{k}}_\text{s}}{2}\right)^{\alpha_\text{I}}\right]
$$

$$
F_{\alpha, \beta}(\theta_i)^{-1} = \Lambda F_\alpha(\theta_i) + (1-\Lambda)F_\beta(\theta_i)
$$

## References:
[atan2](https://nvlabs.github.io/sionna/em_primer.html#id2)

Wikipedia, [atan2](https://en.wikipedia.org/wiki/Atan2), accessed 8 Feb. 2023.

[Balanis](https://nvlabs.github.io/sionna/em_primer.html#id5)
<ol class="upperalpha simple">
- Balanis, Advanced Engineering Electromagnetics, John Wiley & Sons, 2012.
</ol>

Degli-Esposti07([1](https://nvlabs.github.io/sionna/em_primer.html#id24),[2](https://nvlabs.github.io/sionna/em_primer.html#id26),[3](https://nvlabs.github.io/sionna/em_primer.html#id28))

Vittorio Degli-Esposti et al., [Measurement and modelling of scattering from buildings](https://ieeexplore.ieee.org/abstract/document/4052607), IEEE Trans. Antennas Propag, vol. 55, no. 1,  pp.143-153, Jan. 2007.

[Degli-Esposti11](https://nvlabs.github.io/sionna/em_primer.html#id25)

Vittorio Degli-Esposti et al., [Analysis and Modeling on co- and Cross-Polarized Urban Radio Propagation for Dual-Polarized MIMO Wireless Systems](https://ieeexplore.ieee.org/abstract/document/5979177), IEEE Trans. Antennas Propag, vol. 59, no. 11,  pp.4247-4256, Nov. 2011.

[Fresnel](https://nvlabs.github.io/sionna/em_primer.html#id23)

Wikipedia, [Fresnel integral](https://en.wikipedia.org/wiki/Fresnel_integral), accessed 21 Apr. 2023.

[ITURP20402](https://nvlabs.github.io/sionna/em_primer.html#id8)

ITU, [Recommendation ITU-R P.2040-2: Effects of building materials and structures on radiowave propagation above about 100 MHz](https://www.itu.int/rec/R-REC-P.2040/en). Sep. 2021.

ITURP52615([1](https://nvlabs.github.io/sionna/em_primer.html#id16),[2](https://nvlabs.github.io/sionna/em_primer.html#id18))

ITU, [Recommendation ITU-R P.526-15: Propagation by diffraction](https://www.itu.int/rec/R-REC-P.526/en), Oct. 2019.

[Keller62](https://nvlabs.github.io/sionna/em_primer.html#id11)

J.B. Keller, [Geometrical Theory of Diffraction](https://opg.optica.org/josa/abstract.cfm?uri=josa-52-2-116), Journal of the Optical Society of America, vol. 52, no. 2, Feb. 1962.

[Kline](https://nvlabs.github.io/sionna/em_primer.html#id9)
<ol class="upperalpha simple" start="13">
- Kline, An Asymptotic Solution of Maxwells Equations, Commun. Pure Appl. Math., vol. 4, 1951.
</ol>

Kouyoumjian74([1](https://nvlabs.github.io/sionna/em_primer.html#id12),[2](https://nvlabs.github.io/sionna/em_primer.html#id14))

R.G. Kouyoumjian, [A uniform geometrical theory of diffraction for an edge in a perfectly conducting surface](https://ieeexplore.ieee.org/abstract/document/1451581/authors#authors), Proc. of the IEEE, vol. 62, no. 11, Nov. 1974.

Luebbers84([1](https://nvlabs.github.io/sionna/em_primer.html#id15),[2](https://nvlabs.github.io/sionna/em_primer.html#id17),[3](https://nvlabs.github.io/sionna/em_primer.html#id19))
<ol class="upperalpha simple" start="18">
- Luebbers, [Finite conductivity uniform GTD versus knife edge diffraction in prediction of propagation path loss](https://ieeexplore.ieee.org/abstract/document/1143189), IEEE Trans. Antennas and Propagation, vol. 32, no. 1, Jan. 1984.
</ol>

[Luneberg](https://nvlabs.github.io/sionna/em_primer.html#id10)

R.M. Luneberg, Mathematical Theory of Optics, Brown University Press, 1944.

McNamara90([1](https://nvlabs.github.io/sionna/em_primer.html#id13),[2](https://nvlabs.github.io/sionna/em_primer.html#id20),[3](https://nvlabs.github.io/sionna/em_primer.html#id22))

D.A. McNamara, C.W.I. Pistorius, J.A.G. Malherbe, [Introduction to the Uniform Geometrical Theory of Diffraction](https://us.artechhouse.com/Introduction-to-the-Uniform-Geometrical-Theory-of-Diffraction-P288.aspx), Artech House, 1990.

[METIS](https://nvlabs.github.io/sionna/em_primer.html#id21)

METIS Deliverable D1.4, [METIS Channel Models](https://metis2020.com/wp-content/uploads/deliverables/METIS_D1.4_v1.0.pdf), Feb. 2015.

[Tse](https://nvlabs.github.io/sionna/em_primer.html#id7)
<ol class="upperalpha simple" start="4">
- Tse, P. Viswanath, [Fundamentals of Wireless Communication](https://web.stanford.edu/~dntse/wireless_book.html), Cambridge University Press, 2005.
</ol>

[Wiesbeck](https://nvlabs.github.io/sionna/em_primer.html#id1)
<ol class="upperalpha simple" start="14">
- Geng and W. Wiesbeck, Planungsmethoden fr die Mobilkommunikation, Springer, 1998.
</ol>

[Wikipedia](https://nvlabs.github.io/sionna/em_primer.html#id6)

Wikipedia, [Maximum power transfer theorem](https://en.wikipedia.org/wiki/Maximum_power_transfer_theorem), accessed 7 Oct. 2022.

[Wikipedia_Rodrigues](https://nvlabs.github.io/sionna/em_primer.html#id4)

Wikipedia, [Rodrigues rotation formula](https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula), accessed 16 Jun. 2023.

[Pharr](https://nvlabs.github.io/sionna/em_primer.html#id27)
<ol class="upperalpha simple" start="13">
- Pharr, J. Wenzel, G. Humphreys, [Physically Based Rendering: From Theory to Implementation](https://www.pbr-book.org/3ed-2018/contents), MIT Press, 2023.
</ol>



