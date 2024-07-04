
# Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#utility-functions" title="Permalink to this headline"></a>
    
This module provides utility functions for the FEC package. It also provides serval functions to simplify EXIT analysis of iterative receivers.

## (Binary) Linear Codes<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#binary-linear-codes" title="Permalink to this headline"></a>
    
Several functions are provided to convert parity-check matrices into generator matrices and vice versa. Please note that currently only binary codes are supported.
```python
# load example parity-check matrix
pcm, k, n, coderate = load_parity_check_examples(pcm_id=3)
```

    
Note that many research projects provide their parity-check matrices in the  <cite>alist</cite> format <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#mackay" id="id1">[MacKay]</a> (e.g., see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#unikl" id="id2">[UniKL]</a>). The follwing code snippet provides an example of how to import an external LDPC parity-check matrix from an <cite>alist</cite> file and how to set-up an encoder/decoder.
```python
# load external example parity-check matrix in alist format
al = load_alist(path=filename)
pcm, k, n, coderate = alist2mat(al)
# the linear encoder can be directly initialized with a parity-check matrix
encoder = LinearEncoder(pcm, is_pcm=True)
# initalize BP decoder for the given parity-check matrix
decoder = LDPCBPDecoder(pcm, num_iter=20)
# and run simulation with random information bits
no = 1.
batch_size = 10
num_bits_per_symbol = 2
source = BinarySource()
mapper = Mapper("qam", num_bits_per_symbol)
channel = AWGN()
demapper = Demapper("app", "qam", num_bits_per_symbol)
u = source([batch_size, k])
c = encoder(u)
x = mapper(c)
y = channel([x, no])
llr = demapper([y, no])
c_hat = decoder(llr)
```
### load_parity_check_examples<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#load-parity-check-examples" title="Permalink to this headline"></a>

`sionna.fec.utils.``load_parity_check_examples`(<em class="sig-param">`pcm_id`</em>, <em class="sig-param">`verbose``=``False`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#load_parity_check_examples">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.load_parity_check_examples" title="Permalink to this definition"></a>
    
Utility function to load example codes stored in sub-folder LDPC/codes.
    
The following codes are available
 
- 0 : <cite>(7,4)</cite>-Hamming code of length <cite>k=4</cite> information bits and codeword    length <cite>n=7</cite>.
- 1 : <cite>(63,45)</cite>-BCH code of length <cite>k=45</cite> information bits and codeword    length <cite>n=63</cite>.
- 2 : (127,106)-BCH code of length <cite>k=106</cite> information bits and codeword    length <cite>n=127</cite>.
- 3 : Random LDPC code with regular variable node degree 3 and check node degree 6 of length <cite>k=50</cite> information bits and codeword length         <cite>n=100</cite>.
- 4 : 802.11n LDPC code of length of length <cite>k=324</cite> information bits and    codeword length <cite>n=648</cite>.

Input
 
- **pcm_id** (<em>int</em>) – An integer defining which matrix id to load.
- **verbose** (<em>bool</em>) – Defaults to False. If True, the code parameters are
printed.


Output
 
- **pcm** (ndarray of <cite>zeros</cite> and <cite>ones</cite>) – An ndarray containing the parity check matrix.
- **k** (<em>int</em>) – An integer defining the number of information bits.
- **n** (<em>int</em>) – An integer defining the number of codeword bits.
- **coderate** (<em>float</em>) – A float defining the coderate (assuming full rank of
parity-check matrix).




### alist2mat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#alist2mat" title="Permalink to this headline"></a>

`sionna.fec.utils.``alist2mat`(<em class="sig-param">`alist`</em>, <em class="sig-param">`verbose``=``True`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#alist2mat">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.alist2mat" title="Permalink to this definition"></a>
    
Convert <cite>alist</cite> <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#mackay" id="id3">[MacKay]</a> code definition to <cite>full</cite> parity-check matrix.
    
Many code examples can be found in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#unikl" id="id4">[UniKL]</a>.
    
About <cite>alist</cite> (see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#mackay" id="id5">[MacKay]</a> for details):
<blockquote>
<div> 
- <cite>1.</cite> Row defines parity-check matrix dimension <cite>m x n</cite>
- <cite>2.</cite> Row defines int with <cite>max_CN_degree</cite>, <cite>max_VN_degree</cite>
- <cite>3.</cite> Row defines VN degree of all <cite>n</cite> column
- <cite>4.</cite> Row defines CN degree of all <cite>m</cite> rows
- Next <cite>n</cite> rows contain non-zero entries of each column (can be zero padded at the end)
- Next <cite>m</cite> rows contain non-zero entries of each row.

</blockquote>
Input
 
- **alist** (<em>list</em>) – Nested list in <cite>alist</cite>-format <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#mackay" id="id6">[MacKay]</a>.
- **verbose** (<em>bool</em>) – Defaults to True. If True, the code parameters are printed.


Output
 
- **(pcm, k, n, coderate)** – Tuple:
- **pcm** (<em>ndarray</em>) – NumPy array of shape <cite>[n-k, n]</cite> containing the parity-check matrix.
- **k** (<em>int</em>) – Number of information bits.
- **n** (<em>int</em>) – Number of codewords bits.
- **coderate** (<em>float</em>) – Coderate of the code.




**Note**
    
Use <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.load_alist" title="sionna.fec.utils.load_alist">`load_alist`</a> to import alist from a
textfile.
    
For example, the following code snippet will import an alist from a file called `filename`:
```python
al = load_alist(path = filename)
pcm, k, n, coderate = alist2mat(al)
```


### load_alist<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#load-alist" title="Permalink to this headline"></a>

`sionna.fec.utils.``load_alist`(<em class="sig-param">`path`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#load_alist">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.load_alist" title="Permalink to this definition"></a>
    
Read <cite>alist</cite>-file <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#mackay" id="id7">[MacKay]</a> and return nested list describing the
parity-check matrix of a code.
    
Many code examples can be found in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#unikl" id="id8">[UniKL]</a>.
Input
    
**path** (<em>str</em>) – Path to file to be loaded.

Output
    
**alist** (<em>list</em>) – A nested list containing the imported alist data.



### generate_reg_ldpc<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#generate-reg-ldpc" title="Permalink to this headline"></a>

`sionna.fec.utils.``generate_reg_ldpc`(<em class="sig-param">`v`</em>, <em class="sig-param">`c`</em>, <em class="sig-param">`n`</em>, <em class="sig-param">`allow_flex_len``=``True`</em>, <em class="sig-param">`verbose``=``True`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#generate_reg_ldpc">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.generate_reg_ldpc" title="Permalink to this definition"></a>
    
Generate random regular (v,c) LDPC codes.
    
This functions generates a random LDPC parity-check matrix of length `n`
where each variable node (VN) has degree `v` and each check node (CN) has
degree `c`. Please note that the LDPC code is not optimized to avoid
short cycles and the resulting codes may show a non-negligible error-floor.
For encoding, the `LinearEncoder` layer can be
used, however, the construction does not guarantee that the pcm has full
rank.
Input
 
- **v** (<em>int</em>) – Desired variable node (VN) degree.
- **c** (<em>int</em>) – Desired check node (CN) degree.
- **n** (<em>int</em>) – Desired codeword length.
- **allow_flex_len** (<em>bool</em>) – Defaults to True. If True, the resulting codeword length can be
(slightly) increased.
- **verbose** (<em>bool</em>) – Defaults to True. If True, code parameters are printed.


Output
 
- **(pcm, k, n, coderate)** – Tuple:
- **pcm** (<em>ndarray</em>) – NumPy array of shape <cite>[n-k, n]</cite> containing the parity-check matrix.
- **k** (<em>int</em>) – Number of information bits per codeword.
- **n** (<em>int</em>) – Number of codewords bits.
- **coderate** (<em>float</em>) – Coderate of the code.




**Note**
    
This algorithm works only for regular node degrees. For state-of-the-art
bit-error-rate performance, usually one needs to optimize irregular degree
profiles (see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#tenbrink" id="id9">[tenBrink]</a>).

### make_systematic<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#make-systematic" title="Permalink to this headline"></a>

`sionna.fec.utils.``make_systematic`(<em class="sig-param">`mat`</em>, <em class="sig-param">`is_pcm``=``False`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#make_systematic">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.make_systematic" title="Permalink to this definition"></a>
    
Bring binary matrix in its systematic form.
Input
 
- **mat** (<em>ndarray</em>) – Binary matrix to be transformed to systematic form of shape <cite>[k, n]</cite>.
- **is_pcm** (<em>bool</em>) – Defaults to False. If true, `mat` is interpreted as parity-check
matrix and, thus, the last k columns will be the identity part.


Output
 
- **mat_sys** (<em>ndarray</em>) – Binary matrix in systematic form, i.e., the first <cite>k</cite> columns equal the
identity matrix (or last <cite>k</cite> if `is_pcm` is True).
- **column_swaps** (<em>list of int tuples</em>) – A list of integer tuples that describes the swapped columns (in the
order of execution).




**Note**
    
This algorithm (potentially) swaps columns of the input matrix. Thus, the
resulting systematic matrix (potentially) relates to a permuted version of
the code, this is defined by the returned list `column_swap`.
Note that, the inverse permutation must be applied in the inverse list
order (in case specific columns are swapped multiple times).
    
If a parity-check matrix is passed as input (i.e., `is_pcm` is True), the
identity part will be re-arranged to the last columns.

### gm2pcm<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#gm2pcm" title="Permalink to this headline"></a>

`sionna.fec.utils.``gm2pcm`(<em class="sig-param">`gm`</em>, <em class="sig-param">`verify_results``=``True`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#gm2pcm">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.gm2pcm" title="Permalink to this definition"></a>
    
Generate the parity-check matrix for a given generator matrix.
    
This function brings `gm` $\mathbf{G}$ in its systematic form and
uses the following relation to find the parity-check matrix
$\mathbf{H}$ in GF(2)

$$
\mathbf{G} = [\mathbf{I} |  \mathbf{M}]
\Leftrightarrow \mathbf{H} = [\mathbf{M} ^t | \mathbf{I}]. \tag{1}
$$
    
This follows from the fact that for an all-zero syndrome, it must hold that

$$
\mathbf{H} \mathbf{c}^t = \mathbf{H} * (\mathbf{u} * \mathbf{G})^t =
\mathbf{H} * \mathbf{G} ^t * \mathbf{u}^t =: \mathbf{0}
$$
    
where $\mathbf{c}$ denotes an arbitrary codeword and
$\mathbf{u}$ the corresponding information bits.
    
This leads to

$$
\mathbf{G} * \mathbf{H} ^t =: \mathbf{0}. \tag{2}
$$
    
It can be seen that (1) fulfills (2), as it holds in GF(2) that

$$
[\mathbf{I} |  \mathbf{M}] * [\mathbf{M} ^t | \mathbf{I}]^t
 = \mathbf{M} + \mathbf{M} = \mathbf{0}.
$$

Input
 
- **gm** (<em>ndarray</em>) – Binary generator matrix of shape <cite>[k, n]</cite>.
- **verify_results** (<em>bool</em>) – Defaults to True. If True, it is verified that the generated
parity-check matrix is orthogonal to the generator matrix in GF(2).


Output
    
<em>ndarray</em> – Binary parity-check matrix of shape <cite>[n-k, n]</cite>.



**Note**
    
This algorithm only works if `gm` has full rank. Otherwise an error is
raised.

### pcm2gm<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#pcm2gm" title="Permalink to this headline"></a>

`sionna.fec.utils.``pcm2gm`(<em class="sig-param">`pcm`</em>, <em class="sig-param">`verify_results``=``True`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#pcm2gm">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.pcm2gm" title="Permalink to this definition"></a>
    
Generate the generator matrix for a given parity-check matrix.
    
This function brings `pcm` $\mathbf{H}$ in its systematic form and
uses the following relation to find the generator matrix
$\mathbf{G}$ in GF(2)

$$
\mathbf{G} = [\mathbf{I} |  \mathbf{M}]
\Leftrightarrow \mathbf{H} = [\mathbf{M} ^t | \mathbf{I}]. \tag{1}
$$
    
This follows from the fact that for an all-zero syndrome, it must hold that

$$
\mathbf{H} \mathbf{c}^t = \mathbf{H} * (\mathbf{u} * \mathbf{G})^t =
\mathbf{H} * \mathbf{G} ^t * \mathbf{u}^t =: \mathbf{0}
$$
    
where $\mathbf{c}$ denotes an arbitrary codeword and
$\mathbf{u}$ the corresponding information bits.
    
This leads to

$$
\mathbf{G} * \mathbf{H} ^t =: \mathbf{0}. \tag{2}
$$
    
It can be seen that (1) fulfills (2) as in GF(2) it holds that

$$
[\mathbf{I} |  \mathbf{M}] * [\mathbf{M} ^t | \mathbf{I}]^t
 = \mathbf{M} + \mathbf{M} = \mathbf{0}.
$$

Input
 
- **pcm** (<em>ndarray</em>) – Binary parity-check matrix of shape <cite>[n-k, n]</cite>.
- **verify_results** (<em>bool</em>) – Defaults to True. If True, it is verified that the generated
generator matrix is orthogonal to the parity-check matrix in GF(2).


Output
    
<em>ndarray</em> – Binary generator matrix of shape <cite>[k, n]</cite>.



**Note**
    
This algorithm only works if `pcm` has full rank. Otherwise an error is
raised.

### verify_gm_pcm<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#verify-gm-pcm" title="Permalink to this headline"></a>

`sionna.fec.utils.``verify_gm_pcm`(<em class="sig-param">`gm`</em>, <em class="sig-param">`pcm`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#verify_gm_pcm">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.verify_gm_pcm" title="Permalink to this definition"></a>
    
Verify that generator matrix $\mathbf{G}$ `gm` and parity-check
matrix $\mathbf{H}$ `pcm` are orthogonal in GF(2).
    
For an all-zero syndrome, it must hold that

$$
\mathbf{H} \mathbf{c}^t = \mathbf{H} * (\mathbf{u} * \mathbf{G})^t =
\mathbf{H} * \mathbf{G} ^t * \mathbf{u}^t =: \mathbf{0}
$$
    
where $\mathbf{c}$ denotes an arbitrary codeword and
$\mathbf{u}$ the corresponding information bits.
    
As $\mathbf{u}$ can be arbitrary it follows that

$$
\mathbf{H} * \mathbf{G} ^t =: \mathbf{0}.
$$

Input
 
- **gm** (<em>ndarray</em>) – Binary generator matrix of shape <cite>[k, n]</cite>.
- **pcm** (<em>ndarray</em>) – Binary parity-check matrix of shape <cite>[n-k, n]</cite>.


Output
    
<em>bool</em> – True if `gm` and `pcm` define a valid pair of parity-check and
generator matrices in GF(2).



## EXIT Analysis<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#exit-analysis" title="Permalink to this headline"></a>
    
The LDPC BP decoder allows to track the internal information flow (<cite>extrinsic information</cite>) during decoding. This can be plotted in so-called EXIT Charts <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#tenbrinkexit" id="id10">[tenBrinkEXIT]</a> to visualize the decoding convergence.
<img alt="../_images/exit.png" src="https://nvlabs.github.io/sionna/_images/exit.png" />
    
This short code snippet shows how to generate and plot EXIT charts:
```python
# parameters
ebno_db = 2.5 # simulation SNR
batch_size = 10000
num_bits_per_symbol = 2 # QPSK
pcm_id = 4 # decide which parity check matrix should be used (0-2: BCH; 3: (3,6)-instruction_answer 4: instruction_answer 802.11n
pcm, k, n , coderate = load_parity_check_examples(pcm_id, verbose=True)
noise_var = ebnodb2no(ebno_db=ebno_db,
                      num_bits_per_symbol=num_bits_per_symbol,
                      coderate=coderate)
# init components
decoder = LDPCBPDecoder(pcm,
                        hard_out=False,
                        cn_type="boxplus",
                        trainable=False,
                        track_exit=True, # if activated, the decoder stores the outgoing extrinsic mutual information per iteration
                        num_iter=20)
# generates fake llrs as if the all-zero codeword was transmitted over an AWNG channel with BPSK modulation
llr_source = GaussianPriorSource()

# generate fake LLRs (Gaussian approximation)
llr = llr_source([[batch_size, n], noise_var])
# simulate free running decoder (for EXIT trajectory)
decoder(llr)
# calculate analytical EXIT characteristics
# Hint: these curves assume asymptotic code length, i.e., may become inaccurate in the short length regime
Ia, Iev, Iec = get_exit_analytic(pcm, ebno_db)
# and plot the analytical exit curves
plt = plot_exit_chart(Ia, Iev, Iec)
# and add simulated trajectory (requires "track_exit=True")
plot_trajectory(plt, decoder.ie_v, decoder.ie_c, ebno_db)
```

    
Remark: for rate-matched 5G LDPC codes, the EXIT approximation becomes
inaccurate due to the rate-matching and the very specific structure of the
code.

### plot_exit_chart<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#plot-exit-chart" title="Permalink to this headline"></a>

`sionna.fec.utils.``plot_exit_chart`(<em class="sig-param">`mi_a``=``None`</em>, <em class="sig-param">`mi_ev``=``None`</em>, <em class="sig-param">`mi_ec``=``None`</em>, <em class="sig-param">`title``=``'EXIT-Chart'`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#plot_exit_chart">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.plot_exit_chart" title="Permalink to this definition"></a>
    
Utility function to plot EXIT-Charts <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#tenbrinkexit" id="id11">[tenBrinkEXIT]</a>.
    
If all inputs are <cite>None</cite> an empty EXIT chart is generated. Otherwise,
the mutual information curves are plotted.
Input
 
- **mi_a** (<em>float</em>) – An ndarray of floats containing the a priori mutual
information.
- **mi_v** (<em>float</em>) – An ndarray of floats containing the variable node mutual
information.
- **mi_c** (<em>float</em>) – An ndarray of floats containing the check node mutual
information.
- **title** (<em>str</em>) – A string defining the title of the EXIT chart.


Output
    
**plt** (<em>matplotlib.figure</em>) – A matplotlib figure handle

Raises
    
**AssertionError** – If `title` is not <cite>str</cite>.



### get_exit_analytic<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#get-exit-analytic" title="Permalink to this headline"></a>

`sionna.fec.utils.``get_exit_analytic`(<em class="sig-param">`pcm`</em>, <em class="sig-param">`ebno_db`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#get_exit_analytic">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.get_exit_analytic" title="Permalink to this definition"></a>
    
Calculate the analytic EXIT-curves for a given parity-check matrix.
    
This function extracts the degree profile from `pcm` and calculates the
variable (VN) and check node (CN) decoder EXIT curves. Please note that
this is an asymptotic tool which needs a certain codeword length for
accurate predictions.
    
Transmission over an AWGN channel with BPSK modulation and SNR `ebno_db`
is assumed. The detailed equations can be found in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#tenbrink" id="id12">[tenBrink]</a> and
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#tenbrinkexit" id="id13">[tenBrinkEXIT]</a>.
Input
 
- **pcm** (<em>ndarray</em>) – The parity-check matrix.
- **ebno_db** (<em>float</em>) – The channel SNR in dB.


Output
 
- **mi_a** (<em>ndarray of floats</em>) – NumPy array containing the <cite>a priori</cite> mutual information.
- **mi_ev** (<em>ndarray of floats</em>) – NumPy array containing the extrinsic mutual information of the
variable node decoder for the corresponding `mi_a`.
- **mi_ec** (<em>ndarray of floats</em>) – NumPy array containing the extrinsic mutual information of the check
node decoder for the corresponding `mi_a`.




**Note**
    
This function assumes random parity-check matrices without any imposed
structure. Thus, explicit code construction algorithms may lead
to inaccurate EXIT predictions. Further, this function is based
on asymptotic properties of the code, i.e., only works well for large
parity-check matrices. For details see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#tenbrink" id="id14">[tenBrink]</a>.

### plot_trajectory<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#plot-trajectory" title="Permalink to this headline"></a>

`sionna.fec.utils.``plot_trajectory`(<em class="sig-param">`plot`</em>, <em class="sig-param">`mi_v`</em>, <em class="sig-param">`mi_c`</em>, <em class="sig-param">`ebno``=``None`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#plot_trajectory">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.plot_trajectory" title="Permalink to this definition"></a>
    
Utility function to plot the trajectory of an EXIT-chart.
Input
 
- **plot** (<em>matplotlib.figure</em>) – A handle to a matplotlib figure.
- **mi_v** (<em>float</em>) – An ndarray of floats containing the variable node mutual
information.
- **mi_c** (<em>float</em>) – An ndarray of floats containing the check node mutual
information.
- **ebno** (<em>float</em>) – A float denoting the EbNo in dB for the legend entry.




## Miscellaneous<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#miscellaneous" title="Permalink to this headline"></a>

### GaussianPriorSource<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#gaussianpriorsource" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.utils.``GaussianPriorSource`(<em class="sig-param">`specified_by_mi``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#GaussianPriorSource">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.GaussianPriorSource" title="Permalink to this definition"></a>
    
Generates <cite>fake</cite> LLRs as if the all-zero codeword was transmitted
over an Bi-AWGN channel with noise variance `no` or mutual information
(if `specified_by_mi` is True). If selected, the mutual information
denotes the mutual information associated with a binary random variable
observed at the output of a corresponding AWGN channel (cf. Gaussian
approximation).
<img alt="../_images/GaussianPriorSource.png" src="https://nvlabs.github.io/sionna/_images/GaussianPriorSource.png" />
    
The generated LLRs are drawn from a Gaussian distribution with

$$
\sigma_{\text{llr}}^2 = \frac{4}{\sigma_\text{ch}^2}
$$
    
and

$$
\mu_{\text{llr}} = \frac{\sigma_\text{llr}^2}{2}
$$
    
where $\sigma_\text{ch}^2$ is the channel noise variance as defined by
`no`.
    
If `specified_by_mi` is True, this class uses the of the so-called
<cite>J-function</cite> (relates mutual information to Gaussian distributed LLRs) as
proposed in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#brannstrom" id="id15">[Brannstrom]</a>.
Parameters
 
- **specified_by_mi** (<em>bool</em>) – Defaults to False. If True, the second input parameter `no` is
interpreted as mutual information instead of noise variance.
- **dtype** (<em>tf.DType</em>) – Defaults to <cite>tf.float32</cite>. Defines the datatype for internal
calculations and the output. Must be one of the following
<cite>(tf.float16, tf.bfloat16, tf.float32, tf.float64)</cite>.


Input
 
- **(output_shape, no)** – Tuple:
- **output_shape** (<em>tf.int</em>) – Integer tensor or Python array defining the shape of the desired
output tensor.
- **no** (<em>tf.float32</em>) – Scalar defining the noise variance or mutual information (if
`specified_by_mi` is True) of the corresponding (fake) AWGN
channel.


Output
    
`dtype`, defaults to <cite>tf.float32</cite> – 1+D Tensor with shape as defined by `output_shape`.

Raises
 
- **InvalidArgumentError** – If mutual information is not in (0,1).
- **AssertionError** – If `inputs` is not a list with 2 elements.




### bin2int<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#bin2int" title="Permalink to this headline"></a>

`sionna.fec.utils.``bin2int`(<em class="sig-param">`arr`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#bin2int">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.bin2int" title="Permalink to this definition"></a>
    
Convert binary array to integer.
    
For example `arr` = <cite>[1, 0, 1]</cite> is converted to <cite>5</cite>.
Input
    
**arr** (<em>int or float</em>) – An iterable that yields 0’s and 1’s.

Output
    
<em>int</em> – Integer representation of `arr`.



### int2bin<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#int2bin" title="Permalink to this headline"></a>

`sionna.fec.utils.``int2bin`(<em class="sig-param">`num`</em>, <em class="sig-param">`len_`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#int2bin">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.int2bin" title="Permalink to this definition"></a>
    
Convert `num` of int type to list of length `len_` with 0’s and 1’s.
`num` and `len_` have to non-negative.
    
For e.g., `num` = <cite>5</cite>; <cite>int2bin(num</cite>, `len_` =4) = <cite>[0, 1, 0, 1]</cite>.
    
For e.g., `num` = <cite>12</cite>; <cite>int2bin(num</cite>, `len_` =3) = <cite>[1, 0, 0]</cite>.
Input
 
- **num** (<em>int</em>) – An integer to be converted into binary representation.
- **len_** (<em>int</em>) – An integer defining the length of the desired output.


Output
    
<em>list of int</em> – Binary representation of `num` of length `len_`.



### bin2int_tf<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#bin2int-tf" title="Permalink to this headline"></a>

`sionna.fec.utils.``bin2int_tf`(<em class="sig-param">`arr`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#bin2int_tf">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.bin2int_tf" title="Permalink to this definition"></a>
    
Converts binary tensor to int tensor. Binary representation in `arr`
is across the last dimension from most significant to least significant.
    
For example `arr` = <cite>[0, 1, 1]</cite> is converted to <cite>3</cite>.
Input
    
**arr** (<em>int or float</em>) – Tensor of  0’s and 1’s.

Output
    
<em>int</em> – Tensor containing the integer representation of `arr`.



### int2bin_tf<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#int2bin-tf" title="Permalink to this headline"></a>

`sionna.fec.utils.``int2bin_tf`(<em class="sig-param">`ints`</em>, <em class="sig-param">`len_`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#int2bin_tf">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.int2bin_tf" title="Permalink to this definition"></a>
    
Converts (int) tensor to (int) tensor with 0’s and 1’s. <cite>len_</cite> should be
to non-negative. Additional dimension of size <cite>len_</cite> is inserted at end.
Input
 
- **ints** (<em>int</em>) – Tensor of arbitrary shape <cite>[…,k]</cite> containing integer to be
converted into binary representation.
- **len_** (<em>int</em>) – An integer defining the length of the desired output.


Output
    
<em>int</em> – Tensor of same shape as `ints` except dimension of length
`len_` is added at the end <cite>[…,k, len_]</cite>. Contains the binary
representation of `ints` of length `len_`.



### int_mod_2<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#int-mod-2" title="Permalink to this headline"></a>

`sionna.fec.utils.``int_mod_2`(<em class="sig-param">`x`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#int_mod_2">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.int_mod_2" title="Permalink to this definition"></a>
    
Efficient implementation of modulo 2 operation for integer inputs.
    
This function assumes integer inputs or implicitly casts to int.
    
Remark: the function <cite>tf.math.mod(x, 2)</cite> is placed on the CPU and, thus,
causes unnecessary memory copies.
Parameters
    
**x** (<em>tf.Tensor</em>) – Tensor to which the modulo 2 operation is applied.



### llr2mi<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#llr2mi" title="Permalink to this headline"></a>

`sionna.fec.utils.``llr2mi`(<em class="sig-param">`llr`</em>, <em class="sig-param">`s``=``None`</em>, <em class="sig-param">`reduce_dims``=``True`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#llr2mi">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.llr2mi" title="Permalink to this definition"></a>
    
Implements an approximation of the mutual information based on LLRs.
    
The function approximates the mutual information for given `llr` as
derived in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#hagenauer" id="id16">[Hagenauer]</a> assuming an <cite>all-zero codeword</cite> transmission

$$
I \approx 1 - \sum \operatorname{log_2} \left( 1 + \operatorname{e}^{-\text{llr}} \right).
$$
    
This approximation assumes that the following <cite>symmetry condition</cite> is fulfilled

$$
p(\text{llr}|x=0) = p(\text{llr}|x=1) \cdot \operatorname{exp}(\text{llr}).
$$
    
For <cite>non-all-zero</cite> codeword transmissions, this methods requires knowledge
about the signs of the original bit sequence `s` and flips the signs
correspondingly (as if the all-zero codeword was transmitted).
    
Please note that we define LLRs as $\frac{p(x=1)}{p(x=0)}$, i.e.,
the sign of the LLRs differ to the solution in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#hagenauer" id="id17">[Hagenauer]</a>.
Input
 
- **llr** (<em>tf.float32</em>) – Tensor of arbitrary shape containing LLR-values.
- **s** (<em>None or tf.float32</em>) – Tensor of same shape as llr containing the signs of the
transmitted sequence (assuming BPSK), i.e., +/-1 values.
- **reduce_dims** (<em>bool</em>) – Defaults to True. If True, all dimensions are
reduced and the return is a scalar. Otherwise, <cite>reduce_mean</cite> is
only taken over the last dimension.


Output
    
**mi** (<em>tf.float32</em>) – A scalar tensor (if `reduce_dims` is True) or a tensor of same
shape as `llr` apart from the last dimensions that is removed.
It contains the approximated value of the mutual information.

Raises
    
**TypeError** – If dtype of `llr` is not a real-valued float.



### j_fun<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#j-fun" title="Permalink to this headline"></a>

`sionna.fec.utils.``j_fun`(<em class="sig-param">`mu`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#j_fun">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.j_fun" title="Permalink to this definition"></a>
    
Calculates the <cite>J-function</cite> in NumPy.
    
The so-called <cite>J-function</cite> relates mutual information to the mean of
Gaussian distributed LLRs (cf. Gaussian approximation). We use the
approximation as proposed in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#brannstrom" id="id18">[Brannstrom]</a> which can be written as

$$
J(\mu) \approx \left( 1- 2^{H_\text{1}(2\mu)^{H_\text{2}}}\right)^{H_\text{2}}
$$
    
with $\mu$ denoting the mean value of the LLR distribution and
$H_\text{1}=0.3073$, $H_\text{2}=0.8935$ and
$H_\text{3}=1.1064$.
Input
    
**mu** (<em>float</em>) – float or <cite>ndarray</cite> of float.

Output
    
<em>float</em> – <cite>ndarray</cite> of same shape as the input.



### j_fun_inv<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#j-fun-inv" title="Permalink to this headline"></a>

`sionna.fec.utils.``j_fun_inv`(<em class="sig-param">`mi`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#j_fun_inv">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.j_fun_inv" title="Permalink to this definition"></a>
    
Calculates the inverse <cite>J-function</cite> in NumPy.
    
The so-called <cite>J-function</cite> relates mutual information to the mean of
Gaussian distributed LLRs (cf. Gaussian approximation). We use the
approximation as proposed in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#brannstrom" id="id19">[Brannstrom]</a> which can be written as

$$
J(\mu) \approx \left( 1- 2^{H_\text{1}(2\mu)^{H_\text{2}}}\right)^{H_\text{2}}
$$
    
with $\mu$ denoting the mean value of the LLR distribution and
$H_\text{1}=0.3073$, $H_\text{2}=0.8935$ and
$H_\text{3}=1.1064$.
Input
    
**mi** (<em>float</em>) – float or <cite>ndarray</cite> of float.

Output
    
<em>float</em> – <cite>ndarray</cite> of same shape as the input.

Raises
    
**AssertionError** – If `mi` < 0.001 or `mi` > 0.999.



### j_fun_tf<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#j-fun-tf" title="Permalink to this headline"></a>

`sionna.fec.utils.``j_fun_tf`(<em class="sig-param">`mu`</em>, <em class="sig-param">`verify_inputs``=``True`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#j_fun_tf">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.j_fun_tf" title="Permalink to this definition"></a>
    
Calculates the <cite>J-function</cite> in Tensorflow.
    
The so-called <cite>J-function</cite> relates mutual information to the mean of
Gaussian distributed LLRs (cf. Gaussian approximation). We use the
approximation as proposed in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#brannstrom" id="id20">[Brannstrom]</a> which can be written as

$$
J(\mu) \approx \left( 1- 2^{H_\text{1}(2\mu)^{H_\text{2}}}\right)^{H_\text{2}}
$$
    
with $\mu$ denoting the mean value of the LLR distribution and
$H_\text{1}=0.3073$, $H_\text{2}=0.8935$ and
$H_\text{3}=1.1064$.
Input
 
- **mu** (<em>tf.float32</em>) – Tensor of arbitrary shape.
- **verify_inputs** (<em>bool</em>) – A boolean defaults to True. If True, `mu` is clipped internally
to be numerical stable.


Output
    
<em>tf.float32</em> – Tensor of same shape and dtype as `mu`.

Raises
    
**InvalidArgumentError** – If `mu` is negative.



### j_fun_inv_tf<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#j-fun-inv-tf" title="Permalink to this headline"></a>

`sionna.fec.utils.``j_fun_inv_tf`(<em class="sig-param">`mi`</em>, <em class="sig-param">`verify_inputs``=``True`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#j_fun_inv_tf">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.j_fun_inv_tf" title="Permalink to this definition"></a>
    
Calculates the inverse <cite>J-function</cite> in Tensorflow.
    
The so-called <cite>J-function</cite> relates mutual information to the mean of
Gaussian distributed LLRs (cf. Gaussian approximation). We use the
approximation as proposed in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#brannstrom" id="id21">[Brannstrom]</a> which can be written as

$$
J(\mu) \approx \left( 1- 2^{H_\text{1}(2\mu)^{H_\text{2}}}\right)^{H_\text{2}}
$$
    
with $\mu$ denoting the mean value of the LLR distribution and
$H_\text{1}=0.3073$, $H_\text{2}=0.8935$ and
$H_\text{3}=1.1064$.
Input
 
- **mi** (<em>tf.float32</em>) – Tensor of arbitrary shape.
- **verify_inputs** (<em>bool</em>) – A boolean defaults to True. If True, `mi` is clipped internally
to be numerical stable.


Output
    
<em>tf.float32</em> – Tensor of same shape and dtype as the `mi`.

Raises
    
**InvalidArgumentError** – If `mi` is not in <cite>(0,1)</cite>.




References:
tenBrinkEXIT(<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id10">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id11">2</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id13">3</a>)
    
S. ten Brink, “Convergence Behavior of Iteratively
Decoded Parallel Concatenated Codes,” IEEE Transactions on
Communications, vol. 49, no. 10, pp. 1727-1737, 2001.

Brannstrom(<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id15">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id18">2</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id19">3</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id20">4</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id21">5</a>)
    
F. Brannstrom, L. K. Rasmussen, and A. J. Grant,
“Convergence analysis and optimal scheduling for multiple
concatenated codes,” IEEE Trans. Inform. Theory, vol. 51, no. 9,
pp. 3354–3364, 2005.

Hagenauer(<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id16">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id17">2</a>)
    
J. Hagenauer, “The Turbo Principle in Mobile
Communications,” in Proc. IEEE Int. Symp. Inf. Theory and its Appl.
(ISITA), 2002.

tenBrink(<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id9">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id12">2</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id14">3</a>)
    
S. ten Brink, G. Kramer, and A. Ashikhmin, “Design of
low-density parity-check codes for modulation and detection,” IEEE
Trans. Commun., vol. 52, no. 4, pp. 670–678, Apr. 2004.

MacKay(<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id1">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id3">2</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id5">3</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id6">4</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id7">5</a>)
    
<a class="reference external" href="http://www.inference.org.uk/mackay/codes/alist.html">http://www.inference.org.uk/mackay/codes/alist.html</a>

UniKL(<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id2">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id4">2</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id8">3</a>)
    
<a class="reference external" href="https://www.uni-kl.de/en/channel-codes/">https://www.uni-kl.de/en/channel-codes/</a>



