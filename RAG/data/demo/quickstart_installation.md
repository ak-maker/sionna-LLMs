
# Installation<a class="headerlink" href="https://nvlabs.github.io/sionna/installation.html#installation" title="Permalink to this headline"></a>
    
Sionna requires <a class="reference external" href="https://www.python.org/">Python</a> and <a class="reference external" href="https://www.tensorflow.org/">Tensorflow</a>.
In order to run the tutorial notebooks on your machine, you also need <a class="reference external" href="https://jupyter.org/">JupyterLab</a>.
You can alternatively test them on <a class="reference external" href="https://colab.research.google.com/github/nvlabs/sionna/blob/main/examples/Discover_Sionna.ipynb">Google Colab</a>.
Although not necessary, we recommend running Sionna in a <a class="reference external" href="https://www.docker.com">Docker container</a>.

**Note**
    
Sionna requires <a class="reference external" href="https://www.tensorflow.org/install">TensorFlow 2.10-2.15</a> and Python 3.8-3.11.
We recommend Ubuntu 22.04.
Earlier versions of TensorFlow may still work but are not recommended because of known, unpatched CVEs.
    
To run the ray tracer on CPU, <a class="reference external" href="https://llvm.org">LLVM</a> is required by DrJit. Please check the <a class="reference external" href="https://drjit.readthedocs.io/en/latest/firststeps-py.html#llvm-backend">installation instructions for the LLVM backend</a>.
The ray tracing preview requires a recent version of <cite>JupyterLab</cite>. You can upgrade to the latest version via `pip` `install` `--upgrade` `ipykernel` `jupyterlab` (requires restart of <cite>JupyterLab</cite>).
    
We refer to the <a class="reference external" href="https://www.tensorflow.org/install/gpu">TensorFlow GPU support tutorial</a> for GPU support and the required driver setup.

## Installation using pip<a class="headerlink" href="https://nvlabs.github.io/sionna/installation.html#installation-using-pip" title="Permalink to this headline"></a>
    
We recommend to do this within a <a class="reference external" href="https://docs.python.org/3/tutorial/venv.html">virtual environment</a>,
e.g., using <a class="reference external" href="https://docs.conda.io">conda</a>. On macOS, you need to install <a class="reference external" href="https://github.com/apple/tensorflow_macos">tensorflow-macos</a> first.
    
1.) Install the package
```python
pip install sionna
```

    
2.) Test the installation in Python
```python
python
```

```python
>>> import sionna
>>> print(sionna.__version__)
0.16.2
```

    
3.) Once Sionna is installed, you can run the <a class="reference external" href="https://nvlabs.github.io/sionna/examples/Hello_World.html">Sionna “Hello, World!” example</a>, have a look at the <a class="reference external" href="https://nvlabs.github.io/sionna/quickstart.html">quick start guide</a>, or at the <a class="reference external" href="https://nvlabs.github.io/sionna/tutorials.html">tutorials</a>.
    
For a local installation, the <a class="reference external" href="https://github.com/jupyterlab/jupyterlab-desktop">JupyterLab Desktop</a> application can be used. This directly includes the Python installation and configuration.

## Docker-based Installation<a class="headerlink" href="https://nvlabs.github.io/sionna/installation.html#docker-based-installation" title="Permalink to this headline"></a>
    
1.) Make sure that you have Docker <a class="reference external" href="https://docs.docker.com/engine/install/ubuntu/">installed</a> on your system. On Ubuntu 22.04, you can run for example
```python
sudo apt install docker.io
```

    
Ensure that your user belongs to the <cite>docker</cite> group (see <a class="reference external" href="https://docs.docker.com/engine/install/linux-postinstall/">Docker post-installation</a>).
```python
sudo usermod -aG docker $USER
```

    
Log out and re-login to load updated group memberships.
    
For GPU support on Linux, you need to install the <a class="reference external" href="https://github.com/NVIDIA/nvidia-docker">NVIDIA Container Toolkit</a>.
    
2.) Build the Sionna Docker image. From within the Sionna directory, run:
```python
make docker
```

    
3.) Run the Docker image with GPU support
```python
make run-docker gpus=all
```

    
or without GPU:
```python
make run-docker
```

    
This will immediately launch a Docker image with Sionna installed, running JupyterLab on port 8888.
    
4.) Browse through the example notebook by connecting to <a class="reference external" href="http://127.0.0.1:8888">http://127.0.0.1:8888</a> in your browser.

## Installation from source<a class="headerlink" href="https://nvlabs.github.io/sionna/installation.html#installation-from-source" title="Permalink to this headline"></a>
    
We recommend to do this within a <a class="reference external" href="https://docs.python.org/3/tutorial/venv.html">virtual environment</a>,
e.g., using <a class="reference external" href="https://docs.conda.io">conda</a>.
    
1.) Clone this repository and execute from within its root folder:
```python
make install
```

    
2.) Test the installation in Python
```python
python
```

```python
>>> import sionna
>>> print(sionna.__version__)
0.16.2
```