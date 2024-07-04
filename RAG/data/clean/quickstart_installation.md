# Installation

Sionna requires [Python](https://www.python.org/) and [Tensorflow](https://www.tensorflow.org/).
In order to run the tutorial notebooks on your machine, you also need [JupyterLab](https://jupyter.org/).
You can alternatively test them on [Google Colab](https://colab.research.google.com/github/nvlabs/sionna/blob/main/examples/Discover_Sionna.ipynb).
Although not necessary, we recommend running Sionna in a [Docker container](https://www.docker.com).

**Note**

Sionna requires [TensorFlow 2.10-2.15](https://www.tensorflow.org/install) and Python 3.8-3.11.
We recommend Ubuntu 22.04.
Earlier versions of TensorFlow may still work but are not recommended because of known, unpatched CVEs.

To run the ray tracer on CPU, [LLVM](https://llvm.org) is required by DrJit. Please check the [installation instructions for the LLVM backend](https://drjit.readthedocs.io/en/latest/firststeps-py.html#llvm-backend).
The ray tracing preview requires a recent version of <cite>JupyterLab</cite>. You can upgrade to the latest version via `pip` `install` `--upgrade` `ipykernel` `jupyterlab` (requires restart of <cite>JupyterLab</cite>).

We refer to the [TensorFlow GPU support tutorial](https://www.tensorflow.org/install/gpu) for GPU support and the required driver setup.

## Installation using pip

We recommend to do this within a [virtual environment](https://docs.python.org/3/tutorial/venv.html),
e.g., using [conda](https://docs.conda.io). On macOS, you need to install [tensorflow-macos](https://github.com/apple/tensorflow_macos) first.

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


3.) Once Sionna is installed, you can run the [Sionna Hello, World! example](https://nvlabs.github.io/sionna/examples/Hello_World.html), have a look at the [quick start guide](https://nvlabs.github.io/sionna/quickstart.html), or at the [tutorials](https://nvlabs.github.io/sionna/tutorials.html).

For a local installation, the [JupyterLab Desktop](https://github.com/jupyterlab/jupyterlab-desktop) application can be used. This directly includes the Python installation and configuration.

## Docker-based Installation

1.) Make sure that you have Docker [installed](https://docs.docker.com/engine/install/ubuntu/) on your system. On Ubuntu 22.04, you can run for example
```python
sudo apt install docker.io
```


Ensure that your user belongs to the <cite>docker</cite> group (see [Docker post-installation](https://docs.docker.com/engine/install/linux-postinstall/)).
```python
sudo usermod -aG docker $USER
```


Log out and re-login to load updated group memberships.

For GPU support on Linux, you need to install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker).

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

4.) Browse through the example notebook by connecting to [http://127.0.0.1:8888](http://127.0.0.1:8888) in your browser.

## Installation from source

We recommend to do this within a [virtual environment](https://docs.python.org/3/tutorial/venv.html),
e.g., using [conda](https://docs.conda.io).

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
``````