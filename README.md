# Sandbox
A repo that I use to try out things I find interesting.

## Current Interests:
* [JAX](https://github.com/google/jax)
* [Flax](https://github.com/google/flax)
* [Brax](https://github.com/google/brax)
* [Theseus](https://github.com/facebookresearch/theseus)


# How to run:
Some libraries that use JAX will unistall CUDA JAX (If installed via pip not sure about locally installed) and instead use CPU JAX.
As a result pip install order matters...

To make this easy, setup.sh provides an interactive CLI to install most dependencies.

*Note: some of the wheels are only available on linux.*

## CLI Setup Method:
The CLI method depends on Charm's `gum` library (`setup.sh` installs `gum` to your `/tmp` directory)


```
git clone https://github.com/jeh15/sandbox
cd sandbox
sudo ./setup.sh
```

## Manual Method:
### Create venv:
```
python3 -m venv env
/env/bin/pip install --upgrade pip
source /env/bin/activate
```

### Libraries that need to go before CUDA JAX install:
```
pip install brax
```

(Needed to fork Distrax due to out of date requirements.txt for numpy)
```
pip install git+https://github.com/jeh15/distrax.git
```

### Rest of the dependencies:
```
# Installs the wheel compatible with CUDA 12 and cuDNN 8.8 or newer.
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
OR
```
# Installs the wheel compatible with CUDA 11 and cuDNN 8.6 or newer.
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

```
pip install flax
pip install optax
pip install matplotlib
pip install absl-py
```