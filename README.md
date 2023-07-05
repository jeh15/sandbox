# Sandbox
A repo that I use to try out things I find interesting.

## Current Interests:
* [JAX](https://github.com/google/jax)
* [Flax](https://github.com/google/flax)
* [Brax](https://github.com/google/brax)
* [Theseus](https://github.com/facebookresearch/theseus)


# How to run:
Some libraries that use JAX will uninstall CUDA JAX (If installed via pip not sure about locally installed) and instead use CPU JAX.
As a result pip install order matters...

To make this easy, setup.sh provides an interactive CLI to install most dependencies.

*Note: some of the wheels are only available on linux.*

## CLI Setup Method:
The CLI method depends on Charm's `gum` library (`setup.sh` installs `gum` to your `/tmp` directory).


```
git clone https://github.com/jeh15/sandbox
cd sandbox
chmod +x setup.sh
./setup.sh
```

