#!/bin/bash

sh ./install/install.sh

gum='/tmp/tmp.tmp_gum/gum'

clear

$gum style --border normal --margin "1" --padding "1 2" --border-foreground 212 "Virtual Environment Setup: Sandbox"

echo -e "Would you like to setup a $($gum style --italic --foreground 99 'virtual environment')?\n"

CONTINUE="Continue"; EXIT="Exit"
ACTIONS=$($gum choose --limit=1 "$CONTINUE" "$EXIT")

clear

if [[ "$ACTIONS" == "$CONTINUE" ]]
then
    $gum spin -s line --title "Creating virtual environment..." -- sh -c 'python3 -m venv env; env/bin/pip install --upgrade pip'
else
    $gum spin -s line --title "Exiting..." -- sleep 1; exit
fi

clear

echo -e "What version of $($gum style --italic --foreground 212 'JAX')?\n"

GPU="GPU"; CPU="CPU"
ACTIONS=$($gum choose --limit=1 "$GPU" "$CPU")

clear

if [[ "$ACTIONS" == "$GPU" ]]
then
    # TODO(jeh15): Check if nvidia drivers are installed
    echo -e "What $($gum style --italic --foreground 99 'CUDA') version?\n"
    CUDA11="CUDA 11"; CUDA12="CUDA 12"
    CUDA_CHOICE=$($gum choose --limit=1 "$CUDA11" "$CUDA12")
    
    clear
fi

echo -e "$($gum style --italic --foreground 99 'Installing dependencies...')\n"
$gum spin -s line --title "Installing Brax..." -- sh -c 'env/bin/pip install git+https://github.com/jeh15/brax.git'
echo -e "$($gum style --italic --foreground 99 '    > Installed Brax')\n"
$gum spin -s line --title "Installing Distrax..." -- sh -c 'env/bin/pip install git+https://github.com/deepmind/distrax.git'
echo -e "$($gum style --italic --foreground 99 '    > Installed Distrax')\n"
$gum spin -s line --title "Installing jqt..." -- sh -c 'env/bin/pip install git+https://github.com/jeh15/jax_quaternion.git'
echo -e "$($gum style --italic --foreground 99 '    > Installed jqt')\n"

if [[ "$ACTIONS" == "$GPU" ]]
then
    if [[ "$CUDA_CHOICE" == "$CUDA11" ]]
    then
        $gum spin -s line --title "Installing CUDA 11 JAX..." -- sh -c 'env/bin/pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'
        echo -e "$($gum style --italic --foreground 99 '    > Installed CUDA 11 JAX')\n"
    else
        $gum spin -s line --title "Installing CUDA 12 JAX..." -- sh -c 'env/bin/pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'
        echo -e "$($gum style --italic --foreground 99 '    > Installed CUDA 12 JAX')\n"
    fi
else
    $gum spin -s line --title "Installing CPU JAX..." -- sh -c 'env/bin/pip install --upgrade "jax[cpu]"'
    echo -e "$($gum style --italic --foreground 99 '    > Installed CPU JAX')\n"
fi

$gum spin -s line --title "Installing Flax..." -- sh -c 'env/bin/pip install flax'
echo -e "$($gum style --italic --foreground 99 '    > Installed Flax')\n"
$gum spin -s line --title "Installing Optax..." -- sh -c 'env/bin/pip install optax'
echo -e "$($gum style --italic --foreground 99 '    > Installed Optax')\n"
$gum spin -s line --title "Installing JAXopt..." -- sh -c 'env/bin/pip install jaxopt'
echo -e "$($gum style --italic --foreground 99 '    > Installed JAXopt')\n"
$gum spin -s line --title "Installing Orbax..." -- sh -c 'env/bin/pip install orbax-checkpoint'
echo -e "$($gum style --italic --foreground 99 '    > Installed Orbax')\n"
$gum spin -s line --title "Installing matplotlib..." -- sh -c 'env/bin/pip install matplotlib'
echo -e "$($gum style --italic --foreground 99 '    > Installed matplotlib')\n"
$gum spin -s line --title "Installing tqdm..." -- sh -c 'env/bin/pip install tqdm'
echo -e "$($gum style --italic --foreground 99 '    > Installed tqdm')\n"
$gum spin -s line --title "Installing absl-py..." -- sh -c 'env/bin/pip install absl-py'
echo -e "$($gum style --italic --foreground 99 '    > Installed absl-py')\n"

$gum style --border normal --margin "1" --padding "1 2" --border-foreground 212 "Virtual Environment Setup: Sandbox -- $($gum style --bold --foreground 99 'Complete')"

sleep 3; clear; exit