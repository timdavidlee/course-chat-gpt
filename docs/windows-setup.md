# Getting started with Windows

Why? Probably because your gaming GPU is installed on your other machine


## Open WSL

```
```

## install pyenv

```
curl https://pyenv.run | bash
```

## Ensure that Windows 10 22H2 is installed + updated

## Ensure that WSL + Virualization is turned on in Windows 10

- `Turn Windows Features on and Off`
- `Enable windows subsystem`
- `Virtualization Platform`

## Ensure that virtualization is enabled in the bios

## install ubuntu

```
wsl --install -d Ubuntu-20.04
```

```
wsl -d Ubuntu-20.04
```

## Verfiy that device is compatible

```
#powershell
```

## Download and install CUDA from NVIDIA website

https://developer.nvidia.com/cuda-downloads

## install anaconda

https://castorfou.github.io/guillaume_blog/blog/wsl2-cuda-conda.html


## install anaconda (wihtin WSL)

```sh
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
sh Anaconda3-2022.10-Linux-x86_64.sh
# accept all the licenses
```

```sh
conda install cuda -c nvidia
```

## Check if pytorch has access to cuda:

```python
import torch
torch.cuda.is_available()
device_id = torch.cuda.current_device()
torch.cuda.get_device_name(device_id)
# >>> 'NVIDIA GeForce RTX 3090'
```