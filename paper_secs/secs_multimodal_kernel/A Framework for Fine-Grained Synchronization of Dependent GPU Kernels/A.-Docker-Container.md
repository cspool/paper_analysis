# *A. Docker Container*

To run artifact inside a Docker container follow these steps:

Install docker Install docker engine by following steps on [https://docs.docker.com/engine/install/ubuntu/.](https://docs.docker.com/engine/install/ubuntu/)

Install NVIDIA Container Toolkit Install NVIDIA Container Toolkit by following steps on [https:](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) [//docs.nvidia.com/datacenter/cloud-native/container-toolkit/](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) [latest/install-guide.html.](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Create Container Create docker container using the Dockerfile, start the container, and cd to the directory:

```
docker build -t cusync-cgo-24 .
docker run -it --gpus all cusync-cgo-24
cd /cusync
```

Check PyTorch and CUDA Install: Check if torch supports CUDA if torch.cuda.is\_available() returns True:

```
python
>>> import torch
>>> torch.cuda.is_available()
True
```

## *B. Running Natively*

We can also run code natively, which requires installing all dependencies. These steps can be ignored if using docker in above steps.

Linux Installation We recommend using Ubuntu 22.04 as the Linux OS. We have not tested our artifact with any other OS but we believe Ubuntu 20.04 and 23.04 should also work.

Install Dependencies Execute following commands to install dependencies.

```
sudo apt update
sudo apt install gcc linux-headers-$(uname -r)\
make g++ git python3 wget\
unzip python3-pip build-essential cmake
```

Install CUDA We need to install CUDA before proceeding further. In our experiments we used CUDA 12.2 on Ubuntu

22.04. CUDA 12.2 toolkit can be downloaded from [https://developer.nvidia.com/cuda-12-1-0-download-archive.](https://developer.nvidia.com/cuda-12-1-0-download-archive) After installing CUDA, set nvcc and CUDA paths.

```
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH=
"/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```

Check CUDA Installation To check CUDA installation, run nvidia-smi and it should print all GPUs in the system. Otherwise there is a problem with the CUDA installation. Install Pytorch: Install PyTorch using pip.

sudo pip3 install torch torchvision torchaudio

Check Pytorch CUDA Install: Check if torch supports CUDA if torch.cuda.is\_available() returns True:

```
python
>>> import torch
>>> torch.cuda.is_available()
True
```

Obtain source code The source code can be downloaded from [\[7\]](#page-11-9). Latest source code is available from cuSync repository and CGO AE branch:

```
git clone --recurse-submodules \
https://github.com/microsoft/cusync
cd cusync
git checkout cgo-24-ae
```

## *C. Functionality and Reusability*

The README.md contains instructions of how code can be compiled to other NVIDIA GPU architectures, an example and test cases. The functionality can be checked by executing these test cases. To run tests execute:

```
make tests -j
```

If all tests passes then we are ready for reproducing results.

## *D. Reproduce Results*

We will now reproduce our main results of Figure [6,](#page-8-0) [7b,](#page-9-0) and [8.](#page-9-1) All commands should be executed in the cusync directory.

Large Language Model Inference Results [Time 60 mins] Following commands will run all experiments to gather the results

```
cd src/ml-bench/volta_transformer
python3 eval_llm.py mlp gpt3
python3 eval_llm.py attention gpt3
python3 eval_llm.py mlp llama
python3 eval_llm.py attention llama
python3 allreduce_times.py
```

Computer Vision Inference Results [Time 60 mins] Following commands will run all experiments to gather results for Figure [7b.](#page-9-0)

```
cd src/ml-bench/volta_conv2d
python3 eval_conv.py resnet
python3 eval_conv.py vgg
```

Generate Plots [Time 5 mins] Generate all Figures by running below commands:

```
cd src/ml-bench/plots
make -j
```

The current directory will have figures as PDF and they can be checked against figures in the paper.

