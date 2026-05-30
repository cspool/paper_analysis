# <span id="page-14-0"></span>C.1 Computational Artifact <sup>1</sup> Artifact Setup (incl. Inputs)

All source codes and artifacts can be found at the Zenodo repository[1](#page-14-1) . To install HyTiS, unpack the artifact from Zenodo, then execute the script src/install.sh. The detail is listed as follows. proj\_root='pwd'

```
pip install torch==2.3.1 torchvision==0.18.1
–index-url https://download.pytorch.org/whl/cu121
  # install dependency library: triton
  cd ${proj_root}/third_party/
  git clone https://github.com/triton-lang/triton.git
  cd triton
  git reset –hard 52cf1aee47f806585fcb1a88f5b24880ab6f6257
  git apply ../../patchs/triton-patchs/0001-build.patch
  git apply ../../patchs/triton-patchs/0002-hytis.patch
  cd python; pip install -e .
  # install streamk, splitk
  cd ${proj_root}/third_party
  git clone https://github.com/nvidia/cutlass.git; cd
cutlass
  git checkout -b v341 v3.4.1
  cd ${proj_root}/third_party/streamk_cutlass
  TORCH_CUDA_ARCH_LIST=9.0a CUTLASS_DIR='pwd'/../cutlass/
pip install -e . # h100
  TORCH_CUDA_ARCH_LIST=8.0 CUTLASS_DIR='pwd'/../cutlass/
pip install -e . # a100
  #install HyTiS
  cd ${proj_root}; pip install -e .
```

