# A.2.4 Benchmarks. The request workloads and image benchmarks can be found in [§6.1](#page-9-1) and [§6.2.](#page-9-0)

#### A.3 Set-up

```
1 # Pull the Docker image .
2 docker pull jiangxiaoxiao / flashps : latest
3 # Clear the stopped container , if it exists
4 docker kill flashps - ae
5 docker rm flashps - ae
6 # Spin up the container . This may take minutes .
7 docker run -d -- name flashps - ae -- runtime = nvidia -- gpus
        all -- shm - size =16 g -e NVIDIA_VISIBLE_DEVICES =all -e
        CUDA_VISIBLE_DEVICES =0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 -e
        CONDA_DEFAULT_ENV ="" -e CONDA_AUTO_ACTIVATE_BASE =
        false jiangxiaoxiao / flashps sleep infinity
8 # Enter the container
9 docker exec - it flashps - ae zsh
10 # Activate the environment
11 conda activate flashps
```

## A.4 Evaluation workflow

