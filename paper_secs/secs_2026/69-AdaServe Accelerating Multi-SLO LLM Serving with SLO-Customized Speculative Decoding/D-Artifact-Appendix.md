# D Artifact Appendix

#### D.1 Abstract

This appendix accompanies the paper AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding. It describes the software artifact submitted for evaluation. The artifact enables reviewers to reproduce the key experimental results presented in the paper (Figure 8-Figure 15). While the precise hardware configuration used in the original evaluation is not always available, the artifact is designed to produce results that demonstrate the same qualitative trends and support the claims of the paper.

#### **D.2** Getting Started

**D.2.1** How to Access. The artifact is available on GitHub at https://github.com/zikun-li/AdaServe-Artifact-Evaluation and archived on Zenodo at DOI 10.5281/zenodo.17052619.

D.2.2 Hardware dependencies. The experiments require access to modern GPUs with adequate memory capacity. We recommend using either the provided machine (8 × NVIDIA A100-SXM4-40GB GPUs with at least 512 GB RAM) or an equivalent cloud instance (e.g., AWS p4de.24xlarge). All experiments should be executed on x86-64 machines with CUDA 12.4 and Docker configured with the NVIDIA container runtime. The original paper's experiments were conducted on a platform with 4× A100-SXM4-80GB GPUs, which may not always be available. While absolute throughput values differ across hardware, the reproduced experiments follow the same trends.

**D.2.3 Set-Up.** To set up the evaluation environment, reviewers should first clone the artifact repository using:

```
$ git clone -recursive

    https://github.com/zikun-li \
    /AdaServe-Artifact-Evaluation.git
```

Next, the Docker container must be built by running the following command:

```
$ ./docker/build_container.sh
```

Then, the docker container can be started with the following command:

```
$ ./docker/start container.sh
```

After the container is running, reviewers should install the required dependencies and download the Hugging Face models by executing the following command:

```
$ ./docker/setup_adaserve.sh
```

Please note that the above step requires a valid Hugging-Face token for authentication.

Finally, a new terminal can be attached to the running container with

```
$ ./docker/attach_to_container.sh
```

Multiple concurrent terminals may be opened if desired.

**D.2.4 Teardown.** After completing the evaluation, reviewers are requested to clean up the environment by first terminating and removing all containers and associated Docker data using:

```
$ ./docker/cleanup_containers.sh
```

They should then delete the cloned repository and any generated files. These steps ensure that subsequent reviewers have access to a clean evaluation environment.

#### D.3 Evaluation and Expected Results

The artifact reproduces the paper's main evaluation figures. Each script runs a sequence of experiments covering a range of configurations; one configuration typically requires  $\sim 15$  minutes. Results are stored under the results/ directory. Due to the hardware differences described above, exact numerical values may vary, but qualitative trends are preserved.

D.3.1 Figures 8 and 9 (SLO attainment and goodput vs. RPS). To reproduce the results on the Llama models:

```
$ ADASERVE=ON RPS_MIN=2.6 RPS_MAX=4.8

∴ ./exps/fig8,9/run_llama_rps.sh
```

To reproduce the results on the Qwen models:

```
$ ADASERVE=ON RPS_MIN=2.4 RPS_MAX=4.2
```

The parameters RPS\_MIN and RPS\_MAX can be adjusted to cover different ranges of requests per second (RPS). In our evaluation, the minimal RPS is 2.6 and the maximal RPS is 4.8 for LLaMA-3.1-70B-Instruct, while the minimal RPS is 2.4 and the maximal RPS is 4.2 for Qwen2.5-32B-Instruct. The minimal step size is set to 0.2. The results are stored in the directory results/fig8,9/llama/adaserve/and results/fig8,9/qwen/adaserve/.

**D.3.2** Figure 10 (SLO attainment and goodput vs. urgent request proportion). To reproduce the results on the LLaMA models:

```
$ ADASERVE=ON PROP_MIN=0.1 PROP_MAX=0.9

∴ ./exps/fig10/run_llama_prop.sh
```

To reproduce the results on the Qwen models:

```
$ ADASERVE=ON PROP_MIN=0.1 PROP_MAX=0.9

∴ ./exps/fig10/run_qwen_prop.sh
```

The parameters PROP\_MIN and PROP\_MAX can be adjusted to cover different ranges of urgent request proportions. In our evaluation, the minimal proportion is 0.1 and the maximal proportion is 0.9 for both LLaMA-3.1-70B-Instruct and Qwen2.5-32B-Instruct. The minimal step size is set to 0.1. The results are stored in the directories results/fig10/lama/adaserve/ and results/fig10/qwen/adaserve/.

**D.3.3** Figure 11 (SLO attainment and goodput vs. SLO scale). To reproduce the results on the LLaMA models:

```
$ ADASERVE=ON SLO_SCALE_MIN=0.6

→ SLO_SCALE_MAX=1.6 OUTPUT_LENGTH=256

→ ./exps/fig11/run_llama_slo.sh
```

To reproduce the results on the Qwen models:

```
$ ADASERVE=ON SLO_SCALE_MIN=0.6

→ SLO_SCALE_MAX=1.6 OUTPUT_LENGTH=256

→ ./exps/fig11/run_qwen_slo.sh
```

The parameters SLO\_SCALE\_MIN and SLO\_SCALE\_MAX can be adjusted to cover different ranges of SLO scales. In our evaluation, the minimal SLO scale is 0.6 and the maximal

<span id="page-19-0"></span>SLO scale is 1.6 for both LLaMA-3.1-70B-Instruct and Qwen2.5-32B-Instruct. The minimal step size is set to 0.2. The results are stored in the directories results/fig11/llama/adaserve/ and results/fig11/qwen/adaserve/.

**D.3.4** Figure 12 (Speculative accuracy). The data for Figure 12 is collected during the experiments for Figure 8 and Figure 9 and can be found in their corresponding directories. The reported numbers correspond to the line starting with mean\_generated\_tokens\_per\_step at the end of the files.

**D.3.5 Figure 14 (Sensitivity to workload fluctuations).** To reproduce the results on the LLaMA models:

```
$ ADASERVE=ON ./exps/fig14/run_llama_fluc.sh
```

To reproduce the results on the Qwen models:

```
$ ADASERVE=ON ./exps/fig14/run_qwen_fluc.sh
```

The results are stored in the directories results/fig14/llama/adaserve/ and results/fig14/qwen/adaserve/.

**D.3.6 Figure 15 (Latency breakdown).** To reproduce the results on the LLaMA models:

```
$ LLAMA_OVERHEAD=ON
```

To reproduce the results on the Qwen models:

```
$ QWEN_OVERHEAD=ON
```

The results are stored in the directories results/fig15/llama/ and results/fig15/qwen/.