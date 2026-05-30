# *Relation To Contributions*

The artifact A<sup>1</sup> implements and validates the key concept in the APTMoE system. To display it more clearly, we divide the artifact into three parts:

- The static part. It profiles memory usage and execution time by performing fine-tuning of a single MoE layer on both CPU and GPU. This part is used to guide the layerto-stage mapping and hardware affinity for the runtime part. It mainly relates to C1.
- The runtime part. It implements the basic pipeline parallelism fine-tuning. Upon the pipeline parallelism, it implements the hierarchical loading strategy and the demand-priority scheduling strategy in the affinity-aware offloading technique. It mainly relates to C1, C2.

• The gate simulator part. It is responsible for validating the effectiveness of APTMoE on MoE models with all kinds of configurations. First, it takes the expert popularity generator for simulating generalized MoE models. Second, it acquires expert popularity from real model fine-tuning for real case evaluation. It mainly relates to C3.

For the static part, it locates in APTMoE/Static. An instance of class Profilier performs fine-tuning of a single MoE layer on both CPU and GPU, and generates execution time lookup table associated with affinity.

The runtime part of APTMoE is located in APTMoE /Runtime. APTMoE system is based on a pipeline framework (APTMoE/Runtime/PipelineRuntime /pipeline\_runtime.py), which is responsible for pipeline P2P communication and fine-tuning process. The hierarchical loading strategy is implemented in APTMoE/ Runtime/OffloadRuntime, including the three loading phases(offload.py) and the optimal expert-to-device allocation scheme for given popularity(R\_solver.py). The demand-priority scheduling strategy is located in comm\_scheduler.py. It includes a PriorityQueue that manages three loading queues with respect to different loading phases. We use torch.cuda.Event.query() to check if the correspoding loading event is completed. If it is, then launch the highest priority loading in the current queue. We perform two cuda streams(comp\_stream and load\_stream) to execute computation and loading operations concurrently, with the aim of overlapping computation and communication. Besides, we specify a torch.cuda.Event() to each model block to maintain the inter-stream dependency.

For the C<sup>3</sup> of the gate simulator part, we simulate the expert popularity that satisfies a specified power-law distribution for the generalized MoE models, which is located in APTMoE /model/top2gate.py. Also, we execute the real MoE model fine-tuning and abstract the real expert popularity for the real case study, the code is located in APTMoE/ RealCase.

Besides, we use psutil.Process().cpu\_affinity () to bind different numbers of CPU cores to a specific process, so as to set different device topologies.

#### *Expected Results*

APTMoE aims to improve the model size and fine-tuning efficiency under limited number of bandwidth-constrained GPU nodes. We take the state-of-the-art approach, i.e. Mobius, as the major baseline. First, we hope to validate that APTMoE can fine-tune MoE models with the same size as Mobius and this can be validated once these experiments are successfully executed. Second, we hope to validate that APTMoE has better performance compared to Mobius in most model configurations and most device topologies. This is evaluated in two steps:

- Generalized Case Study: We establish MoE models with varying model sizes to conduct performance evaluation, incorporating a simulator for simulating different expert popularity.
- Real Case Study: We inject the real expert popularity of fine-tuning NLLB-MoE and Mixtral-8x7B models on APP dataset into the simulator.

#### *Expected Reproduction Time (in Minutes)*

Since the generalized case study simulates the expert popularity, it can be performed directly. Its execution time depends on the number of fine-tuning iterations, the model configuration, the simulated expert popularity and device topology. It takes less than 10 mins per case.

For the real case study, it needs to run the real fine-tuning process and collect gating results. The expected time of this process is around 30 mins.

*Hardware:* We conduct all these experiments on a cluster with 4 nodes. Each node contains 8 NVIDIA A800 GPUs (40GB) and every four of them connect to a Intel Xeon Gold 6348 CPU with 28 cores. Each node has a total of 1024 GB main memory. The inter-node interconnect is InfiniBand HDR 100 Gbps, and the intra-node interconnect is PCIe. We evaluate on three different device topologies: *C1+G4*, *C1+G2* and *C1+G1*.

*Software:* Ubuntu 22.04.3, Pytorch 2.0.0+cu117, numpy 1.26.4, transformers 4.37.0, psutil 5.9.8.

*Datasets / Inputs:* Basically, we design a simulator to proxy both predictor and gate operation for evaluation. For the generalized model study, we simulate the expert popularity and use the dummy data. For the real case study, we take traces from fine-tuning NLLB-MoE<sup>1</sup> and Mixtral-8x7B<sup>2</sup> models on APP<sup>3</sup> dataset.

*Installation and Deployment:* The artifact depends on Pytorch and the recommended version is 2.0.0+cu117. Also, it relies on numpy 1.26.4, transformers 4.37.0 and psutil 5.9.8.

#### *Artifact Execution*

#### To execution command of the demo:

CUDA\_VISIBLE\_DEVICES=0,1,2,3 torchrun - nproc\_per\_node 4 ./main.py --is\_moe=True --num\_training\_steps=50 --model\_config=S --num\_experts=16 --gini=0.3 --topo=C1+G2 --pipeline=APTMoE. You can use python main.py --help and browse README.md to investigate the meaning of these hyper-parameters and customize the experiment.

The experiment workflow of executing the artifact A<sup>1</sup> for the generalized case is consist of two phases. First, the static part profiles the execution of a single MoE layer to generate the memory usage of a layer and an execution time lookup table, so as to provide guidance for the runtime part. Second, the runtime part performs the pipeline fine-tuning with the given model configuration and parameter settings. The throughput of different approaches will be reported in this phase.

To execute real case experiment, we need to execute the real fine-tuning and acquire the expert popularity, which follows commands in README.md of APTMoE/RealCase. This part will produce expert popularity generated from both predictor and gate throughout all iterations. Also, it reports the predictor accuracy.

*Artifact Analysis (incl. Outputs)*

The throughput of each evaluation is related to a specific model configuration(e.g. MoE-S-16), hardware configuration(e.g. *C1+G2*) and expert popularity(e.g. G=0.3). In most cases, our APTMoE outperforms Mobius, GPipe, and GPipeOffload.

For the predictor in the simulator part, its accuracy improves with the training process, and tends to be stable in seconds.

<sup>1</sup>https://huggingface.co/docs/transformers/main/model doc/nllb-moe

<sup>2</sup>https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1

<sup>3</sup>https://github.com/hendrycks/apps