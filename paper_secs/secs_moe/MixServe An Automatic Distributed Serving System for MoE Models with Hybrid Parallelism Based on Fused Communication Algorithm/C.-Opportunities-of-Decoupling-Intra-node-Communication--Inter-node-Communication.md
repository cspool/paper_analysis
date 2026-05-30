# C. Opportunities of Decoupling Intra-node Communication & Inter-node Communication

We adopt the parallel strategies outlined in the DeepSeek-V3 technical report [1], in which the MoE blocks fully employ EP while retaining the parallel strategy for the Attention blocks. This approach decouples the original EP communication group into intra-node TP groups and inter-node EP groups. We collected the profiling data of the MoE block from a layer of the Decoder in DeepSeek-R1. The results of the Gantt chart in Fig. 4 indicate that while the TP within the nodes introduced AR, it significantly assisted the EP component in sharing a substantial portion of the communication, leading to a significant reduction in the communication overhead of the EP group. Preliminary experimental results indicate that decoupling intra-node and inter-node communication allows for further optimization of communication overhead.

#### III. System Design

We introduce MixServe, a novel automatic distributed serving system that enables efficient deployment of MoE models by TP-EP hybrid parallelism based on a fused AR-A2A communication algorithm. MixServe automatically selects the optimal parallel strategy based on model parameters and network configurations.

#### A. System Overview

Fig. 5 illustrates the system overview of MixServe, which operates in two stages: (i) offline, (ii) online. During the offline stage, MixServe determines the optimal parallelism strategy based on the model's hyperparameters and the configuration of network and hardware resources. During the online stage, MixServe automatically loads and partitions the model weights according to the results of the parallelism strategy analyzed during the offline phase. Additionally, it injects collective communication operators into the model's forward method through the mixed parallel communication groups.

Offline Stage: MixServe first retrieves the model's hyperparameters and presets prompts with varying batch sizes and sequence lengths to obtain profiling data as observations. Subsequently, it uses the configuration of network and hardware resources as input, which includes computational power, as well as intra-node and inter-node network bandwidth and topology, to calculate theoretical values. Both the observations and theoretical values are then input into the analyzer to derive the optimal parallelism strategy. This will provide critical input for the weight loader and partitioner in the online phase.

Online Stage: Based on the optimal parallelism strategy derived from the offline stage, MixServe employs the weight loader to load the corresponding model weight shards through the partitioner. Subsequently, when MixServe initiates the serving service, it initializes the mixed parallel communication group and injects collective communication operators into the appropriate forward method of the MoE models. The serving service manages memory and schedules requests based on the leading vLLM [7] system currently available in the industry.

![](_page_3_Figure_8.jpeg)

Fig. 5: MixServe system overview.

