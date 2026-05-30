# Algorithm 2 Fused AG-Dispatch Pairwise Communication

```
Require: An n-node cluster with m GPUs/NPUs per node; an
       input tensor X \in \mathbb{R}^{\frac{b}{d_{\mathrm{DP}}} \times s \times h} per node; global rank r
Ensure: An output tensor Y \in \mathbb{R}^{\frac{bs}{dEP} \times h} per node
  1: Y \leftarrow \text{empty}(\frac{bs}{d_{\text{EP}}}, h)
2: [X_1, X_2, \cdots, X_m] \leftarrow \text{split}(X, m, -1)
  3: [Y_{11}, Y_{12}, \dots, Y_{mn}] \leftarrow \text{split}(Y, [m, n], [0, 1]) \rightarrow \text{Split } Y
       into m \times n parts along the token and hidden dimension
  4: r_{\text{TP}} \leftarrow r \mod m
  5: [X_{r_{\text{TP}}1}, X_{r_{\text{TP}}2}, \cdots, X_{r_{\text{TP}}n}] \leftarrow \text{route}(X_{r_{\text{TP}}}) \triangleright \text{Calculate the}
       expert map on local TP rank only
  6: for i \leftarrow 1 to n - 1 do async
             r_{\text{to}} \leftarrow (r_{\text{TP}} + im) \mod mn
  8:
              isend(X_{r_{TP}i}, r_{to})
  9:
             r_{\text{from}} \leftarrow (r_{\text{TP}} - im) \mod mn
 10:
             Y_{r_{\text{TP}}r_{\text{from}}} \leftarrow \text{irecv}(r_{\text{from}})
 11: end for
 12: for i \leftarrow 1 to n do async
             Y_{:i} \leftarrow \mathbf{await} \text{ all } \mathsf{gather}(Y_{r_{\mathsf{TP}}i}, \mathsf{TP} \mathsf{group})
14: end for
```

storage space for each rank, corresponding in size to the output. Consequently, the space complexity is  $O(bsh \cdot n_{\rm proc})$  in total. The fused RS-Combine Pairwise communication algorithm we present ingeniously incorporates an overlapping mechanism for communication both within and between nodes, effectively trading off space for time. This approach significantly reduces the communication overhead associated with inference in the MoE models.

2) Fused AG-Dispatch Communication Algorithm: Similarly, it is precisely because the hidden states are replicated in the MoE TP group that they can be sharded within the MoE TP group. The sharding further minimizes the internode Dispatch communication overhead, requiring only the addition of extra intra-node AG communication, analogous to Megatron-based PP. On this basis, it is possible to allow for the overlapping of the intra-node AG communication and inter-node Dispatch communication, in a manner analogous to the Fused RS-Combine algorithm.

Fig. 9b shows the Gantt chart of the fused AG-Dispatch communication algorithm. Apart from the pairwise communication in the first round and the AG communication in the last round, the intra-node and inter-node communication during the remaining rounds can overlap with one another. Alg. 2 describes the detailed communication schedule. In contrast to Alg. 1, the total number of communication rounds both within and between nodes is  $n_{\text{node}} - 1$ , as the local shards in the TP group and EP group do not require communication. Therefore, the time complexity of the algorithm is  $O(n_{\text{node}})$  and the space complexity is O(1), respectively.

TABLE II: Configuration of parallel strategies of baselines.

| Baselines | Parallel Strategies                                  |                    |  |  |  |
|-----------|------------------------------------------------------|--------------------|--|--|--|
| Dascilles | H20                                                  | Ascend 910B        |  |  |  |
|           | TP=8 [PP=2]                                          | TP=8 [PP=4]        |  |  |  |
| vLLM      | TP=8 + DP=2, EP=16                                   | TP=8 + DP=4, EP=32 |  |  |  |
|           | TP=4 + DP=4, EP=16                                   | TP=4 + DP=8, EP=32 |  |  |  |
| Tutel     | TP=8 + DP=2, TP=8 + EP=2<br>TP=4 + DP=4, TP=4 + EP=4 | Not supported      |  |  |  |

#### IV. EVALUATION

#### A. Experimental Setup

**Hardware and Network**: We conduct our experiments on following clusters:

- A cluster of 2 servers with 8 Nvidia H20 GPUs (96 GB) each. The intra-node network is supported by NVLink 4.0 (up to 900 GB/s), while the inter-node network is connected via InfiniBand (400 Gbps).
- A cluster of 4 Atlas 800T A2 servers with 8 Ascend 910B NPUs (64 GB) each. The intra-node network is fullyconnected via HCCS (up to 480 Gbps), while the inter-node network is connected via RoCE (up to 200 Gbps).

**Implementation**: We implement MixServe based on several serving systems, including vLLM [7] (on the Ascend 910B cluster) and Tutel [8] (on the H20 cluster).

Models and Datasets: To evaluate MixServe, we adopt the following SOTA MoE models: (1) DeepSeek-R1 [2], a 671B-parameter MoE model with 256 routed experts and 1 shared expert, where 37B parameters are activated per token; and (2) Qwen3 [3], a 235B-parameter MoE model with 128 experts, with 22B parameters activated per token. We use ShareGPT-V3 [9] for benchmark evaluation, which is a large-scale dataset containing 1.2B tokens of human conversations.

**Baselines**: We compare MixServe with the following baselines: (1) vLLM [7], which utilizes hybrid TP+PP for LLM serving and hybrid DP+EP for distributed MoE model serving; and (2) Tutel [8], which employs hybrid TP+EP for distributed MoE model serving. In addition, we also set up different TP degrees (*i.e.* 4 and 8) for comparative experiments. The specific configurations of parallel strategies for baselines are summarized in Table II.

