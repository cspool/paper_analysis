# D. Fused AR-A2A Communication Algorithm

Building upon the hybrid TP-EP parallelism, we design the fused AR-A2A communication algorithm by employing the principle of mutual overlapping of intra-node and internode communication, guided by the computational dependency relationships.

Fig. 8 illustrates the overall process of the fused AR-A2A communication algorithm. Steps 1-5 demonstrate how hidden states are synchronized by intra-node TP and inter-node EP. Initially, each rank only possesses a partition of the hidden

#### **Algorithm 1** Fused RS-Combine Pairwise Communication

```
Require: An n-node cluster with m GPUs/NPUs per node; an
      input tensor X \in \mathbb{R}^{\frac{bs}{d_{\rm EP}} \times h} per node; global rank r
Ensure: An output tensor Y \in \mathbb{R}^{\frac{b}{d_{\mathrm{DP}}} \times s \times h} each node
  1: Y \leftarrow \text{empty}(\frac{b}{d_{\text{DP}}}, s, h)
2: [X_1, X_2, \cdots, X_m] \leftarrow \text{split}(X, m, -1) \rightarrow \text{Split } X \text{ into } m
      parts along the hidden dimension (the same below)
  3: [Y_1, Y_2, \cdots, Y_m] \leftarrow \text{split}(Y, m, -1)
  4: r_{\text{TP}} \leftarrow r \mod m
                                                                  ▶ Compute TP rank
  5: Initialize tensor list [S_1, S_2, \dots, S_n]
                                                           \triangleright Stage local tensor X_{r_{TP}}
  6: S_1 \leftarrow X_{r_{\text{TP}}}
  7: for i \leftarrow 1 to n - 1 do async
            r_{\text{to}} \leftarrow (r_{\text{TP}} + im) \mod mn
            isend(X_{r_{TP}}, r_{to}) \rightarrow Send X_{r_{TP}} to the same TP rank of
  9:
      the next i-step node asynchronously
            r_{\text{from}} \leftarrow (r_{\text{TP}} - im) \mod mn
            S_{i+1} \leftarrow \text{irecv}(r_{\text{from}}) \rightarrow \text{Receive } X_{r_{\text{TP}}} \text{ from the same}
      TP rank of the previous i-step node asynchronously
 12: end for
                              ▶ Inter-node A2A pairwise communication
 13: for i \leftarrow 1 to n do async
            S_i \leftarrow \mathbf{await} \text{ reduce } \mathbf{scatter}(S_i, \mathsf{TP} \mathsf{group})
            Y_i \leftarrow Y_i + \text{topk\_weights}(S_i)
 15:
```

states (the blue segment). Subsequently, intra-node AG/RS communication (the light green segment) and inter-node A2A communication (the orange segment) are performed to exchange the hidden states asynchronously step by step. Finally, each rank acquires the complete hidden states (the green segment in step 5) after the communication process concludes.

17:  $Y \leftarrow \text{all\_gather}(Y_{r_{\text{TP}}}, \text{TP group})$ 

▶ Intra-node AR communication

1) Fused RS-Combine Communication Algorithm: Fig. 9a and Alg. 1 illustrates the fused RS-Combine communication algorithm. The algorithm is designed to optimize the communication process by overlapping intra-node and inter-node communication. The key steps are as follows: (1) Intra-node RS, (2) Inter-node A2A, (3) Intra-node AG.

Initially, the hidden states at each rank within the node engage in one round of RS communication, temporarily storing the results after weighting them with the top-k weights. Concurrently, a round of communication between nodes is executed using the Pairwise algorithm, enabling each node to acquire the corresponding hidden states as input for the subsequent iteration. Upon completion of the Pairwise algorithm, all weighted results are ultimately combined through AG within the nodes. The Gantt chart in Fig. 9 illustrates the overlapping of communication processes, where the RS and A2A communication are executed concurrently, followed by the AG operation.

In summary, the algorithm necessitates  $n_{\text{node}} - 1$  rounds of communication between nodes and  $n_{\text{node}}$  rounds of communication within each node. The asynchronous mechanism facilitates overlapping communication both within and across nodes, resulting in a time complexity of  $O(n_{\text{node}})$ . Furthermore, the algorithm necessitates the allocation of additional temporary

