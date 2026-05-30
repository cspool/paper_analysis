# Algorithm 1: Dynamic expert clustering.

```
: I: Profiled data, n_e: number of experts, n_l: number of
               layers, max iter: max iterations
   output : C: clusters, c: centroids
1 c \leftarrow \text{random initialization from } I / \star \text{ Initialize centroids } \star /
2 for iter \leftarrow 1 to max\_iter do
        C \leftarrow \{\} / \star \text{ Reset clusters}
        /* Assign each data point to the nearest centroid
        foreach i \in I do
             assigned\_cluster \leftarrow \arg\min_{c} (n_l - \operatorname{overlap}(i, c))
               C[assigned\_cluster] \leftarrow C[assigned\_cluster] \cup \{i\}
        end
             Update centroids based on cluster members
        for j \in 1 \dots n_e do
                 \leftarrow most common experts in each layer of C[j]
10 end
11 return C, c
```

selection statistics (Profiling). With the profiled data, ScaleMoE replicates popular experts to replace unpopular ones to improve clustering efficiency (Replicating Experts). Then, ScaleMoE clusters input tokens having similar expert selection patterns (Dynamic Expert Clustering).

**Profiling.** Since we rely on the per-token expert selection history to predict the selections in the current epoch, it is essential to ensure that expert selection does not vary dramatically between consecutive epochs. To assess the similarity in expert selection, we measure the ratio of changes (i.e., *changing ratio*) between consecutive epochs. Figure 10 shows the results, with the x-axis representing training epochs and the y-axis indicating the changing ratio in expert selection. For instance, when the x-axis value is i, the corresponding changing ratio (# of tokens selecting different experts / # of total experts) between  $i-1^{th}$  and  $i^{th}$  epochs. Here, a lower changing ratio means that consecutive epochs have similar expert selection patterns. The results show that the changing ratio quickly decreases to 6.25% (at epoch-9), indicating our history-based approach is plausible.

ScaleMoE systematically monitors the per-token expert selections. To support various shuffling methods in the data loader, we assign a unique identifier *<batchID*, *sequenceID*, *tokenIndex*, *tokenName>* to each token. The size of this unique ID is small (12B) compared to the hidden dimension (3072B), so this profiling overhead is negligible.

**Replicating Experts.** To improve clustering efficiency, ScaleMoE replicates experts based on profiled data. With the profiled data, ScaleMoE identifies rarely selected experts (i.e.,

![](_page_6_Figure_8.jpeg)

Fig. 11: The overview of topology-aware expert remapping. At runtime, ScaleMoE builds two matrices (i.e., coverage matrix, bandwidth matrix) to find near-optimal cluster mapping.

unpopular experts). Once we identify unpopular experts, we spill them to host pinned memory. To leverage freed GPU memory, ScaleMoE replaces the offloaded unpopular experts with replicas of frequently selected experts (i.e., popular experts). The number of replicas for each expert is determined proportionally to its selection ratio; in other words, popular experts get more replicas. Replicating experts allows more tokens to be transferred locally, and since access to unpopular experts is rare, the performance overhead is negligible.

Dynamic expert clustering. ScaleMoE clusters input tokens with similar expert selection patterns using K-means. For the distance function, we compute the number of overlapping expert selections between two sequences and subtract it from the total sequence length (i.e., a smaller distance indicates that two tokens have more similar expert selections). Algorithm 1 shows our clustering process. ScaleMoE selects random centroids for each cluster (line 1) to avoid clusters being biased to specific points. Then, it iterates the clustering logic until it reaches the maximum number of iterations or the clusters are saturated (lines 2-10). For each iteration, it first computes the distance between the inputs and each cluster. The inputs are assigned to the cluster with the closest centroid, and the centroids of each cluster are updated accordingly. Next, ScaleMoE updates the centroids by choosing the most frequently selected expert for each layer (lines 3–6).

#### C. Topology-aware Expert Remapping

The state-of-the-art distributed training frameworks often overlook the heterogeneous networks, leading to suboptimal performance. To address this, ScaleMoE proposes topology-aware expert remapping to achieve near-optimal performance on various network configurations. Figure 11 shows the high-level overview of our technique. To find the optimal mapping between clusters and GPUs while considering heterogeneous network bandwidths, ScaleMoE constructs two matrices: the coverage matrix (representing the coverage information between clusters) and the bandwidth matrix (representing peer-to-peer network bandwidths between GPUs). With these two matrices, ScaleMoE performs expert remapping using a genetic algorithm.

Coverage matrix & bandwidth matrix. Given that clustering is not always perfect, it is possible that some input tokens' target experts do not exist within the corresponding clusters. In such a case, we need to send those tokens to experts on other

GPUs, which incurs additional network traffic. Depending on cluster-to-GPU mapping, this network traffic can be handled via various network mediums (e.g., NVLink [64], UltraFast Ethernet [67], moderate Ethernet [69]).

To estimate amount of required expert transfers, ScaleMoE builds the *coverage matrix* that represents the overlap between clusters (e.g., how much one cluster can cover another). In the (C x C) coverage matrix (C is the number of clusters), each cell (i, j) represents how well cluster (Ci) can cover the experts required by cluster (C<sup>j</sup> ).

In addition to the coverage matrix, ScaleMoE also builds the *bandwidth matrix* representing the peer-to-peer network bandwidths between node pairs. For any given network configuration, ScaleMoE initially constructs this bandwidth map when the network is idle.

Genetic algorithm. With these two matrices, we can find the optimal mapping between devices (GPUs) and clusters. However, the design exploration space is vast and grows significantly as the number of experts (Ne) increases. To reduce search time, ScaleMoE adopts a heuristic approach using a genetic algorithm to find a near-optimal cluster mapping within a reasonable timeframe. Through the genetic algorithm, ScaleMoE aims to find a solution vector (SV ) containing mapping information from devices (i) to clusters (SV [i]).

$$FitnessFunction = \sum_{i,j=0}^{N_e} \frac{\{(b \cdot s) - CM[SV[i]][SV[j]]\} \cdot h}{(BM[i][j])} \tag{2}$$

Equation 2 shows our fitness function. Here, b, s, and D<sup>H</sup> represent batch size, sequence length, and hidden dimension, respectively. The indices i and j represent the device indices. The term (b · s) − CM[SV [i]][SV [j]]·h represents the size of data that needs to be transferred. We compute the estimated communication time by dividing the data size by the GPU bandwidth (BM[i][j]). In each generation, the SV with the lowest fitness value is selected to minimize communication latency, and the genetic algorithm performs the uniform orderbased crossover and mutations that involve swapping the cluster mapping between two arbitrary positions. Once the genetic algorithm is complete, we map the clusters to the corresponding devices by following the solution vector (SV ).

