# <span id="page-6-0"></span>3.4 Summary of RaBitQ

We summarize the RaBitQ algorithm in Algorithm 1 (the index phase) and Algorithm 2 (the query phase). In the index phase, it takes a set of raw data vectors as inputs. It normalizes the set of vectors based on Section 3.1.1 (line 1), constructs the RaBitQ codebook by sampling a random orthogonal matrix P based on Section 3.1.2 (line 2) and computes the quantization codes  $\bar{\mathbf{x}}_b$  based on Section 3.1.3 (line 3). In the query phase, it takes a raw query vector, a set of IDs of the data vectors and the pre-processed variables about the RaBitQ method as inputs. It first inversely transforms, normalizes and quantizes the raw query vector (line 1-2). We note that the time cost of these steps can be shared by all the data vectors. Then for each input ID of the data vectors, it efficiently computes the value of  $\frac{\langle \bar{\mathbf{o}}, \mathbf{q} \rangle}{\langle \bar{\mathbf{o}}, \mathbf{o} \rangle}$  based on Section 3.3.2, adopts it as an unbiased estimation of  $\langle \mathbf{o}, \mathbf{q} \rangle$  based on Section 3.2 and further computes an

estimated distance between the raw query and the raw data vectors based on Section 3.1.1 (line 3-5).

## Algorithm 1: RaBitQ (Index Phase)

**Input**: A set of raw data vectors

**Output:** The sampled matrix P; the quantization code  $\bar{\mathbf{x}}_b$ ; the pre-computed results of  $\|\mathbf{o}_r - \mathbf{c}\|$  and  $\langle \bar{\mathbf{o}}, \mathbf{o} \rangle$ 

- 1 Normalize the set of vectors (Section 3.1.1)
- <sup>2</sup> Sample a random orthogonal matrix P to construct the codebook  $C_{rand}$  (Section 3.1.2)
- <sup>3</sup> Compute the quantization codes  $\bar{\mathbf{x}}_b$  (Section 3.1.3)
- <span id="page-7-2"></span>4 Pre-compute the values of  $\|\mathbf{o}_r - \mathbf{c}\|$  and  $\langle \bar{\mathbf{o}}, \mathbf{o} \rangle$

## Algorithm 2: RaBitQ (Query Phase)

**Input** :A raw query vector  $\mathbf{q}_r$ ; the sampled matrix P; a set of IDs of the data vectors, their quantization codes  $\bar{\mathbf{x}}_b$  and the results of  $\|\mathbf{o}_r - \mathbf{c}\|$  and  $\langle \bar{\mathbf{o}}, \mathbf{o} \rangle$ 

Output: A set of approximate distances between the raw query and the raw data vectors

- 1 Normalize and inversely transform the raw query vector and obtain q'
- 2 Quantize q' into q (Section 3.3.1)
- 3 **foreach** input ID of the data vectors **do**
- Compute the value of the estimator  $\frac{\langle \bar{\mathbf{o}}, \mathbf{q} \rangle}{\langle \bar{\mathbf{o}}, \mathbf{o} \rangle}$  as an approximation of  $\langle \mathbf{o}, \mathbf{q} \rangle$  (Section 3.3.2)
- <span id="page-7-3"></span>5 Compute an estimated distance between the raw query and the data vector based on Equation (2)

## <span id="page-7-0"></span>4 RABITQ FOR IN-MEMORY ANN SEARCH

Next we present the application of our method to the in-memory ANN search. We note that the popular quantization method PQx4fs has been used in combination with the inverted-file-based indexes such as IVF [45] or the graph-based indexes such as NGT-QG [43] for in-memory ANN search. The combination of a quantization method with IVF can be easily done without much efforts. For example, we can use the quantization method to estimate the distances between the data vectors in the clusters that are probed, which decide those vectors to be re-ranked based on exact distances. In this case, batches of data vectors can be formed and the SIMD-based fast implementation (i.e., PQx4fs) can be adopted. Nevertheless, the combination of a quantization method with graph-based methods such as NGT-QG would require much more efforts in order to make the combined method work competitively in the in-memory setting, which would be of independent interest. This is because in graph-based methods, the vectors to be searched are decided one after one based on the greedy search process in the run-time, and it is not easy to form batches of them so that SIMD-based fast implementation can be adopted. Therefore, we apply our method in combination with IVF index [45] in this paper. We leave it as future work to apply our quantization method in graph-based methods.

We present the workflow of the RaBitQ method with the IVF index as follows. During the index phase, for a set of raw data vectors, the IVF algorithm first clusters them with the KMeans algorithm, builds a bucket for each cluster and assigns the vectors to their corresponding buckets. Our method then normalizes the raw data vectors based on the centroids of their corresponding clusters and feeds the set of the normalized vectors to the subsequent steps of our RaBitQ method. During the query phase, for a raw query vector, the algorithm selects the first  $\overline{N_{probe}}$  clusters whose centroids are nearest to the query. Then for each selected cluster, the algorithm retrieves all the quantization codes and estimates their distances based on the quantization codes, which decide the vectors to be re-ranked based on exact distances.

As for re-ranking [85], PQ and its variants set a fixed hyperparameter which decides the number of data vectors to be re-ranked (i.e., they re-rank the vectors with the smallest estimated distances). Specifically, they retrieve their raw data vectors, compute the exact distances and find out the final NN. In particular, the tuning of the hyper-parameter is empirical and often hard as it can vary largely across different datasets (see Section 5.2.3). In contrast, recall that in our method, there is a sharp error bound as discussed in Section 3.2 (note that the error bound is rigorous and always holds regardless of the data distribution). Thus, we decide the data vectors to be reranked based on the error bound without tuning hyper-parameters. Specifically, if a data vector has its lower bound of the distance greater than the exact distance of the currently searched nearest neighbor, then we drop it. Otherwise, we compute its exact distance for re-ranking. Due to the theoretical error bound, the re-ranking strategy has the guarantee of correctly sending the true NN from the probed clusters to re-ranking with high probability. The empirical verification can be found in Section 5.2.4. We emphasize that the idea of re-ranking based on a bound is not new. There are many studies from the database community that adopt a similar strategy [24, 27, 76, 89, 90, 94] for improving the robustness of similarity search for various data types. We note that beyond the idea of re-ranking based on an error bound, RaBitQ provides rigorous theoretical analysis on the tightness of the bounds and achieves the asymptotic optimality as we have discussed in Section 3.2.2.

Moreover, it is worth noting that re-ranking is a vital step for pushing forward RaBitQ's rigorous error bounds on the distances to the robustness of ANN search. In particular, when the ANN search requires higher accuracy than what RaBitQ can guarantee (e.g., when the true distances from the query to two different data vectors are extremely close to each other), then the estimated distance produced by RaBitQ would be less effective to rank them correctly. Re-ranking, in this case, is necessary for achieving high recall. Note that it is inherently difficult for any methods of distance estimation when the distances are extremely close to each other.

