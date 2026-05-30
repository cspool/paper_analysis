# Algorithm 1 Training MoE models using our LSH-MoE framework

```
Input: X: sequence of tokens
Output: \{Y_{ij} \mid 1 \leq i \leq n, 1 \leq j \leq m\}, where Y_{ij} is the output for tokens in the j-th cluster
assigned to the i-th expert
 1: function MoE_LAYER_WITH_LSH(X)
          Calculate the token-to-expert mapping \zeta using the gating network;
          Dispatch X into \{X_i \mid i=1,2,\ldots,n\} based on \zeta; // X_i are tokens assigned to the i-th
 3:
     expert
 4:
          for i \leftarrow 1, 2, \ldots, n do
                IDX_i \leftarrow LSH(X_i); // Get the LSH bucket for each token
 5:
               Divide X_i into {cluster j \mid j = 1, 2, ..., m} based on IDX;
 7:
                for j \leftarrow 1, 2, \ldots, m do
 8:
                    \overline{\text{cluster}_j} \leftarrow \text{Mean}(\text{cluster}_j); // \text{ Get the centroids for each cluster}
                    \Deltacluster<sub>i</sub> \leftarrow \{x - \overline{\text{cluster}}_i \mid x \in \text{cluster}_i\}; // Get the difference between each token
 9:
     and its cluster centroids
               \begin{array}{l} C_i \leftarrow \{\overline{\mathrm{cluster}}_j \mid j=1,2,\ldots,m\}; \\ \Delta X_i \leftarrow \bigcup_{j=1}^m \Delta \mathrm{cluster}_j; \end{array}
10:
11:
          C \leftarrow \{C_i \mid i = 1, 2, \dots, n\};
12:
          Input \leftarrow \text{all-to-all}(C); // Transmit the cluster centroids through all-to-all
13:
14:
          Output \leftarrow Expert(Input); // Perform computations on centroids
          E(C) \leftarrow \text{all-to-all}(Output); // Transmit the results back through all-to-all
15:
16:
          for (i, j) \leftarrow (1, 2, ..., n) \times (1, 2, ..., m) do
               Y_{ij} \leftarrow \{E(\overline{\text{cluster}}_j) + \Delta \text{Cluster}_{jk} \mid k = 1, 2, \dots, N_j\}; // \text{Apply the residual-based error}
     compensation scheme
18:
          return \{Y\}.
```

