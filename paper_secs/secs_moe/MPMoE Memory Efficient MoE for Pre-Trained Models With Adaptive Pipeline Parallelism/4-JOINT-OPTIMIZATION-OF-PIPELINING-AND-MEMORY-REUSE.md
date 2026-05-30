# 4 JOINT OPTIMIZATION OF PIPELINING AND MEMORY REUSE

As described in Section 3.2 and Section 3.3, we propose micro-batch pipelining to mitigate the latency of communication and memory reuse strategies to reduce the memory footprint. However, each design is influenced by certain configurations. First, the performance of the micro-batch pipelining depends on the granularity of the pipeline, which is determined by n. A pipeline that is too coarse-grained may result in insufficient overlap, while a pipeline that is too fine-grained may lead to underutilization of hardware resources such as GPU utilization and network bandwidth. Second, the latency overhead

```
Algorithm 1: Adaptive Pipeline Granularity Search
```

**Input:** the batch size of tokens B

```
Input: the memory reuse strategy S
   Output: the number of partitions n
   /\star SortedDict(\{n_i: (B_i^{floor}, B_i^{ceil}\}))
 1 global: G = \{2 : (0,1), 8 : (\infty,\infty)\};
2 global: \mathcal{C} = \{\};
 \mathfrak{s} if B in \mathcal C then
       return \mathcal{C}[B];
 5 end
 6 ((B^{floor}, B^{ceil}, n_i) = find\_closest\_B(G, B);
   /* find best n for B
7 if (B > B^{ceil}) then
       \_, B^{ceil} = G[n_{i+1}] ;
       n^{floor} = n_i;
9
       n^{ceil} = n_{i+1};
10
       n = searchBestGran(B, (n^{floor}, n^{ceil}));
11
12 end
13 if (B < B^{floor}) then
       B^{floor}, \_ = G[n_{i-1}] ;
       n^{floor} = n_{i-1};
       n^{ceil} = n_i;
16
       n = searchBestGran(B, (n^{floor}, n^{ceil})) ;
17
18 end
   /* update G
                                                               */
19 if (n! = n^{floor} \& \& n! = n^{ceil}) then
       G[n] = (B, B);
21 else
       G[n] = (min(B, B^{lower}), max(B, B^{upper}));
23 end
24 C[B] = n;
25 return n;
```

of memory reuse is affected by the activation-restoring strategies, which is denoted as S. Consequently, we consider the configuration of MPMoE to be (n,S). To determine the optimal configuration, we explore two methods:

- Profile-Based. This method determines the optimal configuration by profiling performance metrics in the real environment. However, this approach incurs the profiling overhead and the search space for configurations increases with the combination of different pipeline granularity and memory reuse strategies.
- Performance Modeling. Establishing a performance model to estimate the performance of different configurations. This method benefits from its fast speed but may struggle to achieve high accuracy in complex product environments.

In this paper, we aim to optimize pipelining and memory reuse strategies jointly by employing the two aforementioned methods. The performance of both methods will be studied in Section 5.

#### 4.1 Profile-Based

As mentioned above, the profile-based method suffers from profiling overhead. To mitigate this, we can cache and reuse all profiling data. However, since the variable B is dynamic and covers a wide range during the training process of MoE models [35], searching for the optimal (n, S) configuration for every value of B is time-consuming.

In order to reduce the search space, we propose solutions based on two intuitive hypotheses: First, n is monotonically increasing as B increases for each S. As a result, the whole value domain of B can be divided into a set of disjoint ranges. We only need to find the boundaries of each range, which reduces the cost of configuration on n by one to two orders of magnitude. Second, given input with batch size equal to B, the performance of the MoE Layers with respect to n is parabola-like. This is reasonable because a very coarsegrained pipeline leads to insufficient overlap and a finegrained pipeline leads to low utilization of hardware resources.

Specifically, we obtain the best configuration for each S as illustrated in Algorithm 1.  $\mathcal C$  is denoted as the cache of searched results and  $G = \{n_1 : (B_1^{floor}, B_1^{ceil}), n_2 :$  $(B_2^{floor}, B_2^{ceil}), \dots \}$  denotes the boundaries of searched ranges, where  $(B_1^{floor}, B_1^{ceil}), (B_2^{floor}, B_2^{ceil})$  correspond to the ranges whose best granularity are  $n_1, n_2$  respectively. Here G is sorted in ascending order according to n, and Bs in G are monotonically increasing with respect to n. When coming to a new B which does not exist in C, we try to find the range i where  $B \geq B_i^{floor}$  and  $B \leq B_i^{ceil}$  and take  $n_i$  as the best granularity, i.e., lines 6. If not found, we obtain  $n_l$  and  $n_h$  from G where range i-1 and range i+1 are the closest ranges to B(values in range i-1 are smaller than B and in range i+1 are bigger than B). Then, we profile the execution time of the program with different granularities ranging from  $n^{floor}$  to  $n^{ceil}$  and obtain the best granularity n, i.e., lines 7-17. Here we call searchBestGran to search for the optimal configuration from  $n^{floor}$  to  $n^{ceil}$ , i.e. line 11, 17. Because the performance concerning n is parabolalike, we can stop the searching process when meeting the tuning point of n. Finally, we update G according to B, n, i.e. lines 19-23. With more training iterations, the boundaries of ranges are more accurate, the profiling processes are fewer and search scopes are fewer. Besides, we initialize G with  $\{1:(0,1),8:(\infty,\infty)\}$  in line 1, indicating the scope of granularity is limited from 2 to 8. This is reasonable because we find it is applicable to the vast majority of scenarios.

Once the optimal pipeline granularity for each memory reuse strategy is chosen, we can configure the optimal (n,S) by simply comparing the results.

## 4.2 Performance Modeling

As mentioned earlier, the profile-based method relies on time-consuming profiling steps. In this section, we aim

![](_page_7_Figure_9.jpeg)

(a) Paradigm 1. Only consider communication and computation. This is applicable to S4.

![](_page_7_Figure_11.jpeg)

(b) Paradigm 2. This is applicable to forward pass of S1, S2 and S3.

![](_page_7_Figure_13.jpeg)

(c) Paradigm 3. This is applicable to backward pass of S1, S2 and S3.

Fig. 8. The illustration of three pipeline paradigms, where S, R, and C have the same meaning as in Figure 4(b), and M represents memory transfer between CPU and GPU respectively. The left DAG graph of each subfigure represents the dependence between different operations of each micro-batch. The right graph of each subfigure describes the pipeline patterns, each includes 5 phases: P0, the initial phase; P1, the saturating phase; P2, the saturated phase; P3, the melting phase; and P4, the final phase. The estimated execution time of each phase is listed in the lower right corner of each subfigure.

to overcome this limitation by developing a lightweight performance model to estimate the execution time of different configurations, which is able to obtain the optimal configuration efficiently. However, we encounter two challenges when constructing the performance model. First, hardware utilization varies with the volume of data, such as underutilized network bandwidth when the data volume of each partition is too low. Second, as explained in Section 2.3, the execution time of communication, computation, and memory copy operations can impact each other when executed in parallel, despite individually requesting different hardware resources in principle.

To address these challenges, we propose two solutions. First, we have developed a piecewise function model to accurately capture the speeds of communication, computation, and memory copying at different data volumes as shown in Figure 9. In a specific product environment, we execute micro-benchmark programs to profile the speeds of these operations independently, which allows us to establish a single performance model that can be applied

![](_page_8_Figure_2.jpeg)

Fig. 9. The micro-benchmarks for profiling and performing piecewise linear fitting of the performance of specific operations.

to multiple models within the same product environment, thereby minimizing overhead. Second, we utilize the workers presented in Section 2.3 to quantify the interference between different operations. These results enable us to measure the impact of executing operations in parallel and account for their mutual influence on performance.

To simplify the representation of different pipeline patterns, we abstract them into three pipeline paradigms, as depicted in Figure 8. The meaning **S**, **R**, **C** is the same as that in Figure 4(b) and we use the symbol **M** to represent the memory transfer between the CPU and GPU. The Paradigm 1 does not contain **M**, which is applicable for S4. The Paradigm 2 involves **M** which depends on **S** and **C** because the activations offloading to CPU are produced from **S** and **C**. The forward pass of S1 to S3 obeys the paradigm 2. **C** in Paradigm 3 depends on **M** because activations must be transferred to GPU before computation, which is applicable for the backward pass of S1 to S3.

For each paradigm, the pipeline can be divided into five phases. 1) P0, the initial phase, in which only one or two CUDA streams are working as usual; 2) P1, the saturating phase, in which all CUDA streams are launching but not saturated; 3) P2, the saturated phase, in which all CUDA streams are saturated and steady, and there may be multiple P2 stages in the whole pipeline; 4) P3, the melting phase, which is similar to P1; 5) P4, the final phase, which is similar to P0.

The estimated execution time of each phase is provided in the right bottom corner of each subfigure depicted in Figure 8. In each phase, the execution time is determined by the bottleneck CUDA stream. For example, the execution time of P2 in paradigm 1 is determined by the maximum execution time of  $\mathbf{R+S}$  and  $\mathbf{C}$ . For the sake of conciseness, we omit the slowdown factors in the formulation of Figure 8. For instance, the complete formulation of P2's execution time in paradigm 1 is  $max(\frac{t_S+t_R}{\alpha(comm,comp)},\frac{t_C}{\alpha(copm,comm)})$ , where  $\alpha$  comes from Section 2.3.

