# <span id="page-5-0"></span>4 Optimized Scheduling of Tasks

Motivated by the potential overlap between inter-node and intra-node communications, we design a new schedule to pipeline all time-consuming communication tasks (ESP-AllGather, ESP-ReduceScatter, AlltoAll Dispatch/Combine, and Gradient-AllReduce communications) and computing tasks (expert and attention computations) when the group of MP and ESP is aligned with the number of GPUs in a node. In such a scenario, ESP-AllGather and ESP-ReduceScatter are intra-node communications, while AlltoAll Dispatch/Combine and Gradient-Allreduce are inter-node communications.

This scenario is frequently encountered in practice. With respect to the MoE framework, each layer comprises a limited number of experts, but each expert's model is considerably large, preventing it from fitting entirely on a single GPU. For instance, models like Mixtral-8x7B and Qwen1.5-MoE-A2.7B necessitate dividing an expert across multiple GPUs during training. Meanwhile, considering the training hardware system's topology, inter-node communication (via InfiniBand or Ethernet) generally trails behind the faster intra-node communication methods (such as Shared Memory or NVLink). For instance, contemporary GPU clusters such as Nvidia

<span id="page-6-0"></span>![](_page_6_Figure_2.jpeg)

![](_page_6_Figure_3.jpeg)

![](_page_6_Figure_4.jpeg)

![](_page_6_Figure_5.jpeg)

(d) Our proposed schedule FSMoE w/ partitioning the gradient.

**Figure 3.** Backpropagation of four schedules in DP+MP+EP+ESP with the pipeline degree r=4 including (a) the default schedule, (b) an improved Tutel version (Tutel-Improved) where Gradient-AllReduce is overlapped with other dense operations using PipeMoE, (c) our proposed schedule FSMoE without partitioning the gradient, and (d) our proposed schedule FSMoE. The forward process is similar to the backpropagation except for the absence of the Gradient-AllReduce.

H100 DGX servers are equipped with eight 200Gb/s network interface cards (NICs), which collectively offer a peak bandwidth of 800Gb/s (equivalent to 100GB/s) for communication between any two nodes. In contrast, the NVLink within a server enables a bandwidth of 900GB/s, illustrating that the bandwidth within a single node is significantly greater than that between nodes. To balance both accuracy and training speed effectively, a practical approach is to align the MP and ESP with the number of GPUs contained within each node. For instance, when training Mixtral-8x7B with settings of  $N_{MP} = N_{ESP} = 8$  on servers that feature 8 A100-SXM4-80G GPUs, the approach is exactly feasible. This setup can also be simulated using a simulator  $^2$ .

As shown in Fig. 3d, the inputs are split into several chunks and sequentially processed in a pipeline. Notably, Gradient-Allreduce is followed by the AlltoAll Dispatch on the last

<span id="page-6-2"></span>![](_page_6_Figure_11.jpeg)

**Figure 4.** Four cases when scheduling the pipelining of ESP-AllGather/ESP-ReduceScatter, AlltoAll Dispatch/Combine, expert computations and Gradient-AllReduce with the pipeline degree r=2. (a) **Case1:** The AlltoAll communications are slower than intra-node communication and expert computations, but the inter-node communications (AlltoAll and Gradient-AllReduce) are not slower than intra-node communication and expert computations. (b) **Case2:** Expert computations are not slower than inter-node communications and intra-node communications. (c) **Case3:** The AlltoAll communications are not slower than intra-node communication and expert computations. (d) **Case4:** The intra-node communications (AllGather and ReduceScatter) are not slower than inter-node communications and expert computations.

partitioned input as it can also be overlapped with ESP-AllGather/ESP-ReduceScatter and expert computations in the backward phase. The forward phase is similar to the backward phase, except for Gradient-Allreduce. In addition, the optimal pipeline degree varies by phase, necessitating phase-specific solutions. To achieve the new proposed schedule, we first build performance models of different time-consuming computing and communication tasks like PipeMoE [42] and FasterMoE [14]. We then formulate an optimization problem based on the performance model and propose an efficient solution.

#### <span id="page-6-3"></span>4.1 Performance Models

The time required for each chunk in the AlltoAll, AllGather, ReduceScatter, and expert computation processes on inputs divided into r chunks is represented by  $t_{a2a,r}$ ,  $t_{ag,r}$ ,  $t_{rs,r}$ , and  $t_{exp,r}$  respectively. These times are modelled via linear models [42] as follows (will verify in §6.2):

<span id="page-6-4"></span>
$$t_{a2a,r} = \alpha_{a2a} + \frac{n_{a2a}}{r} \cdot \beta_{a2a},$$

$$t_{ag,r} = \alpha_{ag} + \frac{n_{ag}}{r} \cdot \beta_{ag},$$

$$t_{rs,r} = \alpha_{rs} + \frac{n_{rs}}{r} \cdot \beta_{rs},$$

$$t_{exp,r} = \alpha_{exp} + \frac{n_{exp}}{r} \cdot \beta_{exp},$$
(1)

<span id="page-6-1"></span> $<sup>^2</sup> https://llm\hbox{-}system\hbox{-}requirements.streamlit.app/$ 

where  $n_*$  represents the volume of the communication message or the computational workload,  $\alpha_*$  denotes the startup time and  $\beta_*$  represents the time per byte transmitted or per unit of workload processed. Particularly, when each expert computation includes multiple identical general matrix-multiplication (GEMM) operations,  $\alpha_{exp}$  and  $\beta_{exp}$  are determined by multiplying  $\alpha_{gemm}$  and  $\beta_{gemm}$  by the number of these operations.

### <span id="page-7-0"></span>4.2 Optimizing the Pipeline Degree

The performance model for both computation and communication supports optimizing the pipeline degree r to minimize time costs.

Direct optimization of overall time consumption is challenging because it relies on numerous factors. For instance, the start time of an ESP-ReduceScatter is constrained by both ESP-AllGather (inter-node communication contention) and Expert (data dependence). These constraints complicate finding effective solutions. We classify all general cases into four scenarios, as shown in Fig. 4 according to the main source of time consumption in each. For each case, we ease the complexity of the problem by focusing on certain constraints, thereby allowing the optimal solution to be obtained more straightforwardly. Specifically, (a) Case1: The AlltoAll communications are slower than intra-node communication and expert computations, but the inter-node communications (AlltoAll and Gradient-AllReduce) are not slower than intranode communication and expert computations. (b) Case2: Expert computations are not slower than inter-node communications and intra-node communications. (c) Case3: The AlltoAll communications are not slower than intra-node communication and expert computations. (d) Case4: The intra-node communications (AllGather and ReduceScatter) are not slower than inter-node communications and expert computations. In situations where multiple time-consuming factors are equally significant, they can be categorized into one of these cases. For instance, when the time consumption for inter-node communication equals that of computation, it can fall into either Case1 or Case2. Before discussing these scenarios, the paper formulates seven constraints characterizing these cases, presented as follows.

**Q1:** 
$$t_{a2a,r} > t_{aq,r}$$
.

**Q1 is True:** implies AlltoAll consumes more time than All-Gather for the chunked input. Assuming AllGather and ReduceScatter require similar durations, AlltoAll also exceeds ReduceScatter in time consumption.

**Q2:** 
$$r \cdot t_{exp,r} > 2(r-1) \cdot t_{a2a,r}$$
.

**Q2** is **True:** indicates that expert computations exceed the duration of communication tasks, excluding AlltoAll Dispatch for the first and AlltoAll Combine for the last chunk. When **Q1** is **True**, this also applies to AllGather and ReduceScatter for the first and last chunks, respectively.

Q3: 
$$r \cdot t_{exp,r} > (r-1) \cdot (t_{ag,r} + t_{rs,r})$$
.

**Q3 is True:** means that the time cost of expert computations is large enough to affect the time cost when **Q1 is False**.

**Q4:** 
$$t_{qar} > t_{aq,r} + t_{rs,r}$$
.

**Q4 is True:** means that the time cost of Gradient AllReduce is large enough to affect the time cost when **Q1 is True and Q2 is False**.

**Q5:** 
$$t_{qar} > r \cdot t_{exp,r} - 2(r-1) \cdot t_{a2a,r} + t_{aq,r} + t_{rs,r}$$
.

Q5 is True: means that the time cost of Gradient AllReduce is large enough to affect the time cost when Q1 is True and Q2 is True.

**Q6:** 
$$t_{gar} > r \cdot t_{aq,r} + r \cdot t_{rs,r} - 2(r-1) \cdot t_{a2a,r}$$
.

Q6 is True: means that the time cost of Gradient AllReduce is large enough to affect the time cost when Q1 is False and Q3 is False.

Q7: 
$$t_{gar} > t_{aq,r} + t_{rs,r} + r \cdot t_{exp,r} - 2(r-1) \cdot t_{a2a,r}$$
.

**Q7 is True:** means that the time cost of Gradient-AllReduce is large enough to affect the time cost when **Q1 is False and Q3 is True**.

With these constraints, four cases can be represented as follows:

1) Case 1: (Q1 is True, Q2 is False and Q4 is True) or (Q1 is True, Q2 is True and Q5 is True) or (Q1 is False, Q3 is False and Q6 is True) or (Q1 is False, Q3 is True and Q7 is True), which indicates that Gradient-AllReduce is large enough so that the inter-node communications (AlltoAll and Gradient-AllReduce) dominate the time cost in Fig. 4a. So we have

$$t_1^{moe} = 2r \cdot t_{a2a,r} + t_{aar} = 2r\alpha_{a2a} + 2n_{a2a}\beta_{a2a} + t_{aar}. \tag{2}$$

Therefore, to find its minima,  $t_1^*$ , we should solve

minimize: 
$$f_1(r) = t_1^{moe}$$
,  
s.t.  $r \ge 1$ ,  
 $(Q1 \land \neg Q2 \land Q4) \lor (Q1 \land Q2 \land Q5)$   
 $\lor (\neg Q1 \land \neg Q3 \land Q6) \lor (\neg Q1 \land Q3 \land Q7)$ .

2) **Case2:** (Q1 is True, Q2 is True and Q5 is False) or (Q1 is False, Q3 is True and Q7 is False), which indicates that Gradient-Allreduce is too small to influence the time cost and expert computation occupies a dominant position in Fig. 4b. So we have

$$\begin{split} t_{2}^{moe} &= 2t_{a2a,r} + t_{ag,r} + t_{rs,r} + r \cdot t_{exp,r} \\ &= 2\alpha_{a2a} + \frac{2n_{a2a}}{r}\beta_{a2a} + \alpha_{ag} + \frac{n_{ag}}{r}\beta_{ag} \\ &+ \alpha_{rs} + \frac{n_{rs}}{r}\beta_{rs} + r\alpha_{exp} + n_{exp}\beta_{exp}. \end{split}$$

Therefore, to find its minima,  $t_2^*$ , we should solve

minimize: 
$$f_2(r)=t_2^{moe}$$
,  
s.t.  $r\geq 1$ ,  
 $(O1 \wedge O2 \wedge \neg O5) \vee (\neg O1 \wedge O3 \wedge \neg O7)$ .

3) **Case3:** *Q1 is True, Q2 is False and Q4 is False*, which indicates that Gradient-Allreduce and expert computation are

### <span id="page-8-2"></span>Algorithm 1 FindOptimalPipelineDegree

Input:  $\alpha_{a2a}$ ,  $\beta_{a2a}$ ,  $n_{a2a}$ ,  $\alpha_{ag}$ ,  $\beta_{ag}$ ,  $n_{ag}$ ,  $\alpha_{rs}$ ,  $\beta_{rs}$ ,  $n_{rs}$ ,  $\alpha_{exp}$ ,  $\beta_{exp}$ ,  $n_{exp}$ ,  $t_{gar}$ Output: r and  $t^{moe}$ 1:  $r1, t1 = solve(f_1)$   $\Rightarrow$  Solve with SLSQP

2:  $r2, t2 = solve(f_2)$ 3:  $r3, t3 = solve(f_3)$ 4:  $r4, t4 = solve(f_4)$ 5: candidate\_mins = [t1, t2, t3, t4]6: candidates = [r1, r2, r3, r4]7:  $r = \text{candidates}[\text{argmin}(\text{candidate_mins})]$ 8:  $t^{moe} = min(\text{candidate_mins})$ 9: return r and  $t^{moe}$ .

too small to influence the time cost. The communications dominate the time cost. And AlltoAll also takes more time than AllGather and ReduceScatter on a chunked tensor in Fig. 4c. So we have

$$\begin{split} t_3^{moe} &= 2r \cdot t_{a2a,r} + t_{ag,r} + t_{rs,r} \\ &= 2r\alpha_{a2a} + 2n_{a2a}\beta_{a2a} + \alpha_{ag} + \frac{n_{ag}}{r}\beta_{ag} + \alpha_{rs} + \frac{n_{rs}}{r}\beta_{rs}. \end{split}$$

Therefore, to find its minima,  $t_3^*$ , we should solve

minimize: 
$$f_3(r) = t_3^{moe}$$
,  
s.t.  $r \ge 1$ ,  
 $Q1 \land \neg Q2 \land \neg Q4$ .

4) Case4: Q1 is False, Q3 is False and Q6 is False, which indicates that Gradient-Allreduce and expert computation are too small to influence the time cost. And AllGather and ReduceScatter also take more time than AlltoAll on a partitioned tensor. Intra-node communications dominate the time cost in Fig. 4d. So we have

$$\begin{split} t_4^{moe} &= 2t_{a2a,r} + r \cdot t_{ag,r} + r \cdot t_{rs,r} \\ &= 2\alpha_{a2a} + \frac{2n_{a2a}}{r}\beta_{a2a} + r\alpha_{ag} + n_{ag}\beta_{ag} + r\alpha_{rs} + n_{rs}\beta_{rs}. \end{split}$$

Therefore, to find its minima,  $t_4^*$ , we should solve

minimize: 
$$f_4(r) = t_4^{moe}$$
,  
s.t.  $r \ge 1$ ,  
 $\neg Q1 \land \neg Q3 \land \neg Q6$ .

#### 4.3 Algorithm

Algorithm 1 determines the optimal pipeline degree using MoE-related coefficients ( $n_{a2a}$ ,  $n_{ag}$ ,  $n_{rs}$ ,  $n_{exp}$ ) and cluster-related coefficients ( $\alpha_{a2a}$ ,  $\beta_{a2a}$ ,  $\alpha_{ag}$ ,  $\beta_{ag}$ ,  $\alpha_{rs}$ ,  $\beta_{rs}$ ,  $\alpha_{exp}$ ,  $\beta_{exp}$ ). In particular,  $t_{gar}$  is a manually entered value that is set to zero in the forward process and determined by §5 in the backward process. FSMoE supports varied pipeline degrees in both phases. The algorithm executes once before training, following the estimation of cluster-related coefficients. The "solve" function employs a sequential least squares programming (SLSQP) [32] solver. This algorithm is quadratic

convergence in solving  $f_1$ ,  $f_2$ ,  $f_3$  and  $f_4$  (Lines 1-4), and other operations take O(1) time complexity.

#### <span id="page-8-0"></span>4.4 Schedule Forward and Backward Separately

Because of the calculation of gradient w.r.t. the weight and the gradient synchronization among DP workers, the tasks in backpropagation are different from the forward phrase. The optimal pipeline degree thus differs. Therefore, we manually implement the backpropagation by storing the activation of each computational operation and computing the gradient.

Specifically, the parameters  $\alpha_{exp}$ ,  $\beta_{exp}$ , and  $n_{exp}$  in the backward phase are twice those in the forward phase to accommodate the derivatives of both weight and input. Meanwhile,  $t_{gar}$  is set to zero in the forward phase as gradient synchronization does not occur, and it is determined by the algorithm detailed in §5 for the backward phase.

