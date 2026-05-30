# VI. DESIGN OF AUM

This section describes the design considerations and technical details of AUM. It aims to precisely handle the threedimensional accelerator unit variations and fully unleash the efficiency potential of shared processors.

## *A. Design Considerations and Overview*

Given the deficiency of intuitive sharing managers, cooptimizing the performance and efficiency of AU-enabled CPU must consider the three-dimensional AUV:

- 1) *Variation-1: Usage Pattern.* AU usages are variable in different applications and operators, leading to execution performance variations. We need to select efficient AU usages with lightweight indicators and analyze real-time requirements for the following management.
- 2) *Variation-2: Frequency Interference.* Variable AU usages lead to dynamic frequency reductions, causing cascaded performance fluctuations. We need to divide the processor into regions to avoid frequency interference and adjust the regions to satisfy runtime requirements.

3) *Variation-3: Resource Bound.* Applications in different processor regions exhibit variable resource bounds, experiencing performance fluctuations. We need to study a unified model to guide resource tuning online to precisely harvest resources for optimal CPU efficiency.

To handle the complex AUV, we propose AUM, a novel AU-aware resource manager designed to maximize processor efficiency in shared environments. As shown in Figure [11,](#page-6-0) it focuses on system-layer management to harvest AU unexploited resources for shared applications precisely and flexibly. AUM contains two cooperative components to handle the threedimensional AUV offline and online. The *Background AU Profiler* characterizes and summarizes AUV into an offline reference model to guide the *Runtime AU Controller* for precise resource allocation with consideration of runtime status.

## *B. Background AU Profiler*

To portray the complex AUV, AUM selects three key variation indicators based on the analysis above. Firstly, it judges application AU usage via arithmetic intensity (ARI) to categorize operators. Secondly, it divides the processor into regions with different AU usages to assess frequency reductions. Thirdly, it considers specific AU resource bounds via profiling minimal demands. Overall, the three-dimensional information is summarized into the *AUV Model*.

- *1) Usage-aware AU Selecting:* To capture the variable AU usage in different applications, we use ARI to determine the proper AU for different operators. Based on previous analysis [\[36\]](#page-13-6), [\[37\]](#page-13-8), we can calculate the ARI of underlying operations of AU applications, such as QKV mapping with 6(1/d + 3/BL) −1 in the prefill phase and 6(1/d + 3/B) −1 in the decode phase. With larger model dimension d, batch size B, and input length L, operations with higher ARI have higher AU usages, denoted as UAU . UAU captures the variable AU usages analyzed in Section [IV-A.](#page-3-3)
- *2) Frequency-aware Processor Dividing:* To properly manage the frequency interference, *AU-Man* recognizes the compulsory frequency reduction due to variable AU usage and divides the processor cores into three regions based on Section [IV-B.](#page-4-2) (1) High-AU region C<sup>H</sup> with high AU usages and

<span id="page-7-0"></span>![](_page_7_Figure_0.jpeg)

Fig. 12: AU applications vary with processor dividing. All the results are normalized to exclusive performance on all cores.

<span id="page-7-1"></span>![](_page_7_Figure_2.jpeg)

Fig. 13: AU application with different usages varies with last-level cache (LLC) resource allocation. All the results are normalized to performance with all LLC ways.

low frequency  $F_H$ , such as generating prefill tokens. (2) Low-AU region  $C_L$  with low AU usages and moderate frequency  $F_L$ , such as generating decode tokens. (3) None-AU region  $C_N$  with no AU usage and high frequency  $F_N$ , only for shared applications. The  $U_{AU}$  threshold is set based on server-level AU usage distributions. As shown in Figure 12, AU-Man records the AU performance and the frequency lower bounds under different region divisions. The frequency profiles are recorded as processor regions C and frequencies F with  $[C_H, F_H, C_L, F_L, C_N, F_N]$ .

3) Bound-aware Resource Profiling: To capture the distinct resource bounds and provide just the right resources for AU, we record changing resource affinities of variable AU usages based on Section IV-C. For L2-cache and LLC capacity, as well as memory bandwidth, we can profile its variation using resource partitioning interfaces [27], such as Cache Allocation Technology (CAT) for cache ways and Memory Bandwidth Allocation (MBA) for memory bandwidth. As shown in Figure 13, we find that for LLC resources, varying AU usages and underlying platforms induce diverse affinity, showing that we can harvest LLC resources for low-AU operators and high-AU operators on GenA. We can profile the minimal resource demands as a three-tuple  $R_{AU}$ .

Overall, the three-dimensional AU profiles are recorded into a *AUV Model* with consideration of AU application performance and processor power consumption, as shown in

<span id="page-7-2"></span>TABLE III: An example bucket in the AUV Model.

| $U_{AU}$ | $C_{AU}$ | $F_{AU}$ | $R_{L2C}$ | $R_{LLC}$ | $R_{BW}$ | $P^a$ | $P^t$ |
|----------|----------|----------|-----------|-----------|----------|-------|-------|
| High     | 0-11     | 2.1 GHz  | 0-2       | 0-1       | 50%      | 0.42  | 0.31  |
| Low      | 12-15    | 2.8 GHz  | 3-6       | 2-4       | 40%      | 9.12  | 7.19  |
| None     | 16-23    | 3.2 GHz  | 7-15      | 5-15      | 10%      | 13.28 | 9.16  |

Table III. For three frequency regions with variable AU usages, we allocate varying resources and record the performance of high AU application  $P_H$ , low AU application  $P_L$ , and shared application  $P_N$ . To reduce profiling costs, we design the AU Bucket mechanism to discretize the continuous variations. For high/low/none AU usages, we profile three processor divisions with five performance-sensitive resource configurations. For every bucket, we record the 50% average performance  $P^a$ , 90% tail performance  $P^t$ , and processor power consumption  $W_{CPU}$ . The variation profiles guide the runtime AU controller.

#### C. Runtime AU Controller

To perform AU-aware processor management, AUM jointly considers *AUV Model* and runtime information to make adaptive decisions as shown in Algorithm 1. Firstly, it analyzes AU performance to update requirements. Secondly, it selects sharing decisions with maximized processor efficiency. Thirdly, it adjusts unexploited resource harvesting considering AU interference. Overall, the processor resource is properly shared to guarantee AU performance and maximize overall efficiency.

1) Slack-aware SLO Analyzer: To determine the varying AU SLO, AUM computes performance slacks for different AU usages. Firstly, for prefill tokens with high AU usages, better performance improves user satisfaction, so we simply use first-come-first-served (FCFS) to schedule prompts [57]. The runtime SLO for prefill tokens  $SLO_H$  is set as  $d_{TTFT}-t_{wait}$ , where  $d_{TTFT}$  is the TTFT SLO and  $t_{wait}$  is the request waiting time. Secondly, for abundant decode tokens with low AU usages, we track the performance of tokens and adapt to varying request arrival rates at runtime with LAG analysis (a measurement of how far behind the token is compared to an ideal schedule that meets the performance requirements).

We quantify the relationship between the partial execution time at time t of serving request  $i(e_i)$  and its relative deadline, denoted as  $LAG_i$ , as shown in Algorithm 1-Line 3.  $T_i(t)$  is the tokens of request i that have completed by time t. For token  $token \in T_i(t), d_{TPOT}$  is set as the TPOT SLO, and  $e_{token}$  is the recorded execution time for token t, respectively.  $LAG_i$ reflects the real-time status of serving request i and quantifies how far ahead or behind the serving request is compared to the deadline at time t. If every LAG is 0, AU applications are allocated precise resources. The AU application is perfect if every LAG within it is 0, which means all tokens have exactly finished by their deadline so far and AU application does not need more resources. The runtime SLO for decode tokens  $SLO_L$  is set as  $d_{TPOT} + LAG_i$ . Since LAG indicates how far behind (LAG < 0) or ahead (LAG > 0) every AU-accelerated request is, the AU configurations need to be adjusted accordingly for faster and slower execution. AUM obtains the runtime performance requirements in this stage.

Algorithm 1: Workflow of the runtime AU controller.

```
Input: Reference AUV Model M
   Output: AU-aware Resource Sharing Decision
   // Slack-aware SLO analysis
1 SLO_H = d_{TTFT} - t_{wait};
2 SLO_L = d_{TPOT} + LAG_i;
3 LAG_i(token, T_i(t)) = \sum_{token \in T_i(t)} (d_{TPOT} - e_{token});
   // Efficiency-aware Core Switcher
4 E_{CPU} = (\alpha \times P_H + \beta \times P_L + \gamma \times P_N)/W_{CPU};
5 Maximize E_{CPU} s.t. P_H^t < SLO_H and P_L^t < d_{TPOT};
6 U/C/F \leftarrow M(P_H, P_L);
   // Collision-aware Allocation Tuner
7 Continuously monitor AU performance P^m;
8 if P_H^m < SLO_H and P_L^m < SLO_L then
     \delta_{AU} \leftarrow \sum U_{AU} \times SLO/P^m;
      R_{AU} \leftarrow M(P_H^a, P_L^a);
11 end
12 else
      \delta_{AU} \leftarrow \sum U_{AU} \times P^m/SLO;
    R_{AU} \leftarrow M(P_H^t, P_L^t);
15 end
16 if \delta_{AU} > threshold then
  \mid C/F \leftarrow M(\delta_{AU}, P_{AU}, C_{AU}, F_{AU})
18 end
```

- 2) Efficiency-aware Core Switcher: To optimize processor efficiency with varying SLO, we switch processor core configurations with consideration of weighted efficiency. The processor performance-per-watt efficiency  $E_{CPU}$  is computed as the weighted sum of application performance divided by CPU power consumption, as shown in Algorithm 1-Line 4. The prices of application outputs are used to normalize their performance in different regions as  $\alpha$ ,  $\beta$ , and  $\gamma$ . The shared applications are continuously running in the background and their pressures are proportional to the allocated cores  $C_N/F_N$ . For lower management complexity, AUM switches processor cores to different frequency regions to maximize the weighted efficiency and satisfy diverse SLO primarily, as shown in Algorithm 1-Lines 5, 6. The frequency of every region is set as the maximal level below the TDP. Admittedly, finegrained frequency control schemes [66], [77] could further improve processor efficiency via per-core or per-workload power capping. But it would significantly enlarge the optimization space, and our rule-based controller needs to integrate intelligent algorithms to make decisions [103], which leaves as our future work. The processor division and shared application are relatively stable for the resource allocation tuning. AUM decides the processor placement in this stage.
- 3) Collision-aware Allocation Tuner: To avoid dramatic AU performance degradation, AUM tunes resource allocation considering the collision between AU and shared applications. For every control iteration, AUM uses a continuous and lightweight indicator to detect AU application performance like token latency. If the measured AU performance  $P^m$

<span id="page-8-2"></span>TABLE IV: Evaluated AU usage scenarios with different prefill/decode SLOs and average input/output lengths.

| Apps | Dataset        | $d_{TTFT}$ | $d_{TPOT}$ | Input | Output |
|------|----------------|------------|------------|-------|--------|
| cb   | ShareGPT [64]  | 250 ms     | 100 ms     | 755   | 200    |
| сс   | HumanEval [10] | 75 ms      | 150 ms     | 171   | 98     |
| sm   | LongBench [6]  | 1.5 s      | 100 ms     | 1738  | 91     |

guarantees runtime SLO, we can aggressively harvest AU resources for shared applications, using average performance  $P_{AU}^a$  to tune resource allocations. Otherwise, we need to conservatively return resources to AU applications using tail performance  $P_{AU}^t$  to control. Given the varying AU resource affinities, hardware resource that causes minimal AU performance degradation are harvested first. The allocation tuner needs to be refined by considering SLO guarantee of corunning applications under the best-effort LLM serving scenarios. Moreover, we compute the deviation  $\delta_{AU}$  to denote the performance gap and higher AU usages result in greater deviations to be eliminated as shown in Algorithm 1-Lines 9, 13. If the deviation  $\delta_{AU}$  exceeds the threshold, tuning AU resources is not sufficient, and we need to switch the processor division as shown in Algorithm 1-Line 17.

#### VII. EVALUATION OF AU-MAN

<span id="page-8-0"></span>Our evaluation wants to answer three questions: **1.** How does AUM improve CPU overall efficiency? (Section VII-B) **2.** How does AUM guarantee AU application performance given varying requirements? (Section VII-C) **3.** How are the costs and revenues to deploy AUM on AU-enabled CPU? (Sections VII-D and VII-E)

## A. Evaluation Methodology

- 1) Implementations: We implement a prototype of AUM based on xFasterTransformer [21] with two components implemented in Python, acting as a system component. The background profiler records the essential information of newer models with repeated experiments on dedicated nodes. The runtime controller works as a system daemon to monitor the SLO and tune the allocation in production. For hyperparameters of AUM, we select the performance prices  $\alpha$  as 1.8 for high-AU prefill tokens and  $\beta$  as 0.2 for low-AU decode tokens. Different from GPU-based token prices [63], we decide the prices based on CPU time to produce a prefill and decode token.  $\gamma$  for none-AU Compute, OLAP, and SPECibb is set as 1e-3, 1e-6, and 3e-5, respectively. The prices are decided based on CPU time to produce one query on the evaluated platform. These parameters are decided empirically and we conduct a sensitivity experiment to evaluate AUM under different scenarios. We set the derivation  $\delta_{AU}$  threshold as 2 to denote performance collision.
- 2) Workloads: For hardware platforms, AUM experiments are mainly conducted on the SPR platform (GenA) as shown in Table I if not otherwise specified. For evaluated workloads, the evaluated AU applications are LLM serving of llama models [51] similar to Section III-A, and the co-running applications are Compute [39], OLAP [86], and SPECjbb [75]

<span id="page-9-2"></span>![](_page_9_Figure_0.jpeg)

Fig. 14: Comparison of CPU performance-per-watt efficiency with variable AU application scenarios and sharing selections. All the results are normalized to ALL-AU under chatbot scenario. AUM outperforms SOTA sharing baselines by 4.7%.

<span id="page-9-1"></span>TABLE V: Three categories of evaluated baselines.

| Category             | Scheme | Description                       |
|----------------------|--------|-----------------------------------|
| AU-exclusive         | ALL-AU | Utilizing AU CPU without sharing  |
| AUV-oblivious        | SMT-AU | SMT sharing AU CPU                |
| Sharing              | RP-AU  | Partition resources of AU CPU     |
| AU-aware             | AU-UP  | Sharing w/ usage pattern          |
| Resource<br>Managers | AU-FI  | Sharing w/ frequency interference |
|                      | AU-RB  | Sharing w/ resource bound         |
| Managers             | AUM    | Our three-dimensional proposal    |

similar to Section V-A. We evaluate different AU applications use-cases [102] as shown in Table IV (1) ChatGPT-like chatbot (*cb*) [62]; (2) Cursor-like code completion (*cc*) [12]; (3) Summarization (*sm*) [43]. The benchmark selection is similar as previous works [102] and represents datacenter scenarios.

3) Baselines: To understand the performance of AUM, we compare it with three types of baselines as shown in Table V. Firstly, AU-exclusive adopts non-sharing settings and uses the whole AU-enabled processor for LLM serving. Secondly, AUV-oblivious Sharing adopts state-of-the-art resource managers for shared CPU platforms [11], [69]. More specifically, SMT-AU adopts SMT sharing [69] and RP-AU adopts workload-aware resource partitioning [11] for LLM serving and co-located workloads. Thirdly, AU-aware Resource Managers are variants of AUM to investigate the effect of three-dimensional AU awareness.

