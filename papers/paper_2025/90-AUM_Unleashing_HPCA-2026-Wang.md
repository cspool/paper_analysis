2026 IEEE International Symposium on High-Performance Computer Architecture (HPCA) 

## AUM: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving 

Xinkai Wang[1] , Chao Li[1] _†_ , Yiming Zhuansun[1] , Jinyang Guo[1] , Xiaofeng Hou[1] , Jing Wang[1] , Luping Wang[2] _†_ , Weigao Chen[2] , Cheng Huang[2] , Guodong Yang[2] , Liping Zhang[2] and Minyi Guo[1] 1. School of Computer Science, Shanghai Jiao Tong University, 2. Alibaba Group, _†_ Corresponding Authors 

_{_ unbreakablewxk, zsym2019, lazarus, jing618 _}_ @sjtu.edu.cn, _{_ lichao, hou-xf, guo-my _}_ @cs.sjtu.edu.cn 

_{_ chamu.wlp, weigao.cwg, xiaoluo.hc, luren.ygd, liping.z _}_ @alibaba-inc.com 

_**Abstract**_ **—Generative AI, especially LLM, is driving a fundamental shift in software paradigms, prompting cloud providers to build more efficient serving infrastructures. To meet the computational demands of emerging software, modern CPU processors are integrating Accelerator Units (AU) in the pipeline to accelerate key operations, such as Intel AMX for matrix multiplication. Current practices that dedicate AU-enabled CPU exclusively to LLM serving lead to significant resource waste and inferior efficiency. To this end, sharing AU-enabled CPU with general workloads is necessary to harvest redundant resources and improve platform performance-per-watt. However, perfectly sharing AU can be challenging since they introduce three-dimensional variations: variable usage patterns, compulsory frequency interferences, and dissimilar resource bounds. Existing resource managers are oblivious to complex Accelerator Unit Variations (AUV), resulting in performance and efficiency degradations of up to 50% in shared environments.** 

**Therefore, this paper introduces AUM, a novel AU-aware resource manager designed to handle AUV and maximize the efficiency of shared processors. AUM has two cooperative components with three stages for three-dimensional AUV. The background profiler characterizes the usage, frequency, and resource information into a discrete model, guiding the runtime controller to analyze usage-aware requirements, select frequency-aware divisions, and make bound-aware resource decisions. Through extensive evaluations on production AU-enabled CPUs, we show that AUM improves CPU efficiency by 4.7-8.8% while maintaining high-performance AU applications by reducing SLO violations by 7-11% compared with state-of-the-art resource managers.** 

## I. INTRODUCTION 

Recent advancements in Large Language Models (LLMs), such as GPT [62] and LLaMA [51], are fundamentally transforming previous software across various domains, from personal assistants [100] to search engines [54]. This paradigm shift imposes immense pressure on cloud providers to develop cost-efficient serving infrastructure capable of handling billions of user queries [67]. However, the high costs and constrained supply of top-performing accelerators like NVIDIA GPUs [61] and self-designed accelerators [52] have forced major cloud providers, such as Amazon [13], Azure [32], and Inspur [80], to increasingly leverage ubiquitous generalpurpose CPUs for LLM serving [21], [36]. 

To meet the computational demands of AI-driven software, modern general-purpose CPUs are incorporating specialized Accelerator Units (AU) in the execution pipeline, which are 

**==> picture [253 x 143] intentionally omitted <==**

**----- Start of picture text -----**<br>
①Changing Software Requirements<br>Evolving  Text Media Bigdata AI LLM …<br>Application<br>④AUV-oblivious Middleware<br>Management<br>Resource  Gap of AUV<br>Manager Exclusive SMT FT RDT BP<br>③Compiler Support for Usage AUM<br>ISA<br>8086 MMX SSE AVX AMX …<br>Extensions<br>②Variable Hardware Accelerator Units (AU)<br>CPU  ALU/FPU Vec. ALU/FPU FMA TMUL …<br>Func. Units<br>Timeline 1980s 1990s 2000s 2010s 2020s 2030s<br>**----- End of picture text -----**<br>


Fig. 1: The management gap between evolving AU and AU variations (AUV)-oblivious resource managers. 

designed to accelerate key operations, such as Intel AMX[1] [59] and RISC-V M-extension [2] for heavy matrix multiplications in LLM serving [37]. Enabled by dedicated Instruction Set Architecture (ISA) extensions for every emerging AU [24], [70], software applications can easily harness AU capabilities to achieve significant performance gains, as shown in Figure 1. However, CPU execution becomes more heterogeneous due to newly-designed AU alongside conventional functional units, such as integer ALU and vector units [70]. To sidestep management complexities and guarantee AU application performance, current industry practice is to utilize AU-enabled CPU exclusively [41], [80] (i.e., allocating the whole processor to AU applications without sharing), leading to severe resource waste and poor platform efficiency. 

Given the inefficiency of exclusive allocation, sharing AUenabled CPU is essential for maximizing platform efficiency, but effective sharing is non-trivial due to variable behaviors of AU at three abstraction layers, which we term Accelerator Unit Variations ( **AUV** ). _Variation-1: Variable Usage Pattern._ Software-layer AU utilization is highly dynamic in different applications and operators. For instance, two LLM serving phases (i.e., prefill and decode [102]) have distinct AU usages due to different matrix operation dimensions. _Variation2: Compulsory Frequency Interference._ Variable AU usages 

> 1In this paper, we use Intel Advanced Matrix Extension (AMX) for matrix multiplication to represent one of the emerging accelerator units. 

cause compulsory system-layer frequency reduction to respect thermal design power (TDP) limits. The dynamic frequency scaling creates unpredictable performance interferences and makes it difficult to guarantee sharing service-level objectives (SLO). _Variation-3: Dissimilar Resource Bound._ AU exhibits microarchitecture-layer resource affinities distinct from traditional functional units. The imbalanced processor frontend and backend resource bounds lead to underutilized hardware components in different cycles. The three-dimensional AUV are challenging for guaranteeing performance and maximizing efficiency in shared environments. 

However, conventional resource managers fail to handle AUV for contradictory performance and efficiency objectives. To utilize the redundant resources, CPU platforms incorporate workload-aware mechanisms as shown in Figure 1, including Simultaneous Multi-Threading (SMT) variants, such as Frontend Throttling (FT) [14] and Backend Partition (BP) [50], and resource partitioning mechanisms, such as Intel Resource Director Technology (RDT) [11], [27]. We find that simply applying current AUV-oblivious resource sharing managers is unacceptable since they cannot make precise decisions for variable AU, causing 10-50% AU application performance and CPU efficiency degradations. Therefore, we envision a critical management gap between more variable AU and AUV-oblivious resource managers, failing to maximize AU capability and CPU efficiency. 

To bridge this management gap, we propose AUM, a novel **AU** -aware resource **M** anager designed to handle AUV and maximize CPU efficiency in shared environments. More specifically, AUM offers higher manageability with two cooperative components, and every component has three stages for three-dimensional AUV. The _Background AU Profiler_ synthesizes AUV offline from usage patterns, frequency reductions, and resource bounds to obtain the discrete _AUV Model_ . Guided by this model and runtime information, the _Runtime AU Controller_ decides the optimal AU configurations online, including usage-aware performance slacks, frequency-aware processor divisions, and collision-aware resource allocations. Overall, AUM selects AU sharing configurations that maximize CPU efficiency and minimize AU performance slowdowns. 

To evaluate AUM, we have implemented a prototype and evaluated it with various workloads on three production CPU platforms. Evaluation results show that AUM achieves better CPU performance-per-watt efficiency by up to 8.8% and 4.7% compared with state-of-the-art (SOTA) exclusive AU usages and AUV-oblivious sharing methods, respectively. Meanwhile, it maintains high-performance AU applications by reducing SLO violations by up to 11%. AUM induces negligible management overheads and achieves better total cost of ownership, enabling more efficient general-purpose processor management for the AI era. 

In summary, this paper makes the following contributions: 

- **Analysis** : We envision the complex accelerator unit variations and characterize them from three abstraction layers. We show that both AU-exclusive usages and AUV- 

**==> picture [253 x 112] intentionally omitted <==**

**----- Start of picture text -----**<br>
L1-Inst ROB Registers … ALU … AVX Accelerator<br>Port5 Units (AU)<br>Fetch Rename Scheduler ALU FPU …<br>Port1<br>Registers ALU Vec. DIV … AMX<br>Decoder IDQ Port0<br>Physical L1/L2  HT 0 Physical TMUL Accelerator TILECFG<br>Core 0 Cache HT 1 Core 1 PE PE … PE 2D reg0<br>Last-level Cache DSA Accelerator PE PE … PE 2D reg1<br>… …<br>1024 BF16 ops/cycle 8 * 1KB<br>NUMA Memory 0<br>**----- End of picture text -----**<br>


Fig. 2: The block diagram of modern CPU with accelerator units and the implementation of typical AMX. 

   - oblivious sharing resource managers cannot achieve highperformance and efficient processors. 

- **Design** : We introduce AUM, a novel AU-aware resource manager tailored for AU-enabled processors in shared environments. We design two cooperative components to pursue optimal efficiency-performance trade-offs. 

- **Evaluation** : We implement and evaluate AUM thoroughly on production CPU platforms to prove its better performance-per-watt efficiency and performance guarantee compared to SOTA resource managers. 

## II. BACKGROUND 

This section introduces current accelerator units and a prevailing AU-accelerated application, LLM serving. 

## _A. CPU Accelerator Units_ 

To improve the performance of artificial intelligence (AI) applications on CPU platforms, hardware vendors have integrated specialized functional units into recent processors, such as Intel Advanced Matrix Extensions (AMX) [59], ARM Scalable Matrix Extensions (SME) [5], and RISC-V M-extension [2]. They follow the Single-Instruction-MultipleData (SIMD) paradigm to accelerate operations like matrix multiplication. Different from dedicated accelerators (e.g., Data Streaming Accelerator, DSA [40]) outside the CPU pipeline, these accelerator units (AU) on every physical core share the instruction flow, microarchitectural resources, and data access path with normal functional units [24]. The layout of modern processors containing Intel AMX and AVX units is shown in Figure 2, which are the targeted AU in this paper with relatively mature software support. Every AMX unit contains an array of eight 2-dimensional registers (TILECFG) with the size of 1KB and a matrix multiply accelerator (TMUL) that performs 1024 ops per cycle with BF16 precision [25]. The matrix multiplications via TMUL compute _C_ [ _M_ ][ _N_ ]+ = _A_ [ _M_ ][ _K_ ] _∗ B_ [ _K_ ][ _N_ ] with _M ≤_ 16 and _N ≤_ 64. AMX is introduced from SPR [59] with BF16 precision and the following generations add support for more data types like FP16 in GNR [28] and FP8 in DMR [31]. 

## _B. AU-accelerated LLM Serving_ 

Large language models (LLM) have shown impressive achievements and great potential in various creative tasks, 

**==> picture [253 x 129] intentionally omitted <==**

**----- Start of picture text -----**<br>
Layers Blocks Prefill H W [T] Decode H W [T]<br>Decoding Feed Forward<br>GEMM AMX GEMV AVX<br>Transformer Add & Norm<br>… Matrix Operations<br>Transformer Multi-head Prefill Q K [T] Decode Q K [T]<br>Attention<br>Embedding GEMM AMX GEMV AVX<br>LLM Iteration 1 LLM Iteration 2 LLM Iteration 3 Tokens<br>I love  KV  Token-1: KV  Token-2: Token-3:<br>reading  Cache HPCA Cache Papers EOS<br>Prefill Phase Decode Phase<br>**----- End of picture text -----**<br>


TABLE I: Hardware specifications of evaluated CPUs. 

||Platforms|**GenA**|**GenB**|**GenC**|
|---|---|---|---|---|
||_Generation_<br>_CPU_|Sapphire Rapids<br>Xeon 8475B|Sapphire Rapids<br>Xeon Max 9468|Granite Rapids<br>Xeon 6982P-C|
||_# cores/sockets_<br>_AU TFLOPS_<br>_(AVX-512/AMX)_|48 / 2<br>25.6/206.4|48 / 2<br>25.6/206.4|120 / 1<br>32/344|
||_Base Frequency_|2.7 GHz|2.1 GHz|2.8 GHz|
||_L1-I / core_|32 KB|32 KB|64 KB|
||_L1-D / core_|48 KB|48 KB|48 KB|
||_L2 / core_|2 MB|2 MB|2 MB|
||_LLC / socket_|97.5 MB|105 MB|504 MB|
||_Memory_<br>_Memory BW_|DDR5 1TB<br>233.8 GB/s|HBM 128GB<br>588 GB/s|MCR 768GB<br>600 GB/s|



Fig. 3: LLM serving with AU-accelerated matrix operations. 

leading the boom of the AI era [51], [62], [99]. Besides once-for-all heavyweight LLM training, optimizing the LLM serving (i.e., inference) efficiency is gaining more attention in both academia and industry [42], [60]. LLM serving contains two phases: (1) The prefill phase processes all tokens of the input prompt simultaneously to generate the first new token. (2) The decode phase uses the previously generated token as input to produce the following tokens one by one until an end-of-sequence (EOS) token is produced [37]. Two phases are relatively independent and have different characteristics [67], [102]. As shown in Figure 3, LLM iteration passes through repeated transformer layers composed of multiple blocks like multi-head attention to generate every subsequent token. The heavy blocks are composed of matrix operations of different sizes, depending on phases, batch sizes, and input lengths [36]. They can be accelerated by AU-enabled CPU [57] but the most efficient AU choices are changing with matrix dimensions [41]. Many LLM serving frameworks are exploring AU utilization to improve serving performance, such as Ktransformer [41] and xFasterTransformer [33]. 

## III. DEFICIENCY OF EXCLUSIVE AU UTILIZATION 

AU acceleration of diverse workloads is promising, but current exclusive AU usages fall short compared with GPUs in efficiency, calling for shared deployment to fully unleash the efficiency potential of AU-enabled processors. 

## _A. Characterization Methodology_ 

_1) Hardware:_ In Sections III, IV, V, and VII, we evaluate three commercial off-the-shelf AU-enabled CPU platforms with different memory devices: Intel 4th Sapphire Rapids (SPR) [59] released in 2022, with DDR5 memory and HighBandwidth Memory (HBM), as well as 6th Granite Rapids (GNR) [28] released in 2024 with Multiplexer Combined Ranks (MCR) memory. The hardware specifications are shown in Table I. The three platforms are equipped with AVX and AMX units in every physical core, and we calculate AU TFLOPS based on their base frequencies. The main difference among GenA, GenB, and GenC is memory configurations, and GenC further improves AU. 

|0<br>4<br>8<br>12<br>16<br>Speedup w/ AFU||||||
|---|---|---|---|---|---|
||_AU accelerate diverse AI-_||||1398|
||7.5x<br>_based datacenter workloads._||||.x|
||1.69x<br>2.32x<br>2.5x<br>3.91x||||1.07<br>1.22<br>1.29|
||d=256 d=1024 d=2048<br>c=4<br>c=8<br>c=16<br>bs=512 bs=2048bs=8192<br>Faiss<br>Vocoder<br>DeepFM<br>AU-accelerated workloads|||||
|||||||



Fig. 4: AU acceleration of three types of AI workloads under different dimension (d), cores (c), and batch sizes (bs). All the results are normalized to AU-disabled GenC performance. 

_2) Software:_ To accelerate LLM serving with AU, we use Intel xFasterTransformer ( _xft_ ) [21] framework with the latest Intel AMX support. We use _xft_ to serve the open-sourced llama2-7b and llama2-13b [51] with BF16 precision and batch size of 16. To study the cascaded yet different serving phases, prefill and decode [67], [102], we mimic them in _xft_ with varying output length. We use Linux perf [19], pmu-tools [38], and pqos [26] tools to characterize LLM and AU behaviors. Meanwhile, we select various metrics widely used in previous studies [56], [102]. The SLO of LLM serving are time-tofirst-token (TTFT) for the prefill phase, indicating the time to generate the first token, and time-per-output-token (TPOT) for the decode phase, indicating the average time taken to generate subsequent tokens. The performance of LLM serving is measured via tokens per second (Throughput), indicating the generated tokens with performance guarantees. 

## _B. Pros and Cons of Current AU Utilization_ 

Despite LLM serving [57], AU has proven its strengths in diverse AI-based datacenter workloads as shown in Figure 4. Faiss vector search (Faiss) [53], speech vocoder (Vocoder) [23], and deepfm recommendation (DeepFM) [20] are greatly accelerated with AU capabilities under different parameters on the GenC platform. Admittedly, GPU-based datacenters are the main trend in the AI era. But CPU is still worth optimizing for two reasons. Firstly, GPU is inadequate for the booming demand of AI-enabled applications with up to 40% shortage [3], thus utilizing AU to accommodate selective AI workloads could free up the limited GPU for higherpriority workloads such as Azure and AWS [13], [32], [80] and improve cluster sustainability [44]. Secondly, even in a GPU-centric datacenter such as Alibaba Cloud, there are also 

**==> picture [253 x 84] intentionally omitted <==**

**----- Start of picture text -----**<br>
GenA GenB GenC GPU-A100<br>2.8 2.7x 2.4x Exclusively utilizing AU is inefficient<br>2.4 compared with GPU.<br>1.62 1.6x 1.2x 1.8x 1.3x1.5x 2.1x 1.2x1.6x1.3x<br>1.2 1x 1x 1.1x 1x 1<br>0.8<br>Normalized Metrics Perf.: Tokens/s Power: Watt Perf-Per-Watt Perf-Per-Dollar<br>**----- End of picture text -----**<br>


Fig. 5: Inferior performance and efficiency of exclusive AUenabled CPU compared with GPU solutions. 

TABLE II: Comparison of LLM of different architectures and sizes. BB is backend bound and DB is dram bound. Every value is denoted as percentages of Prefill / Decode phases. 

|**Model**<br>**Phi3**[83]|**Size**<br>38B|**Cycle Ratio**<br>178/19|_µ_**op Ratio**<br>41/09|**BB**<br>84/89|**DB**<br>21/53|
|---|---|---|---|---|---|
|**-** <br>**Llama2** [51]<br>**Llama3** [51]|.<br>7B<br>8B|.  .<br>14.4 / 1.5<br>13.1 / 1.3|.  .<br>3.7 / 0.5<br>3.5 / 0.5|<br>92 / 96<br>91 / 96|<br>24 / 59<br>24 / 60|
|**Gemma2** [82]|9B|13.3 / 1.4|3.3 / 0.4|92 / 96|25 / 62|
|**Llama2** [51]|14B|10.9 / 1.2|2.5 / 0.4|94 / 97|29 / 68|
|**Qwen3-A3B** [84]|30B|18.2 / 2.3|4.5 / 1.1|82 / 90|21 / 51|



## _A. Variation-1: Variable Usage Pattern_ 

around 50% idle CPU cores with accelerator units that can be better utilized besides its cooperation with GPUs [95]. Current industrial providers concentrate on the AU application performance and allocate the whole CPU exclusively to avoid management complexity and hardware contention [41], [80]. 

Admittedly, exclusive AU usages without any sharing can maximize its performance, but it causes both low CPU efficiency and waste of redundant hardware resources. Given the enhanced AU on the CPU platform, the improved LLM serving performance is still insufficient compared with highend GPUs like Nvidia A100, as shown in Figure 5. We compare with a single-GPU server driven by FlexGen [57], [73] on performance, power, and cost of the processing units (1 CPU vs. 1 GPU). The absolute performance, power, and cost of GenA are 188 tokens/s, 270 Watt, and $ 7200, respectively. We find that GPU is even more efficient considering the power consumption, with a 2.1x better performance-perwatt compared with GenA and 1.4x better than GenC. But the performance-per-cost of single-GPU servers is worse than that of the high-end CPU platforms. The performance gap would be bigger with model parallelism [74] while the power consumption would be significantly higher for multi-GPU servers and larger models (5kW vs. 500W), which are in line with prior efficiency studies [66], [78]. It shows that utilizing AU exclusively is not an efficient alternative to GPUs in large-scale datacenters. Moreover, exclusive AU usages cause a great waste of hardware resources. Applications with variable AU usages have distinct resource demands and leave redundant processor resources underutilized, which requires proper sharing methods to deploy more general applications and improve overall efficiency [11], [101]. 

Considering the performance pros and efficiency cons of exclusive AU utilization, sharing processors between AU and general workloads is both necessary and worthwhile for more cost-efficient AU-enabled processor management. 

## IV. UNDERSTANDING ACCELERATOR UNIT VARIATIONS 

To better utilize and manage the emerging AU, we need to understand their behaviors and differences. This section systematically characterizes LLM serving with the modern AU, Intel AMX, and demonstrates the three-dimensional AUV that hinder the intuitive sharing management. 

Firstly, AU is used unequally in applications and operators, laying the foundation of AUV. We analyze the variable AU usage patterns to find the causes and proxies. 

_1) AU usage analysis in LLM serving:_ AU is variably utilized in current AI applications. We observe three practical metrics on CPU to characterize AU usage in applications [85]. Firstly, _AMX cycle ratio_ , measured via _tma amx busy_ metric, indicates the proportion of cycles during which AMX is busy with arithmetic operations. We find that AMX usage is varying: the compute-intensive prefill phase (14.4%) is higher than the memory-intensive decode phase (1.5%), and the high-bandwidth platforms (GenB and GenC) are higher. Secondly, _AMX µop ratio_ , measured via _tma fp amx_ divided by _tma fp arith_ , shows the proportion of floating-point operations the AMX has finished. We find that FP operations of prefill are mostly finished by AMX (3.7% / 3.8%), but the decode phase uses AMX for FP operations less (0.5% / 1.5%). Thirdly, the _avx insts_ metric of the decode phase is higher because the vector-size operations are more efficient using AVX rather than AMX [41], showing different AU choices for varying-size matrix multiplications. 

_2) Analysis of LLM Architecture:_ We analyze more LLM of different architectures and sizes: Phi-3-Mini-128K-Instruct of 3.8B size [83], Llama3 of 8B size [51], Gemma2 of 9B size [82], and Qwen3-30B-A3B of 30B size and Mixture-ofExperts (MoE) architecture [84]. Also, larger models would be restricted by the CPU memory bandwidth greatly, such that they are more suitable for CPU-GPU hybrid deployment [36], [41]. We mainly study the AU usage patterns and key backend resource bounds of the six models as shown in Table II. We find that smaller models have higher AU usages and MoE architecture is suitable for AU deployment as well. The memory bound is more vital for larger models, but sparse expert activation of the MoE architecture can relieve the memory pressure. We choose the small-size models for the following experiments since they are suitable for CPU-only serving without GPU requirements [30], and larger LLMs behave similarly on AU usage patterns [57]. 

_3) Analysis of AU usage variations:_ The AU usage differences mainly result from variable operator dimensions. The performance of GEMM operations achieves 40 _._ 57 TFLOPS in the prefill phase but 6 _._ 87 TFLOPS in the decode phase due to variations in matrix dimensions. Most GEMMs in the prefill phase are 8192 _×_ 4096 _×_ 22016, where 8192 denotes 

**==> picture [253 x 105] intentionally omitted <==**

**----- Start of picture text -----**<br>
P refill with AMX D ecode with AVX Compute OLAP<br>A U-enabled Cores A U-disabled Cores SPECjbb<br>3.3<br>~3.2 3.3<br>3.1<br>~3.1<br>2.9 Sharing causes little frequency reduction.<br>2.8<br>Prefill causes great frequency reduction.<br>2.7<br>2.54<br>2.5 2.3<br>0 12 24 36 48 60 72 84 96 0 12 24 36 48 60 72 84<br>AU-enabled Cores for LLM Serving AU Cores w/ SMT Sharing<br>Frequency (GHz) Frequency (GHz)<br>**----- End of picture text -----**<br>


**==> picture [253 x 19] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Impact of usage pattern on AU fre- (b) Sharing interferences af-<br>quency reduction. fect frequency reduction.<br>**----- End of picture text -----**<br>


Fig. 6: Variable frequency reduction due to AU utilization. 

batch size 16 times input sequence length 512. However, most GEMMs in the decode phase are 16 _×_ 4096 _×_ 22016 with the performance of 3 _._ 87 TFLOPS, where 16 denotes batch size for one output token. Meanwhile, although the request arrival rates for user-facing LLM serving are inherently variable, we find that it affects the AU usage pattern via batch size variations since current LLM serving frameworks always adopt continuous batching techniques [33], [42]. AU usage patterns are stable under certain model, phase, and batch size. Therefore, we choose to profile the variation of different batch sizes to reduce the overhead of profiling varying request arrival rates, and we adaptively tune the configurations at runtime based on SLO and performance slack of different requests, similar to prior works have done [11], [102]. 

_**Findings #1:** Variable AU usages in applications initialize AUV and requires usage-aware management._ 

## _B. Variation-2: Compulsory Frequency Interference_ 

Secondly, variable AU usages incur compulsory frequency reduction, excavating AUV. We analyze the frequency variations and interferences on shared cores. 

_1) Compulsory frequency reduction:_ Due to the thermal design power limit, physical cores enabling more power-hungry AU trigger automatic frequency reduction, which changes with AU usage. We use the turbostat [45] tool to record the core frequency of GenA with all-core turbo frequency of 3.2 GHz. Figure 6a presents the impact of AU usages on frequency reduction and varies the number of AU cores with (square) and without (circle) power stressors on the remaining physical cores. We have two observations: (1) The prefill phase with more AMX usages causes greater frequency reduction to 2.5 GHz, which is influenced little by the number of AU cores (green circles). The decode phase with more AVX usages reduces frequency slightly to 3.1 GHz (blue circles). (2) The remaining AU-disabled cores experience no cascaded frequency reduction (gray squares). However, the frequency reduction of AU cores running the decode phase is more severe when co-running power stressors (blue squares). 

_2) Complex frequency interference:_ We find that AUinduced frequency reduction is further affected by sharing interferences, leading to variable cascaded performance degradation. We use all cores for the decode phase and share 

**==> picture [253 x 121] intentionally omitted <==**

**----- Start of picture text -----**<br>
Heavy Ops Light Ops Branch Mispredict Machine Clears<br>Fetch Latency Fetch Bandwidth Memory Bound Core Bound<br>GenA:GEMM 92 3<br>GenA:D 77 19<br>GenA:P 3 70 22<br>GenB:P 8 8 59 22<br>GenC:P 5 55 35<br>SPEC:mcf 16 5 9 70<br>Google:ads 14 3 20 63<br>0% 20% 40% 60% 80% 100%<br>**----- End of picture text -----**<br>


Fig. 7: Comparison of cycle distributions with variable AU usages on evolving platforms. The yellow, gray, blue, and green categories are retiring, bad speculation, frontend bound, and backend bound, respectively. 

different numbers of cores with three types of applications: _Compute_ / _OLAP_ / _OLTP_ (detailed in Section V-A). Figure 6b presents the average frequency of the shared cores, showing that with increasing sharing pressure, relatively low AU usages in the decode phase also induce more severe and variable frequency reduction. Moreover, the abrupt frequency drops on limited cores (12-24) may be caused by heat accumulation on compute-intensive shared cores, and we observe similar drops in repeated experiments with other applications, which excavates the complexity of frequency interference. 

_**Findings #2:** Usage-based frequency reduction worsens AUV, and unstable frequency interference adds difficulties to maximizing the efficiency of AU-enabled processors._ 

## _C. Variation-3: Dissimilar Resource Bound_ 

Thirdly, AU presents microarchitectural resource demand different from traditional functional units, changing with variable AU usages. We adopt the top-down methodology [97] to analyze AU resource bounds and further deep dive into key backend bottlenecks. 

_1) Frontend resources are oversupplied.:_ Frontend resources are responsible for providing _µ_ ops for AU execution, and frontend bound means execution stalls at the L1 instruction cache, fetch unit, and decoder components. As shown in Figure 7, we compare applications with variable AU usages with traditional non-AU applications, _mcf_ benchmark from SPECCPU [8], and _ads_ services from Google [35]. Based on CPU cycle distributions, we mainly have three observations: (1) AU frontend bound is significantly less than the general functional units. AU follows the SIMD paradigm with a smaller instruction working set and fewer i-cache misses, leading to lower fetch latency than before [35] (5% _→_ 1%). (2) Increasing AU usages cause similar frontend bound by comparing pure matrix multiplication (GEMM), prefill, and decode on the GenA platform. (3) Higher bandwidth platforms cause greater frontend bound. 

_2) Backend resources are overloaded.:_ Backend resources are responsible for providing data for AU execution, and backend bound means execution stalls at the execution ports (core bound) and memory access path (memory bound). Despite 

**==> picture [253 x 104] intentionally omitted <==**

**----- Start of picture text -----**<br>
AMX busy Divider DRAM Bound L1 Cache Bound<br>Ports utilization Serializing operation L2 Cache Bound Last Level Cache Bound<br>4.9%<br>19.5% 27.1%<br>19.6% 20.4% 30.3% 13.4% 59.9% 4.5%<br>14.4% 24.2%<br>0.6% 1.5% 4 4.3%<br>Prefill Decode Prefill Decode<br>(a) Core bound breakdown. (b) Memory bound breakdown.<br>**----- End of picture text -----**<br>


Fig. 8: Decomposed AU demands on backend resources. 

abundant frontend resources for AU, backend resources are under strain and restrict AU performance, as shown in Figure 7. To investigate the overload causes, we further decompose the backend resource in Figure 8. 

Figure 8a breaks down the core bound. We find that the instruction window resources [50] (e.g., reorder buffer) are critical for AU execution due to high serializing operations to guarantee instruction sequences. Meanwhile, the decode phase has higher demands on the instruction window resources with higher serializing ratios. As for more severe memory bound, we break down the memory path (L1 cache, L2 cache, LLC, DRAM) in Figure 8b. We find that for the decode phase, DRAM access has the highest influence, specifically on memory bandwidth rather than memory access latency. For the prefill phase, the memory access path across the cache hierarchy and memory affects AU performance similarly. 

_**Findings #3:** AU microarchitectural resource bounds are variable and dissimilar from before, complicating AUV and optimal resource sharing decisions._ 

## V. DEFICIENCY OF AUV-OBLIVIOUS SHARING 

The complex three-dimensional AUV pose new challenges to conventional sharing management. AUV-oblivious sharing based on simultaneous multi-threading (SMT) and resource partitioning (RP) fails to optimize AU-enabled processors for contradictory performance and efficiency goals. 

## _A. Real-world Sharing Scenarios with AU_ 

Workload sharing is a common practice in datacenters to utilize redundant resources and improve overall efficiency [48], [49], [90]. Regarding AU processors, sharing is between AIbased applications with AU and general applications without AU, where AU is not shared for hyperthreads, since AU resources are limited for physical cores. As for sharing priority, LLM serving is considered latency-critical (LC) in this paper and we share it with other best-effort (BE) applications for two reasons. Firstly, LLM applications are mostly user-facing scenarios, leading to its high sensitivity on SLO guarantee. Secondly, it is rare to co-locate many LC workloads in realistic datacenters for better performance guarantee. We select three representative BE applications based on real-world parameters: (1) Memory-intensive online analytical processing ( _OLAP_ ) uses TPC-H [86] to replay joint queries, evaluated by query throughput. (2) Compute-intensive prime number 

**==> picture [253 x 135] intentionally omitted <==**

**----- Start of picture text -----**<br>
LLM TTFT LLM TPOT LLM TTFT LLM TPOT<br>OL AP Throughput Sharing<br>60% 15 50%<br>50% 40%<br>40% 10<br>30%<br>30%<br>20%<br>20% 5<br>10% 10%<br>0% 0 0 %<br>1 2 3 4 5 6 7 SPECjbb Compute<br>Sharing OLAP Workload Pressure Sharing Workloads<br>(a) Impact of sharing pressures on AU (b) Impact of shared ap-<br>and shared applications. plication types.<br>3.24x<br>31.2% 29.7% 43.3%<br>24.7%<br>OLAP Slowdown 2.45x 1.33x 2.6x 1.4x 2.69x 1.49x 2.8x 1.56x 2.82x 1.68x 2.93x 1.76x 1.97x LLM Slowdown 6% 5%<br>Performance Slowdown<br>**----- End of picture text -----**<br>


Fig. 9: Variable SMT impact on AU sharing performance. 

**==> picture [253 x 126] intentionally omitted <==**

**----- Start of picture text -----**<br>
Exclusive TTFT Exclusive TPOT Inclusive TTFT<br>Inclusive TPOT Backend Bound<br>1.8 0.8<br>66.8% 71.3%<br>65.5% 71.8%<br>1.6<br>0.6<br>1.4<br>0.4<br>1.2<br>1 0.2<br>Hyperthread L2 Cache LLC Mem BW<br>Isolated Resources<br>CPU Backend Bound<br>LLM Performance Slowdown<br>**----- End of picture text -----**<br>


Fig. 10: AUV-oblivious resource sharing impact on AU performance. Exclusive means partitioning the single resource. Inclusive means partitioning all resources on the left. 

division ( _Compute_ ) uses sysbench [39] to perform repeated computations, evaluated by event throughput. (3) Complicated Java server ( _SPECjbb_ ) uses SPECjbb 2015 benchmark [75] to perform parallel transactions, evaluated by transaction throughput. The sharing selections are determined before the execution for different AU usage scenarios. 

## _B. Deficiency of Conventional SMT Sharing_ 

SMT-based sharing methods treat AU applications statically and cause variable performance degradations. Conventional SMT uses redundant processor resources to execute multiple threads on one processor core [101], failing to capture dynamic AU usages and interferences as shown in Figure 9. Figure 9a shows that AU performance degrades by more than 200% while the co-running OLAP is affected by more than 40%, resulting from severe memory contentions. WE find that the interferences are unstable with variable AU usages but stable with increased sharing pressures. Figure 9b shows that compute-intensive shared application causes less than 10% AU interference, but sharing _Compute_ experiences 40% degradation due to compulsory frequency reduction. Therefore, AUVoblivious SMT sharing cannot properly control the variable AU behaviors and performance. Moreover, while SMT sharing is known to create contention-based security vulnerabilities for co-running user threads [17], researchers have proposed effective mitigation and isolation techniques without much 

**==> picture [516 x 141] intentionally omitted <==**

**----- Start of picture text -----**<br>
Software-layer AU Applications Sec VI.A  System-layer AU-aware Resource Manager Hardware-layer AU Processor<br>LLM Inference Queue Variation-1:  Variation-2:  Variation-3:  AFU  ROB C C C C<br>Request-1 Prompt Request-2 Token Usage Pattern Frequency Interference Resource Bound Perf. LLC C C C C<br>Res. Alloc.<br>AU Sharing<br>Sec. VI.C SLO Analyzer Performance Core Switcher Processor Allocation Tuner Decisions<br>Runtime AU  Requirement Placement<br>Controller Slack LAG Efficiency Change Collision Adjust AUM<br>Sec VI.B  AU Selecting Processor Dividing Resource Profiling AU  AUV Model<br>Background  [Usage, Frequency, Resource,<br>AU Profiler ARI Usage Frequency Split Bound Divide Bucket Performance, Power]<br>**----- End of picture text -----**<br>


Fig. 11: Considerations of AUV and the workflow of two cooperative components of AUM. 

loss of SMT performance [81], which lead to the widespread deployment of SMT sharing in production datacenters like ours in this paper. 

## _C. Deficiency of Application-aware Resource Partitioning_ 

More advanced RP methods with application awareness cannot handle AUV and achieve optimal efficiency. RP controls shared resources via identifying application interference and allocating key resources (e.g., multi-level caches, memory bandwidth) separately [47]. We isolate three key backend resources between AU-accelerated LLM serving and SPECjbb to observe their exclusive and inclusive effects. As shown in Figure 10, we partition the L2 cache, LLC, and memory bandwidth based on software preferences [11] and normalize LLM serving performance to that under exclusive settings. We find that isolating backend resources separately can slightly relieve AU slowdown, but fails to achieve the optimal decision. Moreover, the critical backend bound is relieved differently by isolating various resources. Therefore, AUV-oblivious resource sharing cannot fully unleash the efficiency potential. 

## VI. DESIGN OF AUM 

This section describes the design considerations and technical details of AUM. It aims to precisely handle the threedimensional accelerator unit variations and fully unleash the efficiency potential of shared processors. 

## _A. Design Considerations and Overview_ 

Given the deficiency of intuitive sharing managers, cooptimizing the performance and efficiency of AU-enabled CPU must consider the three-dimensional AUV: 

- 1) _Variation-1: Usage Pattern._ AU usages are variable in different applications and operators, leading to execution performance variations. We need to select efficient AU usages with lightweight indicators and analyze real-time requirements for the following management. 

- 2) _Variation-2: Frequency Interference._ Variable AU usages lead to dynamic frequency reductions, causing cascaded performance fluctuations. We need to divide the processor into regions to avoid frequency interference and adjust the regions to satisfy runtime requirements. 

- 3) _Variation-3: Resource Bound._ Applications in different processor regions exhibit variable resource bounds, experiencing performance fluctuations. We need to study a unified model to guide resource tuning online to precisely harvest resources for optimal CPU efficiency. 

To handle the complex AUV, we propose AUM, a novel AU-aware resource manager designed to maximize processor efficiency in shared environments. As shown in Figure 11, it focuses on system-layer management to harvest AU unexploited resources for shared applications precisely and flexibly. AUM contains two cooperative components to handle the threedimensional AUV offline and online. The _Background AU Profiler_ characterizes and summarizes AUV into an offline reference model to guide the _Runtime AU Controller_ for precise resource allocation with consideration of runtime status. 

## _B. Background AU Profiler_ 

To portray the complex AUV, AUM selects three key variation indicators based on the analysis above. Firstly, it judges application AU usage via arithmetic intensity (ARI) to categorize operators. Secondly, it divides the processor into regions with different AU usages to assess frequency reductions. Thirdly, it considers specific AU resource bounds via profiling minimal demands. Overall, the three-dimensional information is summarized into the _AUV Model_ . 

_1) Usage-aware AU Selecting:_ To capture the variable AU usage in different applications, we use ARI to determine the proper AU for different operators. Based on previous analysis [36], [37], we can calculate the ARI of underlying operations of AU applications, such as QKV mapping with 6(1 _/d_ + 3 _/BL_ ) _[−]_[1] in the prefill phase and 6(1 _/d_ + 3 _/B_ ) _[−]_[1] in the decode phase. With larger model dimension _d_ , batch size _B_ , and input length _L_ , operations with higher ARI have higher AU usages, denoted as _UAU_ . _UAU_ captures the variable AU usages analyzed in Section IV-A. 

_2) Frequency-aware Processor Dividing:_ To properly manage the frequency interference, _AU-Man_ recognizes the compulsory frequency reduction due to variable AU usage and divides the processor cores into three regions based on Section IV-B. (1) High-AU region _CH_ with high AU usages and 

**==> picture [253 x 85] intentionally omitted <==**

**----- Start of picture text -----**<br>
Prefill:TTFT Decode:TPOT Decode:Throughput<br>8<br>Different dividing provide<br>4 varying AU performance.<br>0<br>0/96 12/84 24/72 36/60 48/48 60/36 72/20 84/12 96/0<br>Decode Cores / Prefill Cores<br>Normalized  Performance<br>**----- End of picture text -----**<br>


Fig. 12: AU applications vary with processor dividing. All the results are normalized to exclusive performance on all cores. 

**==> picture [253 x 158] intentionally omitted <==**

**----- Start of picture text -----**<br>
GenA GenB GenC<br>1.8<br>~80% AU-enabled prefill phase is sensitive to LLC.<br>1.5<br>1.2<br>0.9<br>1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16<br>Last-level caches ways for AU applications<br>1.2<br>1.1 <5% AU-enabled decode phase is insensitive to LLC.<br>1.0<br>0.9<br>1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16<br>Last-level caches ways for AU applications<br>Normalized TTFT<br>Normalized TPOT<br>**----- End of picture text -----**<br>


Fig. 13: AU application with different usages varies with last-level cache (LLC) resource allocation. All the results are normalized to performance with all LLC ways. 

low frequency _FH_ , such as generating prefill tokens. (2) LowAU region _CL_ with low AU usages and moderate frequency _FL_ , such as generating decode tokens. (3) None-AU region _CN_ with no AU usage and high frequency _FN_ , only for shared applications. The _UAU_ threshold is set based on serverlevel AU usage distributions. As shown in Figure 12, _AUMan_ records the AU performance and the frequency lower bounds under different region divisions. The frequency profiles are recorded as processor regions _C_ and frequencies _F_ with _[CH , FH , CL, FL, CN , FN ]_ . 

_3) Bound-aware Resource Profiling:_ To capture the distinct resource bounds and provide just the right resources for AU, we record changing resource affinities of variable AU usages based on Section IV-C. For L2-cache and LLC capacity, as well as memory bandwidth, we can profile its variation using resource partitioning interfaces [27], such as Cache Allocation Technology (CAT) for cache ways and Memory Bandwidth Allocation (MBA) for memory bandwidth. As shown in Figure 13, we find that for LLC resources, varying AU usages and underlying platforms induce diverse affinity, showing that we can harvest LLC resources for low-AU operators and highAU operators on GenA. We can profile the minimal resource demands as a three-tuple _RAU_ . 

Overall, the three-dimensional AU profiles are recorded into a _AUV Model_ with consideration of AU application performance and processor power consumption, as shown in 

TABLE III: An example bucket in the AUV Model. 

||_UAU_<br>High<br>Low<br>None|_CAU_<br>0-11<br>12-15<br>16-23|_FAU_<br>2.1 GHz<br>2.8 GHz<br>3.2 GHz|_RL_2_C_<br>0-2<br>3-6<br>7-15|_RLLC_<br>0-1<br>2-4<br>5-15|_RBW_<br>50%<br>40%<br>10%|_P a_<br>0.42<br>9.12<br>13.28|_P t_<br>0.31<br>7.19<br>9.16|
|---|---|---|---|---|---|---|---|---|



Table III. For three frequency regions with variable AU usages, we allocate varying resources and record the performance of high AU application _PH_ , low AU application _PL_ , and shared application _PN_ . To reduce profiling costs, we design the _AU Bucket_ mechanism to discretize the continuous variations. For high/low/none AU usages, we profile three processor divisions with five performance-sensitive resource configurations. For every bucket, we record the 50% average performance _P[a]_ , 90% tail performance _P[t]_ , and processor power consumption _WCP U_ . The variation profiles guide the runtime AU controller. 

## _C. Runtime AU Controller_ 

To perform AU-aware processor management, AUM jointly considers _AUV Model_ and runtime information to make adaptive decisions as shown in Algorithm 1. Firstly, it analyzes AU performance to update requirements. Secondly, it selects sharing decisions with maximized processor efficiency. Thirdly, it adjusts unexploited resource harvesting considering AU interference. Overall, the processor resource is properly shared to guarantee AU performance and maximize overall efficiency. 

_1) Slack-aware SLO Analyzer:_ To determine the varying AU SLO, AUM computes performance slacks for different AU usages. Firstly, for prefill tokens with high AU usages, better performance improves user satisfaction, so we simply use first-come-first-served (FCFS) to schedule prompts [57]. The runtime SLO for prefill tokens _SLOH_ is set as _dT T F T −twait_ , where _dT T F T_ is the TTFT SLO and _twait_ is the request waiting time. Secondly, for abundant decode tokens with low AU usages, we track the performance of tokens and adapt to varying request arrival rates at runtime with LAG analysis (a measurement of how far behind the token is compared to an ideal schedule that meets the performance requirements). 

We quantify the relationship between the partial execution time at time _t_ of serving request _i_ ( _ei_ ) and its relative deadline, denoted as _LAGi_ , as shown in Algorithm 1-Line 3. _Ti_ ( _t_ ) is the tokens of request _i_ that have completed by time _t_ . For token _token ∈ Ti_ ( _t_ ), _dT P OT_ is set as the TPOT SLO, and _etoken_ is the recorded execution time for token _t_ , respectively. _LAGi_ reflects the real-time status of serving request _i_ and quantifies how far ahead or behind the serving request is compared to the deadline at time _t_ . If every LAG is 0, AU applications are allocated precise resources. The AU application is perfect if every LAG within it is 0, which means all tokens have exactly finished by their deadline so far and AU application does not need more resources. The runtime SLO for decode tokens _SLOL_ is set as _dT P OT_ + _LAGi_ . Since LAG indicates how far behind ( _LAG <_ 0) or ahead ( _LAG >_ 0) every AU-accelerated request is, the AU configurations need to be adjusted accordingly for faster and slower execution. AUM obtains the runtime performance requirements in this stage. 

**Algorithm 1:** Workflow of the runtime AU controller. 

**Input:** Reference AUV Model _M_ **Output:** AU-aware Resource Sharing Decision // Slack-aware SLO analysis **1** _SLOH_ = _dT T F T − twait_ ; **2** _SLOL_ = _dT P OT_ + _LAGi_ ; **3** _LAGi_ ( _token, Ti_ ( _t_ )) =[�] _token∈Ti_ ( _t_ )[(] _[d][T P OT][−][e][token]_[)][;] // Efficiency-aware Core Switcher **4** _ECP U_ = ( _α × PH_ + _β × PL_ + _γ × PN_ ) _/WCP U_ ; **5** Maximize _ECP U_ s.t. _PH[t][< SLO][H]_[and] _[P][ t] L[< d][T P OT]_[ ;] **6** _U/C/F ← M_ ( _PH , PL_ ); // Collision-aware Allocation Tuner **7** Continuously monitor AU performance _P[m]_ ; **8 if** _PH[m][< SLO][H][and][P][ m] L[< SLO][L]_ **[then] 9** _δAU ←_[�] _UAU × SLO/P[m]_ ; **10** _RAU ← M_ ( _PH[a][, P] L[ a]_[)][;] **11 end 12 else 13** _δAU ←_[�] _UAU × P[m] /SLO_ ; **14** _RAU ← M_ ( _PH[t][, P] L[ t]_[)][;] **15 end 16 if** _δAU > threshold_ **then 17** _C/F ← M_ ( _δAU , PAU , CAU , FAU_ ) **18 end** 

_2) Efficiency-aware Core Switcher:_ To optimize processor efficiency with varying SLO, we switch processor core configurations with consideration of weighted efficiency. The processor performance-per-watt efficiency _ECP U_ is computed as the weighted sum of application performance divided by CPU power consumption, as shown in Algorithm 1-Line 4. The prices of application outputs are used to normalize their performance in different regions as _α_ , _β_ , and _γ_ . The shared applications are continuously running in the background and their pressures are proportional to the allocated cores _CN /FN_ . For lower management complexity, AUM switches processor cores to different frequency regions to maximize the weighted efficiency and satisfy diverse SLO primarily, as shown in Algorithm 1-Lines 5, 6. The frequency of every region is set as the maximal level below the TDP. Admittedly, finegrained frequency control schemes [66], [77] could further improve processor efficiency via per-core or per-workload power capping. But it would significantly enlarge the optimization space, and our rule-based controller needs to integrate intelligent algorithms to make decisions [103], which leaves as our future work. The processor division and shared application are relatively stable for the resource allocation tuning. AUM decides the processor placement in this stage. 

_3) Collision-aware Allocation Tuner:_ To avoid dramatic AU performance degradation, AUM tunes resource allocation considering the collision between AU and shared applications. For every control iteration, AUM uses a continuous and lightweight indicator to detect AU application performance like token latency. If the measured AU performance _P[m]_ 

TABLE IV: Evaluated AU usage scenarios with different prefill/decode SLOs and average input/output lengths. 

|**Apps**|**Dataset**|**dTTFT**|**dTPOT**|**Input**|**Output**|
|---|---|---|---|---|---|
|_cb_|ShareGPT [64]|250 ms|100 ms|755|200|
|_cc_|HumanEval [10]|75 ms|150 ms|171|98|
|_sm_|LongBench [6]|1.5 s|100 ms|1738|91|



guarantees runtime SLO, we can aggressively harvest AU resources for shared applications, using average performance _PAU[a]_[to][tune][resource][allocations.][Otherwise,][we][need][to] conservatively return resources to AU applications using tail performance _PAU[t]_[to][control.][Given][the][varying][AU][resource] affinities, hardware resource that causes minimal AU performance degradation are harvested first. The allocation tuner needs to be refined by considering SLO guarantee of corunning applications under the best-effort LLM serving scenarios. Moreover, we compute the deviation _δAU_ to denote the performance gap and higher AU usages result in greater deviations to be eliminated as shown in Algorithm 1-Lines 9, 13. If the deviation _δAU_ exceeds the threshold, tuning AU resources is not sufficient, and we need to switch the processor division as shown in Algorithm 1-Line 17. 

## VII. EVALUATION OF AU-MAN 

Our evaluation wants to answer three questions: **1.** How does AUM improve CPU overall efficiency? (Section VII-B) **2.** How does AUM guarantee AU application performance given varying requirements? (Section VII-C) **3.** How are the costs and revenues to deploy AUM on AU-enabled CPU? (Sections VII-D and VII-E) 

## _A. Evaluation Methodology_ 

_1) Implementations:_ We implement a prototype of AUM based on xFasterTransformer [21] with two components implemented in Python, acting as a system component. The background profiler records the essential information of newer models with repeated experiments on dedicated nodes. The runtime controller works as a system daemon to monitor the SLO and tune the allocation in production. For hyperparameters of AUM, we select the performance prices _α_ as 1.8 for high-AU prefill tokens and _β_ as 0.2 for low-AU decode tokens. Different from GPU-based token prices [63], we decide the prices based on CPU time to produce a prefill and decode token. _γ_ for none-AU Compute, OLAP, and SPECjbb is set as 1e-3, 1e-6, and 3e-5, respectively. The prices are decided based on CPU time to produce one query on the evaluated platform. These parameters are decided empirically and we conduct a sensitivity experiment to evaluate AUM under different scenarios. We set the derivation _δAU_ threshold as 2 to denote performance collision. 

_2) Workloads:_ For hardware platforms, AUM experiments are mainly conducted on the SPR platform (GenA) as shown in Table I if not otherwise specified. For evaluated workloads, the evaluated AU applications are LLM serving of llama models [51] similar to Section III-A, and the co-running applications are _Compute_ [39], _OLAP_ [86], and _SPECjbb_ [75] 

**==> picture [516 x 79] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.3<br>ALL-AU 1.2 Max increase of  Min increase of  Higher is better.<br>1.1 15.3% 1.6%<br>SMT-AU<br>1<br>RP-AU 0.9<br>cb cc sm cb cc sm cb cc sm<br>AUM<br>Compute OLAP SPECjbb Avg.<br>Efficiency<br>Normalized<br>1.05 1.07 1.09 1.14<br>**----- End of picture text -----**<br>


Fig. 14: Comparison of CPU performance-per-watt efficiency with variable AU application scenarios and sharing selections. All the results are normalized to ALL-AU under chatbot scenario. AUM outperforms SOTA sharing baselines by 4.7%. 

TABLE V: Three categories of evaluated baselines. 

|**Category**<br>**AU-exclusive**<br>**AUV-oblivious**<br>**Sharing**<br>**AU-aware**<br>**Resource**<br>**Managers**|**Scheme**<br>ALL-AU<br>SMT-AU<br>RP-AU<br>AU-UP<br>AU-FI<br>AU-RB<br>AUM|**Description**<br>Utilizing AU CPU without sharing<br>SMT sharing AU CPU<br>Partition resources of AU CPU<br>Sharing w/ usage pattern<br>Sharing w/ frequency interference<br>Sharing w/ resource bound<br>Our three-dimensional proposal|
|---|---|---|



similar to Section V-A. We evaluate different AU applications use-cases [102] as shown in Table IV (1) ChatGPT-like chatbot ( _cb_ ) [62]; (2) Cursor-like code completion ( _cc_ ) [12]; (3) Summarization ( _sm_ ) [43]. The benchmark selection is similar as previous works [102] and represents datacenter scenarios. 

_3) Baselines:_ To understand the performance of AUM, we compare it with three types of baselines as shown in Table V. Firstly, _AU-exclusive_ adopts non-sharing settings and uses the whole AU-enabled processor for LLM serving. Secondly, _AUV-oblivious Sharing_ adopts state-of-the-art resource managers for shared CPU platforms [11], [69]. More specifically, _SMT-AU_ adopts SMT sharing [69] and _RP-AU_ adopts workload-aware resource partitioning [11] for LLM serving and co-located workloads. Thirdly, _AU-aware Resource Managers_ are variants of AUM to investigate the effect of threedimensional AU awareness. 

## _B. Improving CPU Efficiency_ 

We first evaluate AUM in improving CPU efficiency, which is the primary design goal. We compare the performanceper-watt efficiency of the CPU platform when sharing AUaccelerated LLM serving with background co-running applications. As shown in Figure 14, AUM achieves the best efficiency on the CPU platform, resulting from precisely harvesting AU underutilized resources for shared applications. On average, AUM improves 8.8%, 6.7%, and 4.7% efficiency compared with AU-exclusive and AUV-oblivious sharing baselines. More specifically, we have two observations. Firstly, sharing AU needs to select co-running applications carefully. Sharing with memory-intensive applications like OLAP has marginal efficiency improvements under diverse scenarios even with precise allocation. Secondly, CPU achieves better efficiency under the code completion scenario with loose SLO requirements via AUV-aware management. 

**==> picture [253 x 89] intentionally omitted <==**

**----- Start of picture text -----**<br>
ALL-AU SMT-AU RP-AU AUM<br>2<br>63% 19% 50% 11% 56%<br>1.5<br>17%<br>1<br>GenA GenB GenC GenA GenB GenC GenA GenB GenC<br>chatbot code completion summarization<br>Normalized Efficiency<br>**----- End of picture text -----**<br>


Fig. 15: Comparison of efficiency on three variable hardware platforms when sharing with SPECjbb. All the results are normalized to ALL-AU on GenA platform. 

**==> picture [253 x 166] intentionally omitted <==**

**----- Start of picture text -----**<br>
ALL-AU SMT-AU RP-AU AUM-UP AUM-FI AUM-RB AUM<br>1.2<br>1.00 0.94 0.95 0.94 0.94 0.95 0.97 (a) Compute<br>0.9<br>1.04 1.00 1.04 1.11 1.07 1.17<br>0.6<br>1.5<br>1.00 (b) OLAP<br>1<br>0.5 0.24 0.26 0.29 0.30 0.31 0.35<br>1.04 1.00 1.04 1.19 1.21 1.28<br>0<br>1.2<br>1.00 (C) SPECjbb<br>0.9 0.79 0.83 0.80 0.82 0.84 0.89<br>0.96 1.00 0.96 1.03 1.03 1.08<br>0.6<br>AU Applications Sharing Applications<br>Normalized  Throughput<br>Normalized  Throughput<br>Normalized  Throughput<br>**----- End of picture text -----**<br>


Fig. 16: Comparison of performance of AU and three types of shared applications, averaged of three scenarios. The performance results of AU and shared applications are normalized to ALL-AU and RP-AU, respectively. 

Further, we investigate the efficiency on hardware platforms with evolving AU. We choose to share with SPECjbb since it presents complex execution and moderate sharing revenues. As shown in Figure 15, newer CPU platforms show better efficiency under various scenarios due to more powerful AU and memory devices, increasing 1.55x on average for exclusive AU usages. The shared AU usages with AUM have higher rates of efficiency increase (63%, 50%, and 56%) compared to AU-exclusive schemes. On the latest GenC platform, AUM increases efficiency by 19%, 11%, and 17% over AU-exclusive baselines. The higher increase rates over GenA result from more resources for tuning and harvesting. 

**==> picture [253 x 88] intentionally omitted <==**

**----- Start of picture text -----**<br>
ALL-AU SMT-AU RP-AU AUM<br>1 AU cannot satisfy  1<br>0.8 strict requirements. 0.9<br>0.6<br>0.8<br>0.4<br>0.2 0.7<br>0 0.6<br>cb cc sm cb cc sm<br>TTFT SLO Guarantee TPOT SLO Guarantee<br>**----- End of picture text -----**<br>


Fig. 17: Comparison of AU application SLO guarantee when sharing with SPECjbb. Left shows the high-AU prefill phase and right shows the low-AU decode phase. AUM shows better performance guarantee than SOTA sharing baselines. 

To understand the detailed performance decomposition of AUM on AU and shared applications, we investigate the decomposed performance as shown in Figure 16. The AU-exclusive scheme achieves the best LLM serving performance but zero sharing performance. The AUV-oblivious sharing schemes cause varying slowdowns. As for variants of AUM, AU-UP only optimizes manipulation of AU applications rather than sharing; AU-FI splits the processor to mostly improve sharing performance; and AU-RB enhances traditional resource partitioning for both co-runners. 

## _C. Guaranteeing AU Performance_ 

We next evaluate AUM in guaranteeing AU performance SLOs under different scenarios. CPU efficiency improvements must guarantee the primary AU applications. As shown in Figure 17, AU-Man achieves better SLO guarantee compared with AUV-oblivious sharing methods. We have two observations. Firstly, AU-enabled CPU should be used as prompt machines under specific scenarios [67]. For _cc_ with strict TTFT SLOs, even using AU exclusively for prefill cannot meet the SLO due to a lack of computing units. For _sm_ with loose TTFT SLO, AUM achieves 93.6% SLO guarantee ratio, which is 11% better than AUV-oblivious schemes. Secondly, AUM achieves better SLO guarantees for low-AU decode performance. For the loose _cc_ scenario, sharing CPU merely causes SLO violations. For the strict _cb_ and _sm_ scenarios, harvesting critical memory bandwidth causes SLO violations, but AUM achieves similar TPOT SLO performance with AU-exclusive schemes, which is 7% better than AUV-oblivious sharing methods. 

## _D. Detailed Analysis of AUM_ 

More specifically, we investigate AUM resource management decisions of two examples, LLC and memory bandwidth, as shown in Figure 18. Different from static AUV-oblivious sharing allocation that gives more resources to AU applications, AUM considers AUV and adopts more flexible resource allocation based on runtime information. It harvests more LLC resources from AU to shared applications, and harvests the high-affinity memory bandwidth adaptively. Better utilization of hardware resources contributes to improved CPU efficiency. 

To evaluate the parameter sensitivity of AUM on efficiency improvements, we change the value of _α/β_ to mimic the 

**==> picture [253 x 104] intentionally omitted <==**

**----- Start of picture text -----**<br>
ALL-AU SMT-AU RP-AU AUM<br>1 1<br>0.8 0.8<br>0.6 0.6<br>0.4 0.4<br>0.2 0.2 AUM allocates memory<br>AUM allocates less LLC. bandwidth more flexibly.<br>0 0<br>0 0.5 1 0 0.5 1<br>CDF CDF<br>LLC Allocation<br>Bandwidth Allocation<br>**----- End of picture text -----**<br>


Fig. 18: Resource allocation Cumulative Distribution Function (CDF) under SPECjbb and chatbot scenario. AUM manages resources more flexibly for shared applications. 

emerging situation where token prices are cheaper and cheaper. Under the default 1 _._ 8 _/_ 0 _._ 2 setting, the efficiency with AUM outperforms 7 _._ 6% over the _SMT-AU_ baseline when sharing _Compute_ with _cc_ . For a lower 0 _._ 9 _/_ 0 _._ 1 setting, AUM exceeds 9 _._ 1% over _SMT-AU_ since it allocates more resources to sharing applications with little sacrifice of AU applications for overall efficiency based on the weighted calculation. 

The runtime overheads of AUM are critical for realistic deployment. The _Background AU Profiler_ works offline, and it takes around 450 AU-enabled executions to converge and construct the _AUV Model_ , including 3 division _×_ 3 sharing _×_ 5 configurations _×_ 10 repetitions. Moreover, the overhead of a single profiling could optimize thousands of processor cores with the same model, thus amortizing the offline resource overheads. The _Runtime AU Controller_ decides resource allocation with one CPU core in less than 1 ms to lookup table, which is negligible compared with 100 ms-scale token SLO. Meanwhile, AUM consumes around 15 MB to store AUV and runtime information, which is also negligible for 256 GB memory capacity. More specifically, for benchmarks with rapidly fluctuating resources like SPECjbb, the runtime controller adapts to the resource collisions and decides the most efficient configurations for every iteration. The limitation of AUM is reliance on runtime controlling rather than online learning to continuously complement the AUV model. 

## _E. Total Cost of Ownership Analysis_ 

Cost-effectiveness has become the top factors that drive next-generation system architecture. We briefly discuss the impact of AUM on the total cost of ownership (TCO), including the capital expenditure (CapEx) for hardware acquisition and the operating expenses (OpEx) for hardware execution. For CapEx, according to 1.3x perf-per-dollar of GPU in Figure 5 and 15% improvements with AUM in Figure 14, CPU with AUM achieves 88% performance-per-CapEx compared with GPU solutions. For OpEx, the maintenance and cooling costs are lower for CPU servers but there are more complex factors affecting the TCO [7], such as machine utilization and amortization. But our conclusion is that if we could unleash the full AU efficiency via AUM, the AU-enabled CPU could cede the limited GPUs to critical scenarios or even become a competitive alternative in the GPU-dominant AI era. 

## VIII. DISCUSSION 

This section discusses four limitations of AUM and potential optimization directions. 

**Large-scale cluster scalability:** Currently, AUM focuses on machine-level scheduling with better arrangement of all processor cores. However, the extracted AUV are easy to be exploited in scale-out clusters. For sharding workloads across multiple servers, we can analyze the AUV of every processor and adopt load balancing [48] to maximize their efficiency separately. The proposed methodology of profilecontrol is applicable to all AU-enabled benchmarks besides LLM serving. It is promising and underway to integrate AUM into cluster-level scheduling for better cluster efficiency. 

**Adaptability to AU operators:** In this paper, AUM focuses on the xFasterTransformer implementation of AU operators based on Intel oneDNN [29]. Current software developers are exploring AU deeply to support unstructured sparsity [1] and memory tiling [41]. The different implementations and optimizations of AU operators further increase AUV. However, AUM could be manually optimized to high-performance AUenabled operators and adapt itself with the runtime controller. 

**Hardware topology adaptability:** In this paper, AUM focuses on analyzing and sharing AMX-enabled CPU cores for better efficiency while it can be further enhanced under different hardware topologies. For different hardware sharing topologies, AUM can handle the trade-off between singlethreading and multi-threading cores under different scenarios based on adaptive processor efficiency calculation. For emerging AU topologies such as sharing SME among physical cores [4], the current assumption that AU is not shared for hyperthreads, as stated in Section V-A, falls short and AUM needs to refine its profiler with a new dimension of contentions on a single AU. For hybrid topologies where CPU AU interacts with dedicated accelerators like GPU and NPU, there are other concurrent works focusing on offloading orchestration and communication optimization, like LIA [36] and ktransformer [9]. Therefore, hybrid topologies are beyond AUM in this paper and are our future works. 

**Challenges of AUM on other ISAs:** The integration of AU into modern CPUs is a universal trend for different vendors besides Intel AVX/AMX [25], such as ARM SVE/SME [5], [76] and RISC-V V/M-extension [2], [71]. We focus on Intel in this paper due to its mature software support of its accelerator units [33], [41]. AUM recognizes the unique AUV, and the three-dimensional designs are independent of underlying hardware component implementations, which can be easily applied to more heterogeneous ISA architectures. 

usage is studied and optimized via core specialization [16]. Meanwhile, many works focus on virtualizing dedicated accelerators outside the CPU cores for more flexible utilization, such as GPU [34], [98], NPU [96], FPGA [15], and video accelerators [58]. Their optimizations are orthogonal to the accelerator units within the CPU pipeline, which is the focus of AUM, but they can further enhance AUM to offer a better sharing infrastructure across CPU and other accelerators. 

**Research on AU-accelerated systems:** LLM serving systems are gaining popularity in both academia and industry [42], [67], [102]. Researchers are exploring AU-based LLM serving from memory optimization [21], [41], [93], sparsity [1] and CPU-GPU computing [36], [56]. Moreover, AU are exploited to accelerate diverse workloads such as scientific applications [22]. AUM builds upon these AU-centric optimization to pursue better overall processor efficiency in the shared environments. 

**Research on CPU sharing:** CPU sharing is a common method to harvest spare resources and improve utilization in cloud data centers [46], [101]. SMT sharing [49], [50] and workload-aware resource partitioning [11], [92] are discussed above. The intelligent sharing decision models, such as interference prediction [48], can further enhance AUM. There are more general works on sharing execution units between cores [55], [72] but AUM focuses on more varying matrix accelerator unit management in the emerging AI era. 

**Research on resource harvesting:** Resource harvesting denotes identifying and utilizing allocated but temporarily idle resources for higher utilization. These works operate at clusterlevel [89], server-level [91], [94], cycle-level [49], [79], and device-level [68]. However, the idle resources in AU-enabled CPU result from its distinct resource demands and are hidden from utilization identifiers [18]. 

## X. CONCLUSION 

To meet the AI-driven software shift, CPUs are incorporating variable accelerator units. We show the deficiency of exclusive AU usages and the challenges of sharing AU usages, summarized into three-dimensional accelerator unit variations. Therefore, we propose AUM, a novel AU-aware resource manager designed to handle AU variations and maximize the efficiency of shared processors. Through extensive evaluations, AUM improves processor efficiency by up to 8.8% while maintaining high-performance AU application by reducing SLO violations by up to 11% compared with SOTA methods. AUM enables a more efficient and high-performance CPU infrastructure in the prevailing AI era. 

## IX. RELATED WORK 

**Research on hardware asymmetry:** Asymmetric hardware is widely studied for accelerating domain-specific tasks. There are three main asymmetries inside the CPU. (1) ISA asymmetry [87] via design space exploration [88]. (2) Processor asymmetry that adopts big.LITTLE architecture [65]. (3) Unit asymmetry that considers specialized accelerator units within the CPU as AUM does. Prior to AMX usage [37], AVX 

## ACKNOWLEDGEMENTS 

We sincerely thank all the anonymous reviewers for their valuable comments that helped us to improve the paper. This work is supported by the National Natural Science Foundation of China (No. U23A6007 and No. U24A20234) and an Alibaba Research Grant. The corresponding authors are Chao Li and Luping Wang. 

## REFERENCES 

- [1] A. F. AbouElhamayed, J. Dotzel, Y. Akhauri, C.-C. Chang, S. Gobriel, J. P. Mu˜noz, V. S. Chua, N. Jain, and M. S. Abdelfattah, “Sparamx: Accelerating compressed llms token generation on amx-powered cpus,” 2025. [Online]. Available: https://arxiv.org/abs/2502.12444 

- [2] S. I. C. D. E. D. U. o. C. B. Andrew waterman, Krste Asanovic, “The risc-v instruction set manual: Volume i,” _Chapter 9. ”M” Standard Extension for Integer Multiplication and Division, Version 2.0_ , 2022. [Online]. Available: https://lists.riscv.org/g/tech-unprivileged/ attachment/535/0/unpriv-isa-asciidoc.pdf 

- [3] M. Arik, “The concern around gpu shortages and how these could impact the ai revolution,” 2023. [Online]. Available: https://www.techuk.org/resource/the-concern-around-gpushortages-and-how-these-could-impact-the-ai-revolution.html 

- [4] ARM, “The c1-sme2 unit,” 2025. [Online]. Available: https: //developer.arm.com/documentation/107831/0102/The-C1-SME2--unit 

- [5] ARM, “Sme and sme2,” 2025. [Online]. Available: https://developer.arm.com/documentation/109246/0101/SMEOverview/SME-and-SME2 

- [6] Y. Bai, X. Lv, J. Zhang, H. Lyu, J. Tang, Z. Huang, Z. Du, X. Liu, A. Zeng, L. Hou, Y. Dong, J. Tang, and J. Li, “Longbench: A bilingual, multitask benchmark for long context understanding,” 2024. [Online]. Available: https://arxiv.org/abs/2308.14508 

- [7] L. A. Barroso, U. H¨olzle, and P. Ranganathan, _The datacenter as a computer: Designing warehouse-scale machines_ . Springer Nature, 2019. 

- [8] J. Bucek, K.-D. Lange, and J. v. Kistowski, “Spec cpu2017: Next-generation compute benchmark,” in _Companion of the 2018 ACM/SPEC International Conference on Performance Engineering (ICPE)_ . New York, NY, USA: Association for Computing Machinery, 2018, p. 41–42. [Online]. Available: https://doi.org/10.1145/3185768. 3185771 

- [9] H. Chen, W. Xie, B. Zhang, J. Tang, J. Wang, J. Dong, S. Chen, Z. Yuan, C. Lin, C. Qiu, Y. Zhu, Q. Ou, J. Liao, X. Chen, Z. Ai, Y. Wu, and M. Zhang, “Ktransformers: Unleashing the full potential of cpu/gpu hybrid inference for moe models,” in _Proceedings of the ACM SIGOPS 31st Symposium on Operating Systems Principles_ , ser. SOSP ’25. New York, NY, USA: Association for Computing Machinery, 2025, p. 1014–1029. [Online]. Available: https://doi.org/10.1145/3731569.3764843 

- [10] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba, “Evaluating large language models trained on code,” 2021. [Online]. Available: https://arxiv.org/abs/2107.03374 

- [11] S. Chen, C. Delimitrou, and J. F. Mart´ınez, “Parties: Qosaware resource partitioning for multiple interactive services,” in _Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)_ , 2019, p. 107–120. [Online]. Available: https: //doi.org/10.1145/3297858.3304005 

- [12] Cursor, “The ai code editor,” 2025. [Online]. Available: https: //cursor.com/en 

- [13] V. G. K. Dylan Souvage and A. Kumar, “How aws and intel make llms more accessible and cost-effective with deepseek,” 2025. [Online]. Available: https://lnkd.in/dhPCgcfU 

- [14] S. Everman and L. Eeckhout, “A memory-level parallelism aware fetch policy for smt processors,” in _Proceedings of the 2007 IEEE 13th International Symposium on High Performance Computer Architecture (HPCA)_ , 2007, p. 240–249. [Online]. Available: https: //doi.org/10.1109/HPCA.2007.346201 

- [15] S. A. Fahmy, K. Vipin, and S. Shreejith, “Virtualized fpga accelerators for efficient cloud computing,” in _2015 IEEE 7th International Conference on Cloud Computing Technology and Science (CloudCom)_ , 2015, pp. 430–435. 

- [16] M. Gottschlag, P. Machauer, Y. Khalil, and F. Bellosa, “Fair scheduling for avx2 and avx-512 workloads,” in _2021 USENIX Annual Technical Conference (USENIX ATC)_ , 2021, pp. 745–758. [Online]. Available: https://www.usenix.org/conference/atc21/presentation/gottschlag 

- [17] B. Gras, K. Razavi, H. Bos, and C. Giuffrida, “Translation leak-aside buffer: Defeating cache side-channel protections with TLB attacks,” in _27th USENIX Security Symposium (USENIX Security 18)_ . Baltimore, MD: USENIX Association, Aug. 2018, pp. 955–972. [Online]. Available: https://www.usenix.org/conference/ usenixsecurity18/presentation/gras 

- [18] B. Gregg, _Systems Performance: Enterprise and the Cloud_ , 1st ed. USA: Prentice Hall Press, 2013. [Online]. Available: https://dl.acm. org/doi/10.5555/2568162 

- [19] B. Gregg, “perf examples,” 2024. [Online]. Available: https: //www.brendangregg.com/perf.html 

- [20] H. Guo, R. Tang, Y. Ye, Z. Li, and X. He, “Deepfm: A factorizationmachine based neural network for ctr prediction,” 2017. [Online]. Available: https://arxiv.org/abs/1703.04247 

- [21] P. He, S. Zhou, W. Huang, C. Li, D. Wang, B. Guo, C. Meng, S. Gui, W. Yu, and Y. Xie, “Inference performance optimization for large language models on cpus,” 2024. [Online]. Available: https://arxiv.org/abs/2407.07304 

- [22] A. Heinecke, G. Henry, M. Hutchinson, and H. Pabst, “Libxsmm: accelerating small matrix multiplications by runtime code generation,” in _Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC)_ , 2016. [Online]. Available: https://doi.org/10.1109/SC.2016.83 

- [23] R. Huang, F. Chen, Y. Ren, J. Liu, C. Cui, and Z. Zhao, “Multisinger: Fast multi-singer singing voice vocoder with a large-scale corpus,” in _Proceedings of the 29th ACM International Conference on Multimedia (MM)_ , 2021, p. 3945–3954. [Online]. Available: https://doi.org/10.1145/3474085.3475437 

- [24] Intel, “Intel® 64 and ia-32 architectures software developer’s manual,” _Volume 1 (3A, 3B, 3C & 3D): Basic Architecture, Chapter 18: Programming With Intel® Advanced Matrix Extensions_ , 2022. [Online]. Available: https://www.intel.com/content/www/us/en/ developer/articles/technical/intel-sdm.html 

- [25] Intel, “Intel® architecture instruction set extensions and future features 64 and ia-32 architectures software developer’s manual,” _Chapter 3: Intel® AMX INSTRUCTION SET REFERENCE_ , 2024. [Online]. Available: https://cdrdv2-public.intel.com/843860/architectureinstruction-set-extensions-programming-reference-dec-24.pdf 

- [26] Intel, “Intel® rdt software package,” 2024. [Online]. Available: https://github.com/intel/intel-cmt-cat 

- [27] Intel, “Intel® resource director technology framework,” 2024. [Online]. Available: https://www.intel.com/content/www/us/en/architecture-andtechnology/resource-director-technology.html 

- [28] Intel, “Intel unveils future-generation xeon with robust performance and efficiency architectures,” 2024. [Online]. Available: https://newsroom.intel.com/artificial-intelligence/intel-unveilsfuture-generation-xeon#gs.ihuopp 

- [29] Intel, “oneapi deep neural network library (onednn),” 2024. [Online]. Available: https://github.com/uxlfoundation/oneDNN 

- [30] Intel, “Overview of accelerating ai inference and llm applications with cpus,” 2025. [Online]. Available: https://www.intel.cn/content/dam/ www/central-libraries/cn/zh/documents/2024-09/24-cmf41-overviewof-accelerating-ai-inference-and-llm-applications-with-cpus.pdf 

- [31] Intel, “Support for next generation intel xeon scalable processors,” 2025. [Online]. Available: https://www.intel.com/content/www/us/en/ developer/articles/technical/next-gen-performance-gcc-15.html 

- [32] Intel, “Unlock your ai potential with a winning combination,” 2025. [Online]. Available: https://cdrdv2-public.intel.com/862506/azure-dv6ai-benchmarks-business-brief-1.pdf 

- [33] Intel, “xfastertransformer,” 2025. [Online]. Available: https://github. com/intel/xFasterTransformer 

- [34] N. Jing, L. Jiang, T. Zhang, C. Li, F. Fan, and X. Liang, “Energyefficient edram-based on-chip storage architecture for gpgpus,” _IEEE Transactions on Computers_ , vol. 65, no. 1, pp. 122–135, 2016. 

- [35] S. Kanev, J. P. Darago, K. Hazelwood, P. Ranganathan, T. Moseley, G.-Y. Wei, and D. Brooks, “Profiling a warehouse-scale computer,” in _Proceedings of the 42nd Annual International Symposium on Computer Architecture (ISCA)_ , 2015, p. 158–169. [Online]. Available: https://doi.org/10.1145/2749469.2750392 

- [36] H. Kim, N. Wang, Q. Xia, J. Huang, A. Yazdanbakhsh, and N. S. Kim, “Lia: A single-gpu llm inference acceleration with cooperative amx-enabled cpu-gpu computation and cxl offloading,” in _Proceedings of the 52nd Annual International Symposium on Computer Architecture (ISCA’25)_ , 2025, p. 544–558. [Online]. Available: https://doi.org/10.1145/3695053.3731092 

- [37] H. Kim, G. Ye, N. Wang, A. Yazdanbakhsh, and N. S. Kim, “Exploiting intel® advanced matrix extensions (amx) for large language model inference,” _IEEE Computer Architecture Letters (CAL)_ , 2024. [Online]. Available: https://doi.org/10.1109/LCA.2024.3397747 

- [38] A. Kleen, “Intel pmu profiling tools,” 2024. [Online]. Available: https://github.com/andikleen/pmu-tools 

- [39] A. Kopytov, “sysbench,” 2025. [Online]. Available: https://github.com/ akopytov/sysbench 

- [40] R. Kuper, I. Jeong, Y. Yuan, R. Wang, N. Ranganathan, N. Rao, J. Hu, S. Kumar, P. Lantz, and N. S. Kim, “A quantitative analysis and guidelines of data streaming accelerator in modern intel xeon scalable processors,” in _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS)_ , 2024, p. 37–54. [Online]. Available: https://doi.org/10.1145/3620665.3640401 

- [41] kvcache ai, “Qwen 3 + ktransformers 0.3 (+amx) = ai workstation/pc,” 2025. [Online]. Available: https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/ AMX.md#qwen-3--ktransformers-03-amx--ai-workstationpc 

- [42] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, and I. Stoica, “Efficient memory management for large language model serving with pagedattention,” in _Proceedings of the 29th Symposium on Operating Systems Principles (SOSP)_ , 2023, p. 611–626. [Online]. Available: https: //doi.org/10.1145/3600006.3613165 

- [43] LangChain, “Summarize text,” 2025. [Online]. Available: https: //python.langchain.com/docs/tutorials/summarization/ 

- [44] Y. Li, Z. Hu, E. Choukse, R. Fonseca, G. E. Suh, and U. Gupta, “Ecoserve: Designing carbon-aware ai inference systems,” 2025. [Online]. Available: https://arxiv.org/abs/2502.05043 

- [45] Linux, “turbostat - report processor frequency and idle statistics,” 2024. [Online]. Available: https://www.linux.org/docs/man8/turbostat.html 

- [46] Z. Liu, J. Leng, Z. Zhang, Q. Chen, C. Li, and M. Guo, “Veltair: towards high-performance multi-tenant deep learning services via adaptive compilation and scheduling,” in _Proceedings of the 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)_ , 2022, p. 388–401. [Online]. Available: https://doi.org/10.1145/3503222.3507752 

- [47] D. Lo, L. Cheng, R. Govindaraju, P. Ranganathan, and C. Kozyrakis, “Heracles: improving resource efficiency at scale,” in _Proceedings of the 42nd Annual International Symposium on Computer Architecture (ISCA)_ , 2015, p. 450–462. [Online]. Available: https://doi.org/10.1145/ 2749469.2749475 

- [48] C. Lu, H. Xu, K. Ye, G. Xu, L. Zhang, G. Yang, and C. Xu, “Understanding and optimizing workloads for unified resource management in large cloud platforms,” in _Proceedings of the Eighteenth European Conference on Computer Systems (EuroSys)_ , 2023, p. 416–432. [Online]. Available: https://doi.org/10. 1145/3552326.3587437 

- [49] Z. Luo, S. Son, S. Ratnasamy, and S. Shenker, “Harvesting memorybound cpu stall cycles in software with msh,” in _18th USENIX Symposium on Operating Systems Design and Implementation (OSDI)_ , 2024, pp. 57–75. [Online]. Available: https://www.usenix.org/ conference/osdi24/presentation/luo 

- [50] A. Margaritov, S. Gupta, R. Gonzalez-Alberquilla, and B. Grot, “Stretch: Balancing qos and throughput for colocated server workloads on smt cores,” in _2019 IEEE International Symposium on High Performance Computer Architecture (HPCA)_ . IEEE, 2019, pp. 15–27. [Online]. Available: https://doi.org/10.1109/HPCA.2019.00024 

- [51] Meta, “Introducing llama 3.2,” 2024. [Online]. Available: https: //www.llama.com/ 

- [52] Meta, “Our next-generation meta training and inference accelerator,” 2024. [Online]. Available: https://ai.meta.com/blog/next-generationmeta-training-inference-accelerator-AI-MTIA/ 

- [53] Meta, “Faiss,” 2025. [Online]. Available: https://github.com/ facebookresearch/faiss 

- [54] Microsoft, “Introducing the new bing. the ai-powered assistant for your search.” 2024. [Online]. Available: https://www.microsoft.com/enus/edge/features/the-new-bing?form=MT00D8 

- [55] S. Mittal and J. S. Vetter, “A survey of cpu-gpu heterogeneous computing techniques,” _ACM Computing Surveys (CSUR)_ , vol. 47, no. 4, pp. 1–35, 2015. 

- [56] S. Na, G. Jeong, B. H. Ahn, A. Jezghani, J. Young, C. J. Hughes, T. Krishna, and H. Kim, “Flexinfer: Flexible LLM inference with CPU computations,” in _Eighth Conference on Machine Learning and Systems (MLSYS)_ , 2025. [Online]. Available: https://openreview.net/forum?id=sFNRNTduKO 

- [57] S. Na, G. Jeong, B. H. Ahn, J. Young, T. Krishna, and H. Kim, “Understanding performance implications of llm inference on cpus,” in _2024 IEEE International Symposium on Workload Characterization (IISWC)_ . IEEE, 2024, pp. 169–180. [Online]. Available: https://doi.org/10.1109/IISWC63097.2024.00024 

- [58] N. C. Nachiappan, H. Zhang, J. Ryoo, N. Soundararajan, A. Sivasubramaniam, M. T. Kandemir, R. Iyer, and C. R. Das, “Vip: virtualizing ip chains on handheld platforms,” in _Proceedings of the 42nd Annual International Symposium on Computer Architecture_ , ser. ISCA ’15. New York, NY, USA: Association for Computing Machinery, 2015, p. 655–667. [Online]. Available: https://doi.org/10.1145/2749469.2750382 

- [59] N. Nassif, A. O. Munch, C. L. Molnar, G. Pasdast, S. V. Lyer, Z. Yang, O. Mendoza, M. Huddart, S. Venkataraman, S. Kandula, R. Marom, A. M. Kern, B. Bowhill, D. R. Mulvihill, S. Nimmagadda, V. Kalidindi, J. Krause, M. M. Haq, R. Sharma, and K. Duda, “Sapphire rapids: The next-generation intel xeon scalable processor,” in _2022 IEEE International Solid-State Circuits Conference (ISSCC)_ , vol. 65, 2022, pp. 44–46. [Online]. Available: https://doi.org/10.1109/ISSCC42614.2022.9731107 

- [60] Nvidia, “Triton inference server,” 2017. [Online]. Available: https: //github.com/triton-inference-server/server 

- [61] Nvidia, “Nvidia h100 tensor core gpu,” 2024. [Online]. Available: https://www.nvidia.com/en-us/data-center/h100/ 

- [62] OpenAI, “Gpt-4,” 2024. [Online]. Available: https://openai.com/index/ gpt-4/ 

- [63] OpenAI, “Api pricing,” 2025. [Online]. Available: https://openai.com/ api/pricing/ 

- [64] OpenAI, “Sharegpt,” 2025. [Online]. Available: https://sharegpt.com/ 

- [65] E. L. Padoin, L. L. Pilla, M. Castro, F. Z. Boito, P. O. Alexandre Navaux, and J.-F. M´ehaut, “Performance/energy trade-off in scientific computing: the case of arm big. little and intel sandy bridge,” _IET Computers & Digital Techniques_ , vol. 9, no. 1, pp. 27–35, 2015. [Online]. Available: https://doi.org/10.1049/iet-cdt.2014.0074 

- [66] P. Patel, E. Choukse, C. Zhang, I. n. Goiri, B. Warrier, N. Mahalingam, and R. Bianchini, “Characterizing power management opportunities for llms in the cloud,” in _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3_ , ser. ASPLOS ’24. New York, NY, USA: Association for Computing Machinery, 2024, p. 207–222. [Online]. Available: https://doi.org/10.1145/3620666.3651329 

- [67] P. Patel, E. Choukse, C. Zhang, A. Shah, I. n. Goiri, S. Maleki, and R. Bianchini, “Splitwise: Efficient generative llm inference using phase splitting,” in _Proceedings of the 51st Annual International Symposium on Computer Architecture (ISCA)_ , 2025, p. 118–132. [Online]. Available: https://doi.org/10.1109/ISCA59077.2024.00019 

- [68] L. Peng, W. Wu, S. Yi, X. Chen, C. Wang, S. Liang, Z. Wang, N. Xiao, Q. Li, M. Zhang, and J. Zhang, “Xharvest: Rethinking high-performance and cost-efficient ssd architecture with cxl-driven harvesting,” in _Proceedings of the 52nd Annual International Symposium on Computer Architecture (ISCA)_ , 2025, p. 434–449. [Online]. Available: https://doi.org/10.1145/3695053.3731028 

- [69] A. Pi, X. Zhou, and C. Xu, “Holmes: Smt interference diagnosis and cpu scheduling for job co-location,” in _Proceedings of the 31st International Symposium on High-Performance Parallel and Distributed Computing (HPDC)_ , 2022, p. 110–121. [Online]. Available: https://doi.org/10.1145/3502181.3531464 

- [70] J. R. Reinders, “Intel® avx-512 instructions,” 2017. [Online]. Available: https://www.intel.com/content/www/us/en/developer/ articles/technical/intel-avx-512-instructions.html 

- [71] RISC-V, “riscv-v-spec,” 2025. [Online]. Available: https://github.com/ riscvarchive/riscv-v-spec 

- [72] R. Rodrigues, I. Koren, and S. Kundu, “Performance and power benefits of sharing execution units between a high performance core and a low power core,” in _2014 27th International Conference on VLSI Design and 2014 13th International Conference on Embedded Systems_ , 2014, pp. 204–209. 

- [73] Y. Sheng, L. Zheng, B. Yuan, Z. Li, M. Ryabinin, B. Chen, P. Liang, C. R´e, I. Stoica, and C. Zhang, “Flexgen: high-throughput generative inference of large language models with a single gpu,” in _Proceedings of the 40th International Conference on Machine Learning_ , ser. ICML’23, 2023. 

- [74] M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, and B. Catanzaro, “Megatron-lm: Training multi-billion parameter language models using model parallelism,” 2020. [Online]. Available: https://arxiv.org/abs/1909.08053 

- [75] SPEC, “Specjbb 2015,” 2024. [Online]. Available: https://www.spec. org/jbb2015/ 

- [76] N. Stephens, S. Biles, M. Boettcher, J. Eapen, M. Eyole, G. Gabrielli, M. Horsnell, G. Magklis, A. Martinez, N. Premillieu, A. Reid, A. Rico, and P. Walker, “The arm scalable vector extension,” _IEEE Micro_ , vol. 37, no. 2, p. 26–39, Mar. 2017. [Online]. Available: https://doi.org/10.1109/MM.2017.35 

- [77] J. Stojkovic, P. A. Misra, Goiri, S. Whitlock, E. Choukse, M. Das, C. Bansal, J. Lee, Z. Sun, H. Qiu, R. Zimmermann, S. Samal, B. Warrier, A. Raniwala, and R. Bianchini, “Smartoclock: Workloadand risk-aware overclocking in the cloud,” in _2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)_ , 2024, pp. 437–451. 

- [78] J. Stojkovic, C. Zhang, I. n. Goiri, E. Choukse, H. Qiu, R. Fonseca, J. Torrellas, and R. Bianchini, _TAPAS: Thermal- and Power-Aware Scheduling for LLM Inference in Cloud Platforms_ . New York, NY, USA: Association for Computing Machinery, 2025, p. 1266–1281. [Online]. Available: https://doi.org/10.1145/3676641.3716025 

- [79] L. Sun, C. Li, X. Hou, T. Huang, C. Xu, X. Wang, G. Bao, B. Sun, S. Rui, and M. Guo, “Jigsaw: Taming bev-centric perception on dual-soc for autonomous driving,” in _2024 IEEE Real-Time Systems Symposium (RTSS)_ , 2024, pp. 280–293. 

- [80] I. SYSTEMS, “Ieit systems launches cpu inference servers to accelerate enterprise ai adoption,” 2025. [Online]. Available: https:// www.linkedin.com/feed/update/urn:li:activity:7308400653385486336/ 

- [81] M. Taram, X. Ren, A. Venkat, and D. Tullsen, “Secsmt: Securing smt processors against contention-based covert channels,” in _31st USENIX Security Symposium (USENIX Security 22)_ . Boston, MA: USENIX Association, Aug. 2022, pp. 3165–3182. [Online]. Available: https: //www.usenix.org/conference/usenixsecurity22/presentation/taram 

- [82] G. Team, “Gemma 2: Improving open language models at a practical size,” 2024. [Online]. Available: https://arxiv.org/abs/2408.00118 

- [83] P.-. Team, “Phi-3 technical report: A highly capable language model locally on your phone,” 2024. [Online]. Available: https: //arxiv.org/abs/2404.14219 

- [84] Q. Team, “Qwen3,” April 2025. [Online]. Available: https://qwenlm. github.io/blog/qwen3/ 

- [85] L. Torvalds, “Perf pmu events on sapphirerapids,” 2023. [Online]. Available: https://github.com/torvalds/linux/blob/master/ tools/perf/pmu-events/arch/x86/sapphirerapids/spr-metrics.json 

- [86] TPC, “Tpc-h version 2 and version 3,” 2024. [Online]. Available: https://www.tpc.org/tpch/ 

- [87] A. Venkat, H. Basavaraj, and D. M. Tullsen, “Composite-isa cores: Enabling multi-isa heterogeneity using a single isa,” in _2019 IEEE International Symposium on High Performance Computer Architecture (HPCA)_ . IEEE, 2019, pp. 42–55. [Online]. Available: https://doi.org/10.1109/HPCA.2019.00026 

- [88] A. Venkat and D. M. Tullsen, “Harnessing isa diversity: design of a heterogeneous-isa chip multiprocessor,” in _Proceeding of the 41st Annual International Symposium on Computer Architecuture (ISCA)_ , 2014, p. 121–132. 

- [89] X. Wang, H. He, Y. Li, C. Li, X. Hou, J. Wang, Q. Chen, J. Leng, M. Guo, and L. Wang, “Not all resources are visible: Exploiting fragmented shadow resources in shared-state scheduler architecture,” in _Proceedings of the 2023 ACM Symposium on Cloud Computing (SoCC)_ , 2023, p. 109–124. [Online]. Available: https://doi.org/10.1145/3620678.3624650 

in _Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2_ , ser. ASPLOS ’25. New York, NY, USA: Association for Computing Machinery, 2025, p. 355–372. [Online]. Available: https://doi.org/10.1145/3676641.3716283 

   - [91] X. Wang, C. Li, L. Sun, Q. Lyu, X. Hou, J. Leng, and M. Guo, “Sheeo: Continuous energy efficiency optimization in autonomous embedded systems,” in _2024 IEEE 42nd International Conference on Computer Design (ICCD)_ , 2024, pp. 496–503. 

   - [92] X. Wang, C. Li, L. Zhang, X. Hou, Q. Chen, and M. Guo, “Exploring efficient microservice level parallelism,” in _2022 IEEE International Parallel and Distributed Processing Symposium (IPDPS)_ , 2022, pp. 223–233. 

   - [93] X. Wang, Y. Zhuansun, C. Li, J. Wang, X. Hou, L. Sun, L. Wang, and M. Guo, “Asymserve: Demystifying and optimizing llm serving efficiency on cpu acceleration units,” in _Advanced Parallel Processing Technologies_ , C. Li, X. Qian, D. Gizopoulos, and B. Grot, Eds. Singapore: Springer Nature Singapore, 2026, pp. 231–245. 

   - [94] Y. Wang, K. Arya, M. Kogias, M. Vanga, A. Bhandari, N. J. Yadwadkar, S. Sen, S. Elnikety, C. Kozyrakis, and R. Bianchini, “Smartharvest: harvesting idle cpus safely and efficiently in the cloud,” in _Proceedings of the Sixteenth European Conference on Computer Systems (EuroSys)_ , 2021, p. 1–16. [Online]. Available: https://doi.org/10.1145/3447786.3456225 

   - [95] Q. Weng, W. Xiao, Y. Yu, W. Wang, C. Wang, J. He, Y. Li, L. Zhang, W. Lin, and Y. Ding, “Mlaas in the wild: Workload analysis and scheduling in large-scale heterogeneous gpu clusters,” in _19th USENIX Symposium on Networked Systems Design and Implementation (NSDI 22)_ . Renton, WA: USENIX Association, Apr. 2022, pp. 945–960. [Online]. Available: https://www.usenix.org/ conference/nsdi22/presentation/weng 

   - [96] Y. Xue, Y. Liu, L. Nai, and J. Huang, “Hardware-assisted virtualization of neural processing units for cloud platforms,” in _2024 57th IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , 2024, pp. 1–16. 

   - [97] A. Yasin, “A top-down method for performance analysis and counters architecture,” in _2014 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)_ . IEEE, 2014, pp. 35–44. [Online]. Available: https://doi.org/10.1109/ISPASS.2014.6844459 

   - [98] H. Yu, A. M. Peters, A. Akshintala, and C. J. Rossbach, “Ava: Accelerated virtualization of accelerators,” in _Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems_ , ser. ASPLOS ’20. New York, NY, USA: Association for Computing Machinery, 2020, p. 807–825. [Online]. Available: https://doi.org/10.1145/3373376. 3378466 

   - [99] L. Zhang, Y. Qian, X. Wang, M. Thakker, D. Wang, J. Yu, H. Wu, Y. Hu, J. Li, Y. Qian, and S. Zhao, “Covomix2: Advancing zero-shot dialogue generation with fully non-autoregressive flow matching,” 2025. [Online]. Available: https://arxiv.org/abs/2506.00885 

   - [100] L. Zhang, Y. Qian, L. Zhou, S. Liu, D. Wang, X. Wang, M. Yousefi, Y. Qian, J. Li, L. He, S. Zhao, and M. Zeng, “Covomix: advancing zeroshot speech generation for human-like multi-talker conversations,” in _Proceedings of the 38th International Conference on Neural Information Processing Systems_ , ser. NIPS ’24. Red Hook, NY, USA: Curran Associates Inc., 2024. 

   - [101] Y. Zhang, M. A. Laurenzano, J. Mars, and L. Tang, “Smite: Precise qos prediction on real-system smt processors to improve utilization in warehouse scale computers,” in _2014 47th Annual IEEE/ACM International Symposium on Microarchitecture_ . IEEE, 2014, pp. 406–418. [Online]. Available: https://doi.org/10.1109/MICRO.2014.53 

   - [102] Y. Zhong, S. Liu, J. Chen, J. Hu, Y. Zhu, X. Liu, X. Jin, and H. Zhang, “Distserve: Disaggregating prefill and decoding for goodput-optimized large language model serving,” in _18th USENIX Symposium on Operating Systems Design and Implementation (OSDI)_ , 2024, pp. 193–210. [Online]. Available: https://www.usenix.org/ conference/osdi24/presentation/zhong-yinmin 

   - [103] L. Zhou, L. N. Bhuyan, and K. K. Ramakrishnan, “Gemini: Learning to manage cpu power for latency-critical search engines,” in _2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , 2020, pp. 637–349. 

- [90] X. Wang, X. Hou, C. Li, Y. Li, D. Liu, G. Xu, G. Yang, L. Zhang, Y. Wu, X. Yuan, Q. Chen, and M. Guo, “Exist: Enabling extremely efficient intra-service tracing observability in datacenters,” 

