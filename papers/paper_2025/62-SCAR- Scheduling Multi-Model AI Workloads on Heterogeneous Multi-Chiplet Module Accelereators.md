2024 57th IEEE/ACM International Symposium on Microarchitecture (MICRO) 

## SCAR: Scheduling Multi-Model AI Workloads on Module Accelerators Heterogeneous Multi-Chiplet 

Mohanad Odema 

Luke Chen 

Hyoukjun Kwon Mohammad Abdullah Al Faruque 

Univ. of California, Irvine Univ. of California, Irvine Univ. of California, Irvine Univ. of California, Irvine Irvine, CA, USA Irvine, CA, USA Irvine, CA, USA Irvine, CA, USA modema@uci.edu panwangc@uci.edu hyoukjun.kwon@uci.edu alfaruqu@uci.edu 

Abstract—Emerging multi-model workloads with heavy models like recent large language models significantly increased the compute and memory demands on hardware. To address such increasing demands, designing a scalable hardware architecture became a key problem. Among recent solutions, the 2.5D silicon interposer multi-chip module (MCM)-based AI accelerator has been actively explored as a promising scalable solution due to their significant benefits in the low engineering cost and composability. However, previous MCM accelerators are based on homogeneous architectures with fixed dataflow, which encounter major challenges from highly heterogeneous multi-model workloads due to their limited workload adaptivity. 

Therefore, in this work, we explore the opportunity in the heterogeneous dataflow MCM AI accelerators. We identify the scheduling of multi-model workload on heterogeneous dataflow MCM AI accelerator is an important and challenging problem due to its significance and scale, which reaches O(10[56] ) even for a two-model workload on 6x6 chiplets. We develop a set of heuristics to navigate the huge scheduling space and codify them into a scheduler, SCAR, with advanced techniques such as interchiplet pipelining. Our evaluation on ten multi-model workload scenarios for datacenter multitenancy and AR/VR use-cases has shown the efficacy of our approach, achieving on average 27.6% and 29.6% less energy-delay product (EDP) for the respective applications settings compared to homogeneous baselines. 

Index Terms—AI accelerators, Multichip modules, Chiplets, Scheduling algorithms, Performance analysis. 

## I. INTRODUCTION 

Recent artificial intelligence (AI) inference workloads have increased their scale in both of the model size (e.g., large language models [7], [69]) and the number of models deployed together (e.g., augmented and virtual reality; AR/VR [38]), which constructs multi-model workloads with heavier models than those in the past. Such trends led to heavy demands on compute capabilities in AI hardware from edge to cloud devices. As an approach to scale up the hardware for AI and increase the compute capability, chiplet-based multi-chip module (MCM) package has emerged as a promising solution [55], [64], [68], [71]. Such MCM packages facilitate the scaling of AI hardware based on their composability and cost-effectiveness, unlike monolithic designs, which are often constrained by fabrication yields, power, heat, and other engineering costs such as verification [50]. 

Researchers have actively explored the MCM for AI, focusing on the dataflow mapping (i.e., loop ordering, parallelization, and tiling) of each layer and workload orchestration onto chiplets considering the network-on-package (NoP) and other communication constraints [55], [64], [68], [71]. For example, 

Simba [64] proposed a scalable MCM inference architecture that enables chiplets to either act as standalone inference engines or collaborate as groups for a layer. Although such works have successfully delivered promising performance and energy efficiency than monolithic designs, they mostly focused on single-model workloads targeting homogeneous chiplets. Unlike single-model workloads, multi-model workloads introduce major challenges to such homogeneous MCMs because of the ML operator heterogeneity (e.g., operator types and tensor sizes) and resulting diverse dataflow preferences [37]. Also, multi-model workloads often involve model level dependency and concurrency [34], [37], [38], [51], [56], which adds complex considerations to the scheduling problem. 

Therefore, considering the new trend with multi-model AI workloads in industry, such as multi-tenancy [23], [40], [72] and AR/VR [38], we explore heterogeneous chiplet-based MCM with AI accelerator chiplets with various dataflows, as a future-proof option. To exploit the benefits of heterogeneous MCM accelerators, we consider inter-layer pipelining to enhance in-package data reuse and reduce offchip traffic. We formulate the scheduling problem and develop effective heuristics to navigate the huge scheduling space, whose problem scale is as big as O(10[56] ) even for a two-model workload (ResNet-50 [24] and UNet [63]) on a 6x6 chiplet MCM AI accelerator system (as in Simba [64]). 

We evaluate ten MCMs including seven heterogeneous MCMs on ten multi-model scenarios: the first five scenarios are curated using MLPerf inference benchmark [62] representing datacenter multi-tenancy scenarios. The models are selected based on recent datacenter model usage trends [23], [29] and the trend of language model adoptions (e.g., GPTL [60]), future-proofing emerging AI workloads such as AI assistant [47]. The other five scenarios are curated for AR/VR usage scenarios from XRBench as a practical use case for edge multi-model workloads [38]. 

The evaluation results show that heterogeneous MCM combined with our scheduling method is promising for heavy multi-model workloads, which is projected by recent trend. Compared to the homogeneous MCM [64] running NVDLA [52] and Shi-diannao [16] style dataflows, heterogeneous MCM, on average, achieved 27.6% and 29.6% less energy-delay product (EDP) in each domain, respectively. We also showcase that our scheduler can identify schedules that can reduce EDP to 0.3× that of single-model schedulers like NN-baton [68]. We summarize our contributions as follows: 

979-8-3503-5057-9/24/$31.00 ©2024 IEEE 565 DOI 10.1109/MICRO61859.2024.00049 

TABLE I 

NOTATION USED IN THE FORMULATION. 

|NOTAT|ION USED IN THE FORMULATION.|
|---|---|
|Notation|Description|
|Sc<br>mi<br>layeri,j|Multi-model workload scenario<br>the i-th model from scenario Sc<br>the j-th layer from model i|
|H<br>C<br>ci<br>DF<br>ndfi<br>BWoffchip<br>BWnop|MCM hardware<br>Set of accelerator chiplets on H<br>the i-th chiplet from C<br>set of supported datafows on H<br>Number of chiplets adopting the i-th datafow<br>offchip bandwidth<br>Network-on-package bandwidth|
|df<br>NP E<br>BWnoc<br>Szmem|Datafow<br>Number of processing engines<br>Network-on-chip bandwidth<br>The memory size in c|
|TW(Sc)<br>tw(Sc)<br>Ts<br>Ttw<br>L(tw(Sc))<br>SG<br>sg(tw(Sc))|The set of time windows for Sc<br>an execution time window for Sc on H<br>time window start<br>time window end<br>set of layers executable on H during tw(Sc)<br>Set of all valid segments for tw(Sc)<br>a layer segment from L(tw(Sc))|
|Sptw(Sc)<br>Spsg(tw)<br>SST W(tw(Sc), H)<br>SSSc(H)<br>sched(Sc,H)|Time window partioning space for Sc<br>Layer segmentation space at tw<br>The scheduling space for tw(Sc) on H<br>The overall scheduling space for Sc on H<br>A scheduling instance for Sc on H|
|Lati<br>j(A)<br>Ei<br>j(A)<br>Szdata<br>nhops<br>nsplits|Latency evaluation for A given identifers i, j<br>Energy evaluation for A given identifers i, j<br>Size of transmission data<br>Number of hops from src to destination<br>Number of time window splits|



## III. SYSTEM MODELING AND PROBLEM FORMULATION 

To develop a systematic approach to navigate complex search space, we formulate the scheduling problem of multimodel workloads on a heterogeneous MCM AI accelerator. 

## A. Base Formulation 

To formulate the MCM scehduling problem, we first define multi-model workload scenario (Sc) and MCM hardware (H). 

We formulate the workload in the granularity of layers in each model. Therefore, we formulate a multi-model workload scenario (Sc) as the collection of layers in the models included in the scenario. Letting the number of models included in Sc as |Sc| and the number of layers included in a model m as |m|, we define Sc as follows: 

## Definition 1. Multi-model Workload Scenario (Sc) 

**==> picture [179 x 12] intentionally omitted <==**

## where layer(i,j) refers to the j-th layer of model i in Sc. 

AI accelerator chiplets consist of a PE array, memory, and on-chip interconnection among memory and PEs. In addition to them, we also include the dataflow in the formulation to model heterogeneous chiplet MCM AI accelerator. Accordingly, we define an AI accelerator chiplet (c) as follows: 

In Definition 2, df refers to the dataflow, NP E is the number of PEs, BWnoc is the NoC bandwidth, BWmem is the chipletlevel shared memory bandwidth, and Szmem is the memory size in c. 

Based on the definition of the chiplet, we formulate the MCM accelerator as the set of chiplets (C = {c1, c2, ..., cNcpl }), NoP, and off-chip interface as follows: 

## Definition 3. MCM AI Accelerator (H) 

**==> picture [127 x 12] intentionally omitted <==**

Unless otherwise stated, we assume the 2D mesh topology for NoP like Simba [64], and chiplets on two sides (left and right) of the packages have off-chip interfaces. 

## B. Workload Partitioning Space 

To reduce the complexity of the scheduling problem, we adopt a multi-level scheduling method, which splits the endto-end workload defined in the layer granularity into coarsegrained layer groups, termed as the time window. Figure 3 shows an example of the time window that contains six layers from Model A and five layers from Model B. 

A time window (tw) is defined by the start time and the duration (TS and Ttw) and a set of assigned layers to the time window, as shown in Definition 4. 

## Definition 4. Time Window (tw) 

For a target workload scenario Sc, a time window tw is defined as follows: 

**==> picture [93 x 11] intentionally omitted <==**

where L = {l|l ∈ Sc} 

The time window describes a set of layers to be executed on an MCM AI accelerator package, which is used for describing package level scheduling. For each chiplet, we define a finergrained group of layers within a time window. We term the sub-set of layers within a time window as segment. 

## Definition 5. Segment (sg) 

For a time window tw(Sc) and its layers L(tw(Sc)), the segment sg(tw(Sc)) is defined as follows: 

**==> picture [137 x 11] intentionally omitted <==**

To develop a systematic optimization algorithm for layer segmentation i each time window, we need to define the conditions of valid layer segments, provided as follows: 

Theorem 1. The validity of segments in a time window For a time window tw(Sc) and its layers L(tw(Sc)), let the set of all segments for tw(Sc) be SG, then SG is valid if the following condition is satisfied: 

� sg = L(tw(Sc)) ∧ [∀sgi = sgj ∈ SG, sgi ∩ sgj = ∅] sg∈SG 

## Definition 2. AI Accelerator Chiplet (c) 

**==> picture [173 x 11] intentionally omitted <==**

Theorem 1 states two conditions (1) the set of segments needs to cover all the layers in their time window for completing assigned layer computations for the time window and 

568 

Authorized licensed use limited to: Access paid by The UC Irvine Libraries. Downloaded on December 28,2024 at 09:36:40 UTC from IEEE Xplore.  Restrictions apply. 

## A. MCM Reconfiguration Engine (MCM-Reconfig) 

The MCM-Reconfig engine at the top-level step receives the multi-model workload descriptions with layer information in each model, layer dependency, and expected latency and energy of each layer on each chiplet class offline-analyzed by MAESTRO [35]. The MCM-Reconfig engine is responsible for the window assignment in Figure 3, which (1) generates candidate time window partitioning strategies via sampling a set of discrete points in time as boundary points, and (2) assigns layers from models to each time window. As the final assignment of layers to chiplets is not known apriori, the decisions in MCM-Reconfig engine are based on expected execution times. Formally, given |DF | dataflow style classes, the expected execution latency for a layer l is: 

**==> picture [202 x 31] intentionally omitted <==**

where ndfi indicates the number of class i chiplets integrated onto the MCM having |C| chiplets in total; Latl→i is layer l latency when scheduled on the class i chiplet, which is retrieved offline from latency database generated by MAESTRO [35], [36]. The average execution time information is utilized in MCM-Reconfig engine for window assignment process illustrated in Figure 3. 

Time Windows Characterization. MCM-Reconfig engine first specifies the number of windows, through a hyperparameter, nsplits, to explore proper cut points for each model. For example, in Figure 3, the model A has a cut after layer 6, which led to having layers 1-6 in Window 1. The worst-case latency experienced by any model is set as the time horizon to be partitioned into periodic time windows. 

Greedy Layer Packing Algorithm. We adopt a first-fit greedy-packing heuristic to assign layers to execution time windows if their execution time is expected to finish within the time window boundaries (see Algorithm 1). Any layer whose execution time lies across two time windows is deferred to the next time window. Through this approach, we enable (i) running low-latency layers in earlier windows (restricts starvation). (ii) dynamically controlling the number of time windows by skipping trivial time windows with no workloads. 

Based on our analysis of the periodic window characterization with greedy layer packing using a workload of UNet and GPT2-L against a layer-optimal approach. We found the rate of EDP improvement stagnated after 4 splits. We set nsplits=4 (5 time windows) as our default unless otherwise stated. 

## B. Provisioner Engine (PROV) 

The PROV engine provides an initial estimate on the number of chiplet needed by each model workload in every time window from a candidate partitioning strategy. PROV assignments are agnostic to the underlying chiplets’ properties (dataflow, resources), and hence we refer to chiplets in this state as nodes. We implement the PROV engine to support exhaustive search or rule-based node distribution assignments. A uniform distribution rule allocates Ni nodes to the i[th] model as follows: 

Algorithm 1 Greedy Layer Packing Algorithm Input: M (workloads), T , C, DF Output: L2W (Layer(s) to windows assignments) 

|Out|put: L2W (Layer(s) to windows assignments)|put: L2W (Layer(s) to windows assignments)|
|---|---|---|
|1:|Function LAYERASSIGNMENT(M, C, T)||
|2:<br>3:<br>4:<br>5:<br>6:<br>7:<br>8:<br>9:|for m∈M do<br>exec<br>win= ()<br>win<br>idx, used<br>cycles= 0,0<br>for l ∈m do<br>E(Lat(l)) = �|DF|<br>i=1<br>ndfi<br>|C| ×Lat(l →i)<br>while True do<br>if win<br>idx==|T | then<br>Slack = None||
|10:<br>11:<br>12:<br>13:<br>14:|else<br>Slack = ρ[win<br>idx]−used<br>cycles<br>if Slack == None or E(Lat(l))<=Slack then<br>exec<br>win += (l,)<br>used<br>cycles += E(Lat(l))||
|15:<br>16:|Break<br>else||
|17:<br>18:<br>19:<br>20:|L2W[win<br>idx][m] =exec<br>win<br>used<br>cycles=T[win<br>idx]<br>exec<br>win= ()<br>win<br>idx += 1||
|21:|L2W[win<br>idx][m] =exec<br>win||



**==> picture [191 x 27] intentionally omitted <==**

where E(Pi) represents the expected value of a target performance optimization metric (latency, energy, EDP) for the model i. E(Pi) is computed in a manner similar to the expectation formula in Equation (1). 

We ensure every model in the time window is assigned at least one node to progress its execution. The rules enable trading off search complexity for coverage. We analyze the efficacy of the uniform distribution compared to the exhaustive search in Section V. 

## C. Segmentation Engine (SEG) 

The SEG module is instantiated every time window to partition topologically sorted model layers into layer segments (Definition 5) that are mappable to computing nodes for exclusive execution throughout the time window. Different segmentation choices reflect various trade-off points between the layer-sqeuencing and layer-pipelining features: the former concerns with execution locality on the same node; the latter specifies inter-layer and -chiplet pipelining opportunities. 

Segmentation Search Space. A segmentation candidate is represented by a sequence of splitting points. Candidate splitting points for a model can be specified after each layer provided to the SEG. Given |Li| and |Ni| as the respective number of layers and number of assigned nodes from the PROV to model workload mi, the max number of segments that can be generated for mi is upper bounded by Ni. Thus, the overall segmentation space complexity becomes O(Πi �NLi−i 1�), We incorporate the following heuristics to manage complexity. 

Heuristic 1. Product to summation reduction. We reduce complexity by leveraging the independence of segments from different models to divide the search into a two-step process: 

571 

Authorized licensed use limited to: Access paid by The UC Irvine Libraries. Downloaded on December 28,2024 at 09:36:40 UTC from IEEE Xplore.  Restrictions apply. 

and 57.6% from Simba (Shi); 33.0% and 28.3% from Simba (NVD). For Scenario 3, Simba (NVD) remained superior, achieving 79.7% EDP reduction Het-Sides. These evaluations follow the trends from our rule-based results in Table IV. Ablation on Greedy Packing Algorithm. Using Scenario 4 and Het-Sides, we test the efficacy of our first-fit greedy layer packing algorithm against a uniform packing baseline, distributing layers uniformly across time windows. Ours achieved 21.8% speedup and 8.6% energy reduction. 

## F. Summary of Results and Main Insights 

- We summarize our main insights and findings as follows. 

- Heterogeneous MCM patterns improve performance for heavy and diverse multi-model workloads (scenarios 4-5). 

- Homogeneous MCM patterns are more suited for small multi-model workloads (scenarios 1-3). 

- Heterogeneous MCM patterns with diverse pipelining options (Het-Sides) are superior to heterogeneous patterns with homogeneous pipelining options (Het-CB). 

- The target optimization objective is crucial in identifying the best integration strategy. In EDP search scenario 4, Het-sides outperformed all other strategies on EDP, but not on pure energy consumption. 

- Topology and number of resources affect the extent of performance improvement for heterogeneous strategies. 

Our findings show that understanding multi-model workload characteristics and usage scenarios is crucial for identifying the best MCM integration strategy for a target objective. 

## VI. DISCUSSION AND LIMITATIONS 

Multi-model optimization targets. We experimented with different optimization targets (latency, EDP, energy) for our scenarios, and showed that the top performing strategy can change based on the target objective. As multi-model workloads evolve, it may be desirable to assign separate optimization targets for different models within a scenario (EDP v. lat). One practical way to achieve this in our framework is by adding a constraint in our EDP search, invalidating schedules that have certain models violate a latency constraint (i.e., the EDP search becomes lower bounded by the latency search). 

Heterogeneous chiplets technology. Heterogeneous chiplet integration has become a viable, cost-effective approach to design state-of-the-art AI systems. Nvidia’s world-class superchips are a successful example of heterogeneous on-package integration (e.g., Grace-Blackwell (1 CPU + 2 GPUs) [14]). The success of these systems and others (AMD’s MI300X [1]) is testament to the hardware manufacturers’ investment in chiplets technology, where through advanced manufacturing processes and heterogeneous integration capabilities, the development of MCM AI accelerators (like Nvidia’s Simba [64]) becomes more accessible, allowing chiplet modifications/replacement in MCM hardware at lower costs without requiring a complete overhaul of the entire package. 

Scheduler Software Integration. SCAR can be integrated on top of existing compiler infrastructure. The advanced scheduling techniques supported by the scheduler (dynamic 

## TABLE VII 

COMPARISON AGAINST PRIOR RELATED SCHEDULING WORKS. 

|COMPARISO|N AGAINST PRIOR RELATED SCHEDULING WORKS.|
|---|---|
|Work|Chiplet-based<br>Multi-<br>Inter-Layer<br>Heterog-<br>Systems<br>Models<br>Pipelining<br>Aware|
|Simba [64]<br>Tangram [19]<br>NN-baton [68]<br>SET [8]<br>Gemini [9]|✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|
|Herald [37]<br>MAGMA [32]<br>Planaria [21]<br>Veltair [42]<br>MoCA [33]|✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|
|This Work|✓<br>✓<br>✓<br>✓|



chiplet regrouping, inter-chiplet pipelining) represent highlevel abstractions of the computational graphs that can be transformed through standard compiler software (e.g., MLIR [39]) to representations suited for the underlying hardware. For example, dynamic chiplets regrouping is correspondent to graph partitioning, where a model’s computational graph is divided into smaller subgraphs, each associated with the set of computing nodes assigned during the corresponding time window. The subgraphs can then be transformed to lower representations covering the details of buffer management, dieto-die communication, memory R/W requests, I/O, all the way to the transformations covering the dataflow features (loop reordering, spatial unrolling) for the specialized accelerators. 

## VII. RELATED WORKS 

Scheduler for Accelerators. Table VII compares our work against prior scheduling works. As shown, the related works can be categorized into two groups: one which has considered aspects of inter-layer pipelining and chiplet-based systems [8], [9], [19], [64], [68], and another that focused on multi-model workloads on heterogeneous platforms [21], [32], [33], [37], [42]. Only this work addressed MCM, multi-model workloads, inter-layer pipelining, and heterogeneous dataflow. 

Multi-chiplet Modules. Several works proposed to address the scalability challenge for DNN acceleration via MCM integration [3], [28], [55], [64], [68]. Most notably, Simba [64] pioneered a scalable deep learning MCM inference accelerator leveraging non-uniform work partitioning, communicationaware data placement, and cross-layer pipelining. Intra- and Inter-layer Parallelism. Prior works explored intra-layer parallelism to maximize DNN performance efficiency by partitioning DNN layers into smaller, parallelizable tiles [25]–[27], [43], [57], [73], [74]. Other works studied the inter-layer scheduling space to compensate for workloads with low degrees of parallelism [6], [8], [19], [31], [45], [53], [78]. 

## VIII. CONCLUSION 

In this work, we explored the scheduling space of a new class of MCM accelerator architecture, heterogeneous MCM AI accelerator, targeting multi-model AI workloads. We identify that the scheduling problem is intractably large but multilevel problem formulation and heuristics we proposed are effective for the large-scale scheduling problem. The results also show that heterogeneous MCM accelerator is beneficial for multi-model workloads, which motivates further exploration. 

576 

Authorized licensed use limited to: Access paid by The UC Irvine Libraries. Downloaded on December 28,2024 at 09:36:40 UTC from IEEE Xplore.  Restrictions apply. 

## REFERENCES 

- [1] AMD. Amd instinct mi300x accelerator. https://www.amd.com/content/ dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinctmi300x-data-sheet.pdf. 

- [2] Apple. Apple Vision Pro Specs. https://www.apple.com/apple-visionpro/specs/, 2024. 

- [3] Akhil Arunkumar, Evgeny Bolotin, Benjamin Cho, Ugljesa Milic, Eiman Ebrahimi, Oreste Villa, Aamer Jaleel, Carole-Jean Wu, and David Nellans. Mcm-gpu: Multi-chip-module gpus for continued performance scalability. ACM SIGARCH Computer Architecture News, 45(2):320– 332, 2017. 

- [4] Eunjin Baek, Dongup Kwon, and Jangwoo Kim. A multi-neural network acceleration architecture. In 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA), pages 940–953. IEEE, 2020. 

- [5] Noah Beck, Sean White, Milam Paraschou, and Samuel Naffziger. ‘zeppelin’: An soc for multichip architectures. In 2018 IEEE International Solid-State Circuits Conference-(ISSCC), pages 40–42. IEEE, 2018. 

- [6] Halima Bouzidi, Mohanad Odema, Hamza Ouarnoughi, Smail Niar, and Mohammad Abdullah Al Faruque. Map-and-conquer: Energy-efficient mapping of dynamic neural nets onto heterogeneous mpsocs. In 2023 60th ACM/IEEE Design Automation Conference (DAC), pages 1–6. IEEE, 2023. 

- [7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 

- [8] Jingwei Cai, Yuchen Wei, Zuotong Wu, Sen Peng, and Kaisheng Ma. Inter-layer scheduling space definition and exploration for tiled accelerators. In Proceedings of the 50th Annual International Symposium on Computer Architecture (ISCA), pages 1–17, 2023. 

- [9] Jingwei Cai, Zuotong Wu, Sen Peng, Yuchen Wei, Zhanhong Tan, Guiming Shi, Mingyu Gao, and Kaisheng Ma. Gemini: Mapping and architecture co-exploration for large-scale dnn chiplet accelerators. In 2024 IEEE International Symposium on High-Performance Computer Architecture (HPCA), pages 156–171. IEEE, 2024. 

- [10] Prasanth Chatarasi, Hyoukjun Kwon, Angshuman Parashar, Michael Pellauer, Tushar Krishna, and Vivek Sarkar. Marvel: a data-centric approach for mapping deep learning operators on spatial accelerators. ACM Transactions on Architecture and Code Optimization (TACO), 19(1):1–26, 2021. 

- [11] Seungbeom Choi, Sunho Lee, Yeonjae Kim, Jongse Park, Youngjin Kwon, and Jaehyuk Huh. Multi-model machine learning inference serving with gpu spatial partitioning. arXiv preprint arXiv:2109.01611, 2021. 

- [12] Seungbeom Choi, Sunho Lee, Yeonjae Kim, Jongse Park, Youngjin Kwon, and Jaehyuk Huh. Serving heterogeneous machine learning models on {Multi-GPU} servers with {Spatio-Temporal} sharing. In 2022 USENIX Annual Technical Conference (USENIX ATC 22), pages 199–216, 2022. 

- [13] Daniel Crankshaw, Xin Wang, Guilio Zhou, Michael J Franklin, Joseph E Gonzalez, and Ion Stoica. Clipper: A {Low-Latency} online prediction serving system. In 14th USENIX Symposium on Networked Systems Design and Implementation (NSDI 17), pages 613–627, 2017. 

- [14] CRN. Nvidia Reveals Next-Gen Blackwell GPUs, Promised To ’Unlock Breakthroughs’ In GenAI. https://www.crn.com/news/componentsperipherals/2024/nvidia-reveals-next-gen-blackwell-gpus-promised-tounlock-breakthroughs-in-genai, 2024. 

- [15] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 

- [16] Zidong Du, Robert Fasthuber, Tianshi Chen, Paolo Ienne, Ling Li, Tao Luo, Xiaobing Feng, Yunji Chen, and Olivier Temam. Shidiannao: Shifting vision processing closer to the sensor. In Proceedings of the 42nd Annual International Symposium on Computer Architecture, pages 92–104, 2015. 

- [17] FacebookResearch. Hrvit-b1. https://github.com/facebookresearch/ HRViT/blob/main/models/hrvit.py#L1125-L1155, 2022. 

- [18] Amin Firoozshahian, Joel Coburn, Roman Levenstein, Rakesh Nattoji, Ashwin Kamath, Olivia Wu, Gurdeepak Grewal, Harish Aepala, Bhasker Jakka, Bob Dreyer, et al. Mtia: First generation silicon targeting 

meta’s recommendation systems. In Proceedings of the 50th Annual International Symposium on Computer Architecture, pages 1–13, 2023. 

- [19] Mingyu Gao, Xuan Yang, Jing Pu, Mark Horowitz, and Christos Kozyrakis. Tangram: Optimized coarse-grained dataflow for scalable nn accelerators. In Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems, pages 807–820, 2019. 

- [20] Liuhao Ge, Zhou Ren, Yuncheng Li, Zehao Xue, Yingying Wang, Jianfei Cai, and Junsong Yuan. 3d hand shape and pose estimation from a single rgb image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10833–10842, 2019. 

- [21] Soroush Ghodrati, Byung Hoon Ahn, Joon Kyung Kim, Sean Kinzer, Brahmendra Reddy Yatham, Navateja Alla, Hardik Sharma, Mohammad Alian, Eiman Ebrahimi, Nam Sung Kim, Cliff Young, and Hadi Esmaeilzadeh. Planaria: Dynamic architecture fission for spatial multitenant acceleration of deep neural networks. In 2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), pages 681–697. IEEE, 2020. 

- [22] Johann Hauswald, Yiping Kang, Michael A Laurenzano, Quan Chen, Cheng Li, Trevor Mudge, Ronald G Dreslinski, Jason Mars, and Lingjia Tang. Djinn and tonic: Dnn as a service and its implications for future warehouse scale computers. ACM SIGARCH Computer Architecture News, 43(3S):27–40, 2015. 

- [23] Kim Hazelwood, Sarah Bird, David Brooks, Soumith Chintala, Utku Diril, Dmytro Dzhulgakov, Mohamed Fawzy, Bill Jia, Yangqing Jia, Aditya Kalro, et al. Applied machine learning at facebook: A datacenter infrastructure perspective. In 2018 IEEE International Symposium on High Performance Computer Architecture (HPCA), pages 620–629. IEEE, 2018. 

- [24] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition, 2015. 

- [25] Kartik Hegde, Po-An Tsai, Sitao Huang, Vikas Chandra, Angshuman Parashar, and Christopher W Fletcher. Mind mappings: enabling efficient algorithm-accelerator mapping space search. In Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, pages 943–958, 2021. 

- [26] Charles Hong, Qijing Huang, Grace Dinh, Mahesh Subedar, and Yakun Sophia Shao. Dosa: Differentiable model-based one-loop search for dnn accelerators. In Proceedings of the 56th Annual IEEE/ACM International Symposium on Microarchitecture, pages 209–224, 2023. 

- [27] Qijing Huang, Minwoo Kang, Grace Dinh, Thomas Norell, Aravind Kalaiah, James Demmel, John Wawrzynek, and Yakun Sophia Shao. Cosa: Scheduling by constrained optimization for spatial accelerators. In 2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA), pages 554–566. IEEE, 2021. 

- [28] Ranggi Hwang, Taehun Kim, Youngeun Kwon, and Minsoo Rhu. Centaur: A chiplet-based, hybrid sparse-dense accelerator for personalized recommendations. In 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA), pages 968–981. IEEE, 2020. 

- [29] Norman P Jouppi, Cliff Young, Nishant Patil, David Patterson, Gaurav Agrawal, Raminder Bajwa, Sarah Bates, Suresh Bhatia, Nan Boden, Al Borchers, et al. In-datacenter performance analysis of a tensor processing unit. In Proceedings of the 44th annual international symposium on computer architecture, pages 1–12, 2017. 

- [30] Ajaykumar Kannan, Natalie Enright Jerger, and Gabriel H Loh. Enabling interposer-based disintegration of multi-core processors. In Proceedings of the 48th international symposium on Microarchitecture, pages 546– 558, 2015. 

- [31] Sheng-Chun Kao, Geonhwa Jeong, and Tushar Krishna. Confuciux: Autonomous hardware resource assignment for dnn accelerators using reinforcement learning. In 2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), pages 622–636. IEEE, 2020. 

- [32] Sheng-Chun Kao and Tushar Krishna. Magma: An optimization framework for mapping multiple dnns on multiple accelerator cores. In 2022 IEEE International Symposium on High-Performance Computer Architecture (HPCA), pages 814–830. IEEE, 2022. 

- [33] Seah Kim, Hasan Genc, Vadim Vadimovich Nikiforov, Krste Asanovi´c, Borivoje Nikoli´c, and Yakun Sophia Shao. Moca: Memory-centric, adaptive execution for multi-tenant deep neural networks. In 2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA), pages 828–841. IEEE, 2023. 

- [34] Seah Kim, Hyoukjun Kwon, Jinook Song, Jihyuck Jo, Yu-Hsin Chen, Liangzhen Lai, and Vikas Chandra. Dream: A dynamic scheduler 

577 

Authorized licensed use limited to: Access paid by The UC Irvine Libraries. Downloaded on December 28,2024 at 09:36:40 UTC from IEEE Xplore.  Restrictions apply. 

for dynamic real-time multi-model ml workloads. In Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 4, pages 73– 86, 2023. 

- [35] Hyoukjun Kwon, Prasanth Chatarasi, Michael Pellauer, Angshuman Parashar, Vivek Sarkar, and Tushar Krishna. Understanding reuse, performance, and hardware cost of dnn dataflow: A data-centric approach. In Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture, pages 754–768, 2019. 

- [36] Hyoukjun Kwon, Prasanth Chatarasi, Vivek Sarkar, Tushar Krishna, Michael Pellauer, and Angshuman Parashar. Maestro: A data-centric approach to understand reuse, performance, and hardware cost of dnn mappings. IEEE micro, 40(3):20–29, 2020. 

- [37] Hyoukjun Kwon, Liangzhen Lai, Michael Pellauer, Tushar Krishna, YuHsin Chen, and Vikas Chandra. Heterogeneous dataflow accelerators for multi-dnn workloads. In 2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA), pages 71–83. IEEE, 2021. 

- [38] Hyoukjun Kwon, Krishnakumar Nair, Jamin Seo, Jason Yik, Debabrata Mohapatra, Dongyuan Zhan, Jinook Song, Peter Capak, Peizhao Zhang, Peter Vajda, et al. Xrbench: An extended reality (xr) machine learning benchmark suite for the metaverse. Proceedings of Machine Learning and Systems, 5, 2023. 

- [39] Chris Lattner, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Jacques Pienaar, River Riddle, Tatiana Shpeisman, Nicolas Vasilache, and Oleksandr Zinenko. Mlir: Scaling compiler infrastructure for domain specific computation. In 2021 IEEE/ACM International Symposium on Code Generation and Optimization (CGO), pages 2–14. IEEE, 2021. 

- [40] Baolin Li, Tirthak Patel, Siddharth Samsi, Vijay Gadepally, and Devesh Tiwari. Miso: exploiting multi-instance gpu capability on multi-tenant gpu clusters. In Proceedings of the 13th Symposium on Cloud Computing, pages 173–189, 2022. 

- [41] Chen Liu, Kihwan Kim, Jinwei Gu, Yasutaka Furukawa, and Jan Kautz. Planercnn: 3d plane detection and reconstruction from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4450–4459, 2019. 

- [42] Zihan Liu, Jingwen Leng, Zhihui Zhang, Quan Chen, Chao Li, and Minyi Guo. Veltair: towards high-performance multi-tenant deep learning services via adaptive compilation and scheduling. In Proceedings of the 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, pages 388–401, 2022. 

- [43] Liqiang Lu, Naiqing Guan, Yuyue Wang, Liancheng Jia, Zizhang Luo, Jieming Yin, Jason Cong, and Yun Liang. Tenet: A framework for modeling tensor dataflow based on relation-centric notation. In 2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA), pages 720–733. IEEE, 2021. 

- [44] Fangchang Ma and Sertac Karaman. Sparse-to-dense: Depth prediction from sparse depth samples and a single image. In 2018 IEEE international conference on robotics and automation (ICRA), pages 4796–4803. IEEE, 2018. 

- [45] Xiaohan Ma, Chang Si, Ying Wang, Cheng Liu, and Lei Zhang. Nasa: accelerating neural network design with a nas processor. In 2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA), pages 790–803. IEEE, 2021. 

- [46] Meta. D2go. https://github.com/facebookresearch/d2go, 2022. 

- [47] Microsoft. Announcing microsoft copilot, your everyday ai companion. https://blogs.microsoft.com/blog/2023/09/21/announcingmicrosoft-copilot-your-everyday-ai-companion/, 2023. 

- [48] Microsoft. Azure openai service. https://azure.microsoft.com/en-us/ products/ai-services/openai-service, 2023. 

- [49] MLCommons. Mlperf inference. https://mlcommons.org/benchmarks/ inference-datacenter/, 2023. 

- [50] Samuel Naffziger, Noah Beck, Thomas Burd, Kevin Lepak, Gabriel H Loh, Mahesh Subramony, and Sean White. Pioneering chiplet technology and design for the amd epyc™and ryzen™processor families: Industrial product. In 2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA), pages 57–70. IEEE, 2021. 

- [51] Sokratis Nikolaidis, Stylianos I. Venieris, and Iakovos S. Venieris. Multitasc: A multi-tenancy-aware scheduler for cascaded dnn inference at the consumer edge. In 2023 IEEE Symposium on Computers and Communications (ISCC), pages 411–416, 2023. 

- [52] NVIDIA. Nvdla deep learning accelerator. http://nvdla.org,2017., 2023. 

- [53] Mohanad Odema, Halima Bouzidi, Hamza Ouarnoughi, Smail Niar, and Mohammad Abdullah Al Faruque. Magnas: A mapping-aware graph neural architecture search framework for heterogeneous mpsoc deployment. ACM Transactions on Embedded Computing Systems, 22(5s):1–26, 2023. 

- [54] Christopher Olston, Noah Fiedel, Kiril Gorovoy, Jeremiah Harmsen, Li Lao, Fangwei Li, Vinu Rajashekhar, Sukriti Ramesh, and Jordan Soyke. Tensorflow-serving: Flexible, high-performance ml serving. arXiv preprint arXiv:1712.06139, 2017. 

- [55] Marcelo Orenes-Vera, Esin Tureci, David Wentzlaf, and Margaret Martonosi. Massive data-centric parallelism in the chiplet era. arXiv preprint arXiv:2304.09389, 2023. 

- [56] Ioannis Panopoulos, Stylianos Venieris, and Iakovos Venieris. Carin: Constraint-aware and responsive inference on heterogeneous devices for single-and multi-dnn workloads. ACM Transactions on Embedded Computing Systems, 2024. 

- [57] Angshuman Parashar, Priyanka Raina, Yakun Sophia Shao, Yu-Hsin Chen, Victor A Ying, Anurag Mukkara, Rangharajan Venkatesan, Brucek Khailany, Stephen W Keckler, and Joel Emer. Timeloop: A systematic approach to dnn accelerator evaluation. In 2019 IEEE international symposium on performance analysis of systems and software (ISPASS), pages 304–315. IEEE, 2019. 

- [58] Jongsoo Park, Maxim Naumov, Protonu Basu, Summer Deng, Aravind Kalaiah, Daya Khudia, James Law, Parth Malani, Andrey Malevich, Satish Nadathur, et al. Deep learning inference in facebook data centers: Characterization, performance optimizations and hardware implications. arXiv preprint arXiv:1811.09886, 2018. 

- [59] Qualcomm. Quacomm hexagon 680. https://www.hotchips.org/wpcontent/uploads/hc archives/hc27/HC27.24-Monday-Epub/HC27. 24.20-Multimedia-Epub/HC27.24.211-Hexagon680-CodrescuQualcomm.pdf, 2015. 

- [60] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. 

- [61] Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, 44(3):1623–1637, 2020. 

- [62] Vijay Janapa Reddi, Christine Cheng, David Kanter, Peter Mattson, Guenther Schmuelling, Carole-Jean Wu, Brian Anderson, Maximilien Breughe, Mark Charlebois, William Chou, et al. Mlperf inference benchmark. In 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA), pages 446–459. IEEE, 2020. 

- [63] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Nassir Navab, Joachim Hornegger, William M. Wells, and Alejandro F. Frangi, editors, Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, pages 234–241, Cham, 2015. Springer International Publishing. 

- [64] Yakun Sophia Shao, Jason Clemons, Rangharajan Venkatesan, Brian Zimmer, Matthew Fojtik, Nan Jiang, Ben Keller, Alicia Klinefelter, Nathaniel Pinckney, Priyanka Raina, et al. Simba: Scaling deep-learning inference with multi-chip-module-based architecture. In Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture, pages 14–27, 2019. 

- [65] Haichen Shen, Lequn Chen, Yuchen Jin, Liangyu Zhao, Bingyu Kong, Matthai Philipose, Arvind Krishnamurthy, and Ravi Sundaram. Nexus: A gpu cluster engine for accelerating dnn-based video analysis. In Proceedings of the 27th ACM Symposium on Operating Systems Principles, pages 322–337, 2019. 

- [66] Yangyang Shi, Yongqiang Wang, Chunyang Wu, Ching-Feng Yeh, Julian Chan, Frank Zhang, Duc Le, and Mike Seltzer. Emformer: Efficient memory transformer based acoustic model for low latency streaming speech recognition. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6783–6787. IEEE, 2021. 

- [67] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1– 9, 2015. 

- [68] Zhanhong Tan, Hongyu Cai, Runpei Dong, and Kaisheng Ma. Nnbaton: Dnn workload orchestration and chiplet granularity exploration for multichip accelerators. In 2021 ACM/IEEE 48th Annual International 

578 

Authorized licensed use limited to: Access paid by The UC Irvine Libraries. Downloaded on December 28,2024 at 09:36:40 UTC from IEEE Xplore.  Restrictions apply. 

Symposium on Computer Architecture (ISCA), pages 1013–1026. IEEE, 2021. 

- [69] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, MarieAnne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 

- [70] Pascal Vivet, Eric Guthmuller, Yvain Thonnart, Gael Pillonnet, Guillaume Moritz, Ivan Miro-Panad`es, Cesar Fuguet, Jean Durupt, Christian Bernard, Didier Varreau, Julian Pontes, Sebastien Thuries, David Coriat, Michel Harrand, Denis Dutoit, Didier Lattard, Lucile Arnaud, Jean Charbonnier, Perceval Coudrain, Arnaud Garnier, Frederic Berger, Alain Gueugnot, Alain Greiner, Quentin Meunier, Alexis Farcy, Alexandre Arriordaz, Severine Cheramy, and Fabien Clermidy. 2.3 a 220gops 96-core processor with 6 chiplets 3d-stacked on an active interposer offering 0.6 ns/mm latency, 3tb/s/mm 2 inter-chiplet interconnects and 156mw/mm 2@ 82%-peak-efficiency dc-dc converters. In 2020 IEEE International Solid-State Circuits Conference-(ISSCC), pages 46–48. IEEE, 2020. 

- [71] Zhenyu Wang, Gopikrishnan Raveendran Nair, Gokul Krishnan, Sumit K Mandal, Ninoo Cherian, Jae-Sun Seo, Chaitali Chakrabarti, Umit Y Ogras, and Yu Cao. Ai computing in light of 2.5 d interconnect roadmap: Big-little chiplets for in-memory acceleration. In 2022 International Electron Devices Meeting (IEDM), pages 23–6. IEEE, 2022. 

- [72] Carole-Jean Wu, David Brooks, Kevin Chen, Douglas Chen, Sy Choudhury, Marat Dukhan, Kim Hazelwood, Eldad Isaac, Yangqing Jia, Bill Jia, et al. Machine learning at facebook: Understanding inference at the edge. In 2019 IEEE international symposium on high performance computer architecture (HPCA), pages 331–344. IEEE, 2019. 

- [73] Yannan Nellie Wu, Po-An Tsai, Angshuman Parashar, Vivienne Sze, and Joel S Emer. Sparseloop: An analytical approach to sparse tensor accelerator modeling. In 2022 55th IEEE/ACM International Symposium on Microarchitecture (MICRO), pages 1377–1395. IEEE, 2022. 

- [74] Qingcheng Xiao, Size Zheng, Bingzhe Wu, Pengcheng Xu, Xuehai Qian, and Yun Liang. Hasco: Towards agile hardware and software co-design for tensor computation. In 2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA), pages 1055–1068. IEEE, 2021. 

- [75] Haoran You, Cheng Wan, Yang Zhao, Zhongzhi Yu, Yonggan Fu, Jiayi Yuan, Shang Wu, Shunyao Zhang, Yongan Zhang, Chaojian Li, et al. Eyecod: eye tracking system acceleration via flatcam-based algorithm & accelerator co-design. In Proceedings of the 49th Annual International Symposium on Computer Architecture, pages 610–622, 2022. 

- [76] Chengliang Zhang, Minchen Yu, Wei Wang, and Feng Yan. {MArk}: Exploiting cloud services for {Cost-Effective},{SLO-Aware} machine learning inference serving. In 2019 USENIX Annual Technical Conference (USENIX ATC 19), pages 1049–1062, 2019. 

- [77] Xinyi Zhang, Cong Hao, Peipei Zhou, Alex Jones, and Jingtong Hu. H2h: heterogeneous model to heterogeneous system mapping with computation and communication awareness. In Proceedings of the 59th ACM/IEEE Design Automation Conference, pages 601–606, 2022. 

- [78] Shixuan Zheng, Xianjue Zhang, Leibo Liu, Shaojun Wei, and Shouyi Yin. Atomic dataflow based graph-level workload orchestration for scalable dnn accelerators. In 2022 IEEE International Symposium on High-Performance Computer Architecture (HPCA), pages 475–489. IEEE, 2022. 

579 

Authorized licensed use limited to: Access paid by The UC Irvine Libraries. Downloaded on December 28,2024 at 09:36:40 UTC from IEEE Xplore.  Restrictions apply. 

