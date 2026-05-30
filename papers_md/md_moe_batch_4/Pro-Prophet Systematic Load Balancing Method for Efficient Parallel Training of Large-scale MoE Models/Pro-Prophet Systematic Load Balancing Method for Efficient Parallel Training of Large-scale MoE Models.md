# Pro-Prophet: A Systematic Load Balancing Method for Efficient Parallel Training of Large-scale MoE Models

Wei Wang, Zhiquan Lai, Shengwei Li, Weijie Liu, Keshi Ge, Ao Shen, Huayou Su, Dongsheng Li

*Abstract*—The size of deep learning models has been increasing to enhance model quality. The linear increase in training computation budget with model size means that training an extremely large-scale model is exceedingly time-consuming. Recently, the Mixture of Expert (MoE) has drawn significant attention as it can scale models to extra-large sizes with a stable computation budget. However, inefficient distributed training of large-scale MoE models hinders their broader application. Specifically, a considerable dynamic load imbalance occurs among devices during training, significantly reducing throughput. Several loadbalancing works have been proposed to address the challenge. System-level solutions draw more attention for their hardware affinity and non-disruption of model convergence compared to algorithm-level ones. However, they are troubled by high communication costs and poor communication-computation overlapping. To address these challenges, we propose a systematic loadbalancing method, Pro-Prophet, which consists of a planner and a scheduler for efficient parallel training of large-scale MoE models. To adapt to the dynamic load imbalance, we profile training statistics and use them to design Pro-Prophet. For lower communication volume, Pro-Prophet planner determines a series of lightweight load-balancing strategies and efficiently searches for a communication-efficient one for training based on the statistics. For sufficient overlapping of communication and computation, Pro-Prophet scheduler schedules the datadependent operations based on the statistics and operation features, further improving the training throughput. We conduct extensive experiments in four clusters and five MoE models. The results indicate that Pro-Prophet achieves up to 2.66x speedup compared to two popular MoE frameworks including Deepspeed-MoE and FasterMoE. Furthermore, Pro-Prophet has demonstrated a load-balancing improvement of up to 11.01x compared to a representative load-balancing work, FasterMoE.

*Index Terms*—Deep learning, mixture of experts, distributed training

# I. INTRODUCTION

R ECENT years, large-scale deep neural networks have achieved superior performance in various domains(e.g., NLP, CV). Previous works have shown that the model capacity is improved with the increased model size, further promoting

The first two authors contributed equally to this work.

Wei Wang, Zhiquan Lai, Shengwei Li, Weijie Liu, Keshi Ge, Ao Shen, Huayou Su and Dongsheng Li are with the National Key Laboratory of Parallel and Distributed Computing, College of Computer, National University of Defense Technology in Changsha, Hunan, China. (Corresponding author: Dongsheng Li)

E-mail: {wwking, zqlai, swli, liuweijie, gekeshi, shenao, shyou, dsli}@nudt.edu.cn

This work is supported by the National Key R&D Program of China (No. 2022YFB4501400) and the National Natural Science Foundation of China under Grant No. 62025208 and 62421002.

the model scaling. However, the substantial computational demand of extra-large models makes the training process excessively time-consuming. As one of the most promising solutions, Mixture of Expert (MoE) enables a nearly constant computational budget as model scaling. Generally, we replace some layers of a foundation model with MoE ones to generate a MoE model. Each MoE layer contains a gate network and a range of sub-modules named *experts*. The gate network can route each input to top-k experts that excel in processing the input. As the k is a super-parameter, the MoE model can be scaled with consistent computational requirements by increasing the number of experts.

1

As the model further scales, the effective collaboration of devices is necessary for extra-large MoE model training. Unfortunately, it is inefficient to train the model with traditional parallelism such as Data Parallelism (DP), Model Parallelism (MP), and Pipeline Parallelism (PP). To overcome the trouble, Gshard [\[1\]](#page-11-0) introduced a specific parallel strategy named *Expert Parallelism* (EP). Nowadays, extra-large MoE models trained with EP have demonstrated the highest accuracy in multiple tasks [\[2\]](#page-11-1), [\[3\]](#page-11-2).

However, training MoE models using EP presents a dynamic load imbalance among devices. For each MoE layer, EP equally divides experts into devices before training and dynamically arranges its inputs according to the gate network during training. Most inputs are transferred to and processed by a few devices, resulting in prolonged communication and computation of inputs. Furthermore, the imbalance varies throughout the training process, making it difficult to resolve.

Numerous attempts in load-balancing have been proposed to improve training throughput. Algorithmic works often restrict the upper bound of each expert's load [\[4\]](#page-11-3), [\[5\]](#page-11-4) or add auxiliary losses to the loss function [\[6\]](#page-11-5), [\[7\]](#page-11-6) for a more balanced load. However, they impact the model convergence and even deteriorate the model quality. Considering the drawback above, systematic solutions of the MoE system draw more attention. Popular systematic works [\[8\]](#page-11-7), [\[9\]](#page-12-0) dynamically readjust the expert placement according to the load, achieving a balanced load without harming the model quality.

However, these systematic solutions struggle to enhance training efficiency effectively due to two drawbacks. 1) Heavy communications of model states (i.e., parameters, gradients and optimizer states [\[10\]](#page-12-1)) are introduced. The previous expert placements introduce a global transfer of parameters and gradients or a whole model states communication. These transferring strategies involve unnecessary communications across devices, hindering the improvement of training efficiency. 2) Devices experience significant communication and computation idle during training. Due to data dependencies among operators, the solutions have to perform some communications and computations sequentially. For example, only the experts have been selected and their model states have been transmitted, their computations of inputs can be launched. And then, the aggregation of gradients occurs only after the computation of gradients is finished. They neglect the potential of communication and computation overlapping, thus significantly influencing device utilization.

In this paper, we propose a systematic load-balancing solution, Pro-Prophet, which overcomes two drawbacks by a planner and scheduler respectively.

To adapt dynamic features presented in the training of a MoE model, we profile the *input distribution* (i.e., the number of inputs processed by each expert) for each MoE layer. We observe that distributions of a MoE layer between adjacent iterations present high similarity. This *locality* is the key to effective load balancing.

To reduce the communication volume, Pro-Prophet planner introduces a series of *lightweight expert placements*. In a lightweight expert placement, each expert is independently allocated to a subset of devices. Communication of parameters and gradients for the expert occurs in these specific devices. Serve to evaluate expert placements, the planner proposes a *performance model* that estimates the execution time of a MoE layer employing a lightweight expert placement. However, it is non-trivial to find the optimal one due to the combinatorial explosion of the number of expert placements. To tackle this, the planner designs a *locality-based greedy algorithm*. The algorithm employs a greedy strategy to search for a communication-efficient expert placement. Besides, its launching frequency is reduced based on the locality, further improving the training throughput.

To exploit the potential of communication-computation overlapping, Pro-Prophet scheduler comprehensively schedules operations based on the locality and the feature of operations. The locality means that we can estimate the input distribution of the upcoming iteration according to the current one. Once the upcoming distribution is obtained, we can promptly determine a communication-efficient expert placement for the upcoming iteration and can transmit the parameters of experts in advance, which provides the opportunity to overlap communications and computations within adjacent iterations. Besides, the gradient aggregation can be scheduled backward for better overlapping. Based on these, the scheduler identifies a scheduling space and designs a *block-wise scheduling strategy* to comprehensively overlap communications and computations.

We implement Pro-Prophet on top of PyTorch and conduct extensive experiments on four different clusters of up to 32 devices with five variant models. The results demonstrate that Pro-Prophet achieves speedups of up to 2.66x compared to two popular MoE frameworks. Additionally, Pro-Prophet has demonstrated load-balancing enhancements of up to 11.01x compared to a representative load-balancing work, FasterMoE.

Our main contributions are summarized as follows:

- We profile input distributions among adjacent iterations and identify a locality that guides the design of Pro-Prophet.
- We design a Pro-Prophet planner that identifies several lightweight expert placements, abstracts a performance model and designs a locality-based greedy algorithm to reduce the heavy communication of model states.
- We propose a Pro-Prophet scheduler, which generates a scheduling space and establishes a block-wise scheduling strategy based on the locality and the feature of operations for comprehensive overlapping of computations and communications.
- We conduct comprehensive experiments for Pro-Prophet on different clusters and models. The results demonstrate that Pro-Prophet achieved up to 1.50x end-to-end speedup and 11.01x load-balancing enhancements with the representative load-balancing method.

## II. BACKGROUND AND MOTIVATION

## <span id="page-1-0"></span>*A. Background*

Recent works in DNN model training have shown that the model capacity can be improved with increasing training data, model scale, and computational budget [\[11\]](#page-12-2). Extraordinary performance has been achieved in several deep learning domains including natural language processing (NLP), computer vision (CV), and so on.

However, significant training overhead comes along with the superior model capacity. The extra large-scale model [\[12\]](#page-12-3)– [\[17\]](#page-12-4) training often takes months on thousands of dedicated accelerators (e.g., MT-NLG [\[18\]](#page-12-5) spends three months to train on over two thousand A100 GPUs), which influences the development of deep learning.

In recent years, dynamic sparse-activated architectures have been proposed to solve the trouble. One of the popular approaches is the Mixture of Experts (MoE), which can significantly improve the model capacity while maintaining a consistent computational budget. Nowadays, MoE has been successfully applied to large language models [\[19\]](#page-12-6)–[\[25\]](#page-12-7). Excellent MoE models that appeared in industry and academia greatly draw researchers' attention. For example, Google has trained a series of MoE models called Glam [\[2\]](#page-11-1). The largest Glam model is seven times larger than GPT-3, but the training cost is less than 1/3 of it. Experiments show that these models achieve higher accuracy than GPT-3 in 29 zero, single, and small sample learning tasks, representing the superiority of MoE models. The other example is GPT-4 [\[26\]](#page-12-8). The technical report of OpenAI indicates that the GPT-4 is a MoE model, which achieved the highest performance in various downstream tasks. Besides, the ChatGPT based on the GPT-4 has caused a tremendous sensation.

Fig. [1](#page-2-0) illustrates the architecture of a MoE model and a MoE layer. The MoE model comprises a stack of non-MoE and MoE layers. A MoE layer consists of two components: 1) a series of experts (3 experts in the figure), where each excels in a specific domain. 2) a gate network, which routes each input to a few experts that are skilled in dealing with this input, rather than all experts. In a MoE layer, for each input, the gate

![](_page_2_Figure_1.jpeg)

<span id="page-2-0"></span>Fig. 1. The structure of a MoE model and MoE layer. The MoE model consists of both MoE and non-MoE layers stacked on top of each other. The MoE layer consists of a series of experts and a gate network for routing input to experts. For each input, the gate network computes the relationship between the input with three experts and allocates it to the top-1 expert for computation.

![](_page_2_Figure_3.jpeg)

<span id="page-2-1"></span>Fig. 2. A workflow of Expert Parallelism (EP) in a MoE layer. Following the gate network, batched inputs are first exchanged via an All-to-All (A2A) operation. After the expert computation on all devices, a second A2A operation is used to pass the expert's outputs back to the device where corresponding inputs were originally located.

network computes the relationship between that input and all the experts. Then it routes the input to top-k (k=1 in the figure) expert(s) for computation. Even as we increase the number of experts (the model size is increased), each input is still routed to a fixed number (k) of experts, and the negligible increased computational budget of the gate network, thereby enabling the scaling of the model with nearly constant computational overhead.

With the increase of the model scale, an isolated device cannot support the training of the MoE model thus various parallelisms have been proposed. Two common parallel approaches are DP and MP. DP equally divides inputs of an iteration across all devices and replicates the model into all devices. In forward propagation (FP), each device computes its local inputs independently by utilizing its model replicas. In the backward propagation (BP), the Allreduce primitive will be performed after the backward computation. Different from DP, MP partitions the model into devices in a specific manner and each device contains a complete copy of the data. The aggregation primitive will be launched whenever required in FP and BP.

For efficient training of a MoE model, Google combines DP and MP into an EP. From the input side, EP adopts the same input partitioning paradigm as DP. From the model side, EP divides the same number of experts to each device and copies other parts of the model (i.e., the gate network and non-MoE

![](_page_2_Figure_8.jpeg)

<span id="page-2-2"></span>Fig. 3. The imbalanced load of experts in an iteration. The model contains 12 MoE layers and each MoE layer contains 16 experts. The vertical axis indicates layer indexes, and the horizontal axis denotes the index of experts. The depth of color represents the proportion of total inputs that an expert handles. Three of the heaviest experts are responsible for over 50% inputs while the three least experts only compute less than 5%.

layer) to all devices.

Fig. 2 illustrates a workflow of EP in a MoE layer. Firstly, the gate network determines top-1 expert for each input. Then inputs are transferred to corresponding devices via an All-to-All (A2A) communication operation [27]–[29]. Subsequently, each device performs the expert computation for collected inputs and then launches another A2A to reorganize the results back to the inputs' original devices for computations of the subsequent non-MoE layer. Nowadays, many popular distributed frameworks support the training of large-scale MoE models using EP [4], [30]–[33].

Even though EP makes it feasible to train extra-large MoE models with up to trillions of parameters, a dynamic load imbalance occurs among devices. Specifically, most of the training inputs are transferred and processed by a few devices. These heavy-load devices vary as the training. Fig. 3 presents the imbalanced load of experts in an iteration. The vertical axis indicates layer indexes, and the horizontal axis denotes the index of experts. The MoE model contains 12 MoE layers and each MoE layer contains 16 experts. Each expert is set into a dedicated device. The depth of color represents the proportion of total inputs processed by an expert. In most MoE layers, the three heaviest experts hold over 50% of inputs, while the three least less than 5%. The unbalanced load of experts means that devices containing light-load experts have to wait for devices containing heavy-load ones, incurring significant under-utilization of devices during training.

#### B. Motivation

A series of methods have been proposed to balance the load. We divide them into algorithmic and systematic methods. From the algorithmic side, researchers constrain the upper bound of inputs received by an expert or add auxiliary losses to the loss function. They change the inputs-to-experts mapping, thus affecting and even deteriorating the model convergence.

Different from algorithmic works, systematic solutions do not affect model convergence and fit into the hardware, thus

![](_page_3_Figure_1.jpeg)

<span id="page-3-1"></span>Fig. 4. The locality of input distributions. The discrepancies between the different colored curves represent the number of inputs received by each of the different experts. It shows that distributions of adjacent iterations remain relatively constant.

attracting extensive attention. These solutions adaptively adjust experts-to-devices mapping based on the device load during training, effectively improving the training efficiency. However, their heavy load-balancing overhead hinders the further improvement of the efficiency.

<span id="page-3-0"></span>TABLE I
TIME BREAKDOWN OF TRAINING. L.B. IS SHORT FOR LOAD BALANCING

| Model      | L.B.  | Search | Place | Reduce | Others |
|------------|-------|--------|-------|--------|--------|
| MoE-GPT-S  | 29.9% | 6.8%   | 11.6% | 11.5%  | 70.1%  |
| MoE-GPT-M  | 29.2% | 3.2%   | 12.5% | 12.5%  | 70.8%  |
| MoE-GPT-L  | 34.5% | 2.6%   | 14.2% | 17.7%  | 64.5%  |
| MoE-GPT-DS | 33.8% | 6.1%   | 13.8% | 13.9%  | 66.2%  |
| MoE-GPT-DM | 37.1% | 6.1%   | 16.1% | 14.9%  | 62.9%  |

As shown in Table I, previous solutions introduce a *Search*, *Place* and *Reduce* processes to balance the load. However, the overhead of load-balancing is up to 37.1%. There are two reasons behind this huge cost. Firstly, they do heavy communication of model states. They have to transfer the parameters and gradients of heavy-load experts among all devices or transmit the whole model states of experts. Secondly, they cannot sufficiently overlap the communication and computations during the training of a MoE model. Due to the data dependency, they have to do communication and computation sequentially.

Locality. Fortunately, we have discovered a property in the training of MoE models. This property makes it possible to address these challenges efficiently. Fig. 4 depicts input distributions in the second MoE layer of a MoE model. The areas with different colors represent inputs received by different experts. It is worth noting that the distribution in other MoE layers follows a similar pattern. As shown in the figure, the slight fluctuation of the distribution occurs across adjacent iterations, which indicates that the load of each expert remains relatively stable in adjacent iterations. This phenomenon suggests that the distribution exhibits a locality among iterations.

#### III. OVERVIEW OF PRO-PROPHET

Motivated by Section II, we propose a systematical loadbalancing approach, Pro-Prophet, which can efficiently balance

![](_page_3_Figure_10.jpeg)

<span id="page-3-2"></span>Fig. 5. The overview of Pro-Prophet. Pro-Prophet is composed of Pro-Prophet planner and Pro-Prophet scheduler. MoE model, locality, and device pool are three inputs of it. Firstly, Pro-Prophet planner searches for a communication-efficient expert placement using its locality-based greedy algorithm. The algorithm iteratively generates and evaluates a lightweight expert placement utilizing its performance model until the load is balanced. Then the execution engine produces a load-balancing workflow based on the planner. Finally, Pro-Prophet scheduler schedules three data-dependent operations to parallel operations for communication and computation overlapping, further improving the training throughput.

the load of devices. The overview of Pro-Prophet is presented in Fig. 5. Pro-Prophet is composed of a planner and a scheduler. MoE model, locality, and device pool are three inputs of it. The device pool defines the topology of devices. The utilization of the locality is the key advantage of Pro-Prophet.

Firstly, Pro-Prophet planner searches for a communicationefficient expert placement from a series of lightweight expert placements using its locality-based greedy algorithm. The algorithm iteratively generates and evaluates a lightweight expert placement utilizing a performance model until the load is balanced.

Then, the execution engine analyzes the procedures of the planner and produces a load-balanced workflow for load balancing.

Finally, after analyzing the workflow, Pro-Prophet scheduler establishes the scheduling space and schedules data-dependent operations (i.e., Plan, Trans, and Agg) to parallel operations (i.e., Para.Op1 and Para.Op2) for communication and computation overlapping. The meaning of operations are presented in Sec IV.

#### IV. PRO-PROPHET PLANNER

## <span id="page-3-3"></span>A. Lightweight Expert Placement

The design of the expert placement is crucial for efficient load balancing. For less communication of model states transferring, the planner introduces a series of lightweight expert placements.

In a lightweight expert placement, each expert is mapped to one or more devices independently. Only the parameters and gradients rather than all model states are transferred among its devices. We use Trans and Agg primitives to describe these two communications respectively. In the forward pass, a

<span id="page-4-1"></span>![](_page_4_Figure_1.jpeg)

![](_page_4_Figure_2.jpeg)

#### <span id="page-4-0"></span>(a) Traditional expert placement.

# <span id="page-4-2"></span>(b) Lightweight expert placement.

Fig. 6. The comparison of a traditional and lightweight expert placement. The load is imbalanced in traditional expert placement. In a lightweight one, each expert is placed into necessary devices to balance the load. The Trans and Agg primitives are involved to communicate their parameters and gradients respectively.

Trans is first launched to transfer the parameters. After that, each device contains the parameters of some expert, thus its local inputs routed to these experts could be computed locally. After the backward computation, the gradients of an expert could be generated in several devices. As each device only maintains the optimizer states of one expert, a Agg primitive is launched to aggregate gradients of each expert to its original device. This design has two advantages: 1) Only part of the model states are communicated. 2) The model states are only communicated among a subset of devices.

Fig 6 illustrates a comparison of a traditional and lightweight expert placement. As shown in Fig. 6a, 5, 2, and 2 inputs are routed to  $E_0$ ,  $E_1$ , and  $E_2$  respectively. After the A2A communication, three devices are responsible for the computation of 5, 2, and 2 inputs as each of the devices only contains parameters of a distinct expert (e.g., Dev. 0 contains  $E_0$ 's parameters), resulting in an imbalanced load among devices. Fig. 6b shows a balanced load achieved by the lightweight expert placement. Experts are mapped to devices according to the routing results produced by the gate network. Parameters of  $E_0$  are sent from Dev. 0 only to Dev. 1 as inputs in Dev. 2 are not routed to  $E_0$ . Similarly, parameters of  $E_1$  are transferred to Dev. 0 and Dev. 1 for their expert computation. It maps experts to necessary devices and only communicates their parameters and gradients, effectively avoiding heavy model states transferring.

## B. Performance model

It's necessary to evaluate lightweight expert placements under various device loads. Therefore, the planner abstracts a performance model to estimate the execution time of a MoE layer employing a lightweight expert placement. Table II presents notations and descriptions used in the performance model.

After employing a lightweight expert placement, a MoE layer performs four A2A communication operations, one forward expert computation operation EFC, one backward computation operation EBC, one Trans operation, and one Agg operation. To accurately evaluate the execution time of the MoE layer, we establish our performance model according to the implementation of operations and hardware characteristics.

**A2A communication.** Tutel [5] presents an efficient A2A implementation used in the training of a MoE model. In this

TABLE II NOTATIONS

<span id="page-4-3"></span>

| Notation       | Description                                               |  |  |  |
|----------------|-----------------------------------------------------------|--|--|--|
| T              | Execution time of an operation                            |  |  |  |
| R              | Inputs received by a device from other devices            |  |  |  |
| $\overline{B}$ | Average communication bandwidth                           |  |  |  |
| H              | Inputs computed in a device                               |  |  |  |
| t              | Computation throughput                                    |  |  |  |
| s              | Number of selected experts should be transferred          |  |  |  |
| n              | Number of devices a selected expert not be transferred to |  |  |  |
| E              | Number of experts in a MoE layer                          |  |  |  |
| D              | Number of devices                                         |  |  |  |

implementation, devices use point-to-point(P2P) communication primitives to achieve the A2A communication operation. Based on this, we define the execution time of an A2A operation as below.

$$T_{A2A}(R) = \max_{i} \frac{R_i \cdot size(input)}{\overline{R}},$$
 (1)

where  $R_i$  is the total number of inputs received by device-i from other devices and size(input) is the size of a input.

**Expert computation.** Next, we formulate the duration of the forward and backward expert computation. In the expert computation procedure, the computations of devices are performed simultaneously. However, computations of different experts are launched sequentially in a device. To depict this characteristic, we define the execution time of FEC as

$$T_{FEC}(H) = \max_{i} \frac{H_i}{t},\tag{2}$$

where  $H_i$  is the number of inputs computed in device-i.

It is widely recognized that the time required for backward computation in DNN training is roughly double that of forward computation, which is the same for MoE model training. Therefore, we define the execution time of BEC as

$$T_{BEC}(H) = 2\max_{i} \frac{H_i}{t},\tag{3}$$

Trans and Agg primitives. Finally, we formulate the overhead of Trans and Agg primitives. The duration time of Trans and Agg primitives depends on two elements. The first element is the number of transferred experts, which determines communication rounds. The second element is the number of devices communicated in a primitive, which influences

the communication scales. Therefore, the TT rans(s, n) and TAgg(s, n) are defined as below.

$$T_{Trans}(s,n) = \frac{s * (D-n) * size(e_j.params)}{D * \overline{B}}, \quad (4)$$

$$T_{Agg}(s,n) = \frac{s * (D-n) * size(e_j.grads)}{D * \overline{B}},$$
 (5)

where the size(e<sup>j</sup> .params) and size(e<sup>j</sup> .grad) are the size of parameters and gradients for the j-th expert.

In summary, the overall execution time of the MoE layer with lightweight expert placement can be represented as

$$T'(R, H, s, n) = 4T_{A2A}(R) + 3T_{FEC}(H) + T_{Trans}(s, n) + T_{Agg}(s, n)$$
(6)

# *C. Locality-based Greedy Algorithm*

The performance model can accurately estimate the execution time of a MoE layer deploying any expert placements. However, it is necessary to determine a communicationefficient one in various load imbalance scenarios. There are 2 <sup>N</sup>∗<sup>E</sup> potential lightweight expert placements. The brute force search algorithm is time-consuming and could be a performance bottleneck.

Therefore, the planner offers an efficient greedy search algorithm shown in Algorithm [1.](#page-5-0) Taking the results of gate network gating, s and n as input, Algorithm [1](#page-5-0) iteratively generates and evaluates for a better expert placement until the load is balanced. Finally, it outputs a communication-efficient expert placement P oE.

Initially, the algorithm estimates the execution time of a MoE layer without implementing any lightweight expert placements and records it as minimum time. Then it employs two greedy strategies to generate a lightweight expert placement that optimizes the load of devices. Specifically, it prioritizes the expert with the higher number of responsible inputs for selection and transfers its parameters to devices that hold more inputs processed by the expert. The algorithm maintains a list of L and n bottoms to record the expert placement. Then the algorithm evaluates the expert placement using the performance model. It updates the minimum time and a counter if the current expert placement achieves a better performance. The search process is repeated until the load is imbalanced. The condition of the balanced load is

$$\max(H) - \min(H) < \alpha \frac{I}{E},\tag{7}$$

where I is the number of inputs training in an iteration and α is a regulable coefficient for different requirements of load balance.

As the search algorithm is required to run during the MoE model training, we define a primitive Plan to describe this search process. As mentioned in Sec. [II,](#page-1-0) the input distributions of adjacent iterations are similar, which inspired us to predict the distribution and reduce the frequency of execution of the algorithm. Based on the inspiration, the planner upgrades the algorithm to a locality-based one. Users can adjust the frequency of the search algorithm flexibly for better training efficiency.

# <span id="page-5-0"></span>Algorithm 1: Greedy search algorithm Input: Inputs-to-experts mapping gating Input: n Result: Communition-efficient expert placement P oE

// Preliminary <sup>1</sup> Toutput ← T ′ (R, H, 0, 0); <sup>2</sup> H, R ← GetH&R(gating); <sup>3</sup> L, n bottoms ← [], []; <sup>4</sup> cnt ← 0;

```
// Iteratively search
5 while not balanced do
     // Get the index of the heaviest
        device
6 i ← arg max
           i
               (H);
7 if i in Used then
8 break;
9 end
10 Used.append(i);
     // Determine n devices saving the
        smallest number of inputs for
        expert-i
11 n bottom ← BottomK(gating, n) ;
12 L.append(i);
13 n bottoms.append(n bottom)
14 s ← size(L)
     // Replace inputs among devices
        according to the expert
        placement
15 H, R ← Replace Inputs(L, n bottoms)
     // Evaluate the expert placement
16 Tchanged ← T
                ′
                (R, H, s, n);
17 if Tchanged < Toutput then
18 Toutput ← Tchanged;
19 cnt = s
20 end
21 end
```

# // Return the communication-efficient expert placement <sup>22</sup> P oE ← Get P oE(L[0 : cnt], n bottoms[0 : cnt])

# V. PRO-PROPHET SCHEDULER

<span id="page-5-1"></span>Previous works introduce a search process (corresponding to Plan primitive), model states transferring (corresponding to Trans and Agg primitives) to balance the load. However, their execution is blocked by other operators due to data dependency, constraining further improvement of training efficiency. In this section, we introduce designs of the scheduler which extensively overlap computation and communication based on the locality described in Sec. [II.](#page-1-0)

# *A. Scheduling space establishment*

<sup>23</sup> RETURN P oE;

As shown in Fig. [1,](#page-2-0) a MoE model consists of both MoE and non-MoE layers stacked on top of each other. We combine

![](_page_6_Figure_1.jpeg)

<span id="page-6-0"></span>Fig. 7. The device state of operators in a MoE block. FEC and BEC are forward and backward computations of the MoE layer. FNEC and BNEC are forward and backward computations of the non-MoE layer. An operator is marked as *comm* and a green rectangle if devices only communicate during its execution. Similarly, we use *comp* and blue rectangle to mark computation operators.

every adjacent pair of MoE layer and non-MoE layer into a MoE block.

Fig. 7 presents the device state of operators in a MoE block. Only primary operations are presented for clarity description. During the FP, two primitives related to load balancing(i.e., Plan and Trans) and three basic operations(i.e., two A2A communications and a forward expert computation FEC) are performed in a MoE layer. For the non-MoE layer, only a forward computation FNEC is executed. After the gate network produces the routing decision and input distribution, a Plan operator is performed to identify a load-balancing strategy based on the load of devices. The Trans primitive can only be launched once the strategy is determined. And then, two A2A communications and an expert computation can be launched. During the BP, the MoE layer executes a Agg operation, two A2A communications, and a backward expert computation BEC. Only a backward computation BNEC is performed in the non-MoE layer, the Agg operation is carried out to aggregate the gradient of experts following the completion of their backward computation.

We label an operation as *comm* if devices communicate during the execution of this operation. Similarly, if an operation computes all the time, we label it as *comp*. Before the launching of Plan, information related to produce a load-balancing strategy is stored in each device. Therefore, there are only computations during Plan. We label it as *comp*. As Trans primitive only exchanges the parameters of experts based on the load balancing strategy, we label it as *comm*. According to the description of Sec. II, the A2A is marked as *comm*, and the computation of the MoE layer and non-MoE layer are tagged as *comp*. As for Agg primitive, gradients of the same expert are aggregated into a device. We flag it as *comm*.

The operators with data dependency are tightly interconnected along the timeline, constraining the scheduling of computation and communication. However, the locality mentioned in Sec. II enables the pre-launch of some data-dependent operations without breaking the data dependency. We can insert the data-independent operation into data-dependent operators to make room for communication and computation overlapping. Specifically, in the case of a MoE layer, the input distribution for the current iteration can be estimated by leveraging the distribution from former iterations, which allows us to produce

the load-balancing strategy by invoking a plan operator within earlier iterations. Subsequently, Trans primitives can be scheduled to earlier locations on the timeline. We also find that the Agg primitive is independent of computation operations later, thus we can schedule it to later positions.

The analysis above provides the potential for computation and communication scheduling. However, there are several constraints for arbitrarily scheduling. Firstly, the estimation of the distribution means that we can establish a load-balancing strategy within a former iteration. As the distribution of the last iteration is necessary for higher estimation accuracy, the earliest position of a plan primitive in i-th iteration is the i-1-th iteration. Secondly, there are two main ways to update parameters. It is necessary to update the expert parameters before the Trans primitive. We can perform the updating procedure layer by layer [34] or update at the end of the BP [8]. For the layer-by-layer updating, we can launch the Trans primitive of a MoE layer of an iteration within the last iteration, which does not apply to concentrated updating works. For the concentrated updating, the Trans primitive could be performed at the end of the BP of the last iteration, which has a similar effect as starting it within this iteration. For the universality of our method, we confine the scheduling of the Trans primitive within a single iteration. Lastly, it is necessary to aggregate the gradients of experts at each iteration. Therefore, the replacing of the Agg primitive is also confined within a single iteration.

Fig. 8 illustrates our scheduling space. The subscript denotes the index of the MoE block of the operation and the superscript denotes its iteration. All Plan computations of iteration j+1 can be scheduled to the A2A communication of iteration j. The Trans primitives from block i+1 to l during the j-th iteration are overlapped with the forward computations of the i-th block, where l is the total number of blocks. Similarly, the Agg primitives from block i+1 to l are orchestrated to overlap with backward computations of the i-th block. Our scheduling space considers a MoE block as a unified reordering entity, thus overcoming the limitations in observation imposed by previous methods.

#### B. Block-wise scheduling strategy

The scheduling space provides extensive strategies for scheduling. However, operator-grained scheduling is far from making full use of overlapping space. We partition the operator into sub-operators and take a scheduling at the sub-operator level. We take a brief example to describe the advantage of sub-operator scheduling in Fig. 9 which shows three types of scheduling for a Trans primitive. The overhead of Trans primitive is varied as the load of devices (e.g., the number of heavy-load experts changed as the training). As shown in Fig. 9a and Fig. 9b, a forward computation of a MoE layer or a non-MoE layer cannot afford the hiding of a Trans primitive as their short duration time. Consequently, the Trans primitive will block the process of the model training. Fig. 9c presents the sub-operators scheduling. The Trans primitive is split into two sub-operators scheduled to two computations respectively. The sub-operator scheduling

![](_page_7_Figure_1.jpeg)

<span id="page-7-0"></span>Fig. 8. Scheduling space. The subscript denotes the index of the MoE block and the superscript denotes that of the iteration. For example,  $Trans_j^{i+1:l}$  denotes the set of Trans operations spanning from the block i+1 to the block l during the iteration j, where l is the number of MoE blocks.

<span id="page-7-2"></span>![](_page_7_Figure_3.jpeg)

<span id="page-7-4"></span><span id="page-7-3"></span>(c) Splitting and scheduling Trans to both the forward expert computation and non-expert computation.

<span id="page-7-1"></span>Fig. 9. Different scheduling strategies for a Trans primitive.

improves the utilization of overlapping space and reduces the communication overhead.

A dynamic scheduling strategy is beneficial as the load of devices fluctuates with the training process. However, the nonnegligible overhead will be introduced if we determine an optimal scheduling strategy that as far as possible hides the overhead of three types of primitives involved by systematical load balancing methods in the runtime. Therefore, we design an offline scheduling policy to overlap communication and computation while avoiding the extra overhead. The policy is founded on static elements within dynamics, not intuition. Our block-wise scheduling strategy is summarized in Algorithm 2. The first primitive is Plan. We schedule the Plan primitive of the *i*-th block in the iteration j + 1 to the A2A communication of the i-th block in the j-th iteration. For the Trans primitive, we overlap it with computations of the former block within an iteration. Specifically, the forward computation of the *i*-th block is responsible for overlapping the Trans primitive of block i + 1. As two computations are executed in the i-th block, we split the Trans primitive into two sub-primitives and launch them simultaneously with <span id="page-7-5"></span>the two computations. Even though the duration of Trans and EFC varies as the device loads, the forward computation overhead of the non-MoE layer and the transferring overhead of an expert's parameters are static. We can estimate them before training and properly split the Trans primitive. The advantage of the estimation is that we can exhaustively fill in the communication idle in the performing of the forward computation of the non-MoE layer. Finally, we schedule the Agg of block i+1 into the backward computation of the i-th one. Similarly, we can estimate the backward computation overhead of the non-MoE layer and do a suitable communication partition.

## C. Effective collaboration with planner

To better integrate the planner and scheduler, we combine the scheduling performed by the scheduler into the performance model of the planner.

Specifically, we define the parallel execution time of the Trans and Agg primitives as  $T_{PTrans}(H,s,n)$  and  $T_{PAgg}(H,s,n)$  respectively. Besides, we denote  $T_{FNEC}$  and  $T_{BNEC}$  as the execution time of FNEC and BNEC. If  $T_{Trans}(s,n)$  can be hidden by  $T_{FEC}(H)$  and  $T_{FNEC}$ , then  $T_{PTrans}(H,s,n)$  is equal to 0. Otherwise,  $T_{PTrans}(H,s,n)$  is equal to  $T_{Trans}(s,n) - T_{Trans}(s,n) - T_{Trans}(s,n)$ . That means

that TP T rans(H, s, n) = max(0, TT rans(s, n) − TF EC (H) − TF NEC ). Similarly, TP Agg(H, s, n) can be expressed as max(0, TAgg(s, n) − TBEC (H) − TBNEC ).

With the above analysis, the overall execution time of the MoE layer estimated by the performance model of the planner is changed as below:

$$T'(R, H, s, n) = 4T_{A2A}(R) + 3T_{FEC}(H) + T_{PTrans}(H, s, n) + T_{PAgg}(H, s, n).$$
(8)

By combining the planner and scheduler, we can achieve a fine-grained pre-allocation of hardware resources to experts, efficiently addressing the load imbalance problem during training.

## VI. EVALUATION

Testbed. We test Pro-Prophet on three types of nodes named *HPWNV*, *HPNV* and *LPWNV* respectively. Each *HPWNV* node is equipped with 2 Intel Xeon CPUs (2.40GHz) and 4 NVIDIA 3090 GPUs with 24GB graphics memory. Each CPU is connected to two GPUs through PCI-Express 3.0. 100Gb/s Infiniband is used for inter-node communication. The difference between *HPWNV* and *HPNV* is that GPUs within a *HPNV* node are connected by NVLink-3.0 connections. Specifically, four GPUs are divided into two groups. Two GPUs within a group are connected by a NVLink connection. The difference between *HPWNV* and *LPWNV* is that the type of GPUs of a *LPWNV* node is 2080Ti.

Models and baselines. As shown in table [III,](#page-8-0) we use five variants of MoE-GPT models in our experiments. All FFN layers are replaced by a MoE layer. The number of experts within a MoE layer is consistent with the number of GPUs.

We compared Pro-Prophet with two representative MoE training systems: 1) Deepspeed-MoE [\[4\]](#page-11-3): Deepspeed-MoE is an efficient MoE framework developed by Microsoft. It exclusively implements EP. 2) FasterMoE [\[8\]](#page-11-7): This training system employs a systematic load balancing method, *dynamic shadowing*, to effectively accelerate the model training.

Default settings. Unless otherwise specified, we fix some training settings. We train MoE models on the cluster consisting of *HPWNV* nodes. We evaluate Pro-Prophet within the first 100 iterations as the input distribution tends to stabilize with the training process.

## *A. End-to-end Performance*

In summary, Pro-Prophet achieves 1.36-2.66x and 1.01- 1.48x speedups compared to Deepspeed-MoE and FasterMoE respectively.

TABLE III MODEL CONFIGURATION.

<span id="page-8-0"></span>

| Name       | Layers | Embedding | Hidden |
|------------|--------|-----------|--------|
| MoE-GPT-S  | 12     | 512       | 1024   |
| MoE-GPT-M  | 12     | 1024      | 2048   |
| MoE-GPT-L  | 12     | 2048      | 4096   |
| MoE-GPT-DS | 24     | 512       | 1024   |
| MoE-GPT-DM | 24     | 1024      | 2048   |

TABLE IV THE OVERALL SPEEDUP ON 4 *HPNV* NODES.

<span id="page-8-1"></span>

|  | K<br>GPUs | Tokens |            | Speedup to DeepspeedMoE |             |
|--|-----------|--------|------------|-------------------------|-------------|
|  |           |        | Model      | FasterMoE               | Pro-Prophet |
|  | 1         | 16384  | MoE-GPT-S  | 1.63                    | 1.98        |
|  |           |        | MoE-GPT-M  | 1.99                    | 2.22        |
|  |           |        | MoE-GPT-L  | 1.62                    | 1.80        |
|  |           |        | MoE-GPT-DS | 1.34                    | 1.70        |
|  | 16        |        | MoE-GPT-DM | 1.68                    | 2.26        |
|  | 2         |        | MoE-GPT-S  | 2.31                    | 2.62        |
|  |           |        | MoE-GPT-M  | 1.82                    | 2.10        |
|  |           |        | MoE-GPT-L  | 1.94                    | 2.23        |
|  |           |        | MoE-GPT-DS | 1.77                    | 1.94        |
|  |           |        | MoE-GPT-DM | 1.84                    | 2.07        |

<sup>1</sup> The label "Token" is the number of tokens trained in an iteration.

Experiments on *HPWNV*. We first evaluate the end-to-end performance of Pro-Prophet on two *HPWNV* clusters. They contain 4 and 8 *HPWNV* nodes respectively. We fixed the number of tokens trained in an iteration to 16384 and 32768.

As shown in previous works, the k is set to 1 or 2 [\[1\]](#page-11-0), [\[6\]](#page-11-5), [\[35\]](#page-12-14) for better balancing between the model quality and training efficiency. We conducted experiments under both values to validate the generality of Pro-Prophet.

Fig. [10a](#page-9-0) and Fig. [10b](#page-9-1) illustrate speedups achieved by Pro-Prophet under five benchmark models with a top-1 gate network. Pro-Prophet achieved 1.47-2.66x end-to-end performance gains in comparison with Deepspeed-MoE. As to FasterMoE, Pro-Prophet achieves performance enhancements of up to 1.31x with an average of 1.19x.

As shown in Fig. [10c](#page-9-2) and Fig. [10d,](#page-9-3) speedups under a top-2 gate network achieved by Pro-Prophet are 1.36-2.37x and 1.05-1.48x compared to Deepspeed-MoE and FasterMoE respectively. The result shows the coarse-grained and blocked manner of FasterMoE will introduce additional runtime overhead and hinder the further improvement of the training efficiency. Our method can precisely pre-allocate resources to experts, thereby avoiding this issue.

Experiments on HPNV and LPWNV clusters. Different hardware conditions significantly affect the effectiveness of the method. For example, the device memory may serve as a constraint on the maximum training tokens in an iteration. Besides, the training issue may be altered by variations in computing throughput and communication bandwidth, causing a significant influence on the effectiveness of methods. To verify the generality of Pro-Prophet, we conduct experiments on diverse hardware environments with varying memory consumption, and the rate of computing throughput and communication bandwidth.

Due to the limited memory capacity compared to the *HPWNV* and *HPNV*, we only train the four smaller models listed in Table [III](#page-8-0) on the *HPWNV* cluster. The number of tokens trained in one iteration is set to 4096.

Table [IV](#page-8-1) shows the speedup results under various models and values of k in a cluster consisting of 4 *HPNV* nodes. The highest speedups are highlighted in the table. With the equipment of NVLink connections in the cluster, communication

<span id="page-9-0"></span>![](_page_9_Figure_1.jpeg)

Fig. 10. End-to-end performance. The numbers in the captions denote speedups achieved by Pro-Prophet over the best baseline.

<span id="page-9-1"></span>TABLE V THE OVERALL SPEEDUP ON A 2 LPWNV NODES.

<span id="page-9-4"></span>

| l K | K GPUs | Tokens | Model -    | Speedup to DeepspeedMoE |             |  |
|-----|--------|--------|------------|-------------------------|-------------|--|
| ıx. |        |        |            | FasterMoE               | Pro-Prophet |  |
|     | 1 8    | 8 4096 | MoE-GPT-S  | 1.20                    | 1.30        |  |
| ١,  |        |        | MoE-GPT-M  | 1.02                    | 1.18        |  |
| 1   |        |        | MoE-GPT-DS | 1.12                    | 1.30        |  |
|     |        |        | MoE-GPT-DM | 0.96                    | 1.26        |  |
|     |        |        | MoE-GPT-S  | 1.56                    | 1.91        |  |
| 2   |        |        | MoE-GPT-M  | 1.29                    | 1.94        |  |
|     |        |        | MoE-GPT-DS | 1.44                    | 1.64        |  |
|     |        |        | MoE-GPT-DM | 1.25                    | 1.58        |  |

<sup>&</sup>lt;sup>1</sup> The label "Token" is the number of tokens trained in an iteration.

![](_page_9_Figure_6.jpeg)

<span id="page-9-5"></span>Fig. 11. Speedups across different layers over the Deepspeed-MoE in the MoE-GPT-M model with different values of k. Pro-Prophet achieves 1.09-1.49x single-layer speedups compared to FasterMoE.

processes such as A2A will be accelerated. Under this condition, Pro-Prophet achieves 1.71-2.63x speedups compared to Deepspeed-MoE and 1.10-1.35x to FasterMoE, demonstrating its adaptability to conditions with higher communication bandwidth. We also test Pro-Prophet on a cluster consisting of 2 *LPWNV* nodes. The results are presented in Table V and the highest speedups are emphasized too. Due to the lower computation ability of the 2080Ti compared to the 3090 GPU, the impact of the computation process becomes more significant. In this environment, Pro-Prophet achieved a speedup of 1.18-1.94x compared to Deepspeed-MoE and 1.08-1.50x compared to FasterMoE, showing its robustness in conditions with lower computation power.

The result of the MoE-GPT-DM model with k=1 shows that Deepspeed-MoE achieves higher performance compared to FasterMoE as FasterMoE transports parameters to unnecessary devices, resulting in additional runtime overhead. However,

<span id="page-9-3"></span><span id="page-9-2"></span>![](_page_9_Figure_10.jpeg)

<span id="page-9-6"></span>Fig. 12. Per-iteration execution time in the MoE-GPT-M model when k=1. Pro-Prophet achieved 1.34x speedup on average compared to fasterMoE.

Pro-Prophet can accurately find a communication-efficient expert placement, thereby avoiding this trouble.

## B. Fine-grained analysis of Pro-Prophet

We conducted a fine-grained analysis of Pro-Prophet according to speedups in a single layer and a single iteration. Experimental results demonstrate that Pro-Prophet enhances training performance in each layer and iteration during training.

**Single-layer speedup.** We first evaluate the single-layer performance of Pro-Prophet. Fig. 11 illustrates the execution time across different layers of three methods on the MoE-GPT-M model. We randomly select the index of layers and use the Pytorch Profiler to collect the training time. As shown in the figure, Pro-Prophet achieves 1.60-2.25x single-layer speedups compared to Deepspeed-MoE and 1.09-1.49x to FasterMoE.

Varying loads of experts across layers occur in the training process. This phenomenon leads to fluctuating speedups for Pro-Prophet. However, it consistently outperforms two baselines in different layers, demonstrating its superior capability of load balancing under diverse load-imbalance conditions.

**Single-iteration speedups.** We also evaluate the single-layer performance of Pro-Prophet. We conduct experiments on the MoE-GPT-M model with k=1. The results are presented in Fig. 12. Compared to fasterMoE, Pro-Prophet achieved 1.34x speedup on average. The iteration time of Pro-Prophet is consistent and lower. This phenomenon can mainly be attributed to the fact that Pro-Prophet is capable of adapting to dynamic situations.

![](_page_10_Figure_1.jpeg)

<span id="page-10-3"></span>Fig. 13. The accuracy of the performance model. The mean estimation error is less than 5%.

#### C. Ablation study

**Necessity of the dynamic adaptation.** Dynamic adaption is necessary for MoE models. It's reasonable to transfer heavyload experts to other GPUs, but it's unclear if Pro-Prophet's dynamic search algorithm is necessary.

To certify the necessity of our algorithm, we compare the planner with two simple dynamic policies. Specifically, two policies transfer 2 and 3 experts with the heaviest load to all GPUs. We named them top2 and top3 respectively. We use PyTorch's topk function to implement these strategies. The overhead of determining the heaviest experts is negligible.

Figure 15 illustrates the latency of three policies in a single iteration with different values of k. As shown in Fig. 15a, the planner gains 1.77-1.82x speedups compared to the top2 policy and 2.04-2.10x speedups to the top3 policy when k=1. The results shown in Fig. 15b demonstrate that the planner gains speedups ranging from 1.38-1.40x compared to different policies when k=2.

The experimental results indicate that fixing the number of experts and passing them to all GPUs does not yield good results. The input distribution changes as training progresses, resulting in different optimal expert placements. Compared to these two dynamic strategies, our algorithm introduces more overhead, but it's necessary for faster training speeds.

Accuracy of performance model. Fig. 13 illustrates the accuracy of the performance model. We compare the estimated time to the real time on A2A, expert computation (EC), Trans and Agg operations. The results show that our mean estimation error is less than 5%.

**Effectiveness of components.** To verify the effectiveness of the components, we conduct incremental experiments. We first turn off all optimizations for Pro-Prophet and use it as a baseline. Based on this, we activate the planner and scheduler sequentially and record the speedup they attain relative to the baseline. Finally, we verify the effective combination of the planner and scheduler mentioned in Sec. V.

Fig. 14 demonstrates speedups on the MoE-GPT-M under different k. Compared to the baseline, the planner gains 1.26x and 1.12x speedups when k=1 and k=2 respectively. These results show that the planner can efficiently and accurately

![](_page_10_Figure_11.jpeg)

<span id="page-10-4"></span>Fig. 14. The effectiveness of components. The baseline is Pro-Prophet without any optimizations. Full is the condition of turning on the effective combination of the planner and scheduler. The planner, scheduler and Full achieve a speedup of 1.19x, 1.075x and 1.025x on average respectively.

<span id="page-10-1"></span>![](_page_10_Figure_13.jpeg)

<span id="page-10-2"></span><span id="page-10-0"></span>Fig. 15. The iteration latency of different policies in the MoE-GPT-M model.

determine a communication-efficient expert placement for well load-balancing under different conditions. Besides, the scheduler gains 1.14X and 1.01x speedups when k=1 and k=2 respectively. These verify that the scheduler can hide the overhead of load-balancing, further improving the training performance. The speedups achieved by the scheduler are significantly influenced by the expert placement produced by the planner. Finally, we test the effectiveness of the effective combination (*Full* in the figure). As the performance model estimates the overlapped execution time, the planner will further balance the load based on the scheduler's capability of communication and computation overlapping. The results demonstrate that it achieves 1.03x and 1.02x speedups under different values of k.

Balance capability. Balance capability serves as a key

![](_page_10_Figure_17.jpeg)

<span id="page-10-5"></span>Fig. 16. The ratio of RB of the planner to FasterMoE in different models and k. The balance degree is the standard deviation of the input distribution tensor. The ratio of the balance degree before and after employing a load-balancing solution is RB of the solution. The planner achieves up to 11.01x ratio of RB.

metric for load-balancing methods. We define the balance degree as the standard deviation of the input distribution tensor. Besides, we denote the ratio of the balance degree before and after employing a load-balancing solution as *RB* to describe its effect on the load.

Fig. [16](#page-10-5) demonstrates the ratio of *RB* of Pro-Prophet planner to that of fasterMoE on different layers in different values of k. It is worth mentioning that the training time for the planner is lower than that of FasterMoE. In most cases, the planner achieves a higher *RB* than fasterMoE. A ratio of *RB* up to 11.01x indicates its ability to enhance training efficiency by fully exploiting the potential of load balancing. In experimental conditions including k=1 with layer=2, k=2 with layers=2 and 5, the ratios of *RB* are below 1, suggesting that the planner tailors expert placement to the actual load, preventing the unnecessary allocation of experts.

In summary, the planner can dynamically determine the load-balancing strategy to maximize training efficiency, showing its superior balance capability.

# VII. RELATED WORK

Hybrid parallelism. Hybrid parallelism strategies [\[36\]](#page-12-15)– [\[41\]](#page-12-16) have been widely used to train large-scale dense models. These hybrid parallelism strategies consist of but are not limited to DP [\[42\]](#page-12-17)–[\[45\]](#page-12-18), TP [\[30\]](#page-12-11), [\[46\]](#page-12-19), PP [\[47\]](#page-12-20), [\[48\]](#page-12-21), and sequence parallelism (SP) [\[49\]](#page-12-22)–[\[52\]](#page-13-0). Unfortunately, it is hard to efficiently train large MoE models utilizing these hybrid parallelism strategies.

To overcome the challenge, a series of works combine EP with above parallelism strategies and effectively improve training efficiency. Switch transformers combines EP with TP straightforwardly and designs a scheme to place the model and data on TPUs. Bagualu [\[53\]](#page-13-1) develops a hybrid parallel strategy that integrates EP and DP tailored for high-performance computing architectures, along with communication and storage optimizations designed to enhance training efficiency. Deepspeed-MoE designs an effective combination of DP, EP and TP for inference but is easy to extend to model training. In the MoE layer, it introduces Allgather and Allreduce primitives to aggregate data and immediate results. Tutel also proposes DP, TP, and EP hybrid parallelism strategies and designs an adaptive parallelism switching method that enables O(1) overhead in runtime switching. Based on the Deepspeed-MoE and Tutel, Parm [\[54\]](#page-13-2) combines DP, EP, and expert-slice parallelism (ESP) and proposes a fine-grained communication scheduling to improve the utilization of communication links. DeepSpeed-TED [\[55\]](#page-13-3) designs a 3-dimensional hybrid parallelism strategy that contains DP of Zero-3, TP of Megatron-LM, and EP of Deepspeed-MoE. Besides, it proposes a memory and communication optimization for better scalability. The methods of Pro-Prophet are compatible with these hybrid parallelism strategies and can help further improve the training efficiency.

Communication schedule. Overlapping communications and computing can enhance hardware utilization and improve the system's throughput [\[56\]](#page-13-4), [\[57\]](#page-13-5). Previous communication scheduling methods [\[37\]](#page-12-23), [\[58\]](#page-13-6) for dense models have demonstrated promising results. In this paragraph, We focus on introducing works designed for MoE models.

Mainstream communication scheduling works focused on pipelining A2A and expert computation. Specifically, they partition an A2A and expert computation operation into suboperators and overlap communication sub-operators with computation ones. Methods implemented on Gshard-like frameworks such as Lina [\[34\]](#page-12-13), Tutel, ScheMoE [\[59\]](#page-13-7), and PipeMoE [\[60\]](#page-13-8) partition computation and communication operators based on the shape of expert computation matrix. FasterMoE is implemented on FastMoE and partitions operators into irregular sub-operators to schedule. Pro-Prophet is compatible with these works as Pro-Prophet allows for overlapping communications and computations at the level of MoE blocks.

# VIII. CONCLUSION

In this paper, we propose Pro-Prophet, a systematic loadbalancing approach for efficient training of MoE models. We observe a locality among input distributions and use it to design the planner and scheduler. Pro-Prophet planner identifies lightweight expert placements and designs a locality-based greedy algorithm to efficiently search for a communicationefficient expert placement using its proposed performance model, effectively reducing the communication overhead. Pro-Prophet scheduler predicts the input distribution based on the locality in the MoE model training and applies blockwise scheduling to overlap communications and computations, further decreasing the communication cost. Our experiments show that Pro-Prophet achieves 1.18-2.66x and 1.01-1.50x speedups compared to Deepspeed-MoE and FasterMoE. Besides, Pro-Prophet achieves a load balancing enhancement of up to 11.01 when compared to FasterMoE.

# REFERENCES

- <span id="page-11-0"></span>[1] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen, "Gshard: Scaling giant models with conditional computation and automatic sharding," in *International Conference on Learning Representations*, 2021.
- <span id="page-11-1"></span>[2] N. Du, Y. Huang, A. M. Dai, S. Tong, D. Lepikhin, Y. Xu, M. Krikun, Y. Zhou, A. W. Yu, O. Firat *et al.*, "Glam: Efficient scaling of language models with mixture-of-experts," in *International Conference on Machine Learning*. PMLR, 2022, pp. 5547–5569.
- <span id="page-11-2"></span>[3] A. Liu, B. Feng, B. Wang, B. Wang, B. Liu, C. Zhao, C. Dengr, C. Ruan, D. Dai, D. Guo *et al.*, "Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model," *arXiv preprint arXiv:2405.04434*, 2024.
- <span id="page-11-3"></span>[4] S. Rajbhandari, C. Li, Z. Yao, M. Zhang, R. Y. Aminabadi, A. A. Awan, J. Rasley, and Y. He, "Deepspeed-moe: Advancing mixture-ofexperts inference and training to power next-generation ai scale," in *International Conference on Machine Learning*. PMLR, 2022, pp. 18 332–18 346.
- <span id="page-11-4"></span>[5] C. Hwang, W. Cui, Y. Xiong, Z. Yang, Z. Liu, H. Hu, Z. Wang, R. Salas, J. Jose, P. Ram *et al.*, "Tutel: Adaptive mixture-of-experts at scale," *Proceedings of Machine Learning and Systems*, vol. 5, pp. 269–287, 2023.
- <span id="page-11-5"></span>[6] W. Fedus, B. Zoph, and N. Shazeer, "Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity," *The Journal of Machine Learning Research*, vol. 23, no. 1, pp. 5232–5270, 2022.
- <span id="page-11-6"></span>[7] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer," in *International Conference on Learning Representations*, 2017.
- <span id="page-11-7"></span>[8] J. He, J. Zhai, T. Antunes, H. Wang, F. Luo, S. Shi, and Q. Li, "Fastermoe: modeling and optimizing training of large-scale dynamic pretrained models," in *Proceedings of the 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming*, 2022, pp. 120–134.

- <span id="page-12-0"></span>[9] X. Nie, X. Miao, Z. Wang, Z. Yang, J. Xue, L. Ma, G. Cao, and B. Cui, "Flexmoe: Scaling large-scale sparse pre-trained model training via dynamic device placement," *Proceedings of the ACM on Management of Data*, vol. 1, no. 1, pp. 1–19, 2023.
- <span id="page-12-1"></span>[10] S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He, "Zero: Memory optimizations toward training trillion parameter models," in *SC20: International Conference for High Performance Computing, Networking, Storage and Analysis*. IEEE, 2020, pp. 1–16.
- <span id="page-12-2"></span>[11] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei, "Scaling laws for neural language models," *arXiv preprint arXiv:2001.08361*, 2020.
- <span id="page-12-3"></span>[12] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "Bert: Pre-training of deep bidirectional transformers for language understanding," *arXiv preprint arXiv:1810.04805*, 2018.
- [13] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, "Exploring the limits of transfer learning with a unified text-to-text transformer," *The Journal of Machine Learning Research*, vol. 21, no. 1, pp. 5485–5551, 2020.
- [14] Z. Yang, Z. Dai, Y. Yang, J. Carbonell, R. R. Salakhutdinov, and Q. V. Le, "Xlnet: Generalized autoregressive pretraining for language understanding," *Advances in neural information processing systems*, vol. 32, 2019.
- [15] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, "Roberta: A robustly optimized bert pretraining approach," *arXiv preprint arXiv:1907.11692*, 2019.
- [16] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever *et al.*, "Language models are unsupervised multitask learners," *OpenAI blog*, vol. 1, no. 8, p. 9, 2019.
- <span id="page-12-4"></span>[17] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell *et al.*, "Language models are few-shot learners," *Advances in neural information processing systems*, vol. 33, pp. 1877–1901, 2020.
- <span id="page-12-5"></span>[18] S. Smith, M. Patwary, B. Norick, P. LeGresley, S. Rajbhandari, J. Casper, Z. Liu, S. Prabhumoye, G. Zerveas, V. Korthikanti *et al.*, "Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model," *arXiv preprint arXiv:2201.11990*, 2022.
- <span id="page-12-6"></span>[19] X. O. He, "Mixture of a million experts," *arXiv preprint arXiv:2407.04153*, 2024.
- [20] S. Zuo, X. Liu, J. Jiao, Y. J. Kim, H. Hassan, R. Zhang, T. Zhao, and J. Gao, "Taming sparsely activated transformer with stochastic experts," *arXiv preprint arXiv:2110.04260*, 2021.
- [21] J. Ludziejewski, J. Krajewski, K. Adamczewski, M. Pioro, M. Krutul, ´ S. Antoniak, K. Ciebiera, K. Krol, T. Odrzyg ´ o´zd´ z, P. Sankowski ´ *et al.*, "Scaling laws for fine-grained mixture of experts," in *Forty-first International Conference on Machine Learning*.
- [22] A. Komatsuzaki, J. Puigcerver, J. Lee-Thorp, C. R. Ruiz, B. Mustafa, J. Ainslie, Y. Tay, M. Dehghani, and N. Houlsby, "Sparse upcycling: Training mixture-of-experts from dense checkpoints," *arXiv preprint arXiv:2212.05055*, 2022.
- [23] F. Xue, Z. Shi, F. Wei, Y. Lou, Y. Liu, and Y. You, "Go wider instead of deeper," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 36, no. 8, 2022, pp. 8779–8787.
- [24] F. Xue, X. He, X. Ren, Y. Lou, and Y. You, "One student knows all experts know: From sparse to dense," *arXiv preprint arXiv:2201.10890*, 2022.
- <span id="page-12-7"></span>[25] B. Zoph, I. Bello, S. Kumar, N. Du, Y. Huang, J. Dean, N. Shazeer, and W. Fedus, "St-moe: Designing stable and transferable sparse expert models," *arXiv preprint arXiv:2202.08906*, 2022.
- <span id="page-12-8"></span>[26] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat *et al.*, "Gpt-4 technical report," *arXiv preprint arXiv:2303.08774*, 2023.
- <span id="page-12-9"></span>[27] R. Sepulchre, D. A. Paley, and N. E. Leonard, "Stabilization of planar collective motion: All-to-all communication," *IEEE Transactions on automatic control*, vol. 52, no. 5, pp. 811–824, 2007.
- [28] S. Kumar, Y. Sabharwal, R. Garg, and P. Heidelberger, "Optimization of all-to-all communication on the blue gene/l supercomputer," in *2008 37th International Conference on Parallel Processing*. IEEE, 2008, pp. 320–329.
- <span id="page-12-10"></span>[29] P. Sanders and J. L. Traff, "The hierarchical factor algorithm for ¨ all-to-all communication," in *Euro-Par 2002 Parallel Processing: 8th International Euro-Par Conference Paderborn, Germany, August 27–30, 2002 Proceedings 8*. Springer, 2002, pp. 799–803.
- <span id="page-12-11"></span>[30] M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, and B. Catanzaro, "Megatron-lm: Training multi-billion parameter language models using model parallelism," *arXiv preprint arXiv:1909.08053*, 2019.

- [31] X. Nie, P. Zhao, X. Miao, T. Zhao, and B. Cui, "Hetumoe: An efficient trillion-scale mixture-of-expert distributed training system," *arXiv preprint arXiv:2203.14685*, 2022.
- [32] J. He, J. Qiu, A. Zeng, Z. Yang, J. Zhai, and J. Tang, "Fastmoe: A fast mixture-of-expert training system," *arXiv preprint arXiv:2103.13262*, 2021.
- <span id="page-12-12"></span>[33] D. Yu, L. Shen, H. Hao, W. Gong, H. Wu, J. Bian, L. Dai, and H. Xiong, "Moesys: A distributed and efficient mixture-of-experts training and inference system for internet services," *IEEE Transactions on Services Computing*, 2024.
- <span id="page-12-13"></span>[34] J. Li, Y. Jiang, Y. Zhu, C. Wang, and H. Xu, "Accelerating distributed moe training and inference with lina," in *2023 USENIX Annual Technical Conference (USENIX ATC 23)*, 2023, pp. 945–959.
- <span id="page-12-14"></span>[35] J. Liu, J. H. Wang, and Y. Jiang, "Janus: A unified distributed training framework for sparse mixture-of-experts models," in *Proceedings of the ACM SIGCOMM 2023 Conference, ACM SIGCOMM 2023, New York, NY, USA, 10-14 September 2023*. ACM, 2023, pp. 486–498.
- <span id="page-12-15"></span>[36] D. Li, H. Wang, E. Xing, and H. Zhang, "Amp: Automatically finding model parallel strategies with heterogeneity awareness," *Advances in Neural Information Processing Systems*, vol. 35, pp. 6630–6639, 2022.
- <span id="page-12-23"></span>[37] Z. Lai, S. Li, X. Tang, K. Ge, W. Liu, Y. Duan, L. Qiao, and D. Li, "Merak: An efficient distributed dnn training framework with automated 3d parallelism for giant foundation models," *IEEE Transactions on Parallel and Distributed Systems*, vol. 34, no. 5, pp. 1466–1478, 2023.
- [38] X. Ye, Z. Lai, S. Li, L. Cai, D. Sun, L. Qiao, and D. Li, "Hippie: A data-paralleled pipeline approach to improve memory-efficiency and scalability for large dnn training," in *50th International Conference on Parallel Processing*, 2021, pp. 1–10.
- [39] S. Li, H. Liu, Z. Bian, J. Fang, H. Huang, Y. Liu, B. Wang, and Y. You, "Colossal-ai: A unified deep learning system for large-scale parallel training," in *Proceedings of the 52nd International Conference on Parallel Processing*, 2023, pp. 766–775.
- [40] J. M. Tarnawski, D. Narayanan, and A. Phanishayee, "Piper: Multidimensional planner for dnn parallelization," *Advances in Neural Information Processing Systems*, vol. 34, pp. 24 829–24 840, 2021.
- <span id="page-12-16"></span>[41] K. Lu, Z. Lai, S. Li, W. Liu, K. Ge, X. Lu, and D. Li, "Parallel intelligent computing: development and challenges," *SCIENTIA SINICA Informationis*, vol. 53, no. 8, pp. 1441–1468, 2023.
- <span id="page-12-17"></span>[42] S. Rajbhandari, O. Ruwase, J. Rasley, S. Smith, and Y. He, "Zeroinfinity: Breaking the gpu memory wall for extreme scale deep learning," in *Proceedings of the international conference for high performance computing, networking, storage and analysis*, 2021, pp. 1–14.
- [43] J. Ren, S. Rajbhandari, R. Y. Aminabadi, O. Ruwase, S. Yang, M. Zhang, D. Li, and Y. He, "Zero-offload: Democratizing billion-scale model training," in *2021 USENIX Annual Technical Conference (USENIX ATC 21)*, 2021, pp. 551–564.
- [44] Y. Zhao, A. Gu, R. Varma, L. Luo, C.-C. Huang, M. Xu, L. Wright, H. Shojanazeri, M. Ott, S. Shleifer *et al.*, "Pytorch fsdp: experiences on scaling fully sharded data parallel," *arXiv preprint arXiv:2304.11277*, 2023.
- <span id="page-12-18"></span>[45] Z. Zhang, S. Zheng, Y. Wang, J. Chiu, G. Karypis, T. Chilimbi, M. Li, and X. Jin, "Mics: near-linear scaling for training gigantic model on public cloud," *arXiv preprint arXiv:2205.00119*, 2022.
- <span id="page-12-19"></span>[46] Z. Bian, Q. Xu, B. Wang, and Y. You, "Maximizing parallelism in distributed training for huge neural networks," *arXiv preprint arXiv:2105.14450*, 2021.
- <span id="page-12-20"></span>[47] W. Liu, Z. Lai, S. Li, Y. Duan, K. Ge, and D. Li, "Autopipe: A fast pipeline parallelism approach with balanced partitioning and microbatch slicing," in *2022 IEEE International Conference on Cluster Computing (CLUSTER)*. IEEE, 2022, pp. 301–312.
- <span id="page-12-21"></span>[48] Y. Duan, Z. Lai, S. Li, W. Liu, K. Ge, P. Liang, and D. Li, "Hph: Hybrid parallelism on heterogeneous clusters for accelerating large-scale dnns training," in *2022 IEEE International Conference on Cluster Computing (CLUSTER)*. IEEE, 2022, pp. 313–323.
- <span id="page-12-22"></span>[49] S. Li, F. Xue, C. Baranwal, Y. Li, and Y. You, "Sequence parallelism: Long sequence training from system perspective," *arXiv preprint arXiv:2105.13120*, 2021.
- [50] V. A. Korthikanti, J. Casper, S. Lym, L. McAfee, M. Andersch, M. Shoeybi, and B. Catanzaro, "Reducing activation recomputation in large transformer models," *Proceedings of Machine Learning and Systems*, vol. 5, pp. 341–353, 2023.
- [51] S. A. Jacobs, M. Tanaka, C. Zhang, M. Zhang, S. L. Song, S. Rajbhandari, and Y. He, "Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models," *arXiv preprint arXiv:2309.14509*, 2023.

- <span id="page-13-0"></span>[52] H. Liu, M. Zaharia, and P. Abbeel, "Ring attention with blockwise transformers for near-infinite context," *arXiv preprint arXiv:2310.01889* , 2023.
- <span id="page-13-1"></span>[53] Z. Ma, J. He, J. Qiu, H. Cao, Y. Wang, Z. Sun, L. Zheng, H. Wang, S. Tang, T. Zheng *et al.*, "Bagualu: targeting brain scale pretrained models with over 37 million cores," in *Proceedings of the 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming*, 2022, pp. 192–204.
- <span id="page-13-2"></span>[54] X. Pan, W. Lin, S. Shi, X. Chu, W. Sun, and B. Li, "Parm: Efficient training of large sparsely-activated models with dedicated schedules," in *IEEE INFOCOM 2024-IEEE Conference on Computer Communications* . IEEE, 2024, pp. 1880–1889.
- <span id="page-13-3"></span>[55] S. Singh, O. Ruwase, A. A. Awan, S. Rajbhandari, Y. He, and A. Bhatele, "A hybrid tensor-expert-data parallelism approach to optimize mixtureof-experts training," in *Proceedings of the 37th International Conference on Supercomputing*, 2023, pp. 203–214.
- <span id="page-13-4"></span>[56] S. Shi, X. Chu, and B. Li, "Mg-wfbp: Efficient data communication for distributed synchronous sgd algorithms," in *IEEE INFOCOM 2019- IEEE Conference on Computer Communications*. IEEE, 2019, pp. 172– 180.
- <span id="page-13-5"></span>[57] C. He, S. Li, M. Soltanolkotabi, and S. Avestimehr, "Pipetransformer: Automated elastic pipelining for distributed training of transformers," *arXiv preprint arXiv:2102.03161*, 2021.
- <span id="page-13-6"></span>[58] S. Li, K. Lu, Z. Lai, W. Liu, K. Ge, and D. Li, "A multidimensional communication scheduling method for hybrid parallel dnn training," *IEEE Transactions on Parallel and Distributed Systems*, 2024.
- <span id="page-13-7"></span>[59] S. Shi, X. Pan, Q. Wang, C. Liu, X. Ren, Z. Hu, Y. Yang, B. Li, and X. Chu, "Schemoe: An extensible mixture-of-experts distributed training system with tasks scheduling," in *Proceedings of the Nineteenth European Conference on Computer Systems*, 2024, pp. 236–249.
- <span id="page-13-8"></span>[60] S. Shi, X. Pan, X. Chu, and B. Li, "Pipemoe: Accelerating mixtureof-experts through adaptive pipelining," in *IEEE INFOCOM 2023-IEEE Conference on Computer Communications*. IEEE, 2023, pp. 1–10.