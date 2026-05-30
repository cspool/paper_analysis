# M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework

Zijian Zhang∗† Jilin University City University of Hong Kong Jilin, China zhangzj2114@mails.jlu.edu.cn

Shuchang Liu Kuaishou Technology Beijing, China liushuchang@kuaishou.com

Jiaao Yu Kuaishou Technology Beijing, China yujiaao596@163.com

Qingpeng Cai Kuaishou Technology Beijing, China cqpcurry@gmail.com

Xiangyu Zhao‡ City University of Hong Kong Hong Kong, China xianzhao@cityu.edu.hk

Chunxu Zhang‡ Jilin University Jilin, China cxzhang19@mails.jlu.edu.cn

Ziru Liu City University of Hong Kong Hong Kong, China ziruliu2-c@my.cityu.edu.hk

Qidong Liu Xi'an Jiaotong University City University of Hong Kong Xi'an, China liuqidong@stu.xjtu.edu.cn

Hongwei Zhao‡ Jilin University Jilin, China zhaohw@jlu.edu.cn

Lantao Hu Kuaishou Technology Beijing, China hulantao@gmail.com

Peng Jiang Kuaishou Technology Beijing, China jp2006@139.com

Kun Gai Unaffiliated Beijing, China gai.kun@qq.com

# ABSTRACT

Multi-domain recommendation and multi-task recommendation have demonstrated their effectiveness in leveraging common information from different domains and objectives for comprehensive user modeling. Nonetheless, the practical recommendation usually faces multiple domains and tasks simultaneously, which cannot be well-addressed by current methods. To this end, we introduce M3oE, an adaptive Multi-domain Multi-task Mixture-of-Experts recommendation framework. M3oE integrates multi-domain information, maps knowledge across domains and tasks, and optimizes multiple objectives. We leverage three mixture-of-experts modules to learn common, domain-aspect, and task-aspect user preferences respectively to address the complex dependencies among multiple domains and tasks in a disentangled manner. Additionally, we design a two-level fusion mechanism for precise control over feature extraction and fusion across diverse domains and tasks. The framework's

# CCS CONCEPTS

• Information systems → Recommender systems.

# KEYWORDS

Recommender System; Multi-Domain; Multi-Task

# ACM Reference Format:

Zijian Zhang, Shuchang Liu, Jiaao Yu, Qingpeng Cai, Xiangyu Zhao, Chunxu Zhang, Ziru Liu, Qidong Liu, Hongwei Zhao, Lantao Hu, Peng Jiang, and Kun Gai. 2024. M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework . In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24), July 14–18, 2024, Washington, DC, USA. ACM, New York, NY, USA, [10](#page-9-0) pages.

<https://doi.org/10.1145/3626772.3657686>

# 1 INTRODUCTION

With the rapid growth of web services and user-generated data, recommendation services have devoted increasing attention to a more precise understanding of user preferences and content recommendation [\[7,](#page-9-1) [8,](#page-9-2) [23,](#page-9-3) [28\]](#page-9-4). As a result of this trend, the research on the

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

SIGIR'24, July 14–18, 2024, Washington, WDC

adaptability is further enhanced by applying AutoML technique, which allows dynamic structure optimization. To the best of the authors' knowledge, our M3oE is the first effort to solve multi-domain multi-task recommendation self-adaptively. Extensive experiments on two benchmark datasets against diverse baselines demonstrate M3oE's superior performance. The implementation code is available to ensure reproducibility[1](#page-0-0) .

<sup>∗</sup>Zijian Zhang is also with the Key Laboratory of Symbolic Computation and Knowledge Engineering of Ministry of Education, Jilin University.

<sup>†</sup>This work is done during Zijian's internship at Kuaishou Technology.

<sup>‡</sup>Corresponding authors.

<sup>©</sup> 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-0431-4/24/07

<https://doi.org/10.1145/3626772.3657686>

<span id="page-0-0"></span><sup>1</sup><https://github.com/Applied-Machine-Learning-Lab/M3oE>

recommender system has gradually moved towards increasingly complex but more practical scenarios like multi-domain [\[5,](#page-9-5) [9,](#page-9-6) [14–](#page-9-7) [16,](#page-9-8) [25\]](#page-9-9) and multi-task [\[1,](#page-9-10) [20,](#page-9-11) [21,](#page-9-12) [26\]](#page-9-13) problems. The multi-domain recommendation problem assumes that users may exhibit similar tastes and behaviors across domains or platforms, indicating the feasibility of auxiliary information transfer [\[37\]](#page-9-14). For example, a user Mike watched Sci-Fi movies on TV, and is now browsing movies on the tablet. Then knowing Mike's movie-watching history on TV would help figure out the preferences of Sci-Fi, and benefit the recommendation on the tablet domain. The same effects also exist in reverse. In the multi-task scenario, users may interact or consume each recommended item in various ways, which brings multiple perspectives to describe a user's preference [\[22\]](#page-9-15). For instance, in a video-sharing platform, users can watch, like, and save videos. In reality, different signals may not always be positively related, e.g., a user watched a video does not necessarily mean that the user likes or will save the video. A more considerate recommender should be able to model and employ these complicated relations between tasks during inference.

In this context, recommender system designs have recently witnessed significant progress in both scenarios. On one hand, the key question in the multi-domain recommendation scenario is how to accurately extract relevant information and efficiently transfer between different domains. For example, DDTCDR [\[14\]](#page-9-7) designs a latent orthogonal mapping to transfer embedding between two domains. STAR [\[25\]](#page-9-9) leverages factorized domain-shared and domainspecific adaptive networks with a star topology to address the domain characteristics and their commonalities. By introducing relevant knowledge from different domains, Multi-Domain Recommendation (MDR) can leverage the beneficial insights to enhance the user personalization modeling in each domain [\[17,](#page-9-16) [27,](#page-9-17) [30,](#page-9-18) [32,](#page-9-19) [36\]](#page-9-20). On the other hand, the multi-task recommendation scenario set up learning tasks for each of the feedback signals (e.g., click-through rate prediction, like rate prediction, and save rate prediction), and the key challenge is how to balance different tasks and collaboratively improve the overall performance. For example, Shared Bottom [\[1\]](#page-9-10) utilizes a shared bottom layer to extract cross-task information. MMoE [\[21\]](#page-9-12) designs common expert networks and task-specific towers for multi-task modeling. By learning user engagement from diverse aspects, Multi-Task Recommendation (MTR) can exploit the shared information between tasks and benefit from a more comprehensive understanding of user behavior [\[12,](#page-9-21) [21,](#page-9-12) [22\]](#page-9-15). Though the tasks may also negatively influence each other like the seesaw phenomenon [\[26,](#page-9-13) [35\]](#page-9-22) and so do the domains [\[2,](#page-9-23) [25\]](#page-9-9), in general, collaborate multiple learning tasks or incorporating information from multiple domains provides a more complete view of user-item interactions and helps improves the generality of recommendation.

Despite the progress of existing MDR and MTR methods, the actual recommendation environment nowadays is usually multidomain and multi-task at the same time [\[38\]](#page-9-24). This involves domaintask interplay that brings new challenges for information transfer and objective balancing. However, limited work except [\[38\]](#page-9-24) explored the combination between MDR and MTR and there is still a gap before reaching a comprehensive Multi-Domain Multi-Task (MDMT) solution. In our empirical study, we observe that merely including an MDR or MTR solution only achieves sub-optimal or imbalanced performance. As an intuitive illustration of this phe-

<span id="page-1-0"></span>![](_page_1_Figure_5.jpeg)

Figure 1: Multi-domain multi-task AUC comparisons on MovieLens. We report the relative improvement of MMoE, STAR, and our M3oE, compared with single-domain singletask MLP baseline. The different colors indicate different performance ranks in the same domain and task.

nomenon, Figure [1](#page-1-0) shows the ranking performance (i.e., AUC) of representative methods of different types of existing solutions: MLP as single-domain single-task solution, MMoE [\[21\]](#page-9-12) as MTR solution for each domain, and STAR [\[25\]](#page-9-9) as MDR solution for each task. Though MMoE and STAR improve the overall performance over the single-domain single-task baseline, they may get worse in some special cases, as indicated by orange grids. Besides, neither MMoE nor STAR is consistently better than the other for any single domaintask pair, indicating their sub-optimality in all domains and tasks. In general, we find it really hard for an MDR or MTR solution to generalize well in the MDMT setting.

In analogy to the domain seesaw [\[2,](#page-9-23) [25\]](#page-9-9) and task seesaw [\[26,](#page-9-13) [35\]](#page-9-22) phenomena, we address this new challenge as MDMT seesaw and further describe it in two aspects:

- The same multi-domain information transfer method may not generalize to different tasks; and
- The same multi-task optimization balancing strategy may not generalize to different domains.

Again, using the video-sharing platform as an example: the domain seesaw addresses how to transfer a user' preferences from TV to tablet domain, and the task seesaw addresses how to balance user's behavior of watching and liking. In contrast, the MDMT seesaw addresses problems like how to transfer a user's preference of watching on TV to augment user's preference of liking on tablet. We argue that the key to this problem lies in how well we generalize the multi-domain multi-task knowledge transfer and integration mechanism, which has been long neglected in existing works.

In this paper, we propose a novel framework M3oE to jointly model multi-domain information extraction and multi-task inferences as a solution to the aforementioned MDMT seesaw challenge. Specifically, it consists of a domain representation extraction layer that implements feature-level multi-domain knowledge transfer, a sophisticated multi-view expert learning layer that extracts and integrates information for specific domains and tasks, and an MDMT objective prediction layer that generates the separate output for each domain-task pair. To better extract transferable information across domains for all tasks, we employ three essential types of experts in the middle expert learning layer to leverage shared common interests, domain-aspect user preferences, and task-aspect user preferences, respectively. As we will discuss in Figure [3,](#page-7-0) this disentangled design helps capture different views of the input. With this multi-view knowledge extracted, we use a flexible two-level

fusion module to control the information aggregation for each domain and task: while the first level controls the integration between domains and between tasks, the second level controls the integration between shared experts, domain experts, and task experts. Furthermore, to pursue the generality for different datasets and recommendation environments, we leverage AutoML to optimize the fusion weights adaptively. Empirically, our proposed M³oE can effectively engage cross-domain cross-task knowledge transfer and integration. As the illustrative example in Figure 1, M³oE (rightmost) achieves consistent improvement over separate MLP, multi-domain, and multi-task baselines. We summarize our main contributions as follows:

- We identify the MDMT seesaw problem in a practical recommendation environment and point out the insufficient generality of sole multi-domain or multi-task solutions.
- To the best of our knowledge, our proposed framework M<sup>3</sup>oE\nis the first general framework to solve MDMT recommendation
  self-adaptively.
- Extensive experiments on two benchmark datasets demonstrate the superior performance of M³oE against state-of-the-art and indepth analysis support its efficacy on knowledge disentanglement and integration.

#### 2 METHODOLOGY

#### 2.1 Problem Definition

Multi-Domain Multi-Task Recommendation. Let  $\mathcal U$  and Irepresent the user set and item set. And the MDMT recommendation problem aims to find a solution that simultaneously optimizes T recommendation tasks on D domains. We assume that all users, items, and tasks may overlap across domains, but each sample is observed in a certain domain (e.g., a user interacts with an item either on TV or tablet, not both). As a result, for a sample in domain  $d \in \{1, ..., D\}$ , we define a comprehensive feature input  $x_d$  that consists of user, item, and context information. Similarly, each sample also observes target labels in T tasks under domain d and there is no label in other domains. In our setting, we consider X-through rate prediction task (e.g., click-through rate) for all tasks and each target label is a binary signal in 1,0 that defines whether the user provides positive feedback (e.g., a click) or not. Then, our research goal is to learn a X-through rate prediction function  $\hat{y}_{d,t} = f^{d,t}(x_d)$ for each task  $t \in \{1, ..., T\}$  and domain  $d \in \{1, ..., D\}$ .

#### 2.2 Framework Overview

As illustrated in Figure 2, we construct a general multi-domain multi-task framework, consisting of the domain representation extraction layer, multi-view expert learning layer, and MDMT objective prediction layer, from bottom to top. In the multi-view expert learning layer, we design the shared expert module, domain expert module, and task expert module, to address the common information learning, domain-aspect user preferences, and task-aspect user preferences, respectively. To achieve the model adaptivity, we assign a two-level fusion mechanism with automatically updated fusion weights. After attaining the comprehensive representation, we infer the user preference for each objective individually. The flexible model architecture and automatic fusion-weights optimization

<span id="page-2-0"></span>![](_page_2_Figure_11.jpeg)

Figure 2: Framework of  $M^3$  oE, focusing on domain d and task t for clarity. Within the multi-view layer, there are three modules arranged from left to right: shared expert module S, domain expert module D, and task expert module T.

of our  ${\rm M}^3{\rm oE}$  enable it to seamlessly adapt to various multi-domain and multi-task settings.

#### 2.3 Domain Representation Extraction Layer

To unify the multi-domain representation and enhance the knowledge transfer, we introduce a domain-specific and common information integration structure for the input features. Specifically, we incorporate fully connected layers to align attribute semantics and learn the representation of multiple domains [25]. In this approach, we assign a domain-specific weight matrix  $\mathbf{W}_d$  to handle domain-specific characteristics and utilize a shared weight matrix  $\mathbf{W}_{sh}$  that processes data from all domains to capture shared patterns. To effectively combine the shared and domain-specific pattern, we perform an element-wise multiplication  $\widehat{\mathbf{W}}_d = \mathbf{W}_d \otimes \mathbf{W}_{sh}$ , followed by the input vector processing. The domain representation extraction operation  $f_{DR}$  on domain data  $\mathbf{x}_d$  can be formulated as follows,

$$f_{\rm DR}(\mathbf{x}_d) = \widehat{\mathbf{W}}_d \mathbf{x}_d + \mathbf{b}_d + \mathbf{b}_{\rm sh} \tag{1}$$

where  $b_d$  and  $b_{sh}$  are the domain-specific and common bias.

After capturing domain common and distinct information, we perform a linear operation parameterized by a weight matrix  $W_c$  and a bias  $b_c$  to map the representation of different domains into the same embedding space. To mitigate the introduction of noisy information from other domains and regulate the unified representation space, we also include a multi-layer neural network  $f_{\rm DA}$  that engages domain-agnostic mapping. Then, given the sample input  $x_d$  of domain d, we extract its domain representation  $h_d$  as follows,

$$\boldsymbol{h}_d = \boldsymbol{W}_c f_{\mathrm{DR}}(\boldsymbol{x}_d) + \boldsymbol{b}_c + f_{\mathrm{DA}}(\boldsymbol{x}_d)$$
 (2)

Note that  $W_{\rm sh}$ ,  $W_c$ , and  $f_{\rm DA}$  learn the patterns from all domains. This would help generate basic features in a unified representation space for later information extraction modules.

# 2.4 Multi-View Expert Learning Layer

With the preprocessed representation  $h_d$ , we also need to find a way to further extract and integrate information for each domain and task. Intuitively, except for the common patterns (e.g., common user interests), domains and tasks may potentially have different views of the given information. In order to extract useful multiview patterns in a disentangled and comprehensive manner, our approach explicitly defines three types of expert networks:

- $\bullet$  The shared expert module  ${\cal S}$  focuses on learning common knowledge that spans across domains and tasks.
- The domain expert module D captures each domain's unique characteristics. Each expert is associated with one domain and is shared across tasks.
- The task expert module T models task-specific characteristics.
   Each task expert is associated with one task and is shared across domains.

For all expert network output, we design a two-level information fusion mechanism that learns to integrate extracted features for specific domains and tasks. The first level learns to aggregate information between shared experts, between domains or between tasks, while the second level learns to aggregate information between three types of experts. Without loss of generality, in the following sections, we show how different experts extract and integrate information for domain d and task t.

2.4.1 **Shared Expert Module**. To capture common patterns across different domains and tasks, we employ multiple expert networks [21] and all experts will process input from multiple domain, *i.e.*,  $\{h_d\}_{d=1}^D$ . In our approach, we incorporate N shared expert networks, each of which is assigned a linear operation with a ReLU activation function. Additionally, we introduce layer normalization to enhance the stability of the module learning process and improve embedding generalization. Note that the normalization technique ensures that the outputs of the expert networks have consistent distributions, promoting better alignment of the shared information. Mathematically, the expert network  $f_E^e$  for any expert  $e \in \{1, \ldots, N\}$  generates the following output:

$$f_E^e(\mathbf{h}_d) = \text{ReLU}(\text{LayerNorm}(\mathbf{W}_e \mathbf{h}_d + \mathbf{b}_e))$$
 (3)

where  $W_e$  and  $b_e$  are trainable weights and bias of  $f_E^e$ .

**Shared Information Fusion.** After learning the common information with N shared expert networks, we propose a gate fusion mechanism to integrate these embeddings as shared information. Concretely, we introduce a linear layer  $f_{\rm gate}(\cdot)$  that generates an N-dimensional softmax weights for the sample-wise weighted sum of shared expert networks output. We allocate  $D \times T$  gate layers to corresponding  $D \times T$  objectives, ensuring that each objective has its own dedicated gate layer. As depicted in the left part of Figure 2, domain d and task t's inference feeds on its corresponding gate-weighted embedding. Hence, we can generate the embedding maintaining the shared information across multiple domains and tasks  $S(h_d)$  as follows,

$$S(\mathbf{h}_d) = f_{\text{gate}}(\mathbf{h}_d) f_E(\mathbf{h}_d)$$
 (4)

where expert  $f_E$  belongs to the shared expert network set S. Notably, we utilize  $D \times T$  gate layers to assign a unique shared information fusion layer  $f_{\text{gate}}(\cdot)$  to each inference objective.

2.4.2 **Domain Expert Module**. The shared experts capture the common patterns, but each domain may have its own view about the input information and various tasks. Following this notion, the domain expert module  $\mathcal{D}$  establish D expert networks and we denoted  $\{f_E^d(\cdot)\}$  for each domain's expert, i.e.,  $\mathcal{D}:=\{f_E^d(\cdot)\}_{d=1}^D$  to address domain-specific information explicitly. The network architecture is similar to that of the shared experts and has separate learnable parameters for different domains.

Multi-Domain Disentanglement and Fusion. Different from shared experts that implicitly learn disentangled views, the domain experts are explicitly defined for input from different domains. Intuitively, for samples  $h_d$  from domain d, we should focus on the output from the corresponding expert of domain *d* and obtain the representation  $f_E^a(\mathbf{h}_d)$ . Besides,  $\mathbf{h}_d$  is in the same representation space with other domains' input, so using other domain experts to process  $h_d$  also becomes valid and potentially augments the information extraction. Specifically, we feed  $\boldsymbol{h}_d$  to the expert networks of other domains and denoted the obtained representations as  $f_E^d(\mathbf{h}_d)$ , where  $\widetilde{d} \in \{1, \dots, D | \widetilde{d} \neq d \}$ . In order to control the incorporation of information from other domain experts' representations, we set an affine balance weight  $\beta_d \in (0,1)$  that determines the balance between the current domain's representation and the information from other domain experts' descriptions. The multi-domain fusion embedding of domain d can be calculated as follows,

$$\mathcal{D}(\boldsymbol{h}_d) = \beta_d \cdot f_E^d(\boldsymbol{h}_d) + \frac{1 - \beta_d}{D - 1} \cdot \sum_{\tilde{d} \neq d}^D f_E^{\tilde{d}}(\boldsymbol{h}_d)$$
 (5)

where we assign equal weights  $\frac{1-\beta_d}{D-1}$  to the representations from other domain expert networks to stabilize the augmented information and facilitate model training. The sum of weights over all domain experts is 1, which guarantees that the weights maintain the desired balance. In other words, we consider unbiased information integration for other domain experts' perspectives, while the expert of the corresponding domain d receives focused attention through  $\beta_d$ 

2.4.3 **Task Expert Module**. To depict user interest from multiple tasks' aspects, we assign T expert networks for the task expert module  $\mathcal{T} := \{f_E^t(\cdot)\}_{t=1}^T$ . Similar to the domain expert module, each task is associated with a dedicated expert network, allowing explicit handling of task-specific information.

**Multi-Task Disentanglement and Fusion.** Different from domain experts, the same input  $h_d$  is processed by all task experts and generates disentangled output. Intuitively, each task should have its own focused view, so we adopt a similar biased fusion strategy where for the task t we calculate the following,

$$\mathcal{T}(\boldsymbol{h}_d) = \beta_t \cdot f_E^t(\boldsymbol{h}_d) + \frac{1 - \beta_t}{T - 1} \cdot \sum_{\tilde{t} \neq t}^T f_{\tilde{t}}^{\tilde{t}}(\boldsymbol{h}_d)$$
 (6)

where  $f_E^t(\cdot)$  is expert network of task t,  $f_E^t(\cdot)$  means expert networks of other tasks. Similar to the domain expert module, we assign unbiased weights to other task expert networks to ensure training stability and assign  $\beta_t$  for the focused task. From the task viewpoint, it is important to note that the data samples from all D domains share the same task expert network. This enables the task

expert module to integrate task-related domain-agnostic information, enhancing the understanding of user preferences within the context of each task.

2.4.4 Multi-View Representation Balancing. Based on the shared expert module, we obtain common patterns  $\mathcal{S}(h_d)$  across multiple domains and multiple tasks. In addition, through the domain expert module and task expert module, we obtain multidomain information  $\mathcal{D}(h_d)$ , and multi-task information  $\mathcal{T}(h_d)$ . As we discussed earlier, performance in certain domains and tasks can benefit from multi-domain modeling and multi-task modeling to varying degrees. At the same time, common knowledge between domains and tasks is also crucial in user preference modeling. Therefore, we propose to balance these three components to achieve the final representation for downstream inference. In specific, we allocate two weights  $\alpha_d \in (0,1)$  and  $\alpha_t \in (0,1)$  to balance the contribution of each component, achieving the comprehensive representation  $\overline{h}_d$  as follows,

$$\overline{h}_d = \mathcal{S}(h_d) + \alpha_d \cdot \mathcal{T}(h_d) + \alpha_t \cdot \mathcal{D}(h_d)$$
 (7)

By adjusting the values of  $\alpha_d$  and  $\alpha_t$ , we can flexibly balance the contributions of the domain-specific, task-specific, and common knowledge components, enabling an adjustable representation that effectively captures the user preferences.

**Discussion of Two-Level Multi-domain Multi-task Fusion.** In this paragraph, we take a close view of the elements of the embedding  $\bar{h}_d$ . In the two-level multi-domain multi-task fusion mechanism, the balance weight  $\beta_t$  controls the trade-off between the domain-specific expert network and the experts from other domains. When  $\beta_d \in (0.5,1)$ , the domain-specific expert network contributes more significantly compared to other domain experts. As  $\beta_d$  approaches 1, it represents a scenario where only the domain-specific expert network is considered without incorporating information from other domains. Conversely,  $\beta_d \in (0,0.5)$  indicates that knowledge transfer from other domain expert networks contributes more than the domain-specific expert network.

Furthermore,  $\alpha_d$  controls the overall weight of the domain expert module for generating domain representation  $\overline{h}_d$ . Higher values of  $\alpha_d$  signify that the domain-aspect patterns play a more significant role compared to the common knowledge and task-specific perspectives. It also further recognizes the efficacy of combination inside the domain expert module. Conversely, lower values of  $\alpha_d$  indicate a stronger reliance on the other two components during training. Based on  $\alpha_d$  and  $\beta_d$ , we achieve precise control of factorized contribution from specific and shared domain experts.

By appropriately weighting and combining these components, we can leverage the advantages of multi-domain modeling, multi-task modeling, and shared knowledge to create a comprehensive and representative embedding that captures user preferences. The concrete balancing and fusion scheme will depend on the concrete requirements and characteristics of the application scenario.

# 2.5 MDMT Objective Prediction Layer

To infer the user preference across different tasks and domains, we introduce multi-domain multi-task objective prediction layer in this subsection. The prediction process for a specific domain and task may depend on the diverse ingredients of domain representation

differently. To accommodate the diverse numerical spaces associated with different domains and tasks, we assign individual MLP prediction towers, denoted as  $f_{tower}^{d,t}(\cdot)$ , for each unique domain and task combination represented as (d,t).

The embeddings generated by the domain expert module and task expert module are fed to the corresponding prediction tower, respectively. The separate inference modules allow for fine-grained control over the domain and task expert networks.

$$f_{tower}^{d,t}(\overline{\boldsymbol{h}}_d) = \boldsymbol{W}_{d,t}^2 \operatorname{ReLU}(\boldsymbol{W}_{d,t}^1 \overline{\boldsymbol{h}}_d + \boldsymbol{b}_{d,t}^1) + \boldsymbol{b}_{d,t}^2$$
(8)

where  $W_{d,t}^2$ ,  $b_{d,t}^2$ ,  $W_{d,t}^1$ , and  $b_{d,t}^1$  are weights and biases of two layers. The model inferences of user preferences on samples from domain d and task t are calculated as,

$$\widehat{\boldsymbol{y}}_{d,t} = \operatorname{Sigmoid}(f_{tower}^{d,t}(\overline{\boldsymbol{h}}_d))$$
 (9

For binary classification tasks, we pick the Sigmoid function as the activation function of the prediction tower.

# 2.6 Optimization by AutoML

The existing multi-domain and multi-task framework depends on manually designed architectures, which suffer from poor generality on new data and tasks. Automated Machine Learning (AutoML) [18] has demonstrated its advanced adaptivity and structural flexibility in allocating architectures and hyper-parameters.

Considering the complex relationships among domains and tasks, we utilize AutoML to optimize the critical domain and task fusion weights  $\alpha_d$ ,  $\alpha_t$ ,  $\beta_d$ , and  $\beta_t$ . Specifically, we generate each weight by a one-dimensional trainable tensor  $\mathbf{e}_w \in \{\mathbf{e}_{\alpha_d}, \mathbf{e}_{\alpha_t}, \mathbf{e}_{\beta_d}, \mathbf{e}_{\beta_t}\}$  with a Sigmoid activation function respectively. Mathematically, the generation of  $w \in \{\alpha_d, \alpha_t, \beta_d, \beta_t\}$  can be formulated as follows,

$$w = \operatorname{Sigmoid}(\boldsymbol{e}_{w}) \tag{10}$$

Then, we optimize these weights along with model training, *i.e.*, Bi-Level Optimization. This end-to-end pipeline allows for the determination of optimal weights that correspond to the specific domains and tasks involved.

2.6.1 Bi-Level Optimization. Let W denote the model parameters of  $M^3$  oE,  $\alpha := \{\alpha_d, \alpha_t\}$ , and  $\beta := \{\beta_d, \beta_t\}$ . This framework could be optimized with a Bi-Level Optimization, which optimizes the two parts of parameters alternatively. We first update the model framework W for one epoch, then we calculate loss based on one batch data and optimize the weights  $\alpha$  and  $\beta$ . Notably, the update of  $\alpha$  and  $\beta$  are based on a mini-batch of training data, which takes trivial computational costs.

$$\min_{\alpha,\beta} \mathcal{L}\left(W^*(\alpha,\beta),\alpha,\beta\right) 
\text{s.t. } W^*(\alpha,\beta) = \arg\min_{W} \mathcal{L}\left(W,\alpha,\beta\right)$$
(11)

We select Binary Cross Entropy as the loss function,

$$\mathcal{L} = \sum_{d}^{|\mathcal{D}|} \sum_{t}^{|\mathcal{T}|} BCE(\widehat{\boldsymbol{y}}_{d,t}, \boldsymbol{y}_{d,t})$$
 (12)

Table 1: Dataset statistics.

<span id="page-5-0"></span>

| Dataset    | I       | MovieLen | s       | KuaiRand-Pure |           |        |  |  |  |
|------------|---------|----------|---------|---------------|-----------|--------|--|--|--|
| Domain     | 1       | 2        | 3       | 1             | 2         | 3      |  |  |  |
| #Users     | 1,325   | 2,096    | 2,619   | 15,398        | 27,049    | 11,809 |  |  |  |
| #Items     | 3,429   | 3,508    | 3,595   | 6,233         | 7,580     | 4,633  |  |  |  |
| #Instances | 210,747 | 395,556  | 393,906 | 178,087       | 2,236,414 | 93,165 |  |  |  |
| Percentage | 21.07%  | 39.55%   | 39.38%  | 7.10%         | 89.18%    | 3.72%  |  |  |  |

#### 3 EXPERIMENT

In this section, we demonstrate the experiment results, including comparison with diverse advanced baselines, visualization of the disentangled and fusion of multi-domain multi-task user preference, the components contribution verification, and key hyperparameters impact analysis.

#### 3.1 Dataset

We evaluate the efficacy of M³oE on two public benchmark recommendation datasets, *i.e.*, MovieLens-1M and KuaiRand-Pure. The datasets' statistics are shown in Table 1. The split ratio of training, validation, and test is 8:1:1.

- MovieLens<sup>2</sup> The MovieLens dataset comprises 1 million ratings for around 3,900 movies and includes 7 user attributes and 2 item attributes. It encompasses a diverse range of data, including ratings and user information from various domains and tasks. It records user demographic information, such as gender, age, occupation *etc.* We use the feature "age" to separate dataset into 3 domains and infer "click" and "like" 2 tasks.
- KuaiRand-Pure<sup>3</sup> This dataset is collected from the short video platform Kuaishou<sup>4</sup>, containing 21 user features, 12 video features, and 17 common features. We select "tab" representing interactions on different tabs to divide the dtaset into 3 domains and address "click" and "long-view" 2 tasks.

#### 3.2 Baseline

To identify the performance comprehensively, we select advanced baselines for comparison. Considering the diverse solutions to accomplish multi-domain and multi-task recommendation, we incorporate the MDR, MTR, and multi-domain multi-task versions of Shared Bottom, MMoE, and PLE. In specific, we denote the suffix "-MTL" for the multi-task setting, which runs D times individually for D domains. Denote "-MDL" for the multi-domain setting, which runs T times individually for T tasks. Denote "-MDMT" for the multi-domain multi-task setting.

- MLP. We utilize MLP architecture as a baseline performance for each specific task and domain. It includes a linear operation with a ReLU activation function and layer normalization.
- Shared Bottom [1]. It is a representative multi-task recommendation model architecture, which enables the sharing of the bottom layer across tasks.
- MMoE [21]. The Multi-gate Mixture of Experts (MMoE) framework utilizes expert networks to address the capture of common

- information in multi-task learning. Each task has its dedicated prediction tower and a gate mechanism for information fusion.
- PLE [26]. Progressive Layered Extraction (PLE) designs taskspecific expert networks for each task, which enriches the information fusion mechanism with task-specific information.
- AdaTT [12]. AdaTT is a multi-task learning approach that incorporates a branch consisting of task-specific experts, where the experts' outputs are linearly combined. AdaTT refers to the overall approach that involves both task-specific and shared modules.
   AdaTT-sp refers to the version without the shared module.
- STAR [25]. It is a multi-domain recommendation framework which designs star topology to integrate domain-specific and shared information effectively.
- M2M [38]. M2M is a multi-scenario multi-task meta-learning approach, which utilizes a meta unit to leverage scenario knowledge, explicitly capturing inter-scenario correlations, along with a meta attention module and meta tower module to capture diverse inter-scenario correlations and enhance the representation of scenario-specific features, respectively.

# 3.3 Experimental Setups

We select the AUC and LogLoss to evaluate the performance and report all results as the average value of 5 repetitive runs using different random seeds. The embedding size is set to 16, and the learning rate is set to 1e-2 for MovieLens and 3e-3 for KuaiRand-Pure. For the expert network, we employ a one-layer MLP, while the prediction layer consists of a two-layer MLP. In the shared expert module, we have 1 expert network for MovieLens, and 4 for KuaiRand-Pure. All expert networks across the three modules share the same structure. To deploy on Movielens or KuaiRand-Pure with 3 domains and 2 tasks, our  $M^3$ oE includes 3 expert networks in the domain expert module (D=3), and 2 expert networks in the task expert module (T=2). The prediction layer consists of 6 towers ( $D\times T$ ) for the different combinations of domains and tasks.

#### 3.4 Overall Performance

We compare M<sup>3</sup>oE with four lines of baselines: single task and domain, multi-task, multi-domain, and multi-domain multi-task recommendation. We calculate Relative Improvement [4, 24], denoted as **RelaImpr**, to show the improvement of M<sup>3</sup>oE over the best baseline. From Table 2, we can safely make following conclusions:

(1) Multi-task methods (b) and multi-domain methods (c) generally outperform the single-domain single-task method MLP (a). This is attributed to the former methods' effective utilization of cross-task or cross-domain knowledge transfer, which offers an advantage in terms of input information compared to MLP, which solely relies on information from a single domain or single task. However, an exception arises in domain 2 on KuaiRand-Pure, which contains a significantly larger number of samples, i.e., 90% of the dataset, compared to the other two domains. As a result, cross-domain methods have to consider the performance balance across domains and yield inferior results. (2) The multi-domain versions of the multi-task baselines, namely ShBot, PLE, and MMoE, consistently outperform their single-domain counterparts. This observation suggests that the performance gap between different domains

<span id="page-5-1"></span><sup>&</sup>lt;sup>2</sup>https://grouplens.org/datasets/movielens/

<span id="page-5-2"></span><sup>3</sup>https://kuairand.com/

<span id="page-5-3"></span><sup>&</sup>lt;sup>4</sup>https://www.kuaishou.com/cn

<span id="page-6-0"></span>Table 2: Overall experiment results with unit of ×10−<sup>2</sup> . "d1, t1" means the result on domain 1 and task 1. "(a)" means training on each domain and each task separately. "(b)" means training multi-task models on each domain respectively. "(c)" means training multi-domain models on each task respectively. "(d)" means training on multi-domain multi-task setting. Best performances are bold, and the next best are underlined. "\*" indicates the statistically significant improvements (i.e., two-sided t-test with < 0.05) over the best baseline. All the results are the average of 5 repetitive runs using different random seeds.

|        |            | AUC for Each Domain and Task |        |        |        |        |               |        |        |        |        | Overall Performance |        |               |           |        |           |
|--------|------------|------------------------------|--------|--------|--------|--------|---------------|--------|--------|--------|--------|---------------------|--------|---------------|-----------|--------|-----------|
|        | Dataset    | MovieLens                    |        |        |        |        | KuaiRand-Pure |        |        |        |        | MovieLens           |        | KuaiRand-Pure |           |        |           |
|        |            | d1, t1                       | d1, t2 | d2, t1 | d2, t2 | d3, t1 | d3, t2        | d1, t1 | d1, t2 | d2, t1 | d2, t2 | d3, t1              | d3, t2 | AUC ↑         | Logloss ↓ | AUC ↑  | Logloss ↓ |
| (a)    | MLP        | 75.02                        | 76.40  | 76.27  | 78.66  | 73.21  | 74.15         | 58.42  | 51.50  | 69.28  | 70.07  | 61.82               | 61.22  | 75.62         | 51.51     | 62.05  | 44.68     |
| (b)    | ShBot-MTL  | 75.36                        | 76.25  | 75.90  | 78.40  | 72.51  | 73.30         | 61.33  | 56.13  | 68.62  | 70.67  | 63.99               | 63.81  | 75.28         | 56.39     | 64.09  | 53.37     |
|        | PLE-MTL    | 75.42                        | 76.23  | 75.87  | 78.32  | 72.73  | 72.49         | 58.56  | 53.48  | 68.08  | 69.39  | 63.63               | 63.32  | 75.18         | 56.49     | 62.74  | 54.05     |
|        | MMOE-MTL   | 75.35                        | 76.96  | 76.58  | 78.73  | 73.19  | 74.63         | 62.03  | 60.19  | 68.81  | 70.91  | 64.08               | 64.32  | 75.90         | 51.27     | 65.05  | 44.07     |
|        | AdaTT      | 72.54                        | 74.24  | 74.12  | 73.01  | 69.75  | 70.25         | 61.92  | 60.59  | 64.86  | 67.28  | 61.40               | 60.65  | 72.32         | 58.72     | 62.78  | 44.41     |
|        | AdaTT-sp   | 72.49                        | 73.51  | 76.04  | 77.59  | 73.04  | 74.26         | 60.01  | 58.28  | 65.75  | 68.06  | 61.15               | 60.35  | 74.49         | 54.65     | 62.27  | 48.80     |
|        | ShBot-MDL  | 75.70                        | 76.50  | 76.58  | 78.10  | 73.88  | 74.76         | 64.29  | 63.36  | 65.83  | 67.22  | 65.88               | 65.47  | 75.92         | 51.53     | 65.34  | 44.04     |
|        | MMOE-MDL   | 75.77                        | 76.50  | 76.71  | 78.37  | 73.92  | 75.16         | 64.33  | 63.00  | 66.16  | 67.39  | 65.52               | 65.27  | 76.08         | 51.48     | 65.28  | 44.28     |
| (c)    | PLE-MDL    | 75.64                        | 76.19  | 76.48  | 78.14  | 73.73  | 74.85         | 64.02  | 62.68  | 66.13  | 67.21  | 65.62               | 65.33  | 75.84         | 52.00     | 65.16  | 44.14     |
|        | STAR       | 75.89                        | 76.85  | 76.87  | 78.47  | 73.68  | 74.70         | 64.12  | 62.96  | 64.91  | 66.32  | 65.73               | 64.28  | 76.08         | 51.11     | 64.72  | 44.68     |
|        | ShBot-MDMT | 75.94                        | 77.25  | 76.56  | 78.63  | 73.97  | 75.21         | 63.32  | 63.55  | 65.80  | 68.04  | 64.45               | 64.89  | 76.26         | 50.96     | 65.01  | 44.91     |
| (d)    | MMOE-MDMT  | 76.07                        | 77.27  | 76.86  | 78.67  | 73.83  | 75.41         | 63.50  | 64.30  | 65.70  | 67.88  | 65.61               | 66.10  | 76.35         | 51.02     | 65.52  | 44.20     |
|        | PLE-MDMT   | 75.83                        | 76.95  | 76.04  | 78.48  | 73.67  | 75.09         | 61.86  | 61.88  | 63.57  | 66.09  | 64.45               | 63.47  | 75.68         | 51.90     | 64.07  | 44.67     |
|        | M2M        | 74.12                        | 75.35  | 75.08  | 77.13  | 72.11  | 73.78         | 61.29  | 62.25  | 60.44  | 63.15  | 61.75               | 61.57  | 74.59         | 54.14     | 61.74  | 44.55     |
| (ours) | M3oE       | 76.61*                       | 78.13* | 77.51* | 79.33* | 74.47* | 76.09*        | 64.85* | 65.89* | 66.03  | 68.31  | 66.21*              | 66.91* | 77.02*        | 50.71*    | 66.37* | 43.76*    |
|        | RelaImpr↑  | 2.07%                        | 3.15%  | 2.38%  | 2.09%  | 2.09%  | 2.68%         | 3.63%  | 11.12% | –      | –      | 2.08%               | 9.31%  | 2.54%         | –         | 5.48%  | –         |

is smaller compared to the gap between different tasks. It also indicates that leveraging cross-domain knowledge is generally easier than leveraging cross-task knowledge. (3) Among the multi-domain methods (c), STAR demonstrates superior performance compared to the other approaches, highlighting the effectiveness of its domain representation learning. This result underscores the leading efficacy of STAR in capturing and leveraging domain-specific and common information for improved recommendation performance in multi-domain scenarios. (4) In the multi-domain multi-task methods (d), M2M demonstrates its capability to leverage dependencies among domains and tasks, as it outperforms MLP on two tasks of domain 1 on the KuaiRand-Pure dataset. However, its performance falls short compared to the MDMT versions of MTR methods, i.e., MMOE-MDMT, ShaBot-MDMT, and PLE-MDMT, highlighting its limited generality and capacity for joint modeling. Furthermore, PLE-MDMT achieves better results on certain objectives than PLE-MTL and PLE-MDL. However, all three methods exhibit a seesaw phenomenon to some extent, and none of these methods consistently achieve a leading performance, emphasizing the need for further enhancements in their model capacity and optimization approaches. (5) Our M3oE achieves consistent and almost all the best performance across all settings, which fully demonstrates its advanced ability to model multi-domain and multi-task jointly. On the MovieLens dataset, M3oE shows the dominant performance by surpassing all baselines on both average and specific objectives. This achievement represents a significant breakthrough in comparison to other approaches, including multi-domain multi-task methods. On the KuaiRand-Pure dataset, it surpasses all baselines on domain 1 and domain 3, except for domain 2 when compared with (a) and certain baselines in (b). This can be attributed to domain 2 accounts for 90% of the samples, allowing methods solely trained on domain 2 to disregard balanced information sharing with other

domains. However, our M3oE's training process does not solely focus on domain 2 but aims to maintain a balanced performance across different domains, leading to remarkable improvements in domain 1 and domain 3, e.g., 11.12% improvement on domain 1 task 2 over the second best baseline. Despite the highly imbalanced distribution of domain samples, our M3oE performs closely to the multi-task methods trained on a single domain. It is worth noting that M3oE achieves the best average AUC, i.e., 5.48% relative improvement over the best baseline, which represents the globally optimal solution. This highlights the effectiveness of our M3oE in achieving high-quality predictions across all domains and tasks simultaneously.

Hence, our M3oE well-addresses the multi-domain multi-task recommendation. Its disentangled common knowledge, domain-aspect user preference, and task-aspect user preference help understand user personalization comprehensively. Besides, the effective twolevel fusion mechanism with adaptive weights precisely controls each factor's contribution to the specific objective. Its self-adaptive optimization of fusion weights empowers the generality without extra human effort.

# 3.5 Visualization

To provide an overview of the effectiveness of M3oE in multidomain multi-task learning, we visualize the disentangled embeddings and fused embeddings. Figure [3](#page-7-0) (a) and (b) show the embeddings learned by the domain expert module on MovieLens and KuaiRand-Pure, respectively, so do (c) and (d) by the task expert module and (e) and (f) by the multi-view expert learning layer. Both domain expert modules yield fused domain embeddings with a similar distribution with domain embedding (domain 1 in (a) and domain 2 in (b)), which means the domain-specific expert plays the dominant role in the knowledge fusion. In terms of the task expert module, domain 2 of KuaiRand-Pure attains fused embedding with

<span id="page-7-0"></span>![](_page_7_Figure_2.jpeg)

Figure 3: T-SNE results on MovieLens domain 1 task 1 (left column) and KuaiRand-Pure domain 2 task 1 (right column).

<span id="page-7-1"></span>Table 3: Components analysis with averaged AUC on all domains and tasks. Results are the average of 5 individual runs.

| Dataset                | MovieLens | KuaiRand-Pure |
|------------------------|-----------|---------------|
| w/o AutoML             | 76.37     | 65.41         |
| Concat modules         | 76.89     | 66.08         |
| Fully gated modules    | 76.92     | 65.87         |
| w/o domain module      | 76.89     | 65.86         |
| w/o task module        | 76.85     | 65.92         |
| w/o domain&task module | 76.80     | 65.80         |
| M <sup>3</sup> oE      | 77.02     | 66.37         |

optimal  $\beta_t$  independent of its components. This indicates different task experts hold different views on the same domain embedding, and the model attains a balanced distribution among multiple experts, which accords with our expectations. Similar phenomena occur on the fused modules embedding of both datasets, where none of the module outputs can replace the fused embedding.

# 3.6 Ablation Study

To further investigate the effectiveness of  $M^3$ oE, we conduct ablation studies by removing key components individually. Specifically, we present the results of the following variants:

- w/o AutoML. To approximate the manually designed fusion weights, we set the two-level fusion weights  $\alpha_d$ ,  $\alpha_t$ ,  $\beta_d$ , and  $\beta_t$  as constraints by fixing them at 0.5.
- Concat modules. We concatenate the three modules outputs  $S(h_d)$ ,  $D(h_d)$ , and  $T(h_d)$ , rather than adding them as  $M^3$ oE.
- Fully gated modules. In order to explore an alternative fusion mechanism, we evaluate the performance of fully gating the outputs of all modules instead of controlling them with weights.
- w/o domain module. We omit the domain expert module.
- w/o task module. We omit the task expert module.
- w/o domain&task module. We omit both domain and task expert modules.

Based on the results presented in Table 3, several conclusions can be drawn: (1) The adaptive fusion weights optimized by AutoML significantly improve performance, demonstrating their contribution to model flexibility and generality. (2) Concatenating the modules' outputs together yields inferior results compared to directly adding them. The latter approach maintains the modules' outputs in the same embedding space, enabling effective information transfer across domains and tasks. (3) The gated module fusion achieves moderate results. We claim that disentangling these modules into different factors and employing an effective fusion mechanism enhances multi-domain multi-task recommendation. It is more effective than fusing by a unified gate. (4) We observe that the importance of domain modules and task modules differs in the two datasets, but removing these two modules clearly hurts performance. Therefore, both modules contribute to performance, and different data depend on both information components in different ways, which emphasizes the importance of our adaptive solution.

#### 3.7 Hyper-Parameter Analysis

In this subsection, we test several key hyper-parameters of  $M^3$ oE to explore their impacts on the performance. Specifically, we tune the learning rate of model parameters from  $\{1e-4, 3e-4, 1e-3, 3e-3, 1e-2\}$ , and present in Figure 4 (a). From the results, we can observe that a relatively larger learning rate prompts the performance in general. Different datasets achieve the best results with different learning rates, with MovieLens peaking at 1e-2 and KuaiRand-Pure peaking at 3e-3.

Besides, we test the number of expert networks N in the shared expert module S from  $\{1, 2, 3, 4, 5, 6\}$ , and illustrates in Figure 4 (b). According to the comparison, we can find that the two datasets reach the optimal result with different N, *i.e.*, N=1 for MovieLens and N=4 for KuaiRand, which shows different multi-domain multitask scenarios prefers the balance between common knowledge, domain-aspect knowledge, and task-aspect knowledge in different degrees. Besides, more expert networks than the optimal value may cause overfitting problems, and performance decreases as N grows.

<span id="page-8-0"></span>![](_page_8_Figure_2.jpeg)

Figure 4: Hyper-parameter analysis results.

#### 4 RELATED WORKS

This section briefly reviews the representative methods in multidomain recommendation, multi-task recommendation, and multidomain multi-task recommendation.

#### 4.1 Multi-Domain Recommendation

MDR methods have gained research attention for their ability to transfer cross-domain knowledge in diverse business scenarios.[3, 10, 13, 25]. Generally, existing MDR methods can be categorized into three groups: cross-domain mapping [14, 34], MTR-based [21, 25, 27], and dynamic weight-based [36]. As a typical crossdomain mapping-based method, DDTCDR [14] utilizes orthogonal mapping for multi-domain embedding transfer so that the auxiliary information can be incorporated. The key idea of MTR-based methods is to treat the multi-domain task as the multiple objectives optimization problem [21, 26]. For example, CausalInt [27] utilizes disentangled representation learning to capture scenario-invariant information, and incorporates a TransNet to use information from different scenarios. Dynamic weight methods aim to generalize domain-specific knowledge by adjusting the weights of domainspecific modules, e.g., AdaSparse [36] introduces adaptively sparse structures with domain-aware neuron-level weighting factors to identify and prune redundant neurons. Unlike MDR methods, our M<sup>3</sup>oE leverages the advantages of multi-objective optimization to simultaneously capture multi-domain and multi-task aspects.

# 4.2 Multi-Task Recommendation

MDR frameworks [19, 21, 22, 26] aim to improve the performance of individual tasks by utilizing common dependencies among tasks. They can be categorized into architecture-based [12, 21] and optimizationbased approaches [22, 35]. SharedBottom [1] is one of the earliest architecture-based methods, which shares the hard structure across tasks. Expert network sharing is another popular paradigm, e.g., MMoE [21] uses fully shared experts to capture information across multiple tasks. PLE [26] incorporates task-specific experts to handle task-specific information, and AdaTT [12] introduces a branch for linearly combining task-specific experts to model tasks jointly. For the optimization-based methods, they primarily address specific challenges, including negative transfer and multi-objective trade-off [29, 31, 33]. To tackle the negative transfer problem, MetaBalance [6] introduces a relaxation factor to balance the gradient magnitude proximity between auxiliary and target tasks. AdaTask [35] addresses gradient conflicts among training objectives by employing a task-specific optimization strategy. ForeMerge [22] proposes balancing the auxiliary task loss weights to alleviate the negative

transfer among tasks. In terms of multi-objective trade-off issues, MTA-F [31] explores the trade-off between group fairness and accuracy in multi-task learning to capture the multi-dimensional Pareto frontier. Wang *et al.* [33] address the trade-off between minimizing task training conflicts and enhancing model's multi-task generalization ability. Existing MTR methods such as MMoE [21] and PLE [26] can be the special cases of our method, which considers both task-specific and common expert networks.

#### 4.3 Multi-Domain Multi-Task Recommendation

Recently, there have been efforts towards multi-domain and multitask recommendations [2, 11, 38], aiming to leverage the benefits of both domain and task simultaneously. M2M [38] employs a meta-unit to capture inter-scenario correlations and scale to new scenarios. It incorporates a meta-attention module for diverse interscenario correlations and a meta-tower module to enhance scenariospecific features. PEPNet [2] leverages personalized prior information and dynamically scales units using gate mechanisms. This enables personalized selection and modification for users across multiple domains and tasks. M3Rec [11] designs a Meta-Item-Embedding Generator (MIEG) and a User-Preference Transformer (UPT) to unify the representation of users and items, which can capture noni.i.d behaviors across scenarios. Existing methods usually rely on complex architectures, which leads to limited generality. In contrast, our framework employs disentangled modules and a self-adaptive two-level fusion mechanism, and optimizes based on data and task characteristics, which is a promising and versatile solution.

# 5 CONCLUSION

Multi-domain recommendation and multi-task recommendation have made great progress in assisting knowledge fusion and improving recommendation accuracy. However, cross-domain and cross-task knowledge transfer is more important in practical recommendation and cannot be handled well by existing methods. In this paper, we first identify the long-neglected cross-domain and cross-task seesaw problem. Then, we propose a framework M³ oE to adaptively address MDMT recommendation for the first time. It aims to solve general cross-domain and cross-task knowledge transfer through effective disentanglement and fusion mechanisms. Extensive experiments on two public datasets demonstrate its outstanding efficacy in solving the MDMT seesaw problem.

### **ACKNOWLEDGMENTS**

This research was partially supported by Kuaishou, Research Impact Fund (No.R1015-23), APRC - CityU New Research Initiatives (No.9610565, Start-up Grant for New Faculty of CityU), CityU - HKIDS Early Career Research Grant (No.9360163), Hong Kong ITC Innovation and Technology Fund Midstream Research Programme for Universities Project (No.ITS/034/22MS), Hong Kong Environmental and Conservation Fund (No. 88/2022), and SIRG - CityU Strategic Interdisciplinary Research Grant (No.7020046, No.7020074). Hongwei Zhao is funded by the Provincial Science and Technology Innovation Special Fund Project of Jilin Province, grant number 20190302026GX, Natural Science Foundation of Jilin Province, grant number 20200201037JC, and the Fundamental Research Funds for the Central Universities, JLU.

# <span id="page-9-0"></span>REFERENCES

- <span id="page-9-10"></span>[1] Rich Caruana. 1997. Multitask learning. Machine learning 28 (1997), 41–75.
- <span id="page-9-23"></span>[2] Jianxin Chang, Chenbin Zhang, Yiqun Hui, Dewei Leng, Yanan Niu, Yang Song, and Kun Gai. 2023. Pepnet: Parameter and embedding personalized network for infusing with personalized prior information. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 3795–3804.
- <span id="page-9-28"></span>[3] Mark Dredze, Alex Kulesza, and Koby Crammer. 2010. Multi-domain learning by confidence-weighted parameter combination. Machine Learning 79 (2010), 123–149.
- <span id="page-9-26"></span>[4] Jingtong Gao, Bo Chen, Menghui Zhu, Xiangyu Zhao, Xiaopeng Li, Yuhao Wang, Yichao Wang, Huifeng Guo, and Ruiming Tang. 2023. Scenario-Aware Hierarchical Dynamic Network for Multi-Scenario Recommendation. arXiv preprint arXiv:2309.02061 (2023).
- <span id="page-9-5"></span>[5] Jingtong Gao, Xiangyu Zhao, Bo Chen, Fan Yan, Huifeng Guo, and Ruiming Tang. 2023. AutoTransfer: Instance Transfer for Cross-Domain Recommendations. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. 1478–1487.
- <span id="page-9-36"></span>[6] Yun He, Xue Feng, Cheng Cheng, Geng Ji, Yunsong Guo, and James Caverlee. 2022. MetaBalance: Improving Multi-Task Recommendations via Adapting Gradient Magnitudes of Auxiliary Tasks. 2205–2215.
- <span id="page-9-1"></span>[7] Folasade Olubusola Isinkaye, Yetunde O Folajimi, and Bolande Adefowoke Ojokoh. 2015. Recommendation systems: Principles, methods and evaluation. Egyptian informatics journal 16, 3 (2015), 261–273.
- <span id="page-9-2"></span>[8] Pengyue Jia, Yejing Wang, Zhaocheng Du, Xiangyu Zhao, Yichao Wang, Bo Chen, Wanyu Wang, Huifeng Guo, and Ruiming Tang. 2024. ERASE: Benchmarking Feature Selection Methods for Deep Recommender Systems. arXiv preprint arXiv:2403.12660 (2024).
- <span id="page-9-6"></span>[9] Pengyue Jia, Yichao Wang, Shanru Lin, Xiaopeng Li, Xiangyu Zhao, Huifeng Guo, and Ruiming Tang. 2024. D3: A Methodological Exploration of Domain Division, Modeling, and Balance in Multi-Domain Recommendations. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 8553–8561.
- <span id="page-9-29"></span>[10] Mahesh Joshi, Mark Dredze, William Cohen, and Carolyn Rose. 2012. Multidomain learning: when do domains matter?. In Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning. 1302–1312.
- <span id="page-9-37"></span>[11] Zerong Lan, Yingyi Zhang, and Xianneng Li. 2023. M3REC: A Meta-based Multiscenario Multi-task Recommendation Framework. In Proceedings of the 17th ACM Conference on Recommender Systems. 771–776.
- <span id="page-9-21"></span>[12] Danwei Li, Zhengyu Zhang, Siyang Yuan, Mingze Gao, Weilin Zhang, Chaofei Yang, Xi Liu, and Jiyan Yang. 2023. AdaTT: Adaptive Task-to-Task Fusion Network for Multitask Learning in Recommendations. 4370–4379.
- <span id="page-9-30"></span>[13] Pengcheng Li, Runze Li, Qing Da, An-Xiang Zeng, and Lijun Zhang. 2020. Improving multi-scenario learning to rank in e-commerce by exploiting task relationships in the label space. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management. 2605–2612.
- <span id="page-9-7"></span>[14] Pan Li and Alexander Tuzhilin. 2020. Ddtcdr: Deep dual transfer cross domain recommendation. In Proceedings of the 13th International Conference on Web Search and Data Mining. 331–339.
- [15] Xinhang Li, Zhaopeng Qiu, Xiangyu Zhao, Zihao Wang, Yong Zhang, Chunxiao Xing, and Xian Wu. 2022. Gromov-wasserstein guided representation learning for cross-domain recommendation. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 1199–1208.
- <span id="page-9-8"></span>[16] Xiaopeng Li, Fan Yan, Xiangyu Zhao, Yichao Wang, Bo Chen, Huifeng Guo, and Ruiming Tang. 2023. Hamur: Hyper adapter for multi-domain recommendation. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management. 1268–1277.
- <span id="page-9-16"></span>[17] Dugang Liu, Chaohua Yang, Xing Tang, Yejing Wang, Fuyuan Lyu, Weihong Luo, Xiuqiang He, Zhong Ming, and Xiangyu Zhao. 2024. MultiFS: Automated Multi-Scenario Feature Selection in Deep Recommender Systems. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining. 434–442.
- <span id="page-9-25"></span>[18] Hanxiao Liu, Karen Simonyan, and Yiming Yang. 2018. Darts: Differentiable architecture search. arXiv preprint arXiv:1806.09055 (2018).
- <span id="page-9-32"></span>[19] Junning Liu, Xinjian Li, Bo An, Zijie Xia, and Xu Wang. 2022. Multi-Faceted Hierarchical Multi-Task Learning for Recommender Systems. 3332–3341.
- <span id="page-9-11"></span>[20] Ziru Liu, Jiejie Tian, Qingpeng Cai, Xiangyu Zhao, Jingtong Gao, Shuchang Liu, Dayou Chen, Tonghao He, Dong Zheng, Peng Jiang, et al. 2023. Multi-task recommendations with reinforcement learning. In Proceedings of the ACM Web Conference 2023. 1273–1282.

- <span id="page-9-12"></span>[21] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H Chi. 2018. Modeling task relationships in multi-task learning with multi-gate mixture-ofexperts. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 1930–1939.
- <span id="page-9-15"></span>[22] Aakarsh Malhotra, Mayank Vatsa, and Richa Singh. 2023. Dropped Scheduled Task: Mitigating Negative Transfer in Multi-task Learning using Dynamic Task Dropping. Transactions on Machine Learning Research (2023).
- <span id="page-9-3"></span>[23] Lalita Sharma and Anju Gera. 2013. A survey of recommendation system: Research challenges. International Journal of Engineering Trends and Technology (IJETT) 4, 5 (2013), 1989–1992.
- <span id="page-9-27"></span>[24] Qijie Shen, Wanjie Tao, Jing Zhang, Hong Wen, Zulong Chen, and Quan Lu. 2021. Sar-net: a scenario-aware ranking network for personalized fair recommendation in hundreds of travel scenarios. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 4094–4103.
- <span id="page-9-9"></span>[25] Xiang-Rong Sheng, Liqin Zhao, Guorui Zhou, Xinyao Ding, Binding Dai, Qiang Luo, Siran Yang, Jingshan Lv, Chi Zhang, Hongbo Deng, et al. 2021. One model to serve all: Star topology adaptive recommender for multi-domain ctr prediction. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 4104–4113.
- <span id="page-9-13"></span>[26] Hongyan Tang, Junning Liu, Ming Zhao, and Xudong Gong. 2020. Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations. In Proceedings of the 14th ACM Conference on Recommender Systems. 269–278.
- <span id="page-9-17"></span>[27] Yichao Wang, Huifeng Guo, Bo Chen, Weiwen Liu, Zhirong Liu, Qi Zhang, Zhicheng He, Hongkun Zheng, Weiwei Yao, Muyu Zhang, et al. 2022. Causalint: Causal inspired intervention for multi-scenario recommendation. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 4090–4099.
- <span id="page-9-4"></span>[28] Yuhao Wang, Ha Tsz Lam, Yi Wong, Ziru Liu, Xiangyu Zhao, Yichao Wang, Bo Chen, Huifeng Guo, and Ruiming Tang. 2023. Multi-task deep recommender systems: A survey. arXiv preprint arXiv:2302.03525 (2023).
- <span id="page-9-33"></span>[29] Yuhao Wang, Ha Tsz Lam, Yi Wong, Ziru Liu, Xiangyu Zhao, Yichao Wang, Bo Chen, Huifeng Guo, and Ruiming Tang. 2023. Multi-Task Deep Recommender Systems: A Survey. arXiv preprint arXiv:2302.03525 (2023).
- <span id="page-9-18"></span>[30] Yuhao Wang, Ziru Liu, Yichao Wang, Xiangyu Zhao, Bo Chen, Huifeng Guo, and Ruiming Tang. 2024. Diff-MSR: A Diffusion Model Enhanced Paradigm for Cold-Start Multi-Scenario Recommendation. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining. 779–787.
- <span id="page-9-34"></span>[31] Yuyan Wang, Xuezhi Wang, Alex Beutel, Flavien Prost, Jilin Chen, and Ed H Chi. 2021. Understanding and improving fairness-accuracy trade-offs in multitask learning. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 1748–1757.
- <span id="page-9-19"></span>[32] Yuhao Wang, Xiangyu Zhao, Bo Chen, Qidong Liu, Huifeng Guo, Huanshuo Liu, Yichao Wang, Rui Zhang, and Ruiming Tang. 2023. PLATE: A Prompt-Enhanced Paradigm for Multi-Scenario Recommendations. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. 1498–1507.
- <span id="page-9-35"></span>[33] Yuyan Wang, Zhe Zhao, Bo Dai, Christopher Fifty, Dong Lin, Lichan Hong, Li Wei, and Ed H Chi. 2022. Can Small Heads Help? Understanding and Improving Multi-Task Generalization. In Proceedings of the ACM Web Conference 2022. 3009– 3019.
- <span id="page-9-31"></span>[34] Huan Yan, Xiangning Chen, Chen Gao, Yong Li, and Depeng Jin. 2019. Deepapf: Deep attentive probabilistic factorization for multi-site video recommendation. TC 2, 130 (2019), 17–883.
- <span id="page-9-22"></span>[35] Enneng Yang, Junwei Pan, Ximei Wang, Haibin Yu, Li Shen, Xihua Chen, Lei Xiao, Jie Jiang, and Guibing Guo. 2023. AdaTask: a task-aware adaptive learning rate approach to multi-task learning (AAAI'23/IAAI'23/EAAI'23). 9 pages.
- <span id="page-9-20"></span>[36] Xuanhua Yang, Xiaoyu Peng, Penghui Wei, Shaoguo Liu, Liang Wang, and Bo Zheng. 2022. AdaSparse: Learning Adaptively Sparse Structures for Multi-Domain Click-Through Rate Prediction. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 4635–4639.
- <span id="page-9-14"></span>[37] Tianzi Zang, Yanmin Zhu, Haobing Liu, Ruohan Zhang, and Jiadi Yu. 2022. A survey on cross-domain recommendation: taxonomies, methods, and future directions. ACM Transactions on Information Systems 41, 2 (2022), 1–39.
- <span id="page-9-24"></span>[38] Qianqian Zhang, Xinru Liao, Quan Liu, Jian Xu, and Bo Zheng. 2022. Leaving no one behind: A multi-scenario multi-task meta learning approach for advertiser modeling. In Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining. 1368–1376.