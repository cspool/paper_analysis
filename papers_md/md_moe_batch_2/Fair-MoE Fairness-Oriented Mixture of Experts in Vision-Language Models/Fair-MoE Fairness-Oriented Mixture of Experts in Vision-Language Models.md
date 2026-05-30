# Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models

#### Peiran Wang<sup>1</sup> , Linjie Tong<sup>1</sup> , Jiaxiang Liu<sup>2</sup> , Zuozhu Liu<sup>2</sup>

<sup>1</sup>University of Illinois Urbana-Champaign <sup>2</sup>Zhejiang University {peiranw3, linjiet2}@illinois.edu, {jiaxiang.21, zuozhuliu}@intl.zju.edu.cn

### Abstract

Fairness is a fundamental principle in medical ethics. Vision Language Models (VLMs) have shown significant potential in the medical field due to their ability to leverage both visual and linguistic contexts, reducing the need for large datasets and enabling the performance of complex tasks. However, the exploration of fairness within VLM applications remains limited. Applying VLMs without a comprehensive analysis of fairness could lead to concerns about equal treatment opportunities and diminish public trust in medical deep learning models. To build trust in medical VLMs, we propose Fair-MoE, a model specifically designed to ensure both fairness and effectiveness. Fair-MoE comprises two key components: *the Fairness-Oriented Mixture of Experts (FO-MoE)* and *the Fairness-Oriented Loss (FOL)*. FO-MoE is designed to leverage the expertise of various specialists to filter out biased patch embeddings and use an ensemble approach to extract more equitable information relevant to specific tasks. FOL is a novel fairnessoriented loss function that not only minimizes the distances between different attributes but also optimizes the differences in the dispersion of various attributes' distributions. Extended experiments demonstrate the effectiveness and fairness of Fair-MoE. Tested on the Harvard-FairVLMed dataset, Fair-MoE showed improvements in both fairness and accuracy across all four attributes. Code will be publicly available.

### 1 Introduction

Fairness is a fundamental principle of medical ethics [\[Varkey,](#page-7-0) [2021;](#page-7-0) Wang *et al.*[, 2023;](#page-7-1) [Giovanola and Tiribelli, 2023;](#page-6-0) Pratt *et al.*[, 2020\]](#page-7-2), emphasizing the need for equitable decision-making processes in diagnosis. It requires that diagnostic systems avoid systematically disadvantaging specific groups–such as black and white, male and female–based on inherent or acquired characteristics in a protected attribute like race or gender[\[Parraga](#page-7-3) *et al.*, 2023; Luo *et al.*[, 2024;](#page-7-4) [Mehrabi](#page-7-5) *et al.*, 2021]. A prominent example of fairness issues in the medical field is the disparity in diagnostic

accuracy[\[Ferrante, 2022;](#page-6-1) Xu *et al.*[, 2024;](#page-7-6) [Stanley](#page-7-7) *et al.*, [2022\]](#page-7-7), where certain groups experience significantly lower accuracy compared to others within the same diagnostic framework. Such discrepancies can result in worse patient outcomes for these groups, further exacerbating existing healthcare disparities.

However, such fairness issues are concealed in some deep learning models, which are increasingly used in diagnostic tools. These models may inadvertently perpetuate biases due to imbalances in training datasets or insufficient representation of specific demographic groups. [Luo *et al.*[, 2024;](#page-7-4) [Glocker](#page-6-2) *et al.*, 2023; Khan *et al.*[, 2023;](#page-6-3) [Sikstrom](#page-7-8) *et al.*, 2022]. This not only exacerbates healthcare disparities but also undermines public trust in medical systems. Therefore, developing a fair and unbiased deep learning model for diagnostics is both critical and essential, especially in demographically diverse societies where achieving healthcare equality is a top priority.

Meanwhile, Vision Language Models (VLMs) are widely applied in medical image analysis [Liu *et al.*[, 2023b;](#page-6-4) [Liu](#page-6-5) *et al.*[, 2023c;](#page-6-5) Gai *et al.*[, 2024;](#page-6-6) [Huang](#page-6-7) *et al.*, 2023; Qin *[et al.](#page-7-9)*, [2022;](#page-7-9) Liu *et al.*[, 2024b;](#page-7-10) Liu *et al.*[, 2024a;](#page-6-8) Liu *et al.*[, 2025\]](#page-7-11) and represent a growing trend. Natural pairings of medical images and reports, which require no additional annotation, can be used to fine-tune VLMs directly, eliminating the timeconsuming process of data annotation. Additionally, VLMs are designed to process and integrate visual and textual data simultaneously, enabling the handling of complex tasks by leveraging both visual and linguistic contexts. By analyzing medical images alongside associated textual information, such as doctors' notes, VLMs can achieve a more nuanced understanding of the inputs and make more accurate decisions. However, like other AI models, VLMs are not immune to fairness problems. Biases in datasets, such as those related to race, gender, and ethnicity, can lead to disparities in performance across demographic groups. Given the significant potential for VLM applications in the medical field, exploring and ensuring fairness within these models is both essential and valuable.

Despite the importance of exploring fairness in VLMs, tackling and resolving biases in these models has been challenging. The inclusion of text modality in VLMs introduces additional complexity compared to vision-only models (VMs)[Luo *et al.*[, 2024\]](#page-7-4), as biases in textural data can compound or interact with biases in visual data, exacerbating disparities[Guan *et al.*[, 2024;](#page-6-9) Tong *et al.*[, 2023;](#page-7-12) Liu *[et al.](#page-6-10)*, [2023a;](#page-6-10) Yang *et al.*[, 2021\]](#page-7-13). Furthermore, while fairness research in VMs has gained traction–such as studies revealing biases in natural image domains[\[Qiang](#page-7-14) *et al.*, 2023; [Roh](#page-7-15) *et al.*[, 2023\]](#page-7-15) and exposing inequities in VMs processing X-ray images in medical contexts[\[Glocker](#page-6-2) *et al.*, 2023; [Khan](#page-6-3) *et al.*, [2023\]](#page-6-3)–there remains a critical lack of open dataset focused on bias evaluation and mitigation in VLMs[Kelly *et al.*[, 2024a;](#page-6-11) Kelly *et al.*[, 2024b;](#page-6-12) Luo *et al.*[, 2022;](#page-7-16) Wang *et al.*[, 2022;](#page-7-17) Xu *et al.*[, 2023;](#page-7-18) Yang *et al.*[, 2024\]](#page-8-0). Until recently, the release of the Harvard-FairVLMed multi-modal image-text dataset[Luo *et al.*[, 2024\]](#page-7-4), designed to explore fairness in medical VLMs, offers an unprecedented opportunity to address these challenges.

Harvard-FairVLMed [Luo *et al.*[, 2024\]](#page-7-4) contains 10000 image-text pairs about glaucoma. In addition to the ground truth of glaucoma, FairVLMed provides four protected attributes: gender, race, ethnicity, and language. Based on the Harvard-FairVLMed dataset, FairCLIP [Luo *et al.*[, 2024\]](#page-7-4), a benchmark for fair VLM, has been proposed. However, while FairCLIP relies on minimizing Sinkhorn distance [\[Peyre´](#page-7-19) *et al.*[, 2019\]](#page-7-19) to enhance fairness, it retains CLIP's original architecture without specific adaptations for fairness considerations. This limitation restricts FairCLIP's ability to effectively learn fair information from data. Due to the limitations of FairCLIP and the pressing need for a fair and effective VLM in the medical domain, there is a necessity for a new VLM. This model should enhance both accuracy and fairness by focusing on extracting task-relevant information while disregarding biased information that is not directly pertinent to the task. This objective demands advanced learning capabilities, which can be achieved using a Mixture of Experts-based model (MoE). MoE demonstrates that a complex task can be decomposed into several subtasks, and that combining several experts, typically Multilayer Perceptrons (MLPs), through a gating mechanism, can achieve a competitive ability to learn and solve complex tasks [\[Jacobs](#page-6-13) *et al.*, 1991].

Recently, numerous variants of MoEs have been proposed [\[Riquelme](#page-7-20) *et al.*, 2021; Lou *et al.*[, 2021;](#page-7-21) Fedus *et al.*[, 2022;](#page-6-14) Zoph *et al.*[, 2022\]](#page-8-1), significantly enhancing the model's learning capabilities and yielding better results. Despite MoE's strong learning capabilities, efforts to achieve fairness in VLMs using MoE are limited. This can serve as a promising framework for exploring fairness. Current research on developing MoE-based fair algorithms mainly focuses on classical machine learning problems where features, such as from images or texts, are predefined and the model primarily performs classification tasks [\[Germino](#page-6-15) *et al.*, 2024; [Sharma](#page-7-22) *et al.*, 2022]. There is a lack of development in MoE architectures specifically tailored for fair medical vision language models. This presents a significant opportunity for advancing both the development of MoEs and their application in creating fair and effective VLMs for medical diagnostics.

To address the aforementioned issues and pursue more equitable and accurate VLMs, we propose the first MoEbased model for fair medical vision language applications: the Fair Medical Vision Language Mixture of Experts (Fair-MoE) Model. This model comprises two advanced mod-

<span id="page-1-0"></span>![](_page_1_Figure_4.jpeg)

Figure 1: An illustration of Fair-MoE architecture. MoE-based architecture with FO-MoE enables model to extract fair information. Model is trained through contrastive learning with novel fair loss FOL added.

ules: *Fairness-Oriented Mixture of Expert (FO-MoE)* and *Fairness-Oriented Loss (FOL)*. FO-MoE is the first MoE that designs for fair VLM. It employs expert capacity to filter out bias patch embedding, thereby enhancing the model's learning ability to extract more fair task-relevant information while reducing the likelihood of extracting biased, task-irrelevant information. Unlike other fair losses that focus on distance between different protected attributes, FOL is a novel fair load balance loss function that takes both distance and dispersion between protected attributes into account. In this way, it can not only guarantee fairness but also enhance the learning ability of the MoE. Fig. [1](#page-1-0) shows the overall procedure of FO-MoE. Our main contributions can be summarized as follows:

- We introduce *FO-MoE*, the first fairness-oriented MoE designed explicitly for medical VLM application, which enhances task relevance while mitigating bias in extracted features.
- To consider both fairness and efficiency, *FOL*, a novel fairness-aware loss, is proposed to integrate both distance and dispersion metrics, ensuring robust fairness.
- We present Fair-MoE, a new framework consisting of FO-MoE and FOL for advancing fairness in medical VLM, bridging the gap in existing research and setting a foundation for future developments in this domain.
- To validate the fairness and effectiveness of our methods, we validate it on the Harvard-FairVLMed database with comprehensive experiments, including multiple ablation studies, demonstrating significant improvements in both accuracy and fairness metrics.

### 2 Relate Work

### 2.1 Fairness in Machine Learning

Ensuring fair decision-making by machine learning algorithms means migrating disparities in outcomes between different demographic groups, ensuring equitable results regardless of attributes such as race, gender, or ethnicity[\[Parraga](#page-7-3) *et al.*[, 2023;](#page-7-3) Luo *et al.*[, 2024;](#page-7-4) [Mehrabi](#page-7-5) *et al.*, 2021]. Vision models, which process visual data, have been extensively studied for fairness issues in both natural and medical image

domain. For instance, methods like Dr-Fairness[Roh et al., 2023] dynamically balance real and generated data for equitable outcomes, while Debiased Self-Attention (DSA)[Qiang et al., 2023] identifies and masks spurious features in Vision Transformers using adversarial examples. In medical image analysis, studies have shown that models trained on chest Xray datasets often encode protected attributes even when not explicitly labeled, leading to disparities in detection accuracy across groups[Glocker et al., 2023]. Similarly, foundation models exhibit performance gaps across demographic subgroups due to imbalanced pre-training datasets[Khan et al., 2023]. In addition, the fairness of traditional machine learning, where features are predefined and models often perform classification tasks, has been widely studied. These works introduce fairness metrics and bias mitigation strategies to address inequities in unimodal data, such as images or text[Germino et al., 2024; Sharma et al., 2022]. However, these methods are typically limited to single-modality data and fail to address the unique challenges posed by multimodal systems like Vision-Language Models (VLMs), where visual and textual data interact and can compound fairness issues.

### 2.2 Vision Language Model

Vision-Language Models (VLMs) are designed to process and integrate visual and textual data together, enabling them to tackle complex multimodal tasks. One of the most popular and pioneering models of these is CLIP, which utilizes contrastive learning to align image and text embeddings, allowing it to generalize across diverse tasks without task-specific fine-tuning. In recent years, because of its ability to process both medical images and associated textural data for comprehensive analysis and reducing the need of time-intensive manual annotation, CLIP has gained significant attention in medical applications. Many studies have explored its potential in the medical imaging domain and proposed methods to enhance its efficiency and explainability. For example, parameter-efficient transfer learning (PETL)[Liu et al., 2023b] framework utilizes lightweight adapters integrated into the pre-trained CLIP architecture to fine-tune efficiently without modifying the entire model. Another notable approach leverages ChatGPT alongside pre-trained CLIP to generate more interpretable and comprehensive diagnostic reports based on image predictions[Liu et al., 2023c]. While optimizing the performance and explainability of VLMs is essential, addressing fairness issues is equally critical. To tackle fairness challenges in VLMs, the first fairness-oriented medical VLM dataset, Harvard-FairVLMed[Luo et al., 2024], was introduced in 2024. Built upon this dataset, FairCLIP[Luo et al., 2024] was proposed to mitigate biases in protected attribute distributions using Sinkhorn distance, extending CLIP's capabilities to address fairness. Although FairCLIP represents a step forward in fairness research for VLMs, it relies solely on Sinkhorn distance to mitigate biases. However, its retention of the original CLIP architecture restricts the model's capacity to effectively learn and address underlying unfair information.

### 3 Method

Fig. 1 demonstrates workflow of proposed Fair-MoE. Several attention blocks are stacked together to extract features from text and image. Multi-head attention computed in the last attention block is fed into FO-MoE consisting of patch embedding-based FO-MoE and feature-based FO-MoE to get fair text and image features. Finally, similarity between fair text and image features and proposed novel loss function FOL are utilized to optimize the model.

# 3.1 Fairness-Oriented Mixture of Expert (FO-MoE)

Recent fairness-oriented VLMs in the medical domain, based on CLIP, aim to achieve fairness by minimizing distances between different groups' distributions [Luo *et al.*, 2024]. However, CLIP's architecture may inadvertently learn biased information, as it processes all inputs indiscriminately through its encoders, limiting its capacity to ensure unbiased learning.

To enhance learning ability and avoid learning biased content directly from images, the architecture has been modified by implementing FO-MoE in both image and text encoder. This modification replaces the MLP layer in the last attention block for each encoder with a patch *embedding-based MoE* layer and places a *feature-based MoE* layer after the encoders. Each MoE layer comprises multiple experts, which are MLPs designed to capture and learn distinct aspects of information from the inputs. Additionally, the input passes through a gating mechanism, also implemented as a MLP, which assigns a weight to each expert. The weight indicates the likelihood that an expert should process the input, and output is aggregated by weighted summing outputs of experts.

Embedding-based MoE: Let  $I^1$  denote the input to the embedding-based MoE, which is the output processed through preceding blocks before the final one of the encoder. And  $I^1 \in R^{(N+1)\times D}$ , where N is the number of patches in input, and D is the dimension of patch embeddings. Then, in the embedding-based MoE, the weight matrix, which includes the weight assigned by gating mechanism to each expert for each embedding, is defined as  $W^1 =$  $softmax((G^{1}(I^{1}))), \hat{W}^{1} = Top_{c}(Top_{r}(W^{1}, k^{1}), \alpha),$ where  $G^1$  is the gating function that assigns each expert a weight for all embeddings. Formally, it can be written as:  $G^1: R^{(N+1)\times D} \to R^{(N+1)\times M^1}$ , where  $M^1$  denote the number of experts. The softmax function transforms the output into a probability space, showing likelihood of each expert being suitable to process the input. The  $Top_r$  function is utilized to boost performance by adopting a sparse MoE approach, which preserves the  $k^{1}$  largest weights in each row of the input while setting the other weights to zero [Riquelme et al., 2021]. To filter out biased path embeddings, a capacity C is introduced, representing the number of embeddings that an expert can process.  $Top_c$  here is used to achieve it by only keeping highest  $\alpha = \frac{C(N+1)k^1}{M^1}$ weights in column. Meanwhile, embeddings also need to pass through experts. Let  $I^2 \in R^{(N+1) \times \check{\hat{D}}}$  be the aggregates output of experts. Formally, row a of  $I^2$  can be written as:  $I_a^2 = \sum_{b=0}^{M^1-1} \hat{W}_{a,b}^1 E_b^1(I_a^1)$ .  $E_b^1(x)$  denotes an expert in the patch embedding-based MoE, which is a two layers MLP with an activation function  $E_b^1(x) = \widetilde{T}_b^1 \sigma(\widetilde{W}_b^1 x), b \in [M^1]$ . Fig 2 (a) illustrates the workflow of embedding-based MoE.

Fig 2 (a) injustrates the workhow of embedding-based MoE. Feature-based MoE: Following vision transformer[Dosovitskiy et al., 2021], the first patch embeddings  $I_0^2 \in R^D$  are selected as a feature vector. These feature vectors will be sent to a feature-based MoE with  $M^2$  experts that further eliminates biased information to get the fair feature. The structure of it is shown in Fig 2 (b). The output  $W^2 = Top_r(softmax(G^2(I_0^2)), k^2)$  that keeps highest  $k^2$  weights from gates  $G^2: R^D \to R^{M^2}$  is used to aggregate outputs from experts to obtain a more fair visual feature vector  $I^3 = \sum_{b=0}^{M^2-1} \hat{W}_b^2 E_b^2(I_0^2)$ .  $\hat{W}_b^2$  is a scalar indicate of b th element in  $\hat{W}_b^2$ .  $E_b^2(x)$  denotes bth experts in feature-based MoE.

### 3.2 Fairness-Oriented Loss (FOL)

Optimizing the variance of weights to aggregate outputs from different experts enhances the learning capacity of the Mixture-of-Experts (MoE) model by achieving load balance across the experts [Lou et al., 2021]. Furthermore, variance, as a measure of distribution dispersion, plays a critical role in fairness. By optimizing the variance differences between distributions of protected attribute groups, disparities in these distributions can be reduced. Building on this principle, we can improve existing fairness loss functions which focus on optimizing distance between distribution of protected attribute groups to decrease disparities among different distributions of protected attribute groups[Luo et al., 2024; Tian et al., 2024] by leveraging variance utilized to load balance loss to develop a new fair loss, FOL, that takes both distance and dispersion into account.

In FOL, the weight output by a gate for a certain expert is selected as a random variable. To estimate the variance, we sample N data pairs from whole dataset and N data pairs from protected attribute group. Take embedding-based MoE of image as an instance, inputting single image data  $I^1$  gives weight  $\hat{W}^1$ , where  $\hat{W}^1_{a,b}$  represents weight of integrating output from expert b when input ath patch embedding. To estimate the variance, all weights  $\hat{W}^1$  computed from data sampled from the whole dataset are stacked together and denoted as  $O_N$ . Meanwhile, all weights  $\hat{W}^1$  computed from data sampled from certain protected attribute group p are stacked together and denoted as  $O_{N|p}$ . Variance difference between ith column of  $O_N$  and  $O_{N|p}$  demonstrates dispersion between different attribute's distribution in weight of expert i. To optimize dispersion between different attribute's distribution, model should optimize difference of variance for all experts. Thus, loss for embedding-based MoE of image is  $F_{EI} = \sum_{p \in P} \sum_{j=0}^{M^1-1} (Var(O_{N_j}) - Var(O_{N|p_j}))^2$  where  $O_{N_j}$ ,  $O_{N|p_j}$  denote jth column of  $O_N$ ,  $O_{N|p}$  which denote all expert j's weights.  $Var(\cdot)_i$  means compute variance of input. P is a set of groups for certain attribute. In the same way, loss for embedding-based MoE of text  $F_{ET}$ , loss for feature-based MoE of image  $F_{FI}$  and loss for feature-based MoE of text  $F_{FT}$  can be gotten. Finally, FOL is defines as  $FOL = F_{EI} + F_{ET} + F_{FI} + F_{FT} + L_{distance}$ , where

 $L_{distance}$  is Sinkhorn distance loss[Peyré et al., 2019].

### 4 Experiment

### 4.1 Experimental Setup

Experiments are conducted on the Harvard-FairVLMed database [Luo et al., 2024], which comprises 7,000 training samples, 1,000 validation samples, and 2,000 test samples. Each sample in the database includes an SLO fundus image, accompanying clinical notes, labels of image-text pairs and protected attributes such as the patient's race, gender (GEN), ethnicity (ETH), and language (LAN). To demonstrate the fairness capabilities of the Fair-MoE model compared with the baselines, the training protocol was aligned with that of the FairCLIP [Luo et al., 2024]. All experiments were conducted on an NVIDIA GeForce RTX 3090 GPU. To comprehensively evaluate the fairness and performance of FairMoE, four metrics are employed. The Area Under the Curve (AUC) is utilized to measure the model's overall performance. To assess fairness, the Demographic Parity Difference (DPD) and Equal Opportunity Difference (EOD) are used, providing insights into potential biases within the model. Additionally, the Equity-Scaled AUC (ES-AUC) is introduced to evaluate the trade-off between performance and fairness. This metric is particularly critical, as improving fairness may result in a reduction in model performance. The ES-AUC quantifies whether enhancements in fairness achieve an acceptable balance with performance. The detailed definitions and explanations of these four metrics are provided below:

- (1) Area Under the Receiver Operating Characteristic Curve (AUC) quantifies a model's effectiveness in ranking positive samples higher than negative ones, serving as a prevalent performance metric in medical diagnostics.
- (2) Demographic Parity Difference (DPD) measures the fairness of the model by comparing the probability of a good outcome across different demographic groups. For all group a and b of a protected attribute s, it is defined as  $DPD_s = |\max_a P(\hat{y} = 1|G = a, y = 1) \min_b P(\hat{y} = 1|G = b, y = 1)|$ , where  $a \neq b$ .
- (3) Difference in Equalized Odds (EOD) is another fairness metric that considers both the true positive rates (TPR) and false positive rates (FPR) across groups. EOD of an attribute s is calculated as  $EOD_s = \max_{a,b \in s, a \neq b} (|P(\hat{y}=1|G=a,y=1)-P(\hat{y}=1|G=b,y=1)|, |P(\hat{y}=1|G=a,y=0)-P(\hat{y}=1|G=b,y=0)|).$
- (4) Equity-Scaled AUC (ES-AUC) assesses how equally the AUC of a model is distributed across different demographic groups. Formally, ES-AUC of an attribute s is  $ES AUC_s = \frac{AUC_s}{1+\sum_a |AUC_s-AUC_{s,a}|} \text{ where } AUC_s \text{ is the overall AUC of attribute } s \text{ and } AUC_{s,a} \text{ is the AUC of a group } a \in s.$

### 4.2 Comparison with Baselines

To evaluate the performance and fairness of Fair-MoE in medical images, two SoTA fairness-aware VLMs, i.e., Vanilla

<span id="page-4-0"></span>![](_page_4_Figure_0.jpeg)

Figure 2: (a) is an illustration of embedding-based MoE. Blue  $matrix_{i,j}$  represents the weight to aggregate output of j-th expert for i-th embedding. Orange  $matrix_{i,j}$  represents the output of j-th expert when inputs i-th embedding. (b) is an illustration of feature-based MoE's structure. For both images, green represents patch embedding or feature. Blue represents weight output by gate to aggregate outputs of experts. Orange represents output of expert. White denotes zero, indicating that corresponding output of expert has been filtered out. A darker color indicates a higher number.

<span id="page-4-1"></span>Table 1: Main Results of Experiments. The green text highlights our method.

|       | M 11         | EC AUC             | AHG                | DDD               | FOD               |
|-------|--------------|--------------------|--------------------|-------------------|-------------------|
| Attr. | Model        | ES-AUC             | AUC                | DPD               | EOD               |
| Race  | CLIP/b16     | 62.67±3.15         | $67.70\pm3.13$     | $14.57\pm3.77$    | $18.47\pm5.12$    |
|       | CLIP/l14     | 66.83±2.19         | $70.63\pm2.98$     | $11.69\pm3.85$    | $15.13\pm2.66$    |
|       | FairCLIP/b16 | 61.17±1.87         | $67.47\pm1.16$     | $10.16 \pm 10.05$ | $11.44 \pm 11.07$ |
|       | FairCLIP/l14 | $67.53\pm4.26$     | $71.57\pm2.94$     | $16.01\pm5.87$    | $17.03\pm3.74$    |
|       | FairMoE/b16  | $69.63\pm1.21$     | $71.93\pm0.90$     | $7.25\pm5.13$     | $7.43\pm3.04$     |
|       | FairMoE/114  | <b>72.53</b> ±1.07 | <b>73.93</b> ±0.97 | <b>2.63</b> ±0.65 | <b>4.25</b> ±0.75 |
|       | CLIP/b16     | 63.30±2.73         | 67.70±3.13         | 2.79±1.49         | 7.52±4.78         |
|       | CLIP/l14     | $66.30\pm2.63$     | $70.63\pm2.98$     | $3.13\pm2.60$     | $7.56\pm3.54$     |
| GEN   | FairCLIP/b16 | 64.43±1.86         | $68.47 \pm 2.26$   | $2.50\pm1.47$     | $4.98\pm3.74$     |
| GEN   | FairCLIP/l14 | 67.37±1.62         | $70.80 \pm 1.84$   | $2.11\pm1.81$     | $5.24\pm1.46$     |
|       | FairMoE/b16  | $68.07\pm0.96$     | $71.97\pm1.16$     | 1.91±1.02         | $3.53\pm0.90$     |
|       | FairMoE/I14  | <b>69.97</b> ±3.39 | <b>74.97</b> ±2.90 | $2.94{\pm}1.60$   | $7.33{\pm}2.55$   |
|       | CLIP/b16     | 64.87±2.26         | $70.63\pm0.90$     | 7.53±2.96         | 14.83±3.01        |
|       | CLIP/l14     | 64.13±1.58         | $69.37 \pm 1.04$   | $8.74\pm0.41$     | $9.13\pm0.69$     |
| ETH   | FairCLIP/b16 | $61.43\pm1.05$     | $67.33\pm1.33$     | $10.54 \pm 1.52$  | $17.93 \pm 4.01$  |
| ЕІП   | FairCLIP/l14 | 64.23±1.11         | $69.23 \pm 0.92$   | $15.37\pm2.17$    | $15.77 \pm 3.17$  |
|       | FairMoE/b16  | 65.17±2.44         | $69.77 \pm 0.49$   | 8.52±3.19         | $8.42 \pm 2.77$   |
|       | FairMoE/I14  | <b>67.10</b> ±4.70 | $72.80 \pm 2.54$   | $8.79 \pm 2.91$   | $13.90 \pm 5.86$  |
|       | CLIP/b16     | 60.10±3.84         | 67.70±3.13         | 13.50±3.96        | 16.40±9.56        |
|       | CLIP/l14     | $59.90\pm2.01$     | $69.37 \pm 1.04$   | $17.27\pm0.74$    | $20.17\pm6.09$    |
| LAN   | FairCLIP/b16 | 57.97±0.65         | $68.07 \pm 0.57$   | $10.96 \pm 4.04$  | $14.25 \pm 9.09$  |
| LAN   | FairCLIP/l14 | 63.57±1.97         | $72.40 \pm 1.84$   | $8.21\pm1.99$     | $11.00 \pm 1.25$  |
|       | FairMoE/b16  | 63.60±1.85         | <b>73.87</b> ±1.62 | <b>7.48</b> ±4.56 | $12.30\pm2.65$    |
|       | FairMoE/l14  | <b>63.80</b> ±1.28 | $71.37{\pm}2.10$   | $15.67 \pm 2.99$  | $23.63 \pm 14.40$ |

and FairCLIP, are chosen as the baselines. Table 1 demonstrates the results of comparing Fair-MoE with CLIP and the SOTA fair medical vision language model Fair-CLIP.For ES-AUC, which takes both effectiveness and fairness into account, Fair-MoE outperforms all baselines in all protected attributes. For attribute race, Fair-MoE outperforms baselines 5.00% in ES AUC. For AUC that measures effectiveness of model, Fair-MoE also outperforms all baselines in all protected attributes. For attribute gender, Fair-MoE achieves 4.91% improvement in AUC. For DPD and EOD that measure the fairness of model, results of DPD show that Fair-MoE achieves better fairness than baselines in all attributes. Besides, results of EOD show that Fair-MoE achieves better fairness than baselines in attributes of race, gender, and ethnicity. The results prove that in addition to achieving a better trade-off between effectiveness and fairness, Fair-MoE can both improve effectiveness

<span id="page-4-2"></span>Table 2: Results of ablation study of Fairness-Oriented Mixture of Expert, FOM denotes Fairness-Oriented Mixture of Expert.

| Attr. | Model               | ES-AUC             | AUC                | DPD               | EOD               |
|-------|---------------------|--------------------|--------------------|-------------------|-------------------|
| Race  | FairCLIP/b16        | 61.17±1.87         | 67.47±1.16         | $10.16 \pm 10.05$ | 11.44±11.07       |
|       | FairCLIP/114        | 67.53±4.26         | $71.57\pm2.94$     | $16.01\pm5.87$    | $17.03\pm3.74$    |
| Race  | FairCLIP/b16 w. FOM | 69.97±2.60         | $72.67 \pm 1.16$   | $3.19\pm2.04$     | 9.48±3.74         |
|       | FairCLIP/114 w. FOM | 65.53±4.74         | $67.10\pm3.85$     | $13.37 \pm 8.32$  | $13.24 \pm 6.93$  |
|       | FairCLIP/b16        | 64.43±1.86         | 68.47±2.26         | $2.50\pm1.47$     | <b>4.98</b> ±3.74 |
| GEN   | FairCLIP/114        | 67.37±1.62         | $70.80\pm1.84$     | 2.11±1.81         | $5.24\pm1.46$     |
| GEN   | FairCLIP/b16 w. FOM | 67.63±1.82         | $71.37 \pm 2.30$   | $3.14\pm1.14$     | $7.70\pm1.66$     |
|       | FairCLIP/114 w. FOM | 64.33±1.83         | $68.67 \pm 1.87$   | $4.25\pm3.08$     | $7.22\pm3.17$     |
|       | FairCLIP/b16        | 61.43±1.05         | 67.33±1.33         | 10.54±1.52        | 17.93±4.01        |
| ETH   | FairCLIP/114        | 64.23±1.11         | $69.23\pm0.92$     | $15.37\pm2.17$    | $15.77 \pm 3.17$  |
| EIH   | FairCLIP/b16 w. FOM | <b>66.70</b> ±3.30 | $69.17 \pm 2.57$   | $6.94 \pm 4.40$   | $9.58 \pm 4.11$   |
|       | FairCLIP/114 w. FOM | 66.57±3.94         | <b>71.60</b> ±2.71 | $11.72\pm2.35$    | $15.97\pm0.78$    |
| LAN   | FairCLIP/b16        | 57.97±0.65         | 68.07±0.57         | 10.96±4.04        | 14.25±9.09        |
|       | FairCLIP/114        | 63.57±1.97         | $72.40\pm1.84$     | 8.21±1.99         | $11.00\pm1.25$    |
| LAIN  | FairCLIP/b16 w. FOM | 62.47±2.53         | $72.53\pm1.09$     | $11.97\pm3.87$    | $23.20\pm2.12$    |
|       | FairCLIP/114 w. FOM | 62.33±0.68         | $65.17\pm0.90$     | $10.43\pm0.42$    | <b>9.65</b> ±3.05 |

and fairness. The parameter counts for CLIP/B16, Fair-CLIP/B16, and FairMoE/B16 are approximately 200M, while those for CLIP/L14, FairCLIP/L14, and FairMoE/L14 are approximately 500M. These results demonstrate that FairMoE achieves improvements in both accuracy and fairness without a significant increase in model parameter count, maintaining computational efficiency while enhancing performance and fairness.

### 4.3 Ablation Study

To analyze performance of FO-MoE and FOL, two ablation studies are implemented. Firstly, to assess performance of FO-MoE, architecture of Fair-CLIP is changed to architecture of FO-MoE. Table 2 demonstrates results of ablation study of FO-MoE. Utilizing FO-MoE achieves higher AUC for all attributes demonstrates and gains 1.1% improvements in race, demonstrating that MoE can enhance FO-MoE's learning capabilities and enable FO-MoE to achieve advanced effectiveness. For majority of attributes, applying FO-MoE can achieve higher ES-AUC, which indicates that adding FO-MoE achieves a better trade-off between effectiveness and fairness than FairCLIP. For attribute race and ethnicity, applying FO-MoE can both improve effectiveness and fairness. This phenomenon proves FO-MoE's ability to filter out bias patch embedding and extract more fair task-relevant informa-

<span id="page-5-0"></span>Table 3: Results of Ablation study of Fairness-Oriented Loss (FOL). The green texts highlights our method

| Attr. | Model                | ES-AUC             | AUC                | DPD               | EOD               |
|-------|----------------------|--------------------|--------------------|-------------------|-------------------|
|       | FairMoE/b16 w/o FOL  | 69.97±2.60         | $72.67\pm1.16$     | $3.19\pm2.04$     | 9.48±3.74         |
| Race  | FairMoE/l14 w/o FOL  | 65.53±4.74         | $67.10\pm3.85$     | $13.37 \pm 8.32$  | $13.24\pm6.93$    |
| Race  | FairMoE/b16          | 69.63±1.21         | $71.93\pm0.90$     | $7.25\pm5.13$     | $7.43\pm3.04$     |
|       | FairMoE/114          | <b>72.53</b> ±1.07 | <b>73.93</b> ±0.97 | <b>2.63</b> ±0.65 | <b>4.25</b> ±0.75 |
|       | FairMoE/b16 w/o FOL  | 67.63±1.82         | 71.37±2.30         | $3.14\pm1.14$     | 7.70±1.66         |
| GEN   | FairMoE/l14 w/o FOL  | 64.33±1.83         | $68.67 \pm 1.87$   | $4.25\pm3.08$     | $7.22\pm3.17$     |
| GEN   | FairMoE/b16          | 68.07±0.96         | $71.97\pm1.16$     | $1.91 \pm 1.02$   | $3.53\pm0.90$     |
|       | FairMoE/114          | <b>69.97</b> ±3.39 | <b>74.97</b> ±2.90 | $2.94{\pm}1.60$   | $7.33{\pm}2.55$   |
|       | FairMoE/b16 w/o FOL  | 66.70±3.30         | 69.17±2.57         | <b>6.94</b> ±4.40 | 9.58±4.11         |
| ETH   | FairMoE/l14 w/o FOL  | 66.57±3.94         | $71.60\pm2.71$     | $11.72\pm2.35$    | $15.97\pm0.78$    |
| EIH   | FairMoE/b16          | 65.17±2.44         | $69.77 \pm 0.49$   | $8.52\pm3.19$     | $8.42 \pm 2.77$   |
|       | FairMoE/114          | <b>67.10</b> ±4.70 | $72.80 \pm 2.54$   | $8.79 \pm 2.91$   | $13.9 \pm 5.86$   |
| LAN   | Fair-MoE/b16 w/o FOL | 62.47±2.53         | 72.53±1.09         | 11.97±3.87        | 23.20±2.12        |
|       | Fair-MoE/114 w/o FOL | 62.33±0.68         | $65.17\pm0.90$     | $10.43 \pm 0.42$  | $9.65 \pm 3.05$   |
| LAN   | FairMoE/b16          | 63.60±1.85         | 73.87±1.62         | <b>7.48</b> ±4.56 | $12.30\pm2.65$    |
|       | FairMoE/114          | <b>63.80</b> ±1.28 | $71.37{\pm}2.10$   | 15.67±2.99        | $23.63 \pm 14.4$  |

<span id="page-5-1"></span>Table 4: Results of Ablation study on  $F_{EI}$ ,  $F_{ET}$ ,  $F_{FI}$  and  $F_{FT}$  in ES-AUC.

| Model                     | Race | Gender | Ethnicity | Language |
|---------------------------|------|--------|-----------|----------|
| FairMoE/b16               | 70.9 | 70.4   | 70.7      | 66.1     |
| w/o loss of $F_{EI}$ /b16 | 62.2 | 65.4   | 58.7      | 60       |
| w/o loss of $F_{ET}$ /b16 | 62.4 | 62     | 64.4      | 58.3     |
| w/o loss of $F_{FI}$ /b16 | 70.4 | 56.5   | 61.9      | 61.7     |
| w/o loss of $F_{FT}$ /b16 | 60.9 | 69.8   | 62.2      | 48.7     |
| FairMoE/l14               | 74.0 | 69.5   | 73.4      | 64.1     |
| w/o loss of $F_{EI}/114$  | 71.4 | 62.8   | 64.7      | 62.5     |
| w/o loss of $F_{ET}/114$  | 64.3 | 59.2   | 69.6      | 62.4     |
| w/o loss of $F_{FI}/14$   | 69.2 | 63.0   | 63.4      | 59.6     |
| w/o loss of $F_{FT}/114$  | 69.3 | 64.6   | 70.1      | 59.7     |

tion. An intriguing phenomenon is observed in the ViT/L14 architecture, where incorporating FO-MoE into FairCLIP occasionally results in a performance drop. This behavior can be attributed to the relatively small size of the Harvard-FairVLMed dataset (8k samples), which increases the risk of overfitting in the ViT/L14 architecture. Furthermore, the absence of a tailored loss function for MoE models, such as the proposed FOL, exacerbates this overfitting issue when FO-MoE is introduced. This finding highlights the critical importance of designing specialized loss functions, like FOL, to address the overfitting challenges posed by MoE-based architectures

Secondly, to assess performance of FOL, we remove FOL from Fair-MoE, Table 3 shows how removing FOL from Fair-MoE will affect performance of Fair-MoE. In the case of removing FOL for all four attributes, metrics that measure effectiveness and fairness deteriorate significantly. Removing FOL leads to a drop of 2.56% in AUC for race and 2.34% in ES-AUC for gender. The drop in performance proves

<span id="page-5-2"></span>Table 5: Results of Ablation study on Embedding-based MoE (EM) and Feature-based MoE (FM) in ES-AUC.

| Model              | Race | Gender | Ethnicity | Language |
|--------------------|------|--------|-----------|----------|
| FairMoE/b16        | 70.9 | 70.4   | 70.7      | 66.1     |
| FairMoE/b16 w/o EM | 66.2 | 68.1   | 53.5      | 62.9     |
| FairMoE/b16 w/o FM | 64.0 | 66.5   | 66.3      | 61.0     |
| FairMoE/114        | 74.0 | 69.5   | 73.4      | 64.1     |
| FairMoE/l14 w/o EM | 68.6 | 66.9   | 62.0      | 62.2     |
| FairMoE/l14 w/o FM | 72.2 | 65.4   | 72.1      | 60.8     |

<span id="page-5-3"></span>Table 6: Results of Ablation study on MoE modules in Text and Image in ES-AUC.

| Model             | Race | Gender | Ethnicity | Language |
|-------------------|------|--------|-----------|----------|
| FairMoE/b16       | 70.9 | 70.4   | 70.7      | 66.1     |
| w/o Text MoE/b16  | 66.8 | 67.2   | 61.3      | 63.6     |
| w/o Image MoE/b16 | 69.4 | 66.8   | 64.6      | 54.8     |
| FairMoE/l14       | 74.0 | 69.5   | 73.4      | 64.1     |
| w/o Text MoE/l14  | 72.1 | 61.3   | 64.0      | 63.8     |
| w/o Image MoE/114 | 66.8 | 65.3   | 64.3      | 58.7     |

that just minimizing the distance between different attributes' distribution is not enough. Thus, optimizing difference between dispersion of attributes' distribution is indispensable to achieve a leap in both effectiveness and fairness. In addition, optimizing dispersion can improve stability of MoE, letting Fair-MoE better filter out bias patch embedding and utilize its supreme learning capacities to extract fair feature. The results of two ablation studies prove the effectiveness of FO-MoE and FOL.

Thirdly, the proposed FOL consists of five components:  $F_{EI}$ ,  $F_{ET}$ ,  $F_{FI}$ ,  $F_{FT}$ , and  $L_{distance}$ . While  $L_{distance}$  is widely adopted for improving fairness, the other four components are introduced for the first time in this work. To evaluate the effectiveness of these components, we perform an ablation study by individually removing  $F_{EI}$ ,  $F_{ET}$ ,  $F_{FI}$ , and  $F_{FT}$  from Fair-MoE. The results of this study, presented in Table 4, highlight the necessity of utilizing all four components. The findings demonstrate that removing any of  $F_{EI}$ ,  $F_{ET}$ ,  $F_{FI}$ , or  $F_{FT}$  results in a decline in ES-AUC, a metric quantifying the trade-off between fairness and performance. These components are specifically designed to guide the corresponding embedding-based FO-MoEs and featurebased FO-MoEs in filtering out bias and irrelevant features while ensuring better load balancing. Eliminating any of these components compromises the fairness and performance of the corresponding MoE layers, as evidenced by the performance drops shown in Table 4.

Fourthly, FO-MoE is composed of two key components: the embedding-based MoE, integrated into the last attention block, and the feature-based MoE, positioned after the encoder. To demonstrate the effectiveness of these components, we conduct an ablation study by separately removing the embedding-based MoE and feature-based MoE from Fair-MoE. The results, presented in Table 5, validate the contributions of both components. Removing the embedding-based MoE results in a noticeable decline in the trade-off between performance and fairness across all four attributes. This is because, without the embedding-based MoE, Fair-MoE loses its capability to filter out bias and task-irrelevant patch embeddings. Consequently, the model is more prone to incorrectly leveraging these biased and irrelevant embeddings during decision-making. Similarly, removing the feature-based MoE also reduces the trade-off between performance and fairness. This highlights the feature-based MoE's critical role in enhancing the model's learning capacity, enabling it to focus on task-relevant information while minimizing the extraction of bias and task-irrelevant features. These findings underscore the importance of both embedding-based and feature-based MoEs in achieving a balance between fairness and performance, as evidenced by the results in Table [5.](#page-5-2)

Fifthly, to investigate whether Fair-MoE benefits both image and text modalities, we conduct an ablation study by removing FO-MoE and FOL from the text modality to evaluate the effectiveness of Fair-MoE in text processing, and similarly, by removing FO-MoE and FOL from the image modality to evaluate its effectiveness in image processing. Table [6](#page-5-3) presents the results of this study. The results show that removing MoE from either the text modality or the image modality leads to a drop in the trade-off between performance and fairness across all four attributes. This indicates that without Fair-MoE, the features extracted from the corresponding text or image contain more bias and task-irrelevant information. These findings confirm the effectiveness of Fair-MoE in enhancing trade-off between fairness and performance in both text and image modalities.

### 5 conclusion

We propose a new algorithm Fair-MoE that can both improve effectiveness and fairness in medical VLMs. Fair-MoE includes two key components: *FO-MoE and FOL*. *FO-MoE* is designed to learn unbiased features and filter out biased information. Meanwhile, *FOL* not only optimizes the distance between different protected attributes but also enhances the dispersion among them, guiding the model towards greater fairness and effectiveness. Extensive experiments demonstrate the superiority of Fair-MoE. Detailed ablation studies provide evidence of the effectiveness of each component within Fair-MoE.

## References

- <span id="page-6-16"></span>[Dosovitskiy *et al.*, 2021] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR*, 2021.
- <span id="page-6-14"></span>[Fedus *et al.*, 2022] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120):1–39, 2022.
- <span id="page-6-1"></span>[Ferrante, 2022] Ricci Lara MA Echeveste R Ferrante. E addressing fairness in artificial intelligence for medical imaging nat. *Commun*, 13(1):4581, 2022.
- <span id="page-6-6"></span>[Gai *et al.*, 2024] Xiaotang Gai, Chenyi Zhou, Jiaxiang Liu, Yang Feng, Jian Wu, and Zuozhu Liu. Medthink: Explaining medical visual question answering via multimodal decision-making rationale. *arXiv preprint arXiv:2404.12372*, 2024.
- <span id="page-6-15"></span>[Germino *et al.*, 2024] Joe Germino, Nuno Moniz, and Nitesh V Chawla. Fairmoe: counterfactually-fair mixture of experts with levels of interpretability. *Machine Learning*, pages 1–21, 2024.
- <span id="page-6-0"></span>[Giovanola and Tiribelli, 2023] Benedetta Giovanola and Simona Tiribelli. Beyond bias and discrimination: redefining the ai ethics principle of fairness in healthcare

- machine-learning algorithms. *AI & society*, 38(2):549– 563, 2023.
- <span id="page-6-2"></span>[Glocker *et al.*, 2023] Ben Glocker, Charles Jones, Melanie ´ Bernhardt, and Stefan Winzeck. Algorithmic encoding of protected characteristics in chest x-ray disease detection models. *EBioMedicine*, 89, 2023.
- <span id="page-6-9"></span>[Guan *et al.*, 2024] Juwei Guan, Jiaxiang Liu, Shuying Huang, and Yong Yang. Eclb: Efficient contrastive learning on bi-level for noisy labels. *Knowledge-Based Systems*, page 112128, 2024.
- <span id="page-6-7"></span>[Huang *et al.*, 2023] Zhi Huang, Federico Bianchi, Mert Yuksekgonul, Thomas J Montine, and James Zou. A visual–language foundation model for pathology image analysis using medical twitter. *Nature medicine*, 29(9):2307–2316, 2023.
- <span id="page-6-13"></span>[Jacobs *et al.*, 1991] Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, and Geoffrey E. Hinton. Adaptive mixtures of local experts. *Neural Computation*, 3(1):79–87, 1991.
- <span id="page-6-11"></span>[Kelly *et al.*, 2024a] Chris Kelly, Luhui Hu, Jiayin Hu, Yu Tian, Deshun Yang, Bang Yang, Cindy Yang, Zihao Li, Zaoshan Huang, and Yuexian Zou. Visiongpt-3d: A generalized multimodal agent for enhanced 3d vision understanding. *arXiv preprint arXiv:2403.09530*, 2024.
- <span id="page-6-12"></span>[Kelly *et al.*, 2024b] Chris Kelly, Luhui Hu, Bang Yang, Yu Tian, Deshun Yang, Cindy Yang, Zaoshan Huang, Zihao Li, Jiayin Hu, and Yuexian Zou. Visiongpt: Visionlanguage understanding agent using generalized multimodal framework. *arXiv preprint arXiv:2403.09027*, 2024.
- <span id="page-6-3"></span>[Khan *et al.*, 2023] Muhammad Osama Khan, Muhammad Muneeb Afzal, Shujaat Mirza, and Yi Fang. How fair are medical imaging foundation models? In *Machine Learning for Health (ML4H)*, pages 217–231. PMLR, 2023.
- <span id="page-6-10"></span>[Liu *et al.*, 2023a] Jiaxiang Liu, Jin Hao, Hangzheng Lin, Wei Pan, Jianfei Yang, Yang Feng, Gaoang Wang, Jin Li, Zuolin Jin, Zhihe Zhao, et al. Deep learning-enabled 3d multimodal fusion of cone-beam ct and intraoral mesh scans for clinically applicable tooth-bone reconstruction. *Patterns*, 4(9), 2023.
- <span id="page-6-4"></span>[Liu *et al.*, 2023b] Jiaxiang Liu, Tianxiang Hu, Yan Zhang, Yang Feng, Jin Hao, Junhui Lv, and Zuozhu Liu. Parameter-efficient transfer learning for medical visual question answering. *IEEE Transactions on Emerging Topics in Computational Intelligence*, 2023.
- <span id="page-6-5"></span>[Liu *et al.*, 2023c] Jiaxiang Liu, Tianxiang Hu, Yan Zhang, Xiaotang Gai, Yang Feng, and Zuozhu Liu. A chatgpt aided explainable framework for zero-shot medical image diagnosis. *arXiv preprint arXiv:2307.01981*, 2023.
- <span id="page-6-8"></span>[Liu *et al.*, 2024a] Jiaxiang Liu, Tianxiang Hu, Huimin Xiong, Jiawei Du, Yang Feng, Jian Wu, Joey Zhou, and Zuozhu Liu. Vpl: Visual proxy learning framework for zero-shot medical image diagnosis. In *Findings of the Association for Computational Linguistics: EMNLP 2024*, pages 9978–9992, 2024.

- <span id="page-7-10"></span>[Liu *et al.*, 2024b] Jiaxiang Liu, Yuan Wang, Jiawei Du, Joey Zhou, and Zuozhu Liu. Medcot: Medical chain of thought via hierarchical expert. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pages 17371–17389, 2024.
- <span id="page-7-11"></span>[Liu *et al.*, 2025] Jiaxiang Liu, Tianxiang Hu, Jiawei Du, Ruiyuan Zhang, Joey Tianyi Zhou, and Zuozhu Liu. Kpl: Training-free medical knowledge mining of visionlanguage models. *arXiv preprint arXiv:2501.11231*, 2025.
- <span id="page-7-21"></span>[Lou *et al.*, 2021] Yuxuan Lou, Fuzhao Xue, Zangwei Zheng, and Yang You. Cross-token modeling with conditional computation. *arXiv preprint arXiv:2109.02008*, 2021.
- <span id="page-7-16"></span>[Luo *et al.*, 2022] Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon, and Tie-Yan Liu. Biogpt: generative pre-trained transformer for biomedical text generation and mining. *Briefings in bioinformatics*, 23(6):bbac409, 2022.
- <span id="page-7-4"></span>[Luo *et al.*, 2024] Yan Luo, Min Shi, Muhammad Osama Khan, Muhammad Muneeb Afzal, Hao Huang, Shuaihang Yuan, Yu Tian, Luo Song, Ava Kouhana, Tobias Elze, et al. Fairclip: Harnessing fairness in vision-language learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 12289– 12301, 2024.
- <span id="page-7-5"></span>[Mehrabi *et al.*, 2021] Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, and Aram Galstyan. A survey on bias and fairness in machine learning. *ACM computing surveys (CSUR)*, 54(6):1–35, 2021.
- <span id="page-7-3"></span>[Parraga *et al.*, 2023] Otavio Parraga, Martin D More, Christian M Oliveira, Nathan S Gavenski, Lucas S Kupssinsku,¨ Adilson Medronha, Luis V Moura, Gabriel S Simoes, and ˜ Rodrigo C Barros. Fairness in deep learning: A survey on vision and language research. *ACM Computing Surveys*, 2023.
- <span id="page-7-19"></span>[Peyre´ *et al.*, 2019] Gabriel Peyre, Marco Cuturi, et al. Com- ´ putational optimal transport: With applications to data science. *Foundations and Trends® in Machine Learning*, 11(5-6):355–607, 2019.
- <span id="page-7-2"></span>[Pratt *et al.*, 2020] Bridget Pratt, Verina Wild, Edwine Barasa, Dorcas Kamuya, Lucy Gilson, Tereza Hendl, and Sassy Molyneux. Justice: a key consideration in health policy and systems research ethics. *BMJ Global Health*, 5(4):e001942, 2020.
- <span id="page-7-14"></span>[Qiang *et al.*, 2023] Yao Qiang, Chengyin Li, Prashant Khanduri, and Dongxiao Zhu. Fairness-aware vision transformer via debiased self-attention. *arXiv preprint arXiv:2301.13803*, 2023.
- <span id="page-7-9"></span>[Qin *et al.*, 2022] Ziyuan Qin, Huahui Yi, Qicheng Lao, and Kang Li. Medical image understanding with pretrained vision language models: A comprehensive study. *arXiv preprint arXiv:2209.15517*, 2022.
- <span id="page-7-20"></span>[Riquelme *et al.*, 2021] Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, Andre Susano Pinto, Daniel Keysers, and Neil Houlsby. ´

- Scaling vision with sparse mixture of experts. *Advances in Neural Information Processing Systems*, 34:8583–8595, 2021.
- <span id="page-7-15"></span>[Roh *et al.*, 2023] Yuji Roh, Weili Nie, De-An Huang, Steven Euijong Whang, Arash Vahdat, and Anima Anandkumar. Dr-fairness: Dynamic data ratio adjustment for fair training on real and generated data. *Transactions on Machine Learning Research*, 2023.
- <span id="page-7-22"></span>[Sharma *et al.*, 2022] Shubham Sharma, Jette Henderson, and Joydeep Ghosh. Feamoe: fair, explainable and adaptive mixture of experts. *arXiv preprint arXiv:2210.04995*, 2022.
- <span id="page-7-8"></span>[Sikstrom *et al.*, 2022] Laura Sikstrom, Marta M Maslej, Katrina Hui, Zoe Findlay, Daniel Z Buchman, and Sean L Hill. Conceptualising fairness: three pillars for medical algorithms and health equity. *BMJ health & care informatics*, 29(1), 2022.
- <span id="page-7-7"></span>[Stanley *et al.*, 2022] Emma AM Stanley, Matthias Wilms, Pauline Mouches, and Nils D Forkert. Fairness-related performance and explainability effects in deep learning models for brain image analysis. *Journal of Medical Imaging*, 9(6):061102–061102, 2022.
- <span id="page-7-23"></span>[Tian *et al.*, 2024] Bowei Tian, Ruijie Du, and Yanning Shen. Fairvit: Fair vision transformer via adaptive masking. *arXiv preprint arXiv:2407.14799*, 2024.
- <span id="page-7-12"></span>[Tong *et al.*, 2023] Linjie Tong, Jiaxiang Liu, Yang Feng, Tianxiang Hu, and Zuozhu Liu. Tsnet: Integrating dental position prior and symptoms for tooth segmentation from cbct images. In *Medical Imaging with Deep Learning, short paper track*, 2023.
- <span id="page-7-0"></span>[Varkey, 2021] Basil Varkey. Principles of clinical ethics and their application to practice. *Medical Principles and Practice*, 30(1):17–28, 2021.
- <span id="page-7-17"></span>[Wang *et al.*, 2022] Zifeng Wang, Zhenbang Wu, Dinesh Agarwal, and Jimeng Sun. Medclip: Contrastive learning from unpaired medical images and text. *arXiv preprint arXiv:2210.10163*, 2022.
- <span id="page-7-1"></span>[Wang *et al.*, 2023] Yue Wang, Yaxin Song, Zhuo Ma, and Xiaoxue Han. Multidisciplinary considerations of fairness in medical ai: A scoping review. *International Journal of Medical Informatics*, page 105175, 2023.
- <span id="page-7-18"></span>[Xu *et al.*, 2023] Shawn Xu, Lin Yang, Christopher Kelly, Marcin Sieniek, Timo Kohlberger, Martin Ma, Wei-Hung Weng, Atilla Kiraly, Sahar Kazemzadeh, Zakkai Melamed, et al. Elixr: Towards a general purpose x-ray artificial intelligence system through alignment of large language models and radiology vision encoders. *arXiv preprint arXiv:2308.01317*, 2023.
- <span id="page-7-6"></span>[Xu *et al.*, 2024] Zikang Xu, Jun Li, Qingsong Yao, Han Li, Mingyue Zhao, and S Kevin Zhou. Addressing fairness issues in deep learning-based medical image analysis: a systematic review. *npj Digital Medicine*, 7(1):286, 2024.
- <span id="page-7-13"></span>[Yang *et al.*, 2021] Yong Yang, Jiaxiang Liu, Shuying Huang, Weiguo Wan, Wenying Wen, and Juwei Guan. Infrared and visible image fusion via texture conditional

- generative adversarial network. *IEEE Transactions on Circuits and Systems for Video Technology*, 31(12):4771– 4783, 2021.
- <span id="page-8-0"></span>[Yang *et al.*, 2024] Deshun Yang, Luhui Hu, Yu Tian, Zihao Li, Chris Kelly, Bang Yang, Cindy Yang, and Yuexian Zou. Worldgpt: a sora-inspired video ai agent as rich world models from text and image inputs. *arXiv preprint arXiv:2403.07944*, 2024.
- <span id="page-8-1"></span>[Zoph *et al.*, 2022] Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. St-moe: Designing stable and transferable sparse expert models. *arXiv preprint arXiv:2202.08906*, 2022.