# Optimizing Distributed Deployment of Mixture-of-Experts Model Inference in Serverless Computing

Mengfan Liu\*, Wei Wang<sup>†</sup>, and Chuan Wu\*

\*Department of Computer Science, The University of Hong Kong

†Department of Computer Science and Engineering, The Hong Kong University of Science and Technology

Email: ml621@connect.hku.hk, weiwa@cse.ust.hk, cwu@cs.hku.hk

Abstract—With the advancement of serverless computing, running machine learning (ML) inference services over a serverless platform has been advocated, given its labor-free scalability and cost effectiveness. Mixture-of-Experts (MoE) models have been a dominant type of model architectures to enable large models nowadays, with parallel expert networks. Serving large MoE models on serverless computing is potentially beneficial, but has been underexplored due to substantial challenges in handling the skewed expert popularity and scatter-gather communication bottleneck in MoE model execution, for costefficient serverless MoE deployment and performance guarantee. We study optimized MoE model deployment and distributed inference serving on a serverless platform, that effectively predict expert selection, pipeline communication with model execution, and minimize the overall billed cost of serving MoE models. Especially, we propose a Bayesian optimization framework with multi-dimensional  $\epsilon$ -greedy search to learn expert selections and optimal MoE deployment achieving optimal billed cost, including: 1) a Bayesian decision-making method for predicting expert popularity; 2) flexibly pipelined scatter-gather communication; and 3) an optimal model deployment algorithm for distributed MoE serving. Extensive experiments on AWS Lambda show that our designs reduce the billed cost of all MoE layers by at least 75.67% compared to CPU clusters while maintaining satisfactory inference throughput. As compared to LambdaML in serverless computing, our designs achieves 43.41% lower cost with a throughput decrease of at most 18.76%.

#### I. INTRODUCTION

Serverless computing is a cloud computing paradigm that the cloud provider elastically manages the provisioning of resources (servers, functions, containers, storage, etc.) to deploy services according to their user demand [1]. Serverless computing has been used for serving data analytic applications such as web services [1] [2]. In recent years, there has been an increasing trend in adopting serverless computing for machine learning (ML) services, particularly for model inference serving. For example, Gillis [3] studies model partitioning and scheduling for large deep neural network (DNN) inference on AWS Lambda [4] and Google Cloud Functions [5]; TETRIS [6] designs a memory-efficient serverless computing runtime for DNN inference.

Deploying ML inference services on a serverless platform is more appealing than using traditional GPU/CPU clusters for several reasons. *First*, it frees ML developers from managing hardware resources and virtual machine/container environments, simplifying service deployment and maintenance [7]. *Second*, its pay-as-you-go pricing model ensures cost

efficiency by charging only for resources actively used in fine granularity, avoiding unnecessary costs for idle resources [8]. *Third*, serverless functions have been provided to support parallelisms needed for large-scale ML model inference like AWS Lambda Functions [4], Google Cloud Functions [5], Azure Functions [9], and Alibaba Cloud Functions [10]. State-of-the-art commercial serverless platforms largely support CPU services [4] [5] [9]. Though GPU-based model inference has been preferred for high serving performance, using CPUs for inference serving has been a viable alternative, given that high-caliber GPUs are often in shortage, CPUs are more available and provide substantial cost savings, while being able to meet application service level objectives (SLOs) [11].

The Mixture-of-Experts (MoE) models have been a dominant type of model architectures to enable large models nowadays [12], achieving high model capacity without increasing computation [13] [14]. To build a large model using the MoE architecture, layers in a representative DNN model (e.g., Transformer) are replaced by MoE layers. Each MoE layer includes a gating network and multiple parallel expert networks. During model inference, each input token to an MoE layer is first evaluated by the gating network, which determines the most relevant expert(s) to handle the token [12]. Then the token is routed to the selected expert(s) for computation, and the processing results are aggregated to produce the MoE layer output. MoE models have been widely used to serve various tasks, e.g., GLaM [13] for speech recognition and SwitchTransformer [12] for text generation.

Serving a large MoE model is resource intensive as it requires substantial memory to deploy the parallel experts. DeepSpeed [14] and Lina [15] adopt expert parallelism to accelerate inference with an expert assigned to a device (mostly GPU) and all-to-all communication to receive inputs from other devices. After deployment, devices incur costs even when idle, making GPU/CPUs more costly than using a serverless platform.

We advocate serving MoE models in a CPU-based serverless platform, for labor and cost-efficient management of inference serving. Serving large MoE models on serverless computing has been underexplored. The main target for MoE model deployment in a serverless platform is to minimize the billed cost of all MoE layers in serving, for memory usage and execution time of serverless functions that run the MoE layers [4] [5] [9] [10]. Two major challenges arise for costminimal distributed deployment of MoE model inference in a serverless platform.

*First*, serverless functions are typically deployed with memory size configured before the service runs, and the skewed, unknown-beforehand expert selection of input tokens complicates proactive memory configuration of the functions. In MoE serving, some experts (each run as a serverless function) receive many tokens for processing, while others do not. Intuitively, popular experts should be run on serverless functions with larger memory while non-popular ones use less memory. Existing MoE serving solutions [\[16\]](#page-12-14) [\[17\]](#page-12-15) in GPU/CPU clusters decide resource assignment for experts during MoE inference, which is infeasible in serverless platforms. Deploying a serverless function takes several minutes. The first time after a serverless function is deployed, it takes a long time for the function to start execution, due to resource initialization (i.e., the cold start issue) [\[2\]](#page-12-0). This would cause long delays in MoE serving if its serverless functions are deployed according to the current demand, degrading efficiency and throughput of inference serving. Therefore, the key challenge here is to efficiently and accurately estimate expert popularity before the MoE inference service starts in a serverless platform. This enables optimizing memory configurations for serverless functions, thereby decreasing the billed cost of MoE serving.

*Second*, the scatter-gather communication for token-toexpert routing and expert processing result aggregation at MoE layers is time-consuming, that may block subsequent operations as non-MoE layers must wait for all experts to complete their computation and communication [\[14\]](#page-12-12). Existing proposals on redesigning MoE scatter-gather communication with pipelining [\[18\]](#page-12-16) [\[19\]](#page-12-17) in CPU/GPU clusters are inadequate in serverless platforms. Direct inter-function transfers in a serverless platform are constrained by platform-specified maximum data transfer size (i.e., payload size), while indirect transfers via external storage (e.g., S3 bucket for AWS Lambda [\[4\]](#page-12-2)) take longer as the data must be saved to external storage and then retrieved. Pipelining data transfer with model execution is infeasible with direct transfers: a serverless function retains no data between invocations (i.e., stateless property) and direct data transfers from other functions require re-invocation of the function each time; model parameters that a serverless function uses are not retained during direct transfers and hence need to be reloaded for each re-invocation, resulting in significant time and memory waste. Indirect transfers between serverless functions rely on external storage, incurring longer communication time and higher cost, adding complexity to pipelining design. This calls for novel communication designs tailored to MoE inference in serverless platforms.

Tackling these challenges, we design a serverless MoE inference solution that effectively predicts expert popularity, pipelines MoE communication with model execution, and optimally deploys MoE model for distributed inference, that minimize the billed cost of all MoE layers in MoE model serving. We propose a Bayesian optimization (BO) framework with multiple ϵ-greedy search (GS) to learn expert popularity and optimize MoE deployment for billed cost minimization. Our main contributions are summarized as follows:

- ▷ We design a novel Bayesian decision-making approach for expert selection prediction, including a comprehensive token feature design, a novel posterior calculation approach, and an adjustable key-value dataset table. We analyze the MoE inference process to extract relevant token features. The posterior calculation incorporates real request distributions to refine the posterior using profiled data. The key-value dataset table is adjusted with new key-value pairs according to feedback from model inference, which are used to update the profiled data probabilities and improve prediction accuracy.
- ▷ We propose several scatter-gather communication designs for a serverless platform: indirect transfers with flexible pipeline operations via external storage, simple indirect transmissions without pipelining, and simple direct invocation of serverless functions without pipelining. As different designs perform the best in terms of execution time under different scenarios, selecting a proper scatter-gather communication design affects billed cost saving in serverless MoE inference.
- ▷ We formulate optimal deployment of distributed MoE inference on a serverless platform into a mixed-integer quadratically constrained programming (MIQCP) problem, which chooses one of the proposed scatter-gather communication designs, sets memory configurations of expert serverless functions, and determines the number of function replicas. We design an optimal deployment selection (ODS) algorithm, achieving a billed cost of all MoE layers upper bounded by a constant ratio of the optimal solutions.
- ▷ We propose a BO framework with multiple ϵ-GS to optimize expert selection predictions and distributed deployment of the MoE model. The BO framework iteratively adjusts the key-value dataset table for expert selection prediction using feedback billed cost of all MoE layers in serving. The expert predictions then serve as input for optimal MoE model deployment, which further provides cost feedback for dataset table update. The multiple ϵ-GS balances exploration of different key-value pairs in the dataset table and exploitation of high-performing key-value pairs, in dataset table update.
- ▷ We implement our designs on AWS Lambda using Pytorch and Optuna packages. Extensive experiments show that our designs reduce the billed cost of all MoE layers by at least 75.67% compared to inference over CPU clusters, while maintaining a satisfactory inference throughput. by at least 43.41% compared to LambdaML [\[20\]](#page-12-18) in serverless computing, with a throughput decrease of at most 18.76%.

# II. BACKGROUND AND MOTIVATION

# *A. Serverless Computing*

![](_page_1_Figure_12.jpeg)

<span id="page-1-0"></span>Fig. 1. Overview of MoE model deployment on a serverless platform.

Serverless computing is a cloud computing paradigm that provides Functions as a Service (FaaS). A serverless function is a piece of code running in a cloud infrastructure for applications or services, without the need of the developer to provision or manage the resources. These functions can be triggered by various events [8], such as inference requests. Serverless functions are stateless, that they retain no data from their previous execution. Serverless functions obtain data inputs mainly in two ways: from output of other functions or from an external storage A function can directly transfer output to another function as input when invoking the latter, and a payload size limits the maximal data transfer size between functions. When the data exceeds the payload size, external storage is used for relaying data between two functions [1] [3]. External storage can also store other data needed by serverless functions, such as model parameters.

Serverless functions, along with external storage, Docker image manager (e.g., ECR [21] for Amazon Lambda [4]), and serverless function deployment manager (e.g., step function [22] on Amazon Lambda [4]) can be used for ML inference. As shown in Fig. 1, deployment of an MoE model for inference serving on a serverless platform involves several steps. First, the MoE model is partitioned: the MoE layer adopts expert parallelism, while the non-MoE layers are grouped and distributed according to model parallelism. Next, each model partition is built as a Docker image, which is then pushed to the Docker image manager, with its model parameters stored in external storage. Finally, each Docker image is assigned to a serverless function and these functions are deployed into the serverless platform by a serverless function manager. After deployment, inference requests from service users are stored in external storage and retrieved by the deployed MoE model for inference serving. During inference, each serverless function loads its model parameters and intermediate computation results from external storage, and saves intermediate results back to external storage.

Current commercial serverless computing platforms (e.g., AWS Lambda [4], Google Cloud Functions [5], Azure [9], and Alibaba Cloud Functions [10]) are CPU-based. Therefore, we focus on CPU-based serverless platforms in this paper.

Function configuration. Memory is the principal lever available to developers for controlling the performance of a serverless function. The memory configuration of a serverless function determines its computing speed, as more memory corresponds to more virtual CPUs [4] [5]. Users can configure the amount of memory to a serverless function and the serverless platform automatically allocates resources based on this configuration during function deployment. Commercial serverless computing providers [4] [5] [9] [10]generally charge users on the used memory during running time of a serverless function at the unit of GB-second (i.e., GBs). External storage and Docker image manager is charged on the size of stored objects, and severless function deployment manager usage is charged on the number of function invocations. We focus on the billed cost of serverless functions as it represents the

![](_page_2_Figure_4.jpeg)

<span id="page-2-0"></span>Fig. 2. Billed cost of all MoE layers and inference throughput of a GPT-2-based MoE model.

primary cost for MoE model inference, given a fixed model size and inference request workload.

#### B. MoE Inference

The gating network routes tokens to experts based on token features, e.g., position in the inference request sequence, meaning of a word token, its roles in the sequence (subject, object, pronoun, verb, etc.) [15]. This routing generally does not restrict the number of tokens to be processed by each expert, resulting in skewed popularity and unbalanced workloads among experts. GShard [23] enforces a threshold for the maximal number of tokens processed by one expert. Zhou et al. [24] allow each expert to select the top-k tokens and operate with a fixed bucket size. Pre-gated MoE [25] modifies the role of a gate function to preemptively select the experts to be activated for the next MoE block. These proposals intervene in token-to-expert selection and hence may change model results [23]–[25]. We do not modify token-to-expert routing decisions in our MoE inference serving.

**Distributed MoE inference.** Expert parallelism is commonly adopted that allocates one device for each expert [14] [15]. For data parallelism in distributed MoE serving, each request batch is distributed across multiple devices for simultaneous processing [12] [23]. These parallelisms necessitate communication across multiple devices for input distribution and output aggregation. FasterMoE [26] applies tensor slicing and pipelining design to overlap all-to-all communication and computation in MoE layers. PipeMoE [18] studies the pipeline degree for tensor slicing to adaptively pipeline communication and computation. MpipeMoE [19] extends tensor slicing to multiple dimensions of the data, further expediting inference. The designs rely on the hardware architecture of GPU/CPU clusters, and thus cannot be adopted on serverless platforms.

ML services in serverless platforms. Extensive research has focused on optimizing ML model serving in serverless platforms. Amps-inf [27] studies model partitioning to minimize inference costs. Ali et al. [28] design a request batch queue to reduce costs, while INFless [29] reduces inference latency by optimizing batch queuing, execution time, and cold start rates. Gillis [3] adopts model partitioning and scheduling for low-latency large DNN model inference, while ServerlessLLM [30] focuses on fast multi-tier checkpoint loading and optimized startup-time scheduling. DNN serving on serverless platforms has been proven effective in meeting serving SLOs and handling varying workloads. However, there is a lack of designs for efficient MoE inference on a serverless platform.

## C. Opportunities and challenges

Opportunity: MoE model inference in a serverless platform can bring substantial cost reduction and satisfactory

![](_page_3_Figure_0.jpeg)

<span id="page-3-0"></span>Fig. 3. Number of tokens with token ID 10424 (from the Enwiki8 dataset) routed to different experts at the 2nd MoE layer in Bert-based MoE model.

inference performance, as compared to CPU cluster-based inference serving. On a serverless platform, expert networks can be individually assigned to serverless functions with different memory size configurations, so that more resources are rent for workload-heavy experts and less for workload-light experts. Further, inference functions are invoked on-demand at fine-grained timescales (i.e., milliseconds) [4] [5]. Serverless functions are only billed on assigned memory when executing; no cost for idle resources is incurred. However, the cost of a CPU cluster typically depends on the amount of resources over a fixed coarse-grained period (e.g., per month or per hour), so that costs may still be incurred for idle resources. Fig. 2 shows the billed cost of all MoE layers and inference throughput, when a GPT-2-based MoE model serves 10,240 tokens from the Enwiki8 [31] dataset. The CPU cluster uses two 64-core AMD EPYC CPUs with 512GB of DRAM for the entire MoE model. Each serverless function for MoE inference on AWS Lambda is allocated 3008 MB of memory. The billed cost of all MoE layers on AWS Lambda is significantly lower than on the CPU cluster. The throughput of MoE inference on the serverless platform is 22.9 tokens per second, substantially exceeding the human reading speed of 3.3 tokens per second [32]. Hence, serverless-based MoE inference presents a viable solution. Two challenges exist on enabling efficient deployment and serving of an MoE model on a serverless platform.

Challenge 1: Skewed, unknown-beforehand expert popularity prevents proactive, accurate memory configuration of serverless functions. Expert selection is unknown before actual token processing by the gating network. Dynamic resource allocation, which decides computational resources for each expert based on real-time expert popularity during inference, has been advocated for MoE serving in GPU/CPU clusters [16] [26]. Serverless functions require a long time to deploy (e.g., 1 minute or longer) and to start execution (e.g., 5 seconds or longer) after deployed [2]; on-demand dynamic function redeployment is largely infeasible, which would substantially slow down inference serving. This necessitates proper configuration of memory sizes when deploying serverless functions before service starts, requiring prior knowledge of expert popularity. Improper memory configuration may either fail to meet the memory demand during inference, or incur higher billed costs due to unnecessary memory usage. FlexMoE [33] and Prophet [34] use average historical expert popularity to adjust resources among experts. Lina [15] predicts expert popularity before model deployment using the maximum a posteriori probability based on historical tokento-expert mapping and token ID information. Cong et al. [35] propose an LSTM-based algorithm trained on historical tokento-expert mapping for expert selection predication. These ex-

![](_page_3_Figure_4.jpeg)

<span id="page-3-1"></span>(a) 256 tokens

Fig. 4. Billed cost of all MoE layers and end-to-end inference time of a Bertbased MoE model on AWS Lambda (tokens from Enwiki8 dataset; payload size 6MB).

pert selection prediction methods may achieve low prediction accuracy as the token-to-expert relationship in historical data is not exploited [33], [34], or are memory-intensive and require a long training time [35]. Lina [15] considers the token distribution but only uses token ID as the token feature. Token ID is insufficient to fully identify a token in token-to-expert mappings, as tokens with the same token ID may be routed to different experts at a MoE layer, as illustrated in Fig. 3.

Challenge 2: MoE scatter-gather communication renders performance bottlenecks in serverless MoE inference. In each MoE layer, scatter communication occurs between the gating network and expert networks, and gather communication between expert networks and the subsequent non-MoE layer, which often renders a bottleneck because the non-MoE layer must wait for all experts to complete their computation and communication. The direct and indirect inter-function communication on a serverless platform results in different billed costs and inference time. In Fig. 4, MoE scatter-gather communication via indirect transfers incurs higher cost and longer inference time than direct transfers when serving a 256-token batch, due to the additional function running time required for storing data in external storage and retrieving data as input. Direct transfers cannot be adopted when serving a 2560-token batch as the payload size is exceeded; the serving cost under indirect transfers is very high. While pipelining communication with computation can typically alleviates this bottleneck in GPU/CPU clusters, pipelining in a serverless platform needs to take its direct or indirect data transfer modes into consideration, and carefully designed to improve efficiency in a serverless platform.

We redesign token features and propose a novel posterior calculation method for expert selection prediction. We design and select the best scatter-gather communication methods for MoE layer inference. We involve all modules in a BO framework for optimal MoE serving deployment.

#### III. DESIGN

#### A. System overview

We consider distributed deployment of an MoE model in a serveless computing platform to serve inference requests. The MoE model consists of MoE layers (each includes a gating network and multiple experts) and non-MoE layers (e.g., feedforward networks, multi-head attention networks [12]). We adopt expert parallelism [14], [15] by assigning each expert to a serverless function. Model parallelism is adopted for non-MoE parts of the model with each non-MoE layer assigned to a serverless function. Model parameters are store in external

![](_page_4_Figure_0.jpeg)

<span id="page-4-0"></span>Fig. 5. System Overview.

storage. We mainly focus on the MoE layers since the non-MoE layers are traditional DNNs extensively studied [3] [36].

We propose a Bayesian Optimization (BO) framework to learn expert selections and optimal deployment of the MoE model for inference serving. The goal is to minimize overall billed cost of all MoE layers in serving. The BO framework consists of a Feedback Processor to adjust expert selection prediction, an Expert Selection Predictor, and a Policy Maker. An illustration is given in Fig. 5. The expert selection predictor provides expert predictions of inference tokens from a realworld dataset on an inference task. This prediction is based on the posterior distribution calculated from these tokens and profiled data. The profiled data records the number of times each token-to-expert mapping occurs across at least 100 samples from the same real-world dataset, organized in a key-value dataset table. With the predictions, the policy maker decides how to deploy the MoE model, configures the memory size of each expert serverless function and adopts our scatter-gather communication design. The feedback processor adjusts the key-value pairs in the profiled dataset table for improving expert selection prediction, using feedback of the billed cost of all MoE layers in serving. In each BO iteration, expert predictor is adjusted, the policy maker decides optimal deployment of the MoE model, and the billed cost is collected for feedback processor's key-value table adjustment.

When the BO algorithm converges, the MoE model is deployed according to the learned expert popularity and optimal deployment policy, and serves real-world inference requests.

## B. Expert selection prediction

In an MoE model, each token in an inference request is routed to top-k experts by the gating network at an MoE layer, based on token features. Token ID is the most commonly used token feature [15], but itself alone is insufficient to fully identify a token in token-to-expert mapping. We carefully design token features by investigating more token information during MoE inference. As Transformer models are typically the backbone of MoE models [12], we focus on Transformer-based MoE models.

For Transformer-based MoE models, token processing mainly occurs in the embedding, encoder and decoder layers. In the embedding layers, each token is embedded with its own information and its position (e.g., word embedding and position embedding). Thus, the token ID and position ID can be extracted as token features. In the encoder and decoder layers, tokens flow through feed-forward networks and one multi-head attention. In the feed-forward networks, no dependencies exist within the token sequence, so token ID and position ID are sufficient as the token features. Each multi-head attention layer contains multiple self-attentions.

Each self-attention calculates the Query, Key, and Value of the tokens to capture dependencies between tokens in the token sequence. The dependencies are quantified by the softmax attention scores, which indicate the relevance of each token to the others. For each token, we extract these dependencies as a token feature. For simplicity, the dependencies for each token are derived as the token ID of the token with the highest sum of softmax attention scores across all self-attentions at a multi-head attention layer, referred to as the attention ID. The attention ID may vary before different MoE layers, aligning with the diverse expert popularities at different MoE layers. Therefore, the token features include the token ID, the position ID, and the attention ID. The token IDs and position IDs are from the input token sequences, known before inference starts. The attention IDs are from the self-attention parts of the non-MoE layer before each MoE layer, known during inference.

Our expert prediction in the BO framework is learned on profiled data, which records the token-to-expert mappings on at least 100 samples from a real-word dataset of inference task. The profiled data are organized in a key-value dataset table where the keys represent token-to-expert mappings and the values denote their occurrence counts. Especially, we design a new posterior calculation method and use the maximum posterior approach to predict expert selection for new tokens, where the posterior represents the probability of an expert given a token [15].

Assume  $\mathbf{f}$  is the token feature vector of a token, in which  $\mathbf{f}_1$  is the token ID,  $\mathbf{f}_2$  is the position ID and  $\mathbf{f}_3$  is the attention ID. The posterior given the token can be expressed as  $\mathcal{P}(\mathbb{N}_{e,i}|\mathbf{f})$  with  $e \in \mathbb{E}$ , where  $\mathbb{N}_{e,i}$  is the i-th expert in the expert set  $\mathbb{N}_e$  at MoE layer e and  $\mathbb{E}$  is the set of MoE layers in the MoE model. For a new token that has not undergone MoE inference, its feature  $\mathbf{f}_1'$  is known but  $\mathbf{f}_3'$  is unknown, The probability of  $\mathbf{f}_2$  at any position is uniform, and the probability of  $\mathbf{f}_3$  at any value can be approximated by the probability of  $\mathbf{f}_1$  at that value, as the attention ID  $\mathbf{f}_3$  is defined as the token ID with the highest attention scores. We can obtain all probabilities related to  $\mathbf{f}_1'$  from the profiled data, the uniform probability  $\mathcal{P}'(\mathbf{f}_2)$  of  $\mathbf{f}_2$  at any value and the probability  $\mathcal{P}'(\mathbf{f}_3)$  of  $\mathbf{f}_3$  at any value from tokens in the same real-world dataset, that have not undergone MoE inference.

To leverage all token features for identifying a token effectively, we use Bayes' theorem to design a new posterior calculation method. The Bayes' theorem  $\mathcal{P}(\mathbb{N}_{e,i}|\mathbf{f}) = \mathcal{P}(\mathbf{f}|\mathbb{N}_{e,i})\mathcal{P}(\mathbb{N}_{e,i})/\mathcal{P}(\mathbf{f})$  describes how the posterior of an expert selection given a token  $\mathcal{P}(\mathbb{N}_{e,i}|\mathbf{f})$  is updated with the prior of the expert  $\mathcal{P}(\mathbb{N}_{e,i})$ , the prior of the token  $\mathcal{P}(\mathbf{f})$ , and the likelihood of the token given the expert  $\mathcal{P}(\mathbf{f}|\mathbb{N}_{e,i})$ . As  $\mathcal{P}(\mathbb{N}_{e,i}|\mathbf{f}'_1) = \mathcal{P}(\mathbf{f}'_1|\mathbb{N}_{e,i})\mathcal{P}(\mathbb{N}_{e,i})/\mathcal{P}(\mathbf{f}'_1)$  and  $\mathcal{P}(\mathbf{f}'_1|\mathbb{N}_{e,i}) = \mathcal{P}(\mathbf{f}'_1,\mathbb{N}_{e,i})/\mathcal{P}(\mathbb{N}_{e,i})$ , we can involve the uniform probability  $\mathcal{P}'(\mathbf{f}_2)$  of  $\mathbf{f}_2$  and the probability  $\mathcal{P}'(\mathbf{f}_3)$  of  $\mathbf{f}_3$  into the likelihood  $\mathcal{P}(\mathbf{f}'_1|\mathbb{N}_{e,i})$  of the expert given  $\mathbf{f}'_1$ , through the joint probability  $\mathcal{P}(\mathbf{f}'_1,\mathbb{N}_{e,i})$ . For simplicity, we multiply  $\mathcal{P}'(\mathbf{f}_2)$  and  $\mathcal{P}'(\mathbf{f}_3)$  with  $\mathcal{P}(\mathbf{f}'_1,\mathbb{N}_{e,i})$  as involvement. Hence, the designed posterior

calculation method is given by:

$$\mathcal{P}(\mathbb{N}_{e,i}|\mathbf{f}_1') = \int_{\mathbf{f}_2} \int_{\mathbf{f}_3} \mathcal{P}^*(\mathbb{N}_{e,i}|\mathbf{f}_1',\mathbf{f}_2,\mathbf{f}_3) \frac{\mathcal{P}^*(\mathbf{f}_1',\mathbf{f}_2,\mathbf{f}_3)\mathcal{P}'(\mathbf{f}_3)}{\mathcal{P}^*(\mathbf{f}_1',\mathbf{f}_2)} d\mathbf{f}_3$$

$$\frac{\mathcal{P}^*(\mathbf{f}_1',\mathbf{f}_2)\mathcal{P}'(\mathbf{f}_2)}{\mathcal{P}^*(\mathbf{f}_1')} d\mathbf{f}_2, \forall e \in \mathbb{E}, \forall i \in \mathbb{N}_e,$$
(1)

where P ∗ (·) represents the probability calculated from profiled data in the key-value dataset table.

Therefore, we can use the maximum a posterior method to predict the expert selection for a token with the known token feature f ′ 1 :

<span id="page-5-0"></span>
$$\hat{i}_e = argmax_{i \in \mathbb{N}_e} \mathcal{P}(\mathbb{N}_{e,i} | \mathbf{f}'_1), \forall e \in \mathbb{E},$$
(2)

where ˆi<sup>e</sup> is the predicted expert at MoE layer e. Eq. [\(2\)](#page-5-0) can be readily extended to top-k expert selection prediction.

## *C. Scatter-gather communication design*

We design pipelined scatter-gather communication at the MoE layer in a serverless platform for better resource utilization and cost reduction.

For a batch of tokens to serve, we set a pipeline degree β to split the batch for each expert into minibatches, where β represents the maximal minibatch size. At each MoE layer, the gating network routes the splitted minibatches to each expert and the next non-MoE layer gathers processed minibatches from each expert. If the minibatch size exceeds the payload limit, indirect transfer through external storage is used; otherwise, direct invocation of the serverless functions is adopted.

Data transfer from the gating network to experts can be pipelined with expert execution based on expert parallelism. Experts do not use pipelining for data transfer to the next non-MoE layer, as this may cause the next non-MoE layer to wait for the execution of experts (e.g., calculation of a minibatch) after receiving the previous processed minibatch. When external storage is involved, with pipelining, the time to download a minibatch from external storage and process this minibatch can overlap with the time to upload the previous processed minibatch to external storage. Note that pipelining is only achievable with indirect transfer via external storage on a serverless platform. Serverless functions are stateless and direct data transfers from other functions require re-invocation of the function each time; model parameters that a serverless function uses are not retained during direct transfers and hence need to be reloaded from external storage for each reinvocation, resulting in significant time and memory waste. We use one block time to represent the maximal overlap time of the indirect upload of a minibatch and the download and calculation of the next minibatch. The block time is determined by the pipeline degree β: a larger β results in fewer minibatches and longer block time.

The benefits of pipelining can be reduced by the access delay to external storage. We design three scatter-gather communication methods tailored for MoE inference on a serverless platform. Let a<sup>e</sup> ∈ A = {1, 2, 3} denote three possible scattergather communication methods at MoE layer e, and we allow all experts at an MoE layer to use the same method to simplify implementation. Two indirect transfer options exist.

- In the first option (a<sup>e</sup> = 1), the gating network splits each expert's input into minibatches and sends them to external storage; at each expert, downloading a minibatch from external storage and calculating this minibatch can be overlapped with uploading the previous processed minibatch to external storage. As shown in Fig. [6\(](#page-6-0)a), after splitting input data into minibatches of one token each, two minibatches for two experts are stored in external storage (step 1); these two minibatches are downloaded from external storage and calculated by two expert serverless functions, while the next two minibatches are stored in external storage (step 2); the processed minibatches are stored in external storage, while the next two minibatches are downloaded from external storage and calculated by two expert serverless functions (step 3); all processed minibatches are stored in external storage (step 4) and then the next non-MoE layer downloads all minibatches from external storage (step 5).
- In the second option (a<sup>e</sup> = 2), the gating network transfers each expert's input via external storage to each expert function, and all processed results are downloaded by the next non-MoE layer, without pipelining. As illustrated in Fig. [6\(](#page-6-0)b), all input data are stored in external storage (step 1); each expert serverless function downloads its input from external storage (step 2), and stores the processed results in external storage (step 3); then the next non-MoE layer downloads all processed results from external storage (step 4).

Fig. [8](#page-6-1) illustrates the execution time of MoE layers under the two communication designs. Stage 1 in both cases represents the time for each expert to start the warm-up function without resource initialization (i.e., short warm start time) and download model parameters, while the gating network concurrently uploads each expert's input to external storage. Stage 2 is the time for each expert to download input from external storage and calculate it, overlapping with the time to start the warmup function and download model parameters in the next non-MoE layer. In stage 3, the next non-MoE layer downloads the processed results.

• When using direct transfers (a<sup>e</sup> = 3), the gating network directly transfers each expert's input to the expert serverless function, and the processed results of each expert are directly transferred to the next non-MoE layer, as shown in Fig. [7.](#page-6-2) Fig. [9](#page-6-3) illustrates its execution time.

The total MoE layer time varies with different communication designs. We will carefully make the choices in our distributed MoE deployment problem.

## <span id="page-5-1"></span>*D. MoE model deployment*

The policy maker decides optimal MoE model deployment by making the following decisions:

• Memory size configuration for each serverless function. Let M be the set of memory size options for each serverless function (e.g., from 128MB to 3008MB on AWS Lambda) xe,i,j ∈ {0, 1} denotes if the j-th option in set M is selected for expert i in MoE layer e (1) or not (0). The processing time of one token at expert i in MoE layer e, is given by:

![](_page_6_Figure_0.jpeg)

Fig. 6. Scatter-gather communication with indirect transfers through external storage: (a) with pipelining; (b) without pipelining. Pipeline degree  $\beta$  is 2.

<span id="page-6-0"></span>![](_page_6_Figure_2.jpeg)

<span id="page-6-2"></span>Fig. 7. Scatter-gather communication with direct function invocation.

![](_page_6_Figure_4.jpeg)

<span id="page-6-1"></span>Fig. 8. Scatter-gather communication time with indirect transfers through external storage: (a) with pipelining; (b) without pipelining.

![](_page_6_Figure_6.jpeg)

<span id="page-6-3"></span>Fig. 9. Scatter-gather communication time with direct function invocation.

$$t_{e,i}^{cal} = s_{e,i} \sum_{j=1}^{|\mathbb{M}|} x_{e,i,j} U_j, \forall e \in \mathbb{E}, \forall i \in \mathbb{N}_e,$$
 (3)

where  $s_{e,i} \in \{0,1\}$  denotes if the expert is selected (1) or not (0) for the token (given based on expert selection prediction), and  $U_j$  is the time to process one token in an expert using the j-th memory size option with the serverless function.

• Expert replication. As the maximal memory size of each serverless function is limited, it is possible that it still takes a long time for a popular expert to process tokens routed to it. Given that an end-to-end inference time target should be met in inference serving, we further consider replicating serverless functions of experts and allow expert replicas to run in parallel for toke processing. Let  $y_{e,i,g} \in \{0,1\}$  denote if expert i in MoE layer e has g serverless function replicas (1) or not (0), where g=1,...,G with G as maximal possible replica number The number of tokens routed to one replica is  $r_{e,i} = \sum_{g=1}^G y_{e,i,g} d_{e,i}/g$ , where  $d_{e,i}$  denotes the number of tokens routed to all replicas of the expert. Let  $D^{in}$  be the size of one token and  $D^p$  be the maximal payload size in the

serverless platform. When  $r_{e,i}D^{in} > D^p$ , direct transfer is not feasible (i.e,  $a_e = 1$  or  $a_e = 2$ ) at MoE layer e.

• Scatter-gather communication method and parameter  $(\beta)$ . We seek to minimize the total billed cost of all MoE layers. The billed cost of the gating network can be ignored here, as it affects the cost of all MoE layers little: the memory size of serverless functions for gating networks does not depend on expert popularity, and the impact of scatter-gather communication methods on the gating network's execution time is also reflected in the experts. Hence, we consider the total billed cost of all MoE layers to be the billed cost of all experts in MoE layers. The billed cost of MoE layer e is then given by  $c_e = (a_e - 2)(a_e - 3)c_{1,e} + (a_e - 1)(a_e - 3)c_{2,e} + (a_e - 1)(a_e - 2)c_{3,e}$ , where the billed cost of MoE layer e under the communication method  $a_e$  is:

$$c_{a_e,e} = \sum_{i \in \mathbb{N}_-} s_{e,i} t_{a_e,e,i} \sum_{i=1}^M x_{e,i,j} \mathbb{M}_j, \forall e \in \mathbb{E}, \forall a_e \in \mathbb{A}, \quad (4)$$

Here  $\mathbb{M}_j$  represents the *j*-th memory size in set  $\mathbb{M}$ , and  $t_{a_e,e,i}$  is the total execution time of all replicas of the expert:

<span id="page-6-4"></span>
$$t_{a_e,e,i} = \sum_{a=1}^{G} y_{e,i,g} gt_{a_e,e,i}^{rep}, \forall e \in \mathbb{E}, \forall i \in \mathbb{N}_e, \forall a_e \in \mathbb{A},$$
 (5)

where  $t_{a_e,e,i}^{rep}$  denotes the execution time of one replica of the expert.  $t_{a_e,e,i}^{rep}$  is related to the scatter-gather communication method chosen. We give formulars of three cases in obtaining  $t_{a_e,e,i}^{rep}$ .

(1) Pipelined indirect transfer ( $a_e = 1$ ). We have

$$t_{1,e,i}^{rep} = T_{e,i}^{h,E} + t_{e,i}^{nblk} + \beta t_{e,i}^{blk}, \forall e \in \mathbb{E}, \forall i \in \mathbb{N}_e,$$
 (6)

where the head time  $T_{e,i}^{h,E}=\frac{P_{e,i}}{B^s}+T^{dl}+T^{str}$  consists of warm start time  $T^{str}$ , the access delay to external storage  $T^{dl}$  and the model download time  $\frac{P_{e,i}}{B^s}+T^{dl}$  with bandwidth  $B^s$  between external storage and serverless function and the parameter size  $P_{e,i}$  of the expert.  $t_{e,i}^{nblk}=T^{dl}+\lceil\frac{r_{e,i}}{\beta}\rceil(\frac{D^o}{B^s})$  includes the time to upload the last minibatch to external storage and  $D^o$  is the size of the processed result of one token by an expert.  $t_{1,e,i}^{blk}=T^{dl}+\beta\max\{\frac{D^{in}}{B^s}+t_{e,i}^{cal},\frac{D^o}{B^s}\}$  denotes one worst-case block time.

The latency, from the earlist time point when expert serverless functions start or the gating network starts to transfer each expert's input, to the latest time when the next non-MoE layer finishes downloading the processed results of all experts from external storage or finishes downloading model parameters with direct transfer, is referred to as MoE-E2E latency  $t_{1,e}^{lat}$ :

$$t_{1,e}^{lat} = \max\{t_{1,e}^{S12}, T_e^{load}\} + t_{1,e}^{S3}, \forall e \in \mathbb{E},\tag{7}$$

where  $T_e^{load}$  includes the time to start the serverless function of non-MoE layer and download the model parameters.

These formulars are derived based on Fig. 8(a) and details omitted due to space limit.

(2) Non-pipelined indirect transfer ( $a_e = 2$ ). We have

$$t_{2,e,i}^{rep} = T_{e,i}^{h,E} + 2T^{dl} + t_{2,e,i}^{data}, \forall e \in \mathbb{E}, \forall i \in \mathbb{N}_e, \tag{8}$$

where  $t_{2,e,i}^{data} = r_{e,i} \left( \frac{D^{in} + D^o}{B^s} + t_{e,i}^{cal} \right)$ .

The MoE-E2E latency is:

$$t_{2,e}^{lat} = \max\{t_{2,e}^{S12}, T_e^{load}\} + t_{2,e}^{S3}, \forall e \in \mathbb{E}.$$
 (9)

These formulars are derived based on Fig. 8(b) and details omitted due to space limit

(3) Direct transfer ( $a_e = 3$ ). We have:

$$t_{3,e,i}^{rep} = T_{e,i}^{h,E} + r_{e,i} (\frac{D^o}{R^f} + t_{e,i}^{cal}), \forall e \in \mathbb{E}, \forall i \in \mathbb{N}_e,$$
 (10)

where  $B^f$  is the bandwidth between serverless functions. The MoE-E2E latency  $t_{3,e}^{lat}$  is:

<span id="page-7-0"></span>
$$t_{3,e}^{lat} = \frac{r_{e,i}D^{in}}{B^f} + \max_{i \in \mathbb{N}_e} \{t_{3,e,i}^{rep}\} + T_e^{load}, \forall e \in \mathbb{E}.$$
 (11

The MoE-E2E latency at MoE layer e is given by  $t_e^{lat} =$  $(a_e-2)(a_e-3)t_{1,e}^{lat}+(a_e-1)(a_e-3)t_{2,e}^{lat}+(a_e-1)(a_e-2)t_{3,e}^{lat}$ Optimal MoE deployment problem: We formulate optimal deployment of MoE model inference in a serverless platform to minimize the billed cost of all MoE layers (i.e.,  $\sum_{e \in \mathbb{E}} c_e$ ), by jointly deciding the communication method (i.e.,  $a_e$ ), selecting memory size configurations (i.e.  $x_{e,i}$ ), deciding expert replication (i.e.  $y_{e,i}$ ), and setting parameter  $\beta$  for pipelined scatter-gather communication.

<span id="page-7-5"></span>
$$\min \sum_{e \in \mathbb{F}} c_e \tag{12a}$$

subject to 
$$5-11$$
, (12b)

$$P_{e,i} + M_{e,i}^{itrm} + r_{e,i}(D^{in} + D^{o})$$

$$\leq \sum_{j=1}^{|\mathbb{M}|} x_{e,i,j} \mathbb{M}_j, \forall e \in \mathbb{E}, \forall i \in \mathbb{N}_e$$

$$T^{head} + T^{tail} + \sum_{e \in \mathbb{E}} (t_e^{lat} + T_e^{NE}) \leq T^{limit},$$
(12d)

$$T^{head} + T^{tail} + \sum_{e \in \mathbb{F}} (t_e^{lat} + T_e^{NE}) \le T^{limit}, \tag{12d}$$

$$1 \le \beta \le \max_{i \in \mathbb{N}_e, e \in \mathbb{E}} r_{e,i},\tag{12e}$$

$$(a_e - 3)(r_{e,i}D^{in} - D^p) \le 0, \forall e \in \mathbb{E}, \forall i \in \mathbb{N}_e, \forall a_e \in \mathbb{A}, \quad (12f)$$

$$\sum_{j=1}^{|\mathbb{M}|} x_{e,i,j} = 1, \forall e \in \mathbb{E}, \forall i \in \mathbb{N}_e,$$
(12g)

$$\sum_{g=1}^{G} y_{e,i,g} = 1, \forall e \in \mathbb{E}, \forall i \in \mathbb{N}_e,$$
(12h)

$$x_{e,i,j} \in \{0,1\}, \forall e \in \mathbb{E}, \forall i \in \mathbb{N}_e, \forall j \in \mathbb{M},$$
 (12i)

$$y_{e,i,g} \in \{0,1\}, \forall e \in \mathbb{E}, \forall i \in \mathbb{N}_e, \forall g \in \mathbb{G}_{e,i},$$
 (12j)

$$\beta \in \mathbb{Z},$$
 (12k)

Here  $M_{e,i}^{itrm}$  is the memory size of intermediate results during an expert's inference.  $T^{limit}$  is the time limit of end-toend MoE model inference (i.e., serving SLO).  $T^{head}$  is the execution time of the serverless function of the first non-MoE layer.  $T^{tail}$  is the execution time of the serverless function of the last non-MoE layer, excluding the time to receive data from the last MoE layer or download them from external storage.  $T_e^{NE}$  is the processing time of non-MoE layer e with the subsequent gating network. (12c) specifies the memory limit of each serverless function, (12d) gives the end-to-end inference time target of the MoE model, (12e) limits the maximal number of tokens in each block in calculating the

worst-case total time, and (12f) prohibits direct transfers when the payload size is below transferred data size between the gating network and each expert, as well as between each expert and the next non-MoE layer.

#### IV. THE BO FRAMEWORK

We now present our BO algorithm for the BO framework to learn expert selection predictions and optimize MoE model deployment, together with an efficient algorithm to solve the optimal MoE model deployment by the policy maker.

#### A. Optimal MoE Deployment Algorithm

Given expert selection results, the optimal MoE deployment problem in (12) is a MIQCP problem with a non-linear objective in (12a), quadratical constraints in (12c) and (12f), binary variables x and y, and integer variable  $\beta$  and a. MIQCP problems are in general NP-hard [37] because of their nonlinearity and discretized variables. We design an efficient algorithm to solve it approximately.

<span id="page-7-6"></span>We first divide problem (12) into three cases by fixing the scatter-gather communication method:  $a_e = 1, \forall e \in \mathbb{E};$  $a_e = 2, \forall e \in \mathbb{E}; \text{ and } a_e = 3, \forall e \in \mathbb{E}.$  We then linearize max functions in (12a), (12d) and (12e) by adding auxilliary variables  $(\max_{h \in \mathbb{H}} \{h\} \text{ can be linearized as } \phi \geq h, \forall h \in \mathbb{H}).$ Then we solve each resulting MIQCP by a solver [38], respectively, and obtain costs  $c_{1,e}$ ,  $\forall e \in \mathbb{E}$ ,  $c_{2,e}$ ,  $\forall e \in \mathbb{E}$ , and  $c_{3,e}, \forall e \in \mathbb{E}$ , from the three solutions. Based on these solutions, we design an Optimal Deployment Selection (ODS) algorithm to decide  $a_e$  for each MoE layer, as given in Alg 1.

<span id="page-7-4"></span><span id="page-7-3"></span><span id="page-7-2"></span><span id="page-7-1"></span>For each MoE layer e, we select the communication method  $\hat{a}_e$  with the lowest cost, set all experts to use the same method, and then calculate the MoE-E2E latency  $\hat{t}_{\hat{a}_e,e}^{lat}$  (lines 4-7). If the new MoE-E2E latency satisfies the end-to-end inference time constraint, the optimal deployment policy for the MoE model is obtained; otherwise, we identify the MoE layer  $\tilde{e}$ with the highest latency, set the cost of the corresponding scatter-gather communication  $\tilde{a}_{\tilde{e}}$  to infinity (lines 10-12), and then iteratively decide  $\hat{a}_e$  for each MoE layer (lines 3-17). At most  $2|\mathbb{E}|$  iterations are needed, as three communication methods provide up to  $3|\mathbb{E}|$  solutions of  $\hat{a}_e$ ,  $\forall e \in \mathbb{E}$ , and selecting  $|\mathbb{E}|$  solutions excludes up to  $2|\mathbb{E}|$  other solutions. If all costs  $c_{a_e,e}$  become infinity, it implies that mixing different communication methods across different MoE layers do not work. In this case, we return the optimal deployment policy with the lowest cost with all MoE layers using the same scatter-gather communication method (lines 18-20).

## B. BO algorithm

BO is a statistical approach for global optimization of a black-box function, including an objective, variables, a surrogate function to simulate the objective, and an acquisition method to update the variables. BO makes no assumptions about the underlying function format and aims to minimize the number of trials to find a near-optimal solution, reducing tuning costs. In our BO algorithm, the black box function corresponds to the billed cost of all MoE layers for MoE inference

**Algorithm 1:** Optimal Deployment Selection Algorithm of an MoE model (ODS)

```
Input: Optimal solutions x_{a,e,i}, y_{a,e,i}, \beta and optimal objective
       c_{a,e}, \forall e \in \mathbb{E}, \forall i \in \mathbb{N}_e of solving three MIQCP problems with
       fixed a, \forall a \in \mathbb{A}
Output: \hat{a}_e, \beta, x_{\hat{a}_e,e,i}, y_{\hat{a}_e,e,i}, \forall e \in \mathbb{E}, \forall i \in \mathbb{N}_e
  1: itr=0.
 2: while itr\leq 2|\mathbb{E}| do
 3:
            for e \in \mathbb{E} do
                  for i \in \mathbb{N}_e do
 4:
                      \hat{a}_e = argmin_{a_e \in \mathbb{A}}[c_{1,e}, c_{2,e}, c_{3,e}];
 5:
                       Calculate the MoE-E2E latency \hat{t}_{\hat{a}_e,e}^{lat}.
  6:
  7:
 8:
            if T^{head} + T^{tail} + \sum_{\substack{e \in \mathbb{E} \\ \hat{a}_e, e}} (\hat{t}^{lat}_{\hat{a}_e, e} + T^{NE}_e) > T^{limit} then
 9:
                  \tilde{a}_{\tilde{e}}, \tilde{e} = argmin_{e \in \mathbb{E}} \hat{t}_{\hat{a}_{e}, e}^{lat};
10:
11:
12:
            else
                 return \hat{a}_e, x_{\hat{a}_e,e,i}, y_{\hat{a}_e,e,i}, \beta, \forall e \in \mathbb{E}, \forall i \in \mathbb{N}_e.
13:
14:
            end if
            itr+=1.
15:
16: end while
17: if itr> 2|\mathbb{E}| then
            \hat{a} = argmin_{a \in \mathbb{A}} \sum_{e \in \mathbb{E}} c_{a,e}
18:
19:
            return \hat{a}, x_{\hat{a},e,i}, y_{\hat{a},e,i}, \beta, \forall e \in \mathbb{E}, \forall i \in \mathbb{N}_e.
20: end if
```

in a serverless platform. Each trial corresponds to one BO iteration which adjusts the key-value table, recomputes expert selection predictions and distributed MoE model deployment.

The objective of our BO is to minimize the billed cost of all MoE layers. The variables are Q key-value pairs in the key-value table  $\Omega(\cdot)$  for adjusting expert selection prediction. The surrogate function uses a Gaussian process to simulate the cost of all MoE layers deployed by the policy maker based on expert selection prediction. For the acquisition method, we design a decaying multi-dimensional  $\epsilon$ -greedy search (GS) to set the variables for the next BO iteration. Traditional single-dimension  $\epsilon$ -GS, as an acquisition function, decays with each iteration and balances exploration of BO variables and exploitation of high-performing BO variable values by selecting the best variable with probability  $1-\epsilon$ and exploring new variable values with probability  $\epsilon$  [39]. However, as multiple key-value pairs in key-value table need to be adjusted together in a single BO trial, these pairs can be viewed as multi-dimensional variables in our BO, which make a single-dimension  $\epsilon$  insufficient to balance exploration and exploitation in all dimensions. We extend  $\epsilon$  to a multidimensional vector  $\epsilon \in \mathbb{R}^Q$ .

In BO learning, our objective, billed cost of all MoE layers, is obtained on several batches of inference data from an open-source real-world dataset [31], [40], [41] (different datasets can be used for different MoE inference tasks). When inaccurate expert selection prediction occurs compared to the ground truth in the profiled data, we set a limited range of key-value pairs to update as  $\mathbb{L}$ , where  $\mathbb{L}$  includes all positive integers for values and all possible token-to-expert mappings with token IDs limited to those present in these batches of data

**Algorithm 2:** Bayesian Optimization Algorithm with Multiple-dimension  $\epsilon$ -Greedy Search

```
Input: MoE model, dataset table \Omega_0, normal range of key-value
        pairs \mathbb{P}, and constants Q, \mu, \alpha, \rho, \rho_1, \rho_2, \rho_3, \lambda, \zeta.
Output: optimal key-value pairs \{\hat{\mathbf{z}}_q, \hat{v}_q\}_{q \in \{1,...,Q\}}
  1: Initialize Q key-value pairs \{\mathbf{z}_{0,q}, v_{0,q}\}_{\forall q \in \{1,...,Q\}} for
        key-value table, \epsilon_0 \in \mathbb{R}^Q for \epsilon-GS, limited range of key-value
        pairs \mathbb{L}, BO historical set \mathbb{B}_0 = \{\}, and the BO trial index
        \tau = 1.
  2: repeat
              \epsilon_{\tau} = \frac{\epsilon_0}{1+\rho\tau}.
  3:
  4:
              \Omega_{\tau}(\mathbf{z}_{\tau-1,q}) = v_{\tau-1,q}, \forall q \in \{1,\ldots,Q\}.
              \ddot{i}_e = argmax_{i \in \mathbb{N}_e} \mathcal{P}(\mathbb{N}_{e,i} | \mathbf{f}'_1), \forall e \in \mathbb{E} \text{ with } \Omega_\tau.
  5:
              c_{\tau,a,e} from three MIQCP solvers, \forall e \in \mathbb{E}, \forall a \in \mathbb{A}.
  6:
              (\hat{a}_e, x_{\hat{a}_e,e,i}, y_{\hat{a}_e,e,i}, \beta)_{\tau} = ODS(c_{\tau,a,e}), \forall a \in \mathbb{A}, \forall e \in \mathbb{E},
  8:
              for j = 1, \ldots, J do
                    for e \in \mathbb{E} do
  9:
10:
                          for i \in \mathbb{N}_e do
                              \begin{aligned} & \textbf{if } |r_{e,i} - R_{e,i}^{real}| > \alpha \\ & \textbf{Append } \mathbf{f}_{j,1}^{\prime} \textbf{ to } \mathbb{L}_{\tau}. \end{aligned}
11:
12:
                                    Append \mathbf{I}_{j,1} to \mathbb{L}_{\tau}.
\nif M^{real} \geq \sum_{j=1}^{|\mathbb{M}|} x_{\tau,\hat{a}_e,e,i,j} \mathbb{M}_j then
\rho' = \rho_1, \, n_{e,i}^{new} = \lceil \frac{M^{real}}{\sum_{j=1}^{|\mathbb{M}|} x_{\tau,\hat{a}_e,e,i,j} \mathbb{M}_j}\nelse if \hat{a}_{\tau,e} = 3 and R_{e,i}^{real} > D^p then
\rho' = \rho_2, \, n_{e,i}^{new} = \lceil \frac{R_{e,i}^{real}}{D^p} \rceil.
13:
14:
15:
16:
17:
                                           \rho' = \rho_3, \, n_{e,i}^{new} = 1.
18:
19.
                                     \epsilon_{\tau,1:\mu Q} = (1 + \rho'\tau)\epsilon_{\tau,1:\mu Q}.
20:
                                     Replicate expert i n_{e,i}^{new} times.
21:
22:
                                end if
23:
                          end for
24:
                    end for
                    Deploy the MoE model using (\hat{a}_e, x_{\hat{a}_e,e,i}, y_{\hat{a}_e,e,i}, \beta)_{\tau};
25:
26:
                    c_{\tau,i} = MoE_{\tau}(input_i).
27:
              end for
              c_{\tau} = \frac{1}{J} \sum_{j=1}^{J} c_{\tau,j}.
28:
              \mathbb{B}_{\tau} \leftarrow \mathbb{B}_{\tau-1} \cup (\{\mathbf{z}_{\tau-1,q}, v_{\tau-1,q}\}_{q \in \{1,\dots,Q\}}, c_{\tau}).
29:
30:
              \mathbf{z}_{\tau,1:\mu Q}, v_{\tau,1:\mu Q} \leftarrow \epsilon_{\tau,1:\mu Q} - \mathsf{GS} \text{ over } \mathbb{B}_{\tau}, \mathbb{L}_{\tau}.
31:
              \mathbf{z}_{\tau,\mu Q+1:Q}, v_{\tau,\mu Q+1:Q} \leftarrow \epsilon_{\tau,1:\mu Q+1:Q} - \mathsf{GS} \text{ over } \mathbb{B}_{\tau}, \mathbb{P}.
              \tau = \tau + 1, initialize \mathbb{L}.
       until The change of \min_{\tau \in 1, ..., \tau} c_{\tau} within \lambda consecutive
        iterations is below \zeta, and record the last iteration index as \eta
34: return \{\hat{\mathbf{z}}_q, \hat{v}_q\}_{q \in \{1,...,Q\}} = argmin_{\tau \in \{1,...,\eta\}} c_{\tau}.
```

for keys. We set a ratio  $\mu$ , and slow down the decay of the split  $\epsilon_{1:\mu Q}$  of vector  $\epsilon$  from the first dimension to the  $\mu Q$ -th dimension by multiplying  $\epsilon_{1:\mu Q}$  with a factor greater than 1. The 1st to  $\mu Q$ -th key-value pairs in our key-value table are then updated using  $\epsilon_{1:\mu Q}$  by adjusting the values of keys in  $\mathbb{L}$ , allowing for more exploration on current low-performing key-value pairs. Meanwhile, the  $\mu Q+1$ -th through the last key-value pairs in the variables are updated by either adjusting the values of keys in  $\mathbb{P}$  or by creating new key-value pairs in  $\mathbb{P}$  using  $\epsilon_{\mu Q+1:Q}$ . Here,  $\mathbb{P}$  is a normal range of key-value pairs to adjust, includeing all positive integers for values, and all possible token-to-expert mappings for keys, where token features can include all possible token IDs assigned by the tokenizer, all possible position IDs, and all possible attention

IDs, and experts are represented by all possible expert indices. This enriches the key-value table by creating new key-value pairs with keys as token-to-expert mappings not present in the profiled data.

The BO algorithm with multi-dimensional  $\epsilon$ -GS is given in Alg. 2. In the  $\tau$ -th BO iteration,  $\epsilon$  decays by being divided by  $1 + \rho \tau$  with a constant  $\rho > 0$  (line 3). The dataset table  $\Omega_{\tau}$  is updated with key-value pairs  $\{\mathbf{z}_{\tau-1,q},v_{\tau-1,q}\}_{\forall q\in\{1,\dots,Q\}}$  (line 4). where  $\mathbf{z} = \{\mathbf{f}, e, i\}$ . Then expert selection is predicted (line 5). The policy maker produces the optimal deployment policy using the ODS algorithm (lines 6-7). For the j-th batch  $input_j$ in BO learning(lines 8-27), at expert i in MoE layer e, if predicted counts  $r^{e,i}$  and real counts  $R^{real}_{e,i}$  of tokens assigned to one replica of this expert exceeds a constant  $\alpha > 0$ , token IDs  $\mathbf{f}'_{i,0}$  of the j-th batch are recorded for limiting the range of key-value pairs to adjust (line 12) and three cases are discussed (lines 10-21): (i) if the minimal memory  $M^{real}$  required by real expert popularityexceeds memory configuration of serverless functions,  $\rho_1 < \rho$  is used to decrease the decay rate  $\epsilon_{\tau,1:\mu Q}$ , and  $n_{e,i}^{new}$  is calculated to replicate expert i  $n_{e,i}^{new}$  times to satisfy the minimal memory  $M^{real}$  (lines 13-14, 20-21); (ii) if the size of transferred tokens exceeds the payload size under direct transfer,  $\rho_2 < \rho_1$  is used to decrease the decay rate  $\epsilon_{\tau,1:\mu Q}$ , and  $n_{e,i}^{new}$  is calculated to replicate expert i  $n_{e,i}^{new}$ times to ensure that the data size transferred to each replica does not exceed the payload size  $D^p$  (lines 15-16, 20-21); (iii) if all constraints in (12) are satisfied,  $\rho_3 < \rho_2$  is used to decrease the decay rate  $\epsilon_{\tau,1:\mu Q}$ , and we do not replicate expert i to avoid increasing cost (lines 17-18, 20-21). Then the billed cost of all MoE layers  $c_{\tau,j}$  is computed on the j-th batch data  $input_i$  on the derived MoE model deployment  $MoE_{\tau}$  (lines 25, 26, 28). Next the historical set  $\mathbb{B}$  in BO to record variables and objectives is updated (line 29), and the key-value pairs to adjust is updated by  $\epsilon$ -GS over the historical set and the range of key-value pairs  $\mathbb{P}$  and  $\mathbb{L}$ (lines 30-31). Then BO iterations repeat until the change of the minimal billed cost of all MoE layers within  $\lambda$  consecutive iterations is below the threshold  $\zeta$  (line 33).

## C. Theoretical Analysis

**Theorem 1.** Alg. I produces feasible MoE deployment in  $O(|\mathbb{E}|)$  time, which achieves a billed cost of all MoE layers upper bounded by a constant ratio of the cost of optimal solutions of (12).

**Theorem 2.** Alg. 2 converges when the BO iteration index satisfies  $\tau > \frac{1+\rho}{\rho-\rho_1}(1-\frac{\delta}{\max_{q\in\{1,\ldots,Q\}}\epsilon_{0,q}})$  with an abitrary small positive constant  $\delta < \max_{q\in\{1,\ldots,Q\}\epsilon_{0,q}}$ .

Theorem 1 indicates that the time complexity of Alg. 1 scales linearly with the number of MoE layers. Theorem 2 shows that the Alg. 2 converges in a constant bound. The sketch of proods are as follows.

Sketch of Proof 1: In theorem 1, the time complexity is analyzed in this section. We prove that the billed cost by the optimal solutions **OPT** of (12) is lower-bounded by the ideal **OPT\_LB**. as  $\sum_{e \in \mathbb{E}} \min_{a_e \in \mathbb{A}} c_{a_e,e}$ . On the other hand, the billed cost achieved by Alg. 1 (i.e., **ALG**) is upper bounded by

![](_page_9_Figure_7.jpeg)

<span id="page-9-0"></span>Fig. 10. Average difference per expert between real and predicted expert selection numbers under different MoE models, datasets and tasks.

 $\begin{array}{lll} \mathbf{ALG\_UP} & \text{as } \sum_{e \in \mathbb{E}} \max_{a_e \in \mathbb{A}} c_{a_e,e}. \text{ From Sec III-D, } c_{a_e,e} > \\ \sum_{i \in \mathbb{N}_e} \{T_{e,i}^{h,E}\} & + \sum_{i \in \mathbb{N}_e} \{d_{e,i}\}(U_{|\mathbb{M}|} + \min\{1/B_s, 1/B_f\}), \\ \text{and } c_{a_e,e} & < \mathbb{M}_{|\mathbb{M}|}G(\sum_{i \in \mathbb{N}_e} \{T_{e,i}^{h,E}\} + \sum_{i \in \mathbb{N}_e} \{d_{e,i}\}(U_1 + \max\{1/B_s, 1/B_f\} + T^{dl}). \end{array} \\ \text{Here the known head time } T_{e,i}^{h,E} \text{ as the first term, the number of tokens multiply the known unit calculation and transfer time as the second term, and } \mathbb{M}_{|\mathbb{M}|} \text{ and } G \text{ denotes the known maximal memory size and number of replicas.} \\ \text{Then } \mathbf{ALG/OPT} \leq \mathbf{ALG\_UP/OPT\_LB} \leq \mathbb{M}_{|\mathbb{M}|}G(U_1 + \max\{1/B_s, 1/B_f\} + T^{dl})/(U_{|\mathbb{M}|} + \min\{1/B_s, 1/B_f\}). \\ \text{Proved.} \end{array}$ 

Sketch of Proof 2: In theorem 2, when  $\epsilon$  decays below an arbitrary small positive constant  $\delta < \max_{q \in \{1, \dots, Q\} \epsilon_{0,q}}$ , we consider the variables achieving the best objective in BO historical set are always selected by GS. As  $\epsilon$  decays at the slowest rate  $(1+\rho_1\tau)/(1+\rho\tau)$ , we set  $\max_{q \in \{1, \dots, Q\} \epsilon_{0,q}} (1+\rho_1\tau)/(1+\rho\tau) < \delta$ . Proved.

#### V. EVALUATION

#### A. Experimental Setup

**Testbed.** We run our experiments on AWS Lambda [4]. To build MoE layer images, we use a Dockerfile to define the environment with Python 3.8 and include packages such as *torch* and *transformers*. We implement the BO algorithm with package *optuna* [42] and MIQCP solvers with package *gurobi* [38]. We use two S3 buckets of size 512MB each for external storage. We adopt 14 discrete memory size configurations for each serverless function: [128, 768, 960, 1152, 1344, 1536, 1728, 1920, 2112, 2304, 2496, 2688, 2880, 3072] MB. We set maximal possible expert replica number as 8.

**MoE Models.** Three common transformer-based dense language models are converted to MoE models with all MLP layers after attention layers converted to MoE layers and a gating network of a linear layer:

- Bert [43]: a 12-layer encoder model with 110 million parameters, converted to 12 MoE layers, with each MoE layer having 4, 8, or 16 experts;
- GPT2 [44]: a 12-layer decoder model with 1.5 billion parameters, converted to 12 MoE layers, with each MoE layer having 4 experts;
- Bert2Bert [45]: a 12-layer encoder-decoder model with 247 million parameters, converted to 24 MoE layers, with each MoE layer having 4 experts.

We run the fill-mask task [46] on Enwik8 [31] and CCnews [47] datasets and the translation task [48] on the Wmt19 [41] dataset for the BERT model. We conduct the text generation task on the Enwik8 and Lambda [40] datasets for the GPT-2 model and on the Enwik8 dataset for Bert2Bert model.

![](_page_10_Figure_0.jpeg)

<span id="page-10-0"></span>Fig. 11. Billed cost and inference throughput of MoE layers with different scatter-gather communication methods on AWS Lambda.

![](_page_10_Figure_2.jpeg)

<span id="page-10-1"></span>Fig. 12. Billed cost of all MoE layers with different MoE deployment algorithms on AWS Lambda.

![](_page_10_Figure_4.jpeg)

<span id="page-10-2"></span>Fig. 13. Ratio of billed cost and expert prediction difference optimized by BO with different acquisition functions over no BO.

![](_page_10_Figure_6.jpeg)

<span id="page-10-3"></span>Fig. 14. Billed cost of all MoE layers and inverse of throughput under different expert selection distributions on AWS Lambda and CPU clusters.

#### B. Expert selection prediction

We first evaluate accuracy of expert selection prediction learned by our BO framework, by calculating the average absolute difference per expert between the real and predicted counts of tokens assigned to each expert. Fig. 10 shows the difference across different MoE models, datasets, and tasks. For a task on a dataset, we use 95% of this dataset for profiling and evaluate the difference on 10,240 tokens in the dataset. Basic Bert MoE represents the Bert MoE model with 4 experts per MoE layer and top-1 routing for the fill-mask task on Enwiki8. The basic GPT2 MoE and the basic Bert2Bert MoE cases are similar. Other cases are variations based on these basic setups, e.g, GPT2 Lambda denotes the GPT2 basic setup but changes the dataset to Lambda [40] dataset. Across all cases, our method outperforms expert prediction in Lina [15], as Lina only uses token ID as token feature while our method incorporates token ID, position ID, and attention ID to capture additional information for more accurate profiled probabilities. Compared to top-1 routing, the results of top-2 routing show that increasing the value of k in top-k gating significantly improves the prediction accuracy, as routing to more experts allows prediction mistakes in one expert to be corrected by the other. When the number of experts increases, the average prediction difference decreases.

#### C. Scatter-gather communication

We next evaluate performance of our different scatter-gather communication designs. We allocate 3008MB of memory to each serverless function and use no expert replicas. Fig. 11 shows the billed cost of MoE layers and throughput of the entire MoE model under different communication methods. The results verify that the optimal scatter-gather communication method varies depending on the number of tokens. For 256 tokens, direct transfer performs best for both Bert MoE and GPT2 MoE. As the number of tokens increases, either pipelined or non-pipelined indirect transfers may perform better, while direct transfer becomes impractical due to payload size limitations; throughput increases because the time for model downloading and serverless function warm-up is distributed over more tokens.

# D. ODS algorithm

We deploy MoE model for inference using 10,240 tokens on AWS Lambda. Fig. 12 illustrates the billed cost of all MoE layers using our ODS algorithm, an MIQCP method and a random selection method, under different inference throughput targets. The MIQCP method uses one MIQCP solver to directly solve (12), and the random method randomly selects the communication method at each MoE layer. We set the target throughput by dividing 10240 tokens by the end-toend latency limit specified in (12). The time limit for searching the optimal solution using the MIQCP approach is set to 180s; for the ODS algorithm with three MIQCP solvers, the search time limit is set to 60s. Our ODS algorithm outperforms other methods. At higher target throughputs, the MIQCP method fails to derive an optimal solution within 180s.

## *E. BO algorithm*

For BO learning, we use 10,240 tokens from the Enwiki8 dataset to simulate the inference requests, set Q = 1000 keyvalue pairs to update in each BO iteration.After BO learning, we test on 100 batches of inference requests from the dataset with batch size of 10240 tokens. Due to the long deployment time on AWS Lambda, we use simulation for this set of evaluation.

Fig. [13](#page-10-2) shows the ratio of the billed cost and expert prediction differences optimized by BO with different acquisition functions, to that of no BO, respectively. No BO means that we do not use the BO algorithm to adjust our expert predictor, and the expert selection is predicted by the unadjusted predictor. The random method randomly adjusts key-value pairs, the single ϵ-greedy sampler uses the same ϵ for all dimensions of the variables, and the TPE [\[49\]](#page-12-47) method samples on promising regions of variable range based on probabilistic modeling. Our multi-dimension ϵ-GS performs best in terms of the billed cost for both Bert MoE and GPT2 MoE models. For BERT MoE, our method achieves the highest expert prediction accuracy. For GPT-2 MoE, our algorithm does not always yield the lowest prediction difference, possibly due to the increase of the number of tokens predicted for the most popular experts: our BO algorithm sets a larger ϵ for key-value pairs where memory is insufficient due to underestimation of expert popularity, resulting in an overestimation of the most popular experts that are most likely to exceed memory limits.

## *F. Algorithm overhead*

We dissect the execution time of the expert selection predictor, the ODS algorithm and the BO algorithm. For expert selection predictor, the time of profiling 100 batches of data is around 28.89 seconds, and the prediction time on 10 batches is around 20.31 seconds. The execution time of the ODS algorithm with three MIQCP solvers is around 2.27s. Our BO algorithm requires around 62.15s per iteration and converges in around 1257.89s.

## *G. Overall performance*

We deploy MoE models on AWS Lambda and a CPU cluster for inference serving of 10,240 tokens. The CPU cluster consists of two 64-core AMD EPYC CPUs with 512GB of DRAM.

Fig. [14](#page-10-3) compares the billed cost of all MoE layers and inference throughput of the entire MoE model under different expert selection distributions (regarding the count of tokens assigned to each expert), among: (1) *Serverless with predicted distribution optimized by BO:* the optimal MoE deployement produced by our BO framework; (2) *Serverless with real expert selection distribution:* the optimal MoE deployment produced based on ground truth of expert selections in the MoE inference; (3) *Serverless with predicted distribution without BO*: the optimal MoE deployment produced using predicted expert selections that is not adjusted by the BO algorithm; (4) *LamdaML [\[20\]](#page-12-18)*, which uses the maximum memory allocation for each serverless function on AWS Lambda (3008MB) for inference serving, requires no expert prediction, and uses no replicas for each expert; (5) *CPU:* the MoE model is deployed in the CPU cluster, with all experts at each MoE layer executing concurrently, requiring no expert prediction; (6) *CPU betterTransformer:* MoE deployment in the CPU cluster accelerated by the CPU inference optimization method betterTransformer [\[50\]](#page-12-48), through sparsity and fused kernels as Flash Attention.

For both Bert MoE and GPT2 MoE, our serverless MoE inference design consistently results in lower billed costs, as compared to MoE inference on the CPU cluster. Specifically, serverless MoE inference with predicted expert selection reduces the billed cost by at least 75.67% compared to CPU cluster-based serving. The throughput in serverless-based MoE serving remains significantly above human reading levels of 3.3 tokens per second. The lower throughput in serverless MoE serving compared to CPU cluster-based serving is mainly because non-MoE layer computation is limited to the 3008MB memory size of each serverless function, which is far less than the 512GB available in a common CPU cluster. Among serverless options, the predicted expert selection distribution optimized by BO outperforms both non-BO methods and over-provisioning with LambdaML. The BO-optimized expert distribution not only reduces the billed cost by at least 43.41% compared to LambdaML with at most an 18.76% decrease in throughput, but also closely aligns with the cost of deployment using the real expert distribution.

# VI. CONCLUSION

This paper studies optimized MoE model deployment and distributed inference serving on a serverless platform, that effectively predicts expert selection, schedules communication with model execution, and minimizes the overall billed cost of serving MoE models. We propose a Bayesian optimization framework with multi-dimensional ϵ-greedy search to learn expert selections and optimal MoE deployment. We design a novel Bayesian decision-making approach for expert selection prediction, propose several scatter-gather communication designs for serverless platforms, and design an ODS algorithm to decide optimal deployment of distributed MoE inference on a serverless platform. Experiments on AWS lambda validate our designs in reducing the billed cost of all MoE layers by at least 75.67% compared to CPU clusters while maintaining satisfactory inference throughput. As compared to LambdaML in serverless computing, our designs achieve 43.41% lower cost with a throughput decrease of at most 18.76%.

# ACKNOWLEDGMENT

This work was supported in part by grants from Hong Kong RGC under the contracts C7004-22G (CRF), C6015- 23G (CRF), 17204423 (GRF), 16210822 (GRF) and T43- 513/23-N (TRS).

## REFERENCES

<span id="page-11-0"></span>[1] C. Jin, Z. Zhang, X. Xiang, S. Zou, G. Huang, X. Liu, and X. Jin, "Ditto: Efficient serverless analytics with elastic parallelism," in *Proceedings of the ACM SIGCOMM 2023 Conference*, 2023, pp. 406–419.

- <span id="page-12-0"></span>[2] H. Zhang, Y. Tang, A. Khandelwal, J. Chen, and I. Stoica, "Caerus:{NIMBLE} task scheduling for serverless analytics," in *18th USENIX Symposium on Networked Systems Design and Implementation (NSDI 21)*, 2021, pp. 653–669.
- <span id="page-12-1"></span>[3] M. Yu, Z. Jiang, H. C. Ng, W. Wang, R. Chen, and B. Li, "Gillis: Serving large neural networks in serverless functions with automatic model partitioning," in *2021 IEEE 41st International Conference on Distributed Computing Systems (ICDCS)*. IEEE, 2021, pp. 138–148.
- <span id="page-12-2"></span>[4] Aws lambda. [Online]. Available: <https://aws.amazon.com/lambda/>
- <span id="page-12-3"></span>[5] Google cloud functions. [Online]. Available: [https://cloud.google.com/](https://cloud.google.com/functions/) [functions/](https://cloud.google.com/functions/)
- <span id="page-12-4"></span>[6] J. Li, L. Zhao, Y. Yang, K. Zhan, and K. Li, "Tetris: Memory-efficient serverless inference through tensor sharing," in *2022 USENIX Annual Technical Conference (USENIX ATC 22)*, 2022.
- <span id="page-12-5"></span>[7] A. Mampage, S. Karunasekera, and R. Buyya, "A holistic view on resource management in serverless computing environments: Taxonomy and future directions," *ACM Computing Surveys (CSUR)*, vol. 54, no. 11s, pp. 1–36, 2022.
- <span id="page-12-6"></span>[8] S. Kounev, N. Herbst, C. L. Abad, A. Iosup, I. Foster, P. Shenoy, O. Rana, and A. A. Chien, "Serverless computing: What it is, and what it is not?" *Communications of the ACM*, vol. 66, no. 9, pp. 80–92, 2023.
- <span id="page-12-7"></span>[9] Microsoft azure. [Online]. Available: <https://learn.microsoft.com/azure/>
- <span id="page-12-8"></span>[10] Alibaba cloud. [Online]. Available: <https://www.alibabacloud.com/>
- <span id="page-12-9"></span>[11] P. He, S. Zhou, C. Li, W. Huang, W. Yu, D. Wang, C. Meng, and S. Gui, "Distributed inference performance optimization for llms on cpus," *arXiv preprint arXiv:2407.00029*, 2024.
- <span id="page-12-10"></span>[12] W. Fedus, B. Zoph, and N. Shazeer, "Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity," *Journal of Machine Learning Research*, vol. 23, no. 120, pp. 1–39, 2022.
- <span id="page-12-11"></span>[13] N. Du, Y. Huang, A. M. Dai, S. Tong, D. Lepikhin, Y. Xu, M. Krikun, Y. Zhou, A. W. Yu, O. Firat *et al.*, "Glam: Efficient scaling of language models with mixture-of-experts," in *International Conference on Machine Learning*. PMLR, 2022, pp. 5547–5569.
- <span id="page-12-12"></span>[14] S. Rajbhandari, C. Li, Z. Yao, M. Zhang, R. Y. Aminabadi, A. A. Awan, J. Rasley, and Y. He, "Deepspeed-moe: Advancing mixture-ofexperts inference and training to power next-generation ai scale," in *International Conference on Machine Learning*. PMLR, 2022, pp. 18 332–18 346.
- <span id="page-12-13"></span>[15] J. Li, Y. Jiang, Y. Zhu, C. Wang, and H. Xu, "Accelerating distributed {MoE} training and inference with lina," in *2023 USENIX Annual Technical Conference (USENIX ATC 23)*, 2023, pp. 945–959.
- <span id="page-12-14"></span>[16] J. He, J. Qiu, A. Zeng, Z. Yang, J. Zhai, and J. Tang, "Fastmoe: A fast mixture-of-expert training system," *arXiv preprint arXiv:2103.13262*, 2021.
- <span id="page-12-15"></span>[17] M. Zhai, J. He, Z. Ma, Z. Zong, R. Zhang, and J. Zhai, "{SmartMoE}: Efficiently training {Sparsely-Activated} models through combining offline and online parallelization," in *2023 USENIX Annual Technical Conference (USENIX ATC 23)*, 2023, pp. 961–975.
- <span id="page-12-16"></span>[18] S. Shi, X. Pan, X. Chu, and B. Li, "Pipemoe: Accelerating mixtureof-experts through adaptive pipelining," in *IEEE INFOCOM 2023-IEEE Conference on Computer Communications*. IEEE, 2023, pp. 1–10.
- <span id="page-12-17"></span>[19] Z. Zhang, D. Yang, Y. Xia, L. Ding, D. Tao, X. Zhou, and D. Cheng, "Mpipemoe: Memory efficient moe for pre-trained models with adaptive pipeline parallelism," in *2023 IEEE International Parallel and Distributed Processing Symposium (IPDPS)*. IEEE, 2023, pp. 167–177.
- <span id="page-12-18"></span>[20] Lambdaml. [Online]. Available: <https://github.com/DS3Lab/LambdaML>
- <span id="page-12-19"></span>[21] Ecr. [Online]. Available: <https://aws.amazon.com/ecr/>
- <span id="page-12-20"></span>[22] Step function. [Online]. Available: [https://aws.amazon.com/](https://aws.amazon.com/step-functions/) [step-functions/](https://aws.amazon.com/step-functions/)
- <span id="page-12-21"></span>[23] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen, "Gshard: Scaling giant models with conditional computation and automatic sharding," *arXiv preprint arXiv:2006.16668*, 2020.
- <span id="page-12-22"></span>[24] Y. Zhou, T. Lei, H. Liu, N. Du, Y. Huang, V. Zhao, A. M. Dai, Q. V. Le, J. Laudon *et al.*, "Mixture-of-experts with expert choice routing," *Advances in Neural Information Processing Systems*, vol. 35, pp. 7103– 7114, 2022.
- <span id="page-12-23"></span>[25] R. Hwang, J. Wei, S. Cao, C. Hwang, X. Tang, T. Cao, M. Yang, and M. Rhu, "Pre-gated moe: An algorithm-system co-design for fast and scalable mixture-of-expert inference," *arXiv preprint arXiv:2308.12066*, 2023.
- <span id="page-12-24"></span>[26] J. He, J. Zhai, T. Antunes, H. Wang, F. Luo, S. Shi, and Q. Li, "Fastermoe: modeling and optimizing training of large-scale dynamic pre-

- trained models," in *Proceedings of the 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming*, 2022, pp. 120–134.
- <span id="page-12-25"></span>[27] J. Jarachanthan, L. Chen, F. Xu, and B. Li, "Amps-inf: Automatic model partitioning for serverless inference with cost efficiency," in *Proceedings of the 50th International Conference on Parallel Processing*, 2021, pp. 1–12.
- <span id="page-12-26"></span>[28] A. Ali, R. Pinciroli, F. Yan, and E. Smirni, "Batch: machine learning inference serving on serverless platforms with adaptive batching," in *SC20: International Conference for High Performance Computing, Networking, Storage and Analysis*. IEEE, 2020, pp. 1–15.
- <span id="page-12-27"></span>[29] Y. Yang, L. Zhao, Y. Li, H. Zhang, J. Li, M. Zhao, X. Chen, and K. Li, "Infless: a native serverless system for low-latency, high-throughput inference," in *Proceedings of the 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems*, 2022, pp. 768–781.
- <span id="page-12-28"></span>[30] Y. Fu, L. Xue, Y. Huang, A.-O. Brabete, D. Ustiugov, Y. Patel, and L. Mai, "{ServerlessLLM}:{Low-Latency} serverless inference for large language models," in *18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)*, 2024, pp. 135–153.
- <span id="page-12-29"></span>[31] Enwik8. [Online]. Available: <http://prize.hutter1.net/>
- <span id="page-12-30"></span>[32] K. Kotani, T. Yoshimi, and H. Isahara, "A machine learning approach to measurement of text readability for efl learners using various linguistic features." *Online Submission*, 2011.
- <span id="page-12-31"></span>[33] X. Nie, X. Miao, Z. Wang, Z. Yang, J. Xue, L. Ma, G. Cao, and B. Cui, "Flexmoe: Scaling large-scale sparse pre-trained model training via dynamic device placement," *Proceedings of the ACM on Management of Data*, vol. 1, no. 1, pp. 1–19, 2023.
- <span id="page-12-32"></span>[34] W. Wang, Z. Lai, S. Li, W. Liu, K. Ge, Y. Liu, A. Shen, and D. Li, "Prophet: Fine-grained load balancing for parallel training of largescale moe models," in *2023 IEEE International Conference on Cluster Computing (CLUSTER)*. IEEE, 2023, pp. 82–94.
- <span id="page-12-33"></span>[35] P. Cong, A. Yuan, S. Chen, Y. Tian, B. Ye, and T. Yang, "Prediction is all moe needs: Expert load distribution goes from fluctuating to stabilizing," *arXiv preprint arXiv:2404.16914*, 2024.
- <span id="page-12-34"></span>[36] F. Romero, Q. Li, N. J. Yadwadkar, and C. Kozyrakis, "{INFaaS}: Automated model-less inference serving," in *2021 USENIX Annual Technical Conference (USENIX ATC 21)*, 2021, pp. 397–411.
- <span id="page-12-35"></span>[37] S. Elloumi and A. Lambert, "Global solution of non-convex quadratically constrained quadratic programs," *Optimization methods and software*, vol. 34, no. 1, pp. 98–114, 2019.
- <span id="page-12-36"></span>[38] Gurobi optimization. [Online]. Available: <https://www.gurobi.com/>
- <span id="page-12-37"></span>[39] G. De Ath, R. M. Everson, J. E. Fieldsend, and A. A. Rahat, "ϵ-shotgun: ϵ-greedy batch bayesian optimisation," *arXiv preprint arXiv:2002.01873*, 2020.
- <span id="page-12-38"></span>[40] Lambda. [Online]. Available: [https://huggingface.co/datasets/cimec/](https://huggingface.co/datasets/cimec/lambada) [lambada](https://huggingface.co/datasets/cimec/lambada)
- <span id="page-12-39"></span>[41] Wmt19. [Online]. Available: [https://huggingface.co/facebook/](https://huggingface.co/facebook/wmt19-en-de) [wmt19-en-de](https://huggingface.co/facebook/wmt19-en-de)
- <span id="page-12-40"></span>[42] Optuna. [Online]. Available: <https://optuna.org/>
- <span id="page-12-41"></span>[43] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "Bert: Pre-training of deep bidirectional transformers for language understanding," *arXiv preprint arXiv:1810.04805*, 2018.
- <span id="page-12-42"></span>[44] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever *et al.*, "Language models are unsupervised multitask learners," *OpenAI blog*, vol. 1, no. 8, p. 9, 2019.
- <span id="page-12-43"></span>[45] C. Chen, Y. Yin, L. Shang, X. Jiang, Y. Qin, F. Wang, Z. Wang, X. Chen, Z. Liu, and Q. Liu, "bert2bert: Towards reusable pretrained language models," *arXiv preprint arXiv:2110.07143*, 2021.
- <span id="page-12-44"></span>[46] Fill-mask task. [Online]. Available: [https://huggingface.co/tasks/](https://huggingface.co/tasks/fill-mask) [fill-mask](https://huggingface.co/tasks/fill-mask)
- <span id="page-12-45"></span>[47] Ccnews. [Online]. Available: [https://huggingface.co/datasets/cc](https://huggingface.co/datasets/cc_news) news
- <span id="page-12-46"></span>[48] Translation task. [Online]. Available: [https://huggingface.co/docs/](https://huggingface.co/docs/transformers/tasks/translation) [transformers/tasks/translation](https://huggingface.co/docs/transformers/tasks/translation)
- <span id="page-12-47"></span>[49] J. Bergstra, R. Bardenet, Y. Bengio, and B. Kegl, "Algorithms for hyper- ´ parameter optimization," *Advances in neural information processing systems*, vol. 24, 2011.
- <span id="page-12-48"></span>[50] better transformer. [Online]. Available: [https://pytorch.org/blog/](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/) [a-better-transformer-for-fast-transformer-encoder-inference/](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/)