# 2 Motivation and Experiment Analysis

## 2.1 Experiment Settings and Metrics

We implement the agent stack using AutoGen [59]. Experiments run on a local device with an RTX 5090 GPU. We evaluate two model pairs: Qwen3-4B (SLM) + GPT-5 (LLM) and Gemma3-4B (SLM) + Claude Sonnet 4 (LLM). Local SLMs are executed with llama.cpp using 4–6 GB VRAM; cloud LLMs are accessed via their public APIs.

We evaluated on five datasets: GSM8K [6], HotPotQA [65], DROP [9], HumanEval [4], and WebShop [66]. These datasets are widely used for evaluating language model-based AI agents [18, 23, 36, 45, 53, 58, 67]. GSM8K has 8,500 grade school math word problems, and its accuracy is measured by the percentage of correct answers. HotPotQA contains complex questions requiring reasoning. DROP contains questions involving numerical operations and discrete reasoning. The accuracy of HotPotQA and DROP is measured by the F1 score. HumanEval contains 164 programming problems, and its accuracy is measured by the pass@1 metric [4]. WebShop contains user purchasing requests and its accuracy is based on BERTScore [69] compared to the ground-truth product descriptions. We consider a response to be similar if the BERTScore is higher than a threshold (e.g., 0.7).

Completion rate is the percentage of requests completed within 5 minutes. The number of subtasks is the number of subtasks to complete a request. SLM usage is the percentage of subtasks processed by SLM per request. In this paper, we focus on text-based AI agents, therefore we use semantic similarity to quantify the alignment of SLM outputs and LLM outputs. We define two outputs as similar if the cosine similarity between their SBERT embeddings [41] exceeds a threshold. Unless otherwise specified, the threshold is set to 0.7, which is empirically determined. AIMS uses a single similarity threshold and Section 4.6 presents the sensitivity study of this threshold. We report the average results of all requests from the two model pairs and for all datasets if not specifically indicated in the figure or table.

![](_page_2_Figure_7.jpeg)

Figure 2. Performance of existing methods.

## 2.2 The Need for a Subtask Scheduler

We conducted experiments comparing the performance of five existing methods: HybridLLM [8], All-SLM, All-LLM, Oracle, and random assignment (Random). In HybridLLM, the routing classifier operates at the subtask level, sending each subtask independently to either the local SLM or the cloud LLM. All-SLM processes an entire request using only the SLM, while All-LLM processes the request entirely using the LLM. Oracle achieves the accuracy threshold (empirically determined as 90% of the All-LLM's accuracy) while maximizing SLM usage by finding the optimal subtask assignment between the SLM and LLM for each user request. It is determined by enumerating all possible assignments for each subtask.

Figure 2 reports macro-averages (across datasets) of SLM usage, accuracy, and completion rate for each method under the two new model pairs. As expected, All-SLM attains the lowest accuracy (42.95% on average), while All-LLM achieves the highest accuracy (93.15%) at the cost of 0% SLM usage. Random yields moderate accuracy (47.65%), and HybridLLM improves to 66.08%. Oracle slightly sacrifices accuracy for substantially higher SLM usage, reaching 84.05% accuracy with 72.25% SLM usage. Compared to Oracle, HybridLLM delivers 17.97% lower accuracy and 31.20% lower SLM usage; compared to All-LLM, HybridLLM is 27.07% lower in accuracy. Completion rates follow the same ordering for both pairs: Oracle ≈ All-LLM > HybridLLM ≈ Random > All-SLM.

In addition, Table 1 reports the percentage of incorrect assignment decisions for HybridLLM and Random relative to Oracle. An assignment is counted as incorrect when a subtask is routed to the SLM while Oracle routes it to the LLM, or vice versa. HybridLLM makes incorrect assignments in 33.80% of cases with Qwen3+GPT-5 and 35.70% with Gemma3+Claude S4; Random is higher at 45.25% and 41.95%,

![](_page_3_Figure_0.jpeg)

**Figure 3.** Similar output percentage across datasets.

respectively. These errors stem from assigning subtasks independently without accounting for inter-subtask dependencies, underscoring the need for holistic, workflow-aware assignment. The results highlight the suboptimality of HybridLLM and Random and the headroom for improved subtask allocation strategies.

**Table 1.** Percentage of incorrect assignments.

| Method    | Qwen3 + GPT-5 | Gemma3 + Claude S4 |
|-----------|---------------|--------------------|
| HybridLLM | 33.80%        | 35.70%             |
| Random    | 45.25%        | 41.95%             |

**Observation 1.** Using the existing HybridLLM inference system in AI agent scenarios, which assigns each subtask independently to either the SLM or LLM, fails to optimize accuracy or SLM usage, highlighting the need for a more advanced approach. (Figure 2 and Table 1)

To investigate subtask-level model assignment and identify which subtasks can be effectively handled by SLM, we conduct two additional experiments, presented in Figure 3: 1) the percentage of user requests where the final outputs are similar when processed entirely by either SLM or LLM (left), and 2) the average percentage of individual subtasks of a request processed by LLM, for which SLM produces similar subtask outputs. The figure shows that 23.3%-44.9% of user requests and 30.2%-68.9% of subtask outputs can be handled by SLMs without compromising accuracy, with percentages varying across datasets. These findings indicate significant opportunities to reduce cloud usage by leveraging SLMs for suitable tasks and subtasks.

**Observation 2.** The SLM can manage certain user requests and subtasks, producing outputs similar to the LLM. (Figure 3)

## 2.3 Impact of Subtask Stage

This experiment assessed the accuracy impact of switching a single subtask from LLM to SLM while keeping the others on LLM, and vice versa, switching one subtask from SLM to LLM while keeping the rest on SLM. To account for user requests with varying numbers of subtasks, we grouped

![](_page_3_Figure_10.jpeg)

![](_page_3_Figure_11.jpeg)

(a) Accuracy  $\Delta$  by switching a subtask between SLM and LLM.

**(b)** The number of subtasks in All-SLM and All-LLM.

**Figure 4.** Subtask behavior under different model assignments.

the subtasks into three relative positions: Early (first third), Middle (middle third), and Late (final third) stages within the request's subtask sequence.

Figure 4a shows the average accuracy impact of switching a subtask at different stages. The results show that switching a subtask from LLM to SLM causes an average accuracy drop of 5.25% in the Early stage, 7.59% in the Middle stage, and 9.53% in the Late stage. Conversely, switching a subtask from SLM to LLM yields accuracy gains of 5.14%, 6.25%, and 9.40% in the Early, Middle, and Late stages, respectively. These findings suggest that SLM can manage early subtasks with minimal accuracy loss, but as tasks progress, leveraging LLM's advanced capabilities becomes increasingly critical.

**Observation 3.** Subtask position can influence the accuracy impact of SLM-LLM switching, with later stages showing greater effects, highlighting its importance in subtask allocation decisions. (Figure 4a)

Figure 4b illustrates the average number of subtasks generated per user request in All-SLM and All-LLM, respectively, across all datasets, considering only the requests that produced correct results. All-SLM generates more subtasks per request than All-LLM, with SLM averaging 6.37 subtasks per user request compared to LLM's 5.9. This indicates that SLM decomposes requests into more granular subtasks due to its limited capability to handle complex requests, whereas LLM's superior ability enables it to generate fewer, more comprehensive subtasks.

![](_page_3_Figure_19.jpeg)

Figure 5. S-L distance illustration.

![](_page_4_Figure_0.jpeg)

Figure 6. S-L distance comparison across subtask sequence.

To evaluate the convergence of SLM and LLM subtasks, we introduce the concept of *S-L distance* for an LLM subtask. This distance represents the number of additional SLM subtasks needed to produce a subtask similar (or match) to this LLM subtask, with their similarity defined as S-L similarity. Figure 5 illustrates the S-L distance, where LLM-generated subtasks are denoted as L1, L2, and L3, while SLM-generated subtasks are denoted as S1, S2, S3, S4, S5, and S6. Dashed lines indicate matched outputs between SLM and LLM subtasks. Subtask L1's S-L distance is 1, indicating that one additional SLM subtask is needed to match it. L2 has an S-L distance of 2, requiring two extra SLM subtasks for a match. L3 directly corresponds to S6, resulting in an S-L distance of 0. If no matched SLM subtask is found for an LLM subtask, its S-L distance is set to infinity. This metric provides insight into the alignment between SLM and LLM outputs during request processing, highlighting how SLM subtask granularity compares to that of LLM at different stages.

We ran each request using both All-LLM and All-SLM and categorized the requests into two groups: those with matched final results between LLM and SLM, and those without. Within each group, we further classified the requests into subgroups based on the number of subtasks (ST) generated (i.e., request length). Figure 6 presents the average S-L distance across LLM subtasks with the same sequence ID (X axis) in each subgroup for matched request group (top) and unmatched request group (bottom), respectively. For example, "4-ST request" represents the request subgroup that has 4 subtasks. As the subtask sequence progresses, the average S-L distance gradually increases in the matched group, while in the unmatched group, many distances reach infinity, indicating significant divergence between SLM and LLM subtask outputs. The results suggest that later LLM subtasks typically require more SLM subtasks to achieve similar outputs. SLM's inability to produce comparable subtask outputs ultimately leads to discrepancies in the final request output compared to the LLM. This observation echoes Observation 3 that the later stage of subtasks is more important to the accuracy of the user request.

![](_page_4_Figure_4.jpeg)

Figure 7. S-L similarity across subtask sequence IDs.

Figure 7 shows the changes of S-L similarity as the request progresses in matched and unmatched scenarios. In matched cases, S-L similarity gradually increases, indicating that SLM outputs align more closely with LLM outputs in later stages. Conversely, in unmatched scenarios, S-L similarity remains low, suggesting persistent deviations. This measurement supports previous findings and highlights the potential for SLM and LLM convergence in subtask sequence, enabling efficient subtask allocation.

**Observation 4.** SLM typically generates more subtasks for a later LLM subtask to gradually converge to similar outputs. Subtask scheduling must account for whether an SLM's output can converge with that of an LLM subtask. (Figures 4b-7)

#### 3 Design of AIMS

#### 3.1 Overview

Motivated by Observation 1, we propose AIMS, which dynamically assigns the subtasks of a request between the SLM and LLM in order to maximize the SLM usage while maintaining the LLM's accuracy of processing the request.

Based on Observation 2 that the SLM may output similar results for a request or for a subtask as the LLM, AIMS lets the SLM process the entire request in the former case. Otherwise, it allocates each subtask to either the SLM or LLM. For each subtask, based on Observation 4, if it finds a convergence point, it uses the SLM until the convergence point. Otherwise, it breaks down the current subtask into smaller sub-subtasks to facilitate the SLM to process them by repeating the above steps for each sub-subtask. To avoid unnecessary LLM execution, the SLM processes decomposed sub-subtasks only if all can be handled by the SLM; otherwise, the original subtask is processed by the LLM. Based on Observation 3, we use the subtask's sequence ID to dynamically adjust the similarity threshold, and also consider it in predicting the subtask's convergence point.

AIMS consists of several offline-trained estimators based on profiled data. With the assistance of the estimators, after

![](_page_5_Figure_0.jpeg)

Figure 8. AIMS decision-making workflow.

receiving a user request, AIMS chooses between the SLM and LLM for the request or its subtasks.

Offline Profiling. AIMS first profiles the AI agent that uses SLM and LLM with various user requests and their corresponding subtasks. The profiling process collects data on the subtask outputs from SLM and LLM. This data is then used to train prediction models, including the *user request classifier*, *subtask predictors* for the SLM and LLM execution, *distance predictor*, and *subtask decomposer*. These trained models are used in the online decision making of AIMS to make informed decisions about task or subtask allocation.

Online Decision Making. After receiving a user request or a subtask, AIMS determines its allocation between SLM and LLM. It first employs the *user request classifier*, which determines whether processing the entire request with the SLM will yield output similar to that of the LLM. If not, AIMS proceeds to subtask-level decision-making for a more granular analysis. At this stage, AIMS employs *subtask similarity evaluator*, *S-L similarity estimator* (SLE), *convergence detector* (CD), and *subtask decomposer* (SD) approaches to determine the suitable model for processing the current subtask. AIMS makes decisions for every subtask using this process until producing the final output.

AIMS follows a fast-path/slow-path design pattern: the fast path executes on the local SLM when predictors indicate the next-step behavior is safe, while the slow path falls back to the cloud LLM for accuracy-critical decisions. Convergence detection and similarity-validated decomposition act as structured recovery paths that safely expand the fraction of execution on the fast path without committing to an irrecoverably divergent subtask.

### 3.2 System Design

Let R denote a given user request, and  $ST_i$  denote the  $i^{th}$  dynamically generated subtask for request R. The workflow of AIMS is depicted in Figure 8. First, based on Observation 2, AIMS employs a hierarchical approach that allows AIMS to leverage the effectiveness of SLM for user requests that can be accurately processed by SLM alone while enabling fine-grained subtask allocation for more complex requests. Specifically, AIMS utilizes a lightweight request-level *classifier* to determine if a user's request can yield similar outputs when executed entirely on SLM. If yes, AIMS opts for SLM for the entire request. If not, AIMS proceeds to the next step to perform subtask-level model assignment. The objective of AIMS is to design a  $router: ST_i \rightarrow \{0,1\}$ , where  $router(ST_i) = 0$  and  $router(ST_i) = 1$  mean that the subtask  $ST_i$  is routed to the SLM, and to the LLM, respectively.

In the subtask-level model assignment, for the  $i^{th}$  subtask, AIMS takes the following steps:

- 1) **Subtask Similarity Evaluator (SSE).** Based on Observation 2, AIMS estimates the output similarity of the  $i^{th}$  subtask using LLM and SLM. If they are similar, AIMS uses SLM for this subtask. Otherwise, AIMS employs the following three steps based on Observation 4:
- 2) **S-L Similarity Evaluator (SLE).** It first estimates the S-L distance *d* for the current subtask, followed by an evaluation of the S-L similarity. If the S-L similarity meets the required threshold (i.e., 0.7), the subtask is assigned to SLM, as it is expected to ultimately reach a similar subtask in LLM; otherwise, it proceeds to the next step.
- 3) **Convergence Detection (CD).** If a convergence point cannot be identified for the current subtask, it may still be found for a subsequent one. Hence, AIMS continuously estimates the outputs of SLM and LLM subtasks, comparing

each SLM-LLM subtask pair until a convergence point is identified where the S-L similarity meets the threshold. If a convergence point is identified, all subtasks preceding it are executed by the SLM. Otherwise, the process moves to the next step.

4) Subtask Decomposition (SD). If convergence is not detected by CD, AIMS employs its Subtask Decomposer (SD) model, which is trained offline (Section 3.3), to predict a sequence of simpler sub-subtasks for the current complex subtask. For example, if a HotpotQA user request asks for the maternal grandfather of the Titanic director, an AI agent, after identifying the director (James Cameron) and his mother (Shirley Cameron née Lowe), might generate a complex subtask (): 'Verify Shirley Cameron's father, including corroborating biographical details, to confirm his identity as James Cameron's maternal grandfather.' The SD might then decompose this into more granular steps: 1: 'Search for Shirley Cameron's father'; 2: 'Extract father's full name'; 3: 'Find key biographical details for verification (e.g., birth/death dates)'; 4: 'Confirm and state the maternal grandfather's name'. AIMS then uses its subtask predictors ( and ) to evaluate if the SLM can adequately handle each predicted sub-subtask. Only if all predicted subsubtasks are deemed suitable for the SLM (i.e., yield similar outputs to the LLM predictions) does AIMS direct the AI agent to execute this decomposed sequence using the actual SLM; otherwise, the original defaults to the LLM. This allows AIMS, as a system-level scheduler, to guide the AI agent towards a more granular execution path that leverages the SLM's capability to handle simpler steps.

## 3.3 Offline Profiling

In the offline profiling phase, AIMS collects data to fine-tune models that predict the performance of different subtask allocations between SLM and LLM. The profiling is conducted on user requests from historical trace datasets (e.g., GSM8K and HotPotQA) and their corresponding subtasks.

Data Collection: For each user request , we generate a binary tree of subtasks, where each node represents a subtask and each edge represents using a model (SLM or LLM) to process the parent subtask. Starting from the root node, which represents the initial user request, we process the subtask using both the SLM and LLM, creating two child nodes. For each child node, we then recursively process the corresponding subtask using both SLM and LLM, further creating child nodes until a predefined depth (e.g., 15 subtasks) is reached or the model thinks the request is finished. At each leaf node, we profile the output of executing the subtask using the selected model. In addition to the subtask-level profiling, we also profile the performance of executing the entire user request using SLM and LLM. The similarity of the final results from the two models is collected. Moreover, we use the SLM to generate multiple smaller subtasks

for each original subtask, creating a dataset of subtask decomposition. Using the collected profiled data, we train the following models.

User Request Classifier (URC): For each user request in the profiled data, we use the user request as the input feature and the similarity score between the outputs generated by processing the entire user request using All-SLM and All-LLM as the target variable. We then train the user request classifier model using this input-output data.

Subtask Predictor (SP): We train two separate models, and , which learn to predict the next subtask when the current subtask is processed using SLM and LLM, respectively. For each node in the binary tree of subtasks generated during the data collection process, we use the subtask at that node as the input feature and the subtask generated by applying SLM or LLM to the current subtask as the target output for the two models, respectively.

Distance Predictor (DP): The distance predictor predicts the S-L distance. For each request in the profiled data, we extract the content of the LLM subtask and its sequence ID as the input features and its corresponding S-L distance as the target label. We then train the distance predictor model using the input and output data to predict the S-L distance. Subtask Decomposer (SD): The subtask decomposer is trained to break down a complex subtask into smaller, more manageable sub-subtasks. It takes a subtask and the predicted next subtask from for this subtask as inputs, and outputs a sequence of sub-subtasks, aiming to ensure that the output of the last sub-subtask is similar to the predicted next subtask from . The subtask decomposer is trained using data derived from decomposing subtasks from user request.

For the above models, we use ModernBERT [55] as the base for URC and DP, and Qwen3-0.6B [64] for SP and SD, fine-tuning them with LoRA [17]. These models are then used in the subsequent components of AIMS to make informed decisions about task or subtask allocation. The estimator stack requires approximately 2 GB of VRAM, which can easily be accommodated on a modern gaming laptop with 8-16 GB of GPU memory.

## 3.4 Online Decision Making

The online decision-making process comprises two stages: request-level decision-making and subtask-level decisionmaking, as detailed below.

3.4.1 Request-Level Decision Making. AIMS first uses the user request classifier to process an incoming user request. It leverages the knowledge learned during the offline profiling phase to identify user requests that can be accurately processed by the SLM alone, avoiding unnecessary subtask-level allocation. Specifically, AIMS feeds the request into the user request classifier model. The classifier predicts a similarity score between 0 and 1, indicating the expected similarity between the results of processing the request solely

using SLM versus the LLM. If the predicted similarity score is above a predefined threshold (e.g., 0.7), AIMS processes the entire user request using SLM, bypassing the subtask-level allocation. Otherwise, AIMS initializes the current subtask as the root request and enters the subtask-level routing stage.

**3.4.2 Subtask-Level Decision Making.** If the *user request classifier* determines that a user request requires subtask-level allocation, AIMS proceeds to the SSE to process the subtask.

Subtask Similarity Evaluator (SSE): The subtask similarity evaluator compares the outputs of the SLM and the LLM for each subtask, assessing their similarity and making appropriate model assignments based on the stage of the user request. For each subtask  $ST_i$  in the user request R, AIMS feeds the current subtask into the  $SP_{SLM}$  and  $SP_{LLM}$  models. The  $SP_{SLM}$  and  $SP_{LLM}$  models generate the predicted next subtasks for the SLM and the LLM, respectively. The subtask similarity evaluator then estimates the similarity of the predicted next subtasks as introduced in Section 2.1. If the similarity is above a predefined threshold  $\kappa$ , the subtask  $ST_i$  is assigned to the SLM. The similarity threshold  $\kappa$ is determined through empirical analysis during the offline profiling phase. For each SLM-LLM pair, we analyze a set of requests and their subtasks, measuring the relationship between threshold values and final accuracy. The sensitivity analysis of the  $\kappa$  threshold can be found in Section 4.6. Organizations deploying AIMS can fine-tune these thresholds during their offline profiling phase based on their specific accuracy requirements and cost constraints.

Here, instead of using a constant similarity threshold, based on Observation 3, we set the threshold adaptively based on the subtask's sequence ID. The threshold  $\kappa$  is smaller at the early stages of a request, permitting loose comparisons, and increases as the request progresses, making it more stringent in the later stages. Using the experimental data collected for Figure 4a, we compute the threshold as  $\kappa = threshold_{base} + \min(ID, 5) \cdot 0.02$ , where  $threshold_{base} = 0.6$ . All parameters are determined empirically.

If the similarity is below the threshold  $\kappa$ , the simplest way is to use LLM. However, directly using LLM results in lower SLM usage. Observation 4 suggests that while the SLM output may differ from the LLM output for the current subtask, a subsequent subtask could still produce a result similar to the LLM. Therefore, we employ three approaches: the *S-L similarity evaluator*, the *convergence detector*, and the *subtask decomposer*. The S-L similarity evaluator identifies when a future SLM subtask matches the current LLM subtask. The convergence detector identifies future subtasks where the outputs of the SLM and LLM are similar. The subtask decomposer breaks down the current subtask into smaller subtasks to increase the likelihood of successful processing by the SLM. The details are presented in the following.

**S-L Similarity Evaluator (SLE)**: Guided by Observation 4, the S-L distance metric helps determine if a future SLM subtask matches the current LLM subtask, which is crucial for deciding whether to process a subtask using SLM or LLM. Thus, the S-L similarity evaluator dynamically adjusts the similarity threshold during task processing based on the progress stage of the user request. It receives the current subtask  $ST_i$  and its sequence ID as inputs and uses the distance predictor model to estimate the S-L distance d between the outputs of the SLM and the LLM, considering the current subtask's content. The  $SP_{SLM}$  model then predicts the output for the  $(i+d)^{th}$  subtask, while the  $SP_{LLM}$  model predicts the output for the  $i^{th}$  subtask. These outputs are compared using the predefined similarity threshold  $\kappa$  (same threshold as in the subtask similarity evaluator). If their similarity is higher than  $\kappa$ , the subtask  $ST_i$  is assigned to the SLM; otherwise, we proceed to the next component.

Convergence Detector (CD): Guided by Observations 3 and 4, convergence detector identifies a future convergence point between the outputs of SLM and LLM. Starting from subtask  $ST_i$ , the convergence detector uses  $SP_{SLM}$  and  $SP_{LLM}$  to predict future subtasks iteratively. It compares the similarity of each pair of the SLM and LLM predictions using the same similarity metric and threshold as previous components. It continues this process for a predefined number of future subtasks or until the end of the sequence. If multiple convergence points are found, convergence detector selects the last one to increase the use of SLM. All subtasks from  $ST_i$  up to the identified convergence point are then assigned to SLM. If no convergence is detected, we proceed to the next component.

Subtask Decomposer (SD): Guided by Observation 4, which highlights the granularity and step-by-step nature of SLM processing, we design the *subtask decomposer*. It breaks down a complex subtask into smaller sub-subtasks, making them easier for the SLM to process. It takes a current subtask  $ST_i$ as input and uses the *subtask decomposer* model, which is trained during the offline profiling phase, to generate a sequence of sub-subtasks, denoted by  $\{SST_1, SST_2, \dots, SST_m\}$ . AIMS then evaluates each sub-subtask  $SST_i$  to determine its suitability for processing by the SLM. Specifically, AIMS inputs  $SST_i$ 's content into both  $SP_{SLM}$  and  $SP_{LLM}$  models, which then predict the next sub-subtask. If the similarity of the two predicted next sub-subtasks exceeds the predefined threshold  $\kappa$ , the sub-subtask is deemed suitable for SLM processing. While we could allocate each sub-subtask individually to the SLM or LLM, this may increase the number of LLM calls. To avoid this, we allocate the entire group of decomposed sub-subtasks or the original subtask as a single unit. That is, only when all sub-subtasks are found suitable for the SLM, AIMS assigns all sub-subtasks  $\{SST_1, SST_2, \dots, SST_m\}$ 

to SLM to produce the output for the original subtask. Conversely, if any sub-subtask is unsuitable for SLM, AIMS assigns the subtask  $ST_i$  to the LLM.

## Algorithm 1 AIMS online decision making process.

```
Require: User request R
Ensure: Final result
 1: if URC(R) predicts similar outputs then
        Process R using SLM
 3:
    else
        for each subtask ST_i generated do
 4:
            if SSE(ST_i) predicts similar outputs then
 5:
                Process ST_i using SLM
 6:
            else if SLE(ST_i) finds high S-L similarity then
 7:
               Process ST_i and next d subtasks using SLM
 8:
            else if CD(ST_i) finds convergence point then
 9:
                Process subtasks to convergence using SLM
10:
            else
11:
                sub\_subtasks = SD(ST_i)
12:
                if one sub_subtask's outputs in SP<sub>SLM</sub> and
13:
    SP_{LLM} are dissimilar then
                    Process ST_i using LLM
14:
                else
15:
                    Process each sub_subtask using SLM
16:
                end if
17:
            end if
18:
        end for
19:
    end if
```

21: return Final result

**3.4.3** Algorithm for the Process. Algorithm 1 shows the pseudocode of the decision-making process of AIMS. When a user request is received, the user request classifier (URC) first predicts the similarity score of running the entire request on SLM and LLM. If the similarity score exceeds a threshold, the request is processed entirely by SLM (lines 1-2). Otherwise, it advances to subtask-level decision-making (line 4). Specifically, the subtask similarity evaluator (SSE) compares the predicted outputs of the subtask from SLM and LLM using  $SP_{SLM}$  and  $SP_{LLM}$ . If the outputs are similar, the subtask is assigned to SLM (lines 5-6). If not, the S-L similarity estimator (SLE) component, using the distance predictor (DP), estimates the S-L distance of the current subtask (d). It then uses  $SP_{SLM}$ and  $SP_{LLM}$  to predict the S-L similarity. If it is higher than the threshold, SLM is used (lines 7-8); otherwise, the process moves to convergence detector (CD). It attempts to identify a convergence point, where the outputs of the SLM and LLM are similar. If it is found, SLM is used until the convergence point (lines 9-10). If not, the subtask decomposer (SD) breaks the subtask into smaller sub-subtasks, and the process of the SSE repeats for each sub-subtask. If one sub-subtask's outputs in the SLM and in the LLM are not similar, LLM

![](_page_8_Figure_4.jpeg)

Figure 9. Accuracy and SLM usage.

processes  $ST_i$  (lines 13-14); otherwise, the SLM processes all the sub-subtasks (line 16).

#### 4 Performance Evaluation

#### 4.1 Experiment Settings

The experiment settings are the same as those in Section 2.1 unless otherwise specified. We employed GPT-5+Qwen3 4B as the LLM-SLM pair to generate 1000 subtask traces (i.e., the sequence of subtasks generated by the AI agent from an initial user request to its final resolution) using WorFBench [38] and GSM8K [6] to fine-tune the models in AIMS. The finetuning process for all estimators took approximately 2 hours on a cloud-based Nvidia A100, a one-time cost that enables subsequent effective decision-making. To evaluate the performance and generalization capabilities of AIMS, we test on nine benchmarks: HotpotQA, GSM8K, DROP, HumanEval, WebShop, MATH [15], WorFBench [38], WebArena [71], and ToolBench [39], though we trained the AIMS in only two of the datasets. The baseline methods for comparison are HybridLLM [8] (explained in Section 2.2) and Minions [31]. Minions follows a collaborative cloud-edge execution model where a planner generates subtasks, and a local small model first attempts each subtask, escalating to a cloud LLM when the local model is uncertain, with uncertainty measured by the average log-probability of the generated tokens.

#### 4.2 Overall Performance

Accuracy and SLM usage. Figure 9 summarizes accuracy and SLM usage. We used three representative datasets to demonstrate the performance, *HotpotQA*, *HumanEval* and *WebShop*, under Qwen3-4B+GPT-5. On *HotpotQA*, AIMS reaches 90.75% vs 76.35% (HybridLLM, +14.40%) and 84.20% (Minions, +6.55%), with higher SLM usage than both (81.85% vs 68.40% / 74.10%). On *HumanEval*, AIMS is 91.45% vs 73.65% (+17.80%) and 86.10% (+5.35%); Minions uses slightly more SLM (86.95%) than AIMS (84.10%) but trails in accuracy. On *WebShop*, AIMS attains 51.20% vs 34.75% (+16.45%) and 43.10% (+8.10%), while SLM usage is comparable (81.25% vs

![](_page_9_Figure_0.jpeg)

**Figure 10.** End-to-end latency breakdown across nine benchmarks (Qwen3-4B + GPT-5 on RTX 5090). AIMS achieves latency comparable to HybridLLM/Minions while remaining faster than All-LLM; routing + network overhead is a small fraction of total time.

**Table 2.** Normalized remote cost per 100 requests (All-LLM = 1.0). Costs are computed via token-footprint accounting; API prices are as of September 2025.

| Dataset   | HybridLLM | Minions | AIMS (ours) |
|-----------|-----------|---------|-------------|
| HotpotQA  | 0.32      | 0.26    | 0.18        |
| GSM8K     | 0.38      | 0.26    | 0.18        |
| DROP      | 0.29      | 0.22    | 0.15        |
| HumanEval | 0.33      | 0.13    | 0.16        |
| WebShop   | 0.29      | 0.20    | 0.21        |
| MATH      | 0.28      | 0.21    | 0.15        |
| WebArena  | 0.23      | 0.15    | 0.17        |
| WorFBench | 0.31      | 0.24    | 0.21        |
| ToolBench | 0.26      | 0.20    | 0.19        |
| Macro avg | 0.29      | 0.20    | 0.17        |

74.35% / 83.10%). These gains reflect AIMS's holistic, position-aware routing, avoiding the independent or myopic assignments that limit HybridLLM and the heuristics in Minions.

Higher accuracy with greater SLM share translates to better application outcomes at lower cloud usage. For example, on *WebShop*, AIMS's 51.20% success (vs 34.75% HybridLLM / 43.10% Minions) is achieved while using the SLM more, reducing GPT-5 usage and cost; on *WebArena*, AIMS improves task success to 57.90% (vs 28.60% / 45.70%) with higher SLM usage; on *ToolBench*, chain correctness rises to 78.35% (vs 59.40% / 66.10%) with less reliance on the LLM. Taken together, for the same budget, AIMS completes more tasks, and to hit the same accuracy target, it requires fewer LLM tokens, demonstrating end-to-end benefits beyond per-step accuracy.

**Average latency.** We measured average end-to-end latency on the desktop edge (RTX 5090) and found that *AIMS* does not compromise speed. Figure 10 reports the mean across datasets: *All-SLM* 11.14 s, *HybridLLM* 12.98 s, *Minions* 14.21 s, *AIMS* 13.33 s, and *All-LLM* 15.82 s. Thus, AIMS is consistently faster than All-LLM, slower than All-SLM (as expected), and comparable to HybridLLM and Minions. The scheduler's

**Table 3.** Generalization summary across AIMS settings.

| Setting      | ΔAcc<br>HLLM | ∆Acc<br>Minions | Cost<br>vs All-LLM | Latency<br>vs All-LLM |
|--------------|--------------|-----------------|--------------------|-----------------------|
| Model pairs  |              |                 |                    |                       |
| Qwen+GPT     | +12.35       | +6.86           | 0.17×              | 0.82×                 |
| Gemma+Claude | +13.14       | +6.18           | 0.22×              | 0.76×                 |
| Hardware     |              |                 |                    |                       |
| RTX 5090     | _            | _               | _                  | 0.79×                 |
| iPhone 15    | +10.65       | +6.32           | 0.19×              | 2.52×                 |

Model-pair rows: macro accuracy deltas (%) vs baselines and normalized cost ratios per 100 tasks. Hardware rows: average end-to-end latency ratios across datasets using the same routing policy/checkpoints (Qwen3-4B + GPT-5 unless noted). RTX 5090 vs iPhone 15 differ in throughput; routing/SLM usage are unchanged.

decision overhead is small (about 3-7% of AIMS's total time), and measured average network hop latency is 0.58 s, which is negligible relative to end-to-end latency. Overall, offloading selected subtasks to the local SLM reduces reliance on cloud decoding without incurring noticeable runtime overheads. **Remote Cost.** Remote monetary cost is dominated by the cloud LLM; we treat on-device SLM execution as free and ignore edge-cloud transfer fees. We compute cost using token-footprint accounting (prompt + completion tokens) multiplied by API prices (prices as of Sep 2025). To reduce dependence on volatile pricing, Table 2 reports normalized remote cost relative to the All-LLM baseline (All-LLM = 1.0 per dataset). Averaged across nine benchmarks, HybridLLM costs 0.29×, Minions 0.20×, and AIMS 0.17× of All-LLM, corresponding to 83% savings vs. All-LLM, 41% vs. HybridLLM, and 15% vs. Minions while maintaining higher accuracy than HybridLLM and Minions.

#### 4.3 Generalizability Evaluation

Table 3 summarizes transfer across *model pairs* and *hardware*. For models, AIMS improves accuracy by +12.35% over HybridLLM and +6.86% over Minions at  $0.17\times$  the cost of All–LLM with Qwen+GPT, and by +13.14% / +6.18% at  $0.22\times$  cost with Gemma3 + Claude Sonnet 4, demonstrating robust

accuracy—cost gains across pairs. To isolate the impact of device throughput, we keep AIMS's routing policy, thresholds, and SLM usage fixed across hardware and change only the execution platform. Unless noted otherwise, the hardware results use the Qwen3-4B (SLM) + GPT-5 (LLM) pair, with the same estimator checkpoints. On RTX 5090, AIMS achieves 0.79× the All-LLM end-to-end latency; on iPhone 15, AIMS is 2.52× due to lower on-device throughput. Importantly, this is a *latency* effect: the allocation policy is unchanged across hardware. Any minor accuracy differences on mobile occur only in rare cases where the on-device runtime cannot accommodate the full context (e.g., truncation), not because the routing logic changes.

#### 4.4 Ablation Study

Ablation—components. Table 4 quantifies the effect of removing AIMS components. Relative to AIMS (full) (77.62% accuracy, 83.58% SLM), removing the subtask decomposer (w/o SD) lowers accuracy to 76.04% (-1.58%) and reduces SLM usage to 78.04% (-5.54%), indicating SD's role in exposing safe SLM opportunities. Further disabling convergence detection (URC+SSE+SLE) drops accuracy and SLM usage more sharply (73.90%, -3.72%; 75.52%, -8.06%), showing CD's contribution to stability. Removing SLE as well (URC+SSE) yields 72.48% (-5.14%) and 72.36% (-11.22%). With URC only, accuracy falls to 70.10% (-7.52%) and SLM usage to 71.98% (-11.60%), revealing that single-stage request routing is too conservative. Finally, w/o URC retains accuracy (76.82%, -0.82%) but incurs the *largest SLM* reduction (70.18%, -13.44%), confirming URC's value as an initial filter that safely increases SLM share. Overall, each component materially helps decide when a subtask can be moved to the SLM without harming accuracy.

**Ablation—training data.** Figure 11 varies the fraction of training traces used by the estimators ( $\{0, 10, 25, 50, 100\}\%$ ). The AIMS system components are trained on subtask traces generated from WorFBench and GSM8K using the indicated fraction of training traces, and evaluated as macro-averages over the other seven benchmarks (HotpotQA, DROP, HumanEval, WebShop, MATH, WebArena, ToolBench). As shown in Figure 11, accuracy improves monotonically from  $68.3\% \rightarrow 77.5\%$  while SLM usage rises from  $69.5\% \rightarrow 83.6\%$ . Gains are most pronounced up to 50% of traces (76.1% accuracy, 81.2% SLM), after which improvements taper, indicating diminishing returns and suggesting that moderate data budgets already capture most of AIMS's benefit.

#### 4.5 Performance of Estimators

Table 5 reports estimator accuracy and latency share. We train on WorFBench and GSM8K and evaluate generalization on the remaining benchmarks. We optionally apply *continual fine-tuning (CFT)* every 500 new requests: we collect the most recent batch of 500 execution traces and perform incremental LoRA updates to the estimator stack (URC, DP,

![](_page_10_Figure_6.jpeg)

**Figure 11.** Training-data ablation for AIMS (Qwen3-4B + GPT-5).

**Table 4.** Ablation study of AIMS components.

| Variant              | Acc. (%) ↑ |       | SLM (%) ↑ |        |
|----------------------|------------|-------|-----------|--------|
| variani              | Value      | Δ     | Value     | Δ      |
| AIMS (full)          | 77.62      | _     | 83.58     | _      |
| w/o SD               | 76.04      | -1.58 | 78.04     | -5.54  |
| URC+SSE+SLE (no CD)  | 73.90      | -3.72 | 75.52     | -8.06  |
| URC+SSE (no CD, SLE) | 72.48      | -5.14 | 72.36     | -11.22 |
| w/ URC (URC only)    | 70.10      | -7.52 | 71.98     | -11.60 |
| w/o URC              | 76.82      | -0.80 | 70.18     | -13.40 |

the subtask predictors, and the subtask decomposer), then continue serving with the updated checkpoints. In Table 5, generalization accuracy without CFT is reported as-is; accuracy with CFT is shown in parentheses. On training data, estimator accuracy is consistently high (82.1–84.1%). Without CFT, generalization accuracy remains strong (75.1–77.9%); with CFT (in parentheses) it improves to 78.1–81.3%, a +3–4% gain. Latency-wise, the subtask predictors dominate estimator compute when invoked (SP<sub>SLM</sub> 42.4%, SP<sub>LLM</sub> 38.8%), while URC, the Distance Predictor, and the Convergence Detector are lightweight (6.4%, 7.1%, and 5.3%), enabling fast decisions.

**Table 5.** Performance of AIMS's estimators.

| Component                    | A     | Lat.           |       |
|------------------------------|-------|----------------|-------|
| Component                    | Train | Generalization | ratio |
| User Request Classifier      | 84.1  | 77.9 (81.3)    | 6.4%  |
| Subtask Predictor $SP_{SLM}$ | 83.9  | 76.1 (80.1)    | 42.4% |
| Subtask Predictor $SP_{LLM}$ | 82.1  | 75.1 (78.1)    | 38.8% |
| Distance Predictor           | 83.6  | 76.2 (79.4)    | 7.1%  |
| Convergence Detector         | 82.4  | 75.8 (79.1)    | 5.3%  |

Training: WorFBench, GSM8K; Generalization: Other benchmarks

![](_page_11_Figure_0.jpeg)

![](_page_11_Figure_1.jpeg)

Figure 12. Sensitivity of AIMS on its parameters.

Table 6. Comparison of mechanisms in AIMS, HybridLLM, and Minions.

|                        | HybridLLM [8]             | Minions [31]              | AIMS                                                                    |
|------------------------|---------------------------|---------------------------|-------------------------------------------------------------------------|
| Core idea              | classifier-based routing  | confidence-based routing  | adaptive iteration-level routing                                        |
| Decision granularity   | per subtask (independent) | per subtask (independent) | per subtask (dependency-aware)                                          |
| Position-aware routing | No                        | No                        | Yes (adaptive $\kappa(i)$ )                                             |
| Dependency handling    | No                        | No                        | Yes (subtask prediction + convergence detector + subtask decomposition) |

#### 4.6 Sensitivity Testing

We analyze two routing thresholds with macro averages across datasets: the *request-level* threshold  $\tau_{\text{req}}$  (Fig. 12a) and the *subtask-similarity* threshold  $\tau_{\text{sse}}$  (Fig. 12b). Both exhibit a monotonic trade-off: as the threshold increases, *accuracy* rises while *SLM usage* falls.

**Request threshold** ( $\tau_{\rm req}$ ). Sweeping from 0.50 to 0.78 increases accuracy from 74.25% to 77.90% (+3.65%) while reducing SLM usage from 88.40% to 76.30% (-12.10%). A balanced operating point is  $\tau_{\rm req} \in [0.66, 0.70]$ , which keeps accuracy  $\geq$ 77.10% with SLM usage  $\approx$ 79–81%.

**Subtask similarity** ( $\tau_{\rm sse}$ ). Sweeping from 0.60 to 0.80 increases accuracy from 75.10% to 77.65% (+2.55%) while reducing SLM usage from 86.20% to 78.55% (-7.65%). The knee appears around  $\tau_{\rm sse} \in [0.70, 0.74]$ , offering near-max accuracy at materially higher SLM share than more conservative settings. Overall, higher thresholds push AIMS toward conservative routing (more LLM), improving accuracy at the cost of SLM efficiency; mid-range values deliver the best accuracy–SLM balance for both thresholds.

## 4.7 Understanding Performance Gains

To explain the performance gains of AIMS over HybridLLM and Minions in Section 4.2, we analyze where the baselines lose accuracy or cost-efficiency relative to AIMS.

**Mechanism Comparison.** Table 6 compares AIMS, HybridLLM, and Minions in terms of their mechanisms. HybridLLM uses classifier-based routing, where an ML classifier forwards each subtask to an SLM or LLM. Minions uses confidence-based routing: an SLM first attempts a subtask and escalates to a cloud LLM if confidence is low. Both handle subtasks independently, ignoring subtask position or dependencies. In contrast, AIMS uses adaptive iteration-level routing that is position- and workflow-aware: it considers how routing decisions affect subsequent subtasks, becomes more conservative for later subtasks, and incorporates convergence detection and subtask decomposition to achieve high accuracy and cost-efficiency. These differences are critical because subtasks are interdependent; an early routing mistake can affect the final result, and late-stage subtasks are often most accuracy-sensitive.

Experiment Results Analysis. In our experiments in Section 4.2, for each request, we record the sequence of subtasks generated by the agent along with the routing decision (SLM or LLM) made at each step by each method. To better understand the behavioral differences among the methods, we analyze the execution traces of 100 requests, where AIMS successfully outputs correct results but both HybridLLM and Minions produce incorrect results. An incorrect result is defined as a final output that fails the task-specific correctness criterion (Section 2.1), e.g., incorrect final answer, failed test case, or similarity score below a threshold. By comparing routing decisions and subtask sequences across the three

methods, we identify two main trace patterns that explain why the baselines fail while AIMS succeeds.

- Early divergence accumulation: An early subtask may appear simple, leading Minions and HybridLLM to offload it to the SLM. However, the resulting state change can alter subsequent subtasks, causing the downstream subtask chain to drift. AIMS mitigates this by checking the predicted subtask's SLM-LLM output similarity in making routing decision and by searching for future convergence when the SLM-LLM output similarity is high.
- Late-stage sensitivity: In tasks with long subtask sequences, such as WebArena and ToolBench, later subtasks often involve final action selection, argument validation, or consistency checks. HybridLLM and Minions do not consider subtask position, leading to aggressive late-stage offloading where divergence is more impactful, producing incorrect final results. AIMS addresses this by tightening its similarity threshold for later subtasks, favoring LLM to ensure accuracy.

Performance Difference Breakdown. We consider HybridLLM or Minions to exhibit performance differences when they either produce an incorrect final outcome or produce a correct final outcome but require more cloud LLM invocations than AIMS, resulting in lower SLM usage and higher cost. For each performance difference case, we analyze the routing decisions and resulting subtask sequences to identify the dominant cause among three categories: early divergence accumulation, late-stage sensitivity, and lack of convergence and decomposition handling. The first two categories correspond to incorrect final outcomes and are explained in the previous paragraph, while the third category corresponds to cases with correct final outcomes but higher cost, where the baseline incurs more cloud LLM invocations because it cannot exploit convergence or decomposition opportunities used by AIMS. We analyze 150 cases and calculate the percentage of each cause among these cases for each method.

For HybridLLM, the gap relative to AIMS is dominated by early divergence accumulation (53.18%), followed by latestage sensitivity (29.64%) and lack of convergence and decomposition handling (17.18%). For Minions, late-stage sensitivity dominates (45.86%), with early divergence accumulation (30.07%) and lack of convergence and decomposition handling (24.07%). These patterns align with the mechanism differences summarized in Table 6: both HybridLLM and Minions treat subtasks independently and lack awareness of subtask position or dependencies. Overall, AIMS's advantage comes from accounting for subtask dependencies and position when offloading subtasks to the SLM, improving both accuracy and cost-efficiency.

