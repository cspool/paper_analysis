# <span id="page-5-2"></span>5 Experiments

In this section, we validate the effectiveness of our method by considering two use cases: expert pruning and expert activation pruning. In Sec. [5.1,](#page-6-0) we introduce the experimental settings. In Sec. [5.2,](#page-6-1) we investigate the first use case, expert pruning, by applying EEP to reduce the total number of experts. In Sec. [5.3,](#page-7-0) we further explore expert activation pruning, applying EEP to maintain performance while reducing the number of active experts by changing the top-2 routing weights to top-1. We also examine a composite case where both the total number of experts and the number of active experts are reduced. In Sec. [5.4,](#page-8-0) we present the experimental results on larger and more diverse datasets, as well as performance on out-of-distribution datasets, to validate the generalization ability of EEP. In Sec. [5.5,](#page-8-1) we profile memory usage and inference speed to demonstrate that our method achieves significant improvements compared to the full SMoE models. In Sec. [5.6](#page-9-0) we provide insights on the observation of fewer experts but higher performance. More results, including experiments on larger datasets and other models, can be found in App. [D.](#page-17-0)

<span id="page-6-2"></span>Table 1: Results of expert pruning on Mixtral 8×7B-Instruct. **Bold** values indicate the best across all methods; <u>underlined</u> values show the best without parameter updates (i.e., excluding EEP (Prune+Merge)).

| Expert | Method               | COPA        | MultiRC     | WIC         | WSC  | RTE  | BoolQ       | СВ          | ReCoRD | DROP | SQuAD       | Avg.        |
|--------|----------------------|-------------|-------------|-------------|------|------|-------------|-------------|--------|------|-------------|-------------|
| Num=8  | Full Model           | 89.0        | 83.0        | 51.8        | 63.5 | 73.2 | 77.4        | 51.7        | 50.3   | 30.6 | 53.4        | 62.4        |
| Num=4  | Random               | 63.8        | 49.4        | 37.6        | 43.3 | 45.1 | 50.2        | 38.7        | 35.1   | 27.4 | 58.3        | 44.9        |
|        | Frequency [37]       | 63.0        | 74.8        | 36.0        | 34.6 | 18.1 | 71.0        | 30.4        | 41.6   | 29.9 | 58.2        | 45.8        |
|        | Soft Activation [37] | 73.0        | 30.6        | 51.4        | 37.5 | 41.9 | 40.4        | 17.9        | 36.8   | 33.3 | 10.2        | 37.3        |
|        | NAEE [34]            | 87.0        | 76.0        | 52.6        | 64.5 | 61.7 | 77.2        | 51.7        | 50.4   | 30.6 | 53.0        | 60.5        |
|        | EEP (Prune Only)     | 95.0        | 81.2        | 57.8        | 67.3 | 74.0 | 82.8        | 69.6        | 60.0   | 37.3 | 75.2        | 70.3        |
|        | EEP (Prune+Merge)    | <b>99.0</b> | <b>84.6</b> | <b>65.0</b> | 73.1 | 76.9 | <b>84.8</b> | <b>75.0</b> | 63.6   | 39.7 | <b>80.6</b> | 74.2        |
| Num=2  | Random               | 36.8        | 22.3        | 13.6        | 15.0 | 28.4 | 15.5        | 38.6        | 16.9   | 18.3 | 36.9        | 24.2        |
|        | Frequency [37]       | 51.0        | 17.6        | 8.8         | 1.9  | 48.4 | 30.6        | 35.7        | 10.4   | 14.9 | 9.2         | 24.9        |
|        | Soft Activation [37] | 33.0        | 18.2        | 49.4        | 18.5 | 15.2 | 1.8         | 32.1        | 4.4    | 11.7 | 50.0        | 23.4        |
|        | NAEE [34]            | 75.0        | 42.4        | 48.4        | 49.0 | 54.5 | 49.8        | 19.6        | 42.0   | 31.2 | 58.2        | 47.0        |
|        | EEP (Prune Only)     | 76.0        | 63.8        | 51.8        | 63.5 | 64.3 | 70.6        | 58.9        | 47.2   | 37.1 | 64.0        | 59.7        |
|        | EEP (Prune+Merge)    | <b>93.0</b> | <b>71.6</b> | <b>58.6</b> | 65.4 | 69.0 | <b>75.6</b> | <b>66.1</b> | 47.2   | 38.4 | <b>70.2</b> | <b>65.6</b> |

#### <span id="page-6-0"></span>5.1 Experimental settings

Our main results are based on the popular SMoE models Mixtral 8×7B [20]. We also include a larger model, Mixtral 8×22B [20], to demonstrate the generalization of our methods. We use the "Instruct" version of these models for the generation tasks. We select tasks from the SuperGLUE dataset, as well as several other generation tasks, including SQuAD [41] and DROP [13]. For each individual dataset, we randomly sample a subset from the training set to conduct evolutionary search and use the test set for evaluation. Additional details can be found in App. A.

**Evaluation.** We adopt a generation-based evaluation approach for all datasets. Specifically, we use the instruction fine-tuned model to generate answers directly in response to the given questions and apply template matching to determine the correctness of the answers. Our evaluation protocol primarily follows the implementation of OpenCompass [11] for the design of question prompts, types of templates, and matching criteria, with a few modifications to better suit the Mixtral family of models. All experiments use the same evaluation settings. Examples of prompts and model outputs can be found in App. E and App. F.

Baselines. Since our method aims to compress the instruction fine-tuned SMoE models on down-stream tasks, we consider the zero-shot performance as our main baseline to show that EEP can achieve a significant decrease on the memory footprint and/or computation overhead during the inference time while maintain or even achieve better performance. For the use case of decreasing the total number of experts, we additionally compare EEP with four other types of baseline to demonstrate the effectiveness of the designed search space and the evolutionary-search-based tuning method: (1) Random selection of pruned experts, (2&3) Pruning the experts with the lowest frequency of being activated or the lowest soft activation values [37], and (4) NAEE [34], which exhaustively evaluates the loss between the full model and all pruning choices for each layer and select the one with the lowest loss. For the use case of decreasing the active number of experts, we select the dynamic skipping method proposed by NAEE [34] as an additional baseline. More details are given in App. A.

#### <span id="page-6-1"></span>5.2 Reducing the total number of experts

We apply EEP to search for the optimal pruning configuration, parameterized by the router mapping matrix  $W_{\rm RM}$  and the expert merging matrix  $W_{\rm EM}$ , for maintaining 4 experts and 2 experts. EEP (Prune Only) indicates the results from solely conducting the expert pruning phase as described in Sec. 4.2. In contrast, EEP (Prune + Merge) shows the results after the complete evolutionary search process. The results are shown in Tab. 1, and we discuss them below. Random is conducted 30 times, and we present the mean results here, deferring the complete results to App. D.4.

**EEP fully exploits expert-wise redundancy on downstream tasks**. Based on the results obtained from the pruning phase of EEP, retaining only 4 experts allows the model to achieve better performance and lower computational costs simultaneously on most datasets, except for MultiRC. Even with a particularly low budget of retaining only 2 experts, EEP can still achieve comparable or even better performance than the full model on five datasets, with some datasets showing significant

<span id="page-7-1"></span>Table 2: Results of expert pruning on Mixtral 8 × 22B-Instruct. Bold values indicate the best across all methods; underlined values show the best without parameter updates (i.e., excluding EEP (Prune+Merge)).

| Budget | Method               | WIC  | WSC  | BoolQ | CB   | SQuAD | Avg. |
|--------|----------------------|------|------|-------|------|-------|------|
| Num=8  | Full Model           | 68.2 | 81.7 | 90.2  | 46.5 | 45.8  | 66.5 |
| Num=4  | Random               | 27.0 | 30.2 | 37.8  | 34.6 | 37.2  | 33.4 |
|        | Frequency [37]       | 0.0  | 38.5 | 76.6  | 57.1 | 50.6  | 30.6 |
|        | Soft Activation [37] | 25.2 | 60.6 | 6.4   | 60.7 | 54.2  | 41.4 |
|        | NAEE [34]            | 64.0 | 68.3 | 78.4  | 33.9 | 52.4  | 59.4 |
|        | EEP (Prune Only)     | 70.2 | 84.2 | 89.6  | 75.0 | 71.4  | 78.1 |
|        | EEP (Prune+Merge)    | 72.2 | 87.5 | 89.6  | 78.6 | 74.0  | 80.4 |
| Num=2  | Random               | 13.9 | 10.1 | 11.0  | 24.9 | 15.6  | 15.1 |
|        | Frequency [37]       | 0.0  | 0.0  | 0.0   | 0.0  | 0.0   | 0.0  |
|        | Soft Activation [37] | 2.4  | 1.9  | 3.6   | 19.6 | 52.6  | 16.0 |
|        | NAEE [34]            | 34.0 | 32.7 | 45.0  | 16.1 | 50.0  | 30.6 |
|        | EEP (Prune Only)     | 57.8 | 63.5 | 76.0  | 50.0 | 71.0  | 63.7 |
|        | EEP (Prune+Merge)    | 59.6 | 65.4 | 76.4  | 58.9 | 75.0  | 67.1 |

improvements over the best baseline (e.g., 58.9 vs. 51.7 on CB and 64.0 vs. 53.4 on SQuAD). For the remaining datasets, model collapse is avoided.

EEP is more effective than other baseline methods for selecting pruned experts. Comparing the results of other methods, we find that EEP is more effective for identifying the optimal pruning pattern. Random sampling of experts results in low mean accuracy and high variance. Pruning experts based on selection frequency also performs poorly on most datasets and has a high probability of collapse under high sparsity. NAEE can nearly maintain the performance of the full model when retaining four experts. However, EEP surpasses all methods by a large margin across all datasets.

Expert merging brings significant improvements after pruning. As shown in the last row for each pruning rate in Tab. [1,](#page-6-2) the results after expert merging exceed those obtained through the expert pruning phase alone. Specifically, expert merging achieves a general improvement on almost all datasets. On WIC, CB, and SQuAD under both pruning rates, and on WSC when four experts are retained, the accuracy improvement reaches 5%∼7%, demonstrating its effectiveness in restoring the knowledge of pruned experts and enhancing individual experts. Additionally, we find expert merging to be an effective method for fine-tuning SMoE LLMs (i.e., keeping the number of total and active experts); the results of this are presented in Tab. [9.](#page-18-1)

Generality across models. With the promising results of Mixtral 8×7B-Instruct model, we further apply EEP to a larger model: Mixtral 8×22B-Instruct [\[20\]](#page-11-4), Qwen1.5-MoE-A2.7B-Chat [\[4\]](#page-10-2), and Qwen2-MoE-A14B-Chat [\[40\]](#page-13-2). We conduct experiments on fewer datasets due to the constraint of computational resource. Results are shown at Tab. [2,](#page-7-1) Tab. [7,](#page-17-1) and Tab. [8,](#page-18-2) respectively. EEP also achieves a strong improvement and above observations are still held, which indicates the scaling-up ability of EEP towards large SMoE models.

#### <span id="page-7-0"></span>5.3 Reducing the number of active experts

Next, we present the experimental results for the second use case: decreasing the number of active experts. We modify the number of active experts by changing the top-k from k = 2 to 1 while applying EEP to restore model performance. We evaluate our method with two different total numbers of experts (8 and 4). The results are presented in Tab. [3.](#page-8-2) We summarize the observations below.

EEP can improve individual experts through expert merging, allowing a single expert to handle the inference. Keeping the total number of experts at 8 and reducing the number of active experts to 1 consistently leads to a decline in baseline performance. However, by optimizing the model with EEP, we introduce a reliable improvement that mitigates this gap, resulting in comparable or even better performance than the full model. It is important to note that when the total number of experts is maintained, there is no expert pruning phase; only expert merging is applied for EEP.

<span id="page-8-2"></span>Table 3: Results of active expert pruning on Mixtral 8 × 7B. Bold values show the best performance. "Active" indicates the average number of experts active per token. Avg. stands for average.

| Total | Active       | Method                     | WIC          | WSC          | BoolQ        | CB           | SQuAD        | Avg.         |
|-------|--------------|----------------------------|--------------|--------------|--------------|--------------|--------------|--------------|
|       | 2            | Full Model                 | 51.8         | 63.5         | 77.4         | 51.7         | 53.4         | 59.6         |
| 8     | 1<br>1.4∼1.5 | Full Model<br>Dyn [34]     | 50.8<br>50.0 | 48.1<br>59.6 | 66.0<br>72.8 | 48.2<br>46.4 | 43.8<br>44.8 | 51.4<br>54.7 |
|       | 1            | EEP                        | 59.2         | 70.2         | 79.0         | 66.1         | 51.8         | 65.3         |
| 4     | 1<br>1.4∼1.5 | NAEE [34]<br>NAEE+Dyn [34] | 48.6<br>43.4 | 20.2<br>61.5 | 56.2<br>36.2 | 33.9<br>53.6 | 51.8<br>53.4 | 42.1<br>49.6 |
|       | 1            | EEP                        | 55.8         | 70.2         | 74.4         | 64.3         | 72.0         | 67.3         |

<span id="page-8-3"></span>Table 4: Results of expert pruning on Mixtral 8×7B-Instruct on MMLU dataset. Bold values indicate the best performance; underlined values show the best without updating remaining parameters.

| Budget | Method               | IID (50 val. sets) | OOD (7 unseen datasets) |
|--------|----------------------|--------------------|-------------------------|
| Num=8  | Full Model           | 60.7               | 72.6                    |
|        | Random               | 53.0±9.6           | 64.6±10.0               |
|        | Frequency [37]       | 35.2               | 35.0                    |
| Num=6  | Soft Activation [37] | 54.3               | 65.6                    |
|        | NAEE [34]            | 57.5               | 69.4                    |
|        | EEP (Prune Only)     | 59.6               | 71.4                    |
|        | EEP (Prune+Merge)    | 61.8               | 71.3                    |
|        | Random               | 45.1±6.1           | 50.3±10.7               |
|        | Frequency [37]       | 26.6               | 25.2                    |
| Num=4  | Soft Activation [37] | 46.7               | 53.1                    |
|        | NAEE [34]            | 53.5               | 63.6                    |
|        | EEP (Prune Only)     | 55.4               | 62.4                    |
|        | EEP (Prune+Merge)    | 56.9               | 64.6                    |

The two use cases can be combined through EEP. By retaining fewer experts and simultaneously reducing the number of active experts, we achieve significant savings *in both GPU memory and inference time* (see Sec. [5.5\)](#page-8-1). EEP can be directly applied in this scenario. Results show that with 4 total experts and 1 active expert, EEP achieves performance comparable to or even better than the full model.

#### <span id="page-8-0"></span>5.4 In-distribution and out-of-distribution generalization on diverse datasets

In this section, we further test EEP on a larger dataset, MMLU, to validate the generalization ability of EEP. We randomly split all 57 datasets in MMLU into two subsets containing 50 and 7 datasets, as the base dataset and the out-of-distribution (OOD) test dataset, respectively. We further divide each dataset in the larger subset into training and validation sets. We conduct our EEP on the training sets and use both the validation sets and the OOD test dataset to evaluate the performance of the searched patterns. Results are shown in Tab. [4.](#page-8-3) We find that EEP outperforms baseline methods on both the base dataset and the OOD test dataset. This indicates that EEP possesses the ability to handle large and diverse datasets and exhibits a certain level of generalization capability.

