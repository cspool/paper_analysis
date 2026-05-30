# <span id="page-15-2"></span>A Additional Details on Experimental Settings

#### A.1 Ours setting

Search Space. As mentioned in Sec. [5,](#page-5-2) to avoid optimizing too many parameters, we split the weights of all experts into several groups. The merging coefficients WEM and WRM within the same group are shared. Most of our main results are obtained by uniformly splitting all weights into four groups based on their depth, except for the experiments on the RTE, ReCoR, and DROP datasets in Tab. [1.](#page-6-2) We find that for these datasets, setting each layer as an independent group performs significantly better than using only four groups during the pruning phase. More detailed results can be found in App. [D.5.](#page-18-3) For other datasets, we maintain the current setting without exploring other configurations, as it consistently yields good performance.

Search Process. We apply a two-stage search method as discussed in Sec. [4.2.](#page-4-1) The pruning phase consists of 40 iterations, followed by 160 iterations for the expert merging phase. At each iteration, we evaluate the accuracy on the training set and use this metric as the score for all individuals of merging coefficients in the population. Examples of the performance curve over the search iterations are provided in App. [D.5.](#page-18-3)

Selected Datasets for OOD Evaluation. In Sec. [5.4,](#page-8-0) we randomly select 7 datasets for OOD test. These datasets are: (1) *lukaemon\_mmlu\_electrical\_engineering*, (2) *lukaemon\_mmlu\_professional\_accounting*, (3) *lukaemon\_mmlu\_high\_school\_macroeconomics*, (4) *lukaemon\_mmlu\_high\_school\_computer\_science*, (5) *lukaemon\_mmlu\_business\_ethics*, (6) *lukaemon\_mmlu\_miscellaneous*, and (7) *lukaemon\_mmlu\_high\_school\_psychology*.

#### A.2 Baselines

To evaluate the effectiveness of reducing the total number of experts, we compare our method against four baseline approaches: (1) Random selection of pruned experts, (2) pruning experts with the lowest frequency of activation, (3) pruning experts with the lowest soft activation values, and (4) NAEE [\[34\]](#page-12-4), which exhaustively evaluates the discrepancy between the full model and all pruning choices for each layer and selects the one with the lowest discrepancy. For reducing the number of active experts, we adopt the dynamic skipping scheme from NAEE as a baseline approach.

For random selection, we uniformly sample a corresponding number of experts from all 8 experts in each layer. The full results with error margins for random selection are presented in Tab. [11.](#page-19-2)

For the frequency-based method, we run the model on the training set and count the number of times each expert is activated. We then prune the experts with the lowest frequency in each layer.

For the soft activation method, we run the model on the training set and accumulate the router weighting (soft activation value) for each expert. We then prune the experts with the lowest accumulated values in each layer.

For NAEE, we enumerate all pruning choices for each layer and select the one with the smallest output discrepancy compared to the full model. We use a batch of calibration data with a size of 64 to calculate the discrepancy. For the dynamic skipping scheme, we run the model on the entire training set to determine the median value of the ratio between the two largest routing weights for each layer. During validation, we dynamically skip the expert with the second-largest routing weight if the ratio between its weight and the largest weight is below the threshold. This results in an average of approximately 1.5 active experts.

#### Size of current SMoE LLMs

Tab. 6 shows the basic parameter information of modern SMoE Large LLMs.

<span id="page-16-0"></span>Table 6: Active Parameters, Total Parameters, and Parameters of the Experts for Various Models

| Model              | <b>Active Parameters</b> | <b>Total Parameters</b> | Parameters of Experts |
|--------------------|--------------------------|-------------------------|-----------------------|
| Mixtral 8x7B       | 13B                      | 47B                     | 45B                   |
| Mixtral 8x22B      | 39B                      | 141B                    | 136B                  |
| Grok-1             | 79B                      | 314B                    | 313B                  |
| DBRX               | 36B                      | 132B                    | 128B                  |
| Qwen 1.5-MoE-A2.7B | 2.7B                     | 14.3B                   | 13.2B                 |
| Qwen 2-57B-A14B    | 14B                      | 57B                     | 49B                   |

## **Algorithm Details**

Alg. 1 presents the details of EEP. The notations are consistent with those in Sec. 4.2. For the Crossover operation, we combine the merging coefficients of the parent models along the dimension of the retained experts. For the Mutate operation, we perturb the merging coefficients. Specifically, during the pruning phase, we randomly replace the pruned experts with other experts and set the router weights accordingly. In the expert merging phase, we perturb the merging coefficients element-wise by adding Gaussian noise.

```
Algorithm 1 Evolutionary Search of EEP
```

```
\Theta = \{ \bm{\theta}_1^l, \bm{\theta}_2^l, \cdots, \bm{\theta}_E^l \}_{l=1}^L: Full set of expert weights across all L SMoE blocks.
     \mathcal{F}: The metric evaluator.
Symbols:
     P: The whole Population of matrix configurations.
     CP: The Candidate Parents set of each loop, from which a parent configuration is selected.
     NG: The Next Generation newly mutated from the parent configurations in each loop.
     W = \{W_{EM}^l, W_{RM}^l\}_{l=1}^L: Full set of the search parameters across all L SMoE blocks.
Hyperparameters:
     Epoch: Number of loops for the entire search process.
     \mathbf{M}_{CP}: Maximum size of the candidate parents set CP.
     Iter: Maximum number of mutations in each loop.
Search Process:
 1: P \leftarrow \emptyset
 2: Initialize a set of random matrices W_{\text{init}}, ensuring that each row is a one-hot vector.
 3: P \leftarrow P \cup \{(\mathbf{W}_{init}, \mathcal{F}(\mathbf{W}_{init}))\}
 4: for r = Expert Pruning Phase, Expert Matching Phase do
         for t = 1, \dots, Iters do
            NG \leftarrow \emptyset
 6:
            for i = 1, \cdots, Epochs do
 7:
               CP \leftarrow \{ \mathbf{W}_i | \hat{\mathcal{F}}(\mathbf{W}_i \cdot \Theta) \text{ ranks within the top } min(\mathbf{M}_{CP}, |P|) \text{ in } P \}
 8:
               \boldsymbol{W}_f, \boldsymbol{W}_m \xleftarrow{\text{Random Sample}} CP
 9:
                \dot{\boldsymbol{W}_{new}} \leftarrow \text{Mutate}(\text{Crossover}(\boldsymbol{W}_f, \boldsymbol{W}_m))
10:
               NG \leftarrow NG \cup \{(\grave{W}_{new}, \mathcal{F}(\grave{W}_{new}))\}
11:
12:
            end for
            P \leftarrow P \cup NG
13:
14:
         end for
15: end for
16: W^* \leftarrow \arg\min \mathcal{F}(W)
17: return W^*
```

<span id="page-17-1"></span>Table 7: Results of expert pruning on Qwen1.5-MoE-A2.7B-Chat. Bold values indicate the best performance; underlined values show the best without updating remaining parameters. For NAEE, due to the excessive number of combinatorial possibilities, we only randomly select 5k of them for each layer.

| Budget | Method               | WIC      | WSC      | BoolQ    | CB       | SQuAD     | Avg. |
|--------|----------------------|----------|----------|----------|----------|-----------|------|
| Num=60 | Full Model           | 51.4     | 46.2     | 73.6     | 32.1     | 68.6      | 54.4 |
| Num=30 | Random               | 3.7±12.1 | 7.6±14.3 | 8.1±12.9 | 5.6±8.4  | 19.5±23.0 | 8.9  |
|        | Frequency [37]       | 55.6     | 9.6      | 2.4      | 0.0      | 17.9      | 21.7 |
|        | Soft Activation [37] | 51.4     | 30.8     | 0.4      | 44.6     | 28.0      | 31.0 |
|        | NAEE [34]            | 0.0      | 0.0      | 1.6      | 0.0      | 34.6      | 7.2  |
|        | EEP (Prune Only)     | 59.8     | 59.6     | 78.0     | 71.4     | 70.6      | 67.9 |
|        | EEP (Prune+Merge)    | 62.6     | 66.3     | 81.4     | 76.9     | 71.4      | 71.7 |
| Num=15 | Random               | 1.4±5.9  | 0.5±1.3  | 2.0±4.1  | 4.3±10.6 | 1.1±3.4   | 1.9  |
|        | Frequency [37]       | 0.0      | 0.0      | 7.8      | 16.1     | 0.0       | 4.9  |
|        | Soft Activation [37] | 26.2     | 3.9      | 0.0      | 0.0      | 25.4      | 11.1 |
|        | NAEE [34]            | 0.0      | 1.0      | 5.2      | 0.0      | 0.0       | 1.2  |
|        | EEP (Prune Only)     | 51.0     | 36.5     | 45.4     | 60.7     | 57.6      | 50.2 |
|        | EEP (Prune+Merge)    | 54.4     | 63.5     | 58.2     | 58.9     | 76.9      | 62.7 |

