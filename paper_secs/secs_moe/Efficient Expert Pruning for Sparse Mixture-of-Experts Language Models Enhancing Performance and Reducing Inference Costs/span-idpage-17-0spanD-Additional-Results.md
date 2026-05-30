# <span id="page-17-0"></span>D Additional Results

#### D.1 Results with other models

In this section, we further apply EEP to the Qwen 1.5 [\[4\]](#page-10-2) and Qwen 2 [\[40\]](#page-13-2) SMoE models. Results can be found in Tab. [7](#page-17-1) and Tab. [8.](#page-18-2) The same observations in Sec. [5](#page-5-2) hold for these models: (1) EEP selects better pruning patterns than other baseline methods without updating the remaining parameters, and (2) expert merging brings improvements in most cases.

For the Qwen1.5-MoE-A2.7B-Chat [\[4\]](#page-10-2), we notice that other methods are prone to collapse. Conversely, the situation is the opposite for the Qwen2-MoE-A14B-Chat model [\[40\]](#page-13-2). Most baseline methods can maintain the performance of the full model with an extremely low number of experts retained. In face, we observe that the experts in the Qwen2-MoE-A14B-Chat model are specifically homogeneous, as the model's performance is largely maintained even when only one random expert is activated per token. However, according to the information provided in their technical report, both Qwen1.5-MoE-A2.7B andQwen2-MoE-A14B employ upcycling and 64 experts per layer. We thus speculate that other training configurations, such as sizes and optimizer hyperparameters, lead to different final statuses. Nevertheless, EEP always achieves comparable or better performance than the full model and outperforms all baseline methods across settings, demonstrating its adaptability to different SMoE models.

#### D.2 Fine-tuning using EEP

EEP can also be applied to fine-tune the model without pruning. As shown in Tab. [9,](#page-18-1) the effectiveness of EEP in fine-tuning demonstrates the efficiency of expert merging. Notably, EEP does not compute gradients and can therefore be executed on devices capable of inference.

## D.3 Profiling Results

We notice that the speedup ratio brought by pruning experts is influenced by the batch size. Additionally, in different stages of the generation process, the speedup ratio is also different. Therefore, we report more detailed profiling results of Mixtral 8 × 7B model in Tab. [10.](#page-19-3)

<span id="page-18-2"></span>Table 8: Results of expert pruning on Qwen2-MoE-A14B-Chat. Bold values indicate the best performance; underlined values show the best without updating remaining parameters. For NAEE, due to the excessive number of pruning patterns, we only randomly select 2k of them for each layer.

| Budget | Method               | WIC      | WSC      | BoolQ    | CB        | SQuAD    | Avg. |
|--------|----------------------|----------|----------|----------|-----------|----------|------|
| Num=64 | Full Model           | 60.2     | 68.3     | 88.8     | 67.9      | 74.4     | 71.9 |
|        | Random               | 55.3±7.1 | 61.6±5.6 | 78.7±7.3 | 35.4±17.6 | 79.7±2.4 | 62.1 |
|        | Frequency [37]       | 58.8     | 59.6     | 79.4     | 46.4      | 78.2     | 64.5 |
|        | Soft Activation [37] | 60.8     | 64.4     | 82.6     | 14.3      | 75.2     | 59.5 |
| Num=8  | NAEE [34]            | 56.6     | 60.6     | 82.6     | 41.1      | 81.2     | 64.4 |
|        | EEP (Prune Only)     | 61.8     | 72.1     | 85.8     | 76.8      | 85.6     | 76.4 |
|        | EEP (Prune+Merge)    | 63.4     | 75.0     | 85.8     | 85.7      | 87.0     | 79.4 |
|        | Random               | 56.5±1.9 | 59.8±5.2 | 79.1±4.0 | 32.1±15.0 | 78.0±2.4 | 61.1 |
|        | Frequency [37]       | 56.8     | 60.6     | 83.2     | 17.9      | 80.0     | 59.7 |
|        | Soft Activation [37] | 59.2     | 61.5     | 81.6     | 17.9      | 77.6     | 59.6 |
| Num=4  | NAEE [34]            | 55.0     | 61.5     | 75.8     | 21.4      | 79.6     | 58.7 |
|        | EEP (Prune Only)     | 62.0     | 65.4     | 84.6     | 69.6      | 80.6     | 72.4 |
|        | EEP (Prune+Merge)    | 63.8     | 72.1     | 85.8     | 80.4      | 84.2     | 77.3 |
|        | Random               | 56.4±1.4 | 58.2±3.7 | 77.8±4.5 | 26.5±9.6  | 76.4±1.9 | 59.1 |
|        | Frequency [37]       | 58.0     | 60.6     | 79.6     | 42.9      | 72.4     | 62.7 |
|        | Soft Activation [37] | 57.4     | 65.4     | 71.4     | 62.5      | 76.8     | 66.7 |
| Num=2  | NAEE [34]            | 55.6     | 56.7     | 73.4     | 16.1      | 75.0     | 55.4 |
|        | EEP (Prune Only)     | 59.2     | 68.3     | 83.4     | 67.9      | 82.0     | 72.2 |
|        | EEP (Prune+Merge)    | 61.0     | 70.2     | 84.4     | 76.8      | 83.8     | 75.2 |
|        | Random               | 56.6±1.3 | 56.3±2.7 | 78.7±1.5 | 23.5±5.9  | 75.2±1.6 | 58.1 |
|        | Frequency [37]       | 52.2     | 62.5     | 78.6     | 35.7      | 77.0     | 61/  |
|        | Soft Activation [37] | 57.8     | 63.5     | 77.4     | 42.9      | 76.0     | 63.5 |
| Num=1  | NAEE [34]            | 57.6     | 56.7     | 78.6     | 16.1      | 73.6     | 56.5 |
|        | EEP (Prune Only)     | 57.8     | 65.4     | 82.6     | 57.1      | 81.4     | 68.5 |
|        | EEP (Prune+Merge)    | 59.4     | 69.2     | 84.0     | 82.1      | 82.8     | 75.5 |

Table 9: Results of fine-tuning on Mixtral 8 × 7B using EEP.

<span id="page-18-1"></span>

| Method   | WSC  | WIC  | RTE  | BoolQ | CB   | Record | SQuAD | DROP | Average |
|----------|------|------|------|-------|------|--------|-------|------|---------|
| Baseline | 63.5 | 51.8 | 73.2 | 77.4  | 51.7 | 50.3   | 53.4  | 30.6 | 56.5    |
| EEP      | 78.8 | 69.2 | 78.7 | 86.2  | 80.4 | 63.0   | 78.4  | 51.5 | 73.2    |

#### <span id="page-18-0"></span>D.4 Random search

We demonstrate the full results of the random pruning baseline with error margin in Tab. [11](#page-19-2) and Tab. [12.](#page-20-1) From the results we can find that random pruning is extremely unstable, especially under low expert number budget, which indicates the challenge of the expert pruning.

#### <span id="page-18-3"></span>D.5 Ablation study

The hyperparameters of EEP include the number of groups that share the same coefficients, and the number of search iterations.

Number of Groups. We uniformly split all expert weights into a number of groups. We evaluate the results when there are 4 groups (the merging coefficients are shared across layers within the group) and 32 groups (i.e., the merging coefficients of each layer are effectively independent) on RTE, ReCoRD, and DROP. Results are shown in Tab. [13.](#page-20-2) We observe that more groups achieve much better performance in the pruning phase, especially when the number of experts is extremely low. However, dividing weights into more groups introduces more parameters to optimize, which may be detrimental to the expert merging phase. It is validated that the improvements brought by expert

Table 10: Profiling the inference speedup of Mixtral  $8 \times 7B$ .

<span id="page-19-3"></span>

| Total | Active | Method     | P             | refill Spee | dup    | Decode Speedup |       |        |  |
|-------|--------|------------|---------------|-------------|--------|----------------|-------|--------|--|
|       |        |            | BS=1          | BS=32       | BS=256 | BS=1           | BS=32 | BS=256 |  |
| 8     | 2      | Full Model | 1.0×          | 1.0×        | 1.0×   | 1.0×           | 1.0×  | 1.0×   |  |
| Ü     | 1      | EEP        | 1.05×         | 1.58×       | 1.63×  | 1.34×          | 1.06× | 1.02×  |  |
| 4     | 2      | EEP        | 1.47×         | 1.02×       | 1.03×  | 1.05×          | 1.60× | 1.29×  |  |
|       | 1      | EEP        | 1.75×         | 1.77×       | 1.72×  | 1.37×          | 1.60× | 1.33×  |  |
| 2     | 2      | EEP        | $2.00 \times$ | 1.20×       | 1.03×  | 1.15×          | 2.43× | 1.53×  |  |

Table 11: Error margin of ramdom pruning on Mixtral  $8 \times 7B$ .

<span id="page-19-2"></span>

| Expert | Method     | COPA            | MultiRC   | WIC             | WSC             | RTE             | BoolQ           | CB              | ReCoRD    | DROP     | SQuAD     |
|--------|------------|-----------------|-----------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------|----------|-----------|
| Num=8  | Full Model | 89.0            | 83.0      | 51.8            | 63.5            | 73.2            | 77.4            | 51.7            | 50.3      | 30.6     | 53.4      |
| Num=4  | Random     | 63.8±17.5       | 49.4±18.0 | 37.6±17.9       | 43.3±20.8       | 45.1±11.9       | 50.2±21.3       | 38.7±13.8       | 35.1±12.7 | 27.4±4.6 | 58.3±11.6 |
| Num=2  | Random     | $36.8 \pm 14.6$ | 22.3±8.4  | $13.6 \pm 14.8$ | $15.0 \pm 18.1$ | $28.4 \pm 13.4$ | $15.5 \pm 17.1$ | $38.6 \pm 10.8$ | 16.9±7.4  | 18.3±3.2 | 36.9±12.6 |

merging with 4 groups are larger than those with 32 groups. Taking all these factors into account, we use 32 groups for these three datasets and keep 4 groups for the rest of the experiments.

**Search Iterations.** We plot the Accuracy-Iteration curve in Fig. 5. We report the best accuracy among all evaluated merging coefficients at each iteration. From the figure, we can see that the evolutionary search in the pruning phase is effective and efficient, finding good pruning configurations from poor initialization within only 40 iterations. The expert merging phase can further improve performance based on the pruning results.

#### <span id="page-19-1"></span>**D.6** Router Pattern

In Sec. 5.6, we demonstrate the changes in expert activation patterns using the statistics from the first transformer block in a Mixtral  $8 \times 7B$ -Instruct model. Additionally, in this section, we provide the statistics for the  $15^{th}$  transformer block Fig. 6 and the  $31^{st}$  transformer block Fig. 7.

#### **D.7** Demonstration of Searched Patterns

We demonstrate the final searched patterns (pruning + merging) in Fig. 8. There is always one highlighted block in each row, which corresponds to the primarily retained experts in the pruning phase, while other values are close to zero. This shows that the merging matrix does not deviate significantly from the discrete matrix obtained in the pruning phase. However, these slight changes bring significant improvements. Additionally, we observe negative coefficients in some positions, indicating that the knowledge from certain experts may not benefit the downstream task.

#### <span id="page-19-0"></span>E Prompt

We list the prompt we used for each dataset in Tab. 14. We follow the default prompt in the Opencompass codebase [11].

Table 12: Results of random pruning on Mixtral  $8 \times 22B$ .

<span id="page-20-1"></span>

| Budget | Method     | WIC       | WSC       | BoolQ     | СВ        | SQuAD     |
|--------|------------|-----------|-----------|-----------|-----------|-----------|
| Num=8  | Full Model | 68.2      | 81.7      | 90.2      | 46.5      | 45.8      |
| Num=4  | Random     | 27.0±24.7 | 30.2±23.7 | 37.8±32.7 | 34.6±14.1 | 37.2±26.2 |
| Num=2  | Random     | 13.9±15.1 | 10.1±13.2 | 11.0±12.9 | 24.9±15.6 | 15.6±20.3 |

Table 13: Results with different number of coefficient groups.

<span id="page-20-2"></span>

| Group Number | Expert | Method                    | RTE          | DROP         | ReCoRD       |
|--------------|--------|---------------------------|--------------|--------------|--------------|
| 4            | Num=4  | Prune Only<br>Prune+Merge | 62.8<br>71.5 | 35.5<br>38.9 | 59.2<br>63.2 |
|              | Num=2  | Prune Only<br>Prune+Merge | 53.8<br>61.7 | 25.3<br>27.5 | 36.0<br>38.8 |
| 32           | Num=4  | Prune Only<br>Prune+Merge | 74.0<br>76.9 | 37.3<br>39.7 | 60.0<br>63.6 |
|              | Num=2  | Prune Only<br>Prune+Merge | 64.3<br>69.0 | 37.1<br>38.4 | 47.2<br>47.2 |

Table 14: Prompts for all datasets.

<span id="page-20-3"></span>

| Dataset | Prompt                                                                                                                                              |
|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| WIC     | Sentence 1: <sentence1>\nSentence 2: <sentence2> Are 'Areada': in the characters are the sense? In A. Yeah R. Noha Areagan.</sentence2></sentence1> |
|         | Are ' <word>' in the above two sentences the same?\nA. Yes\nB. No\nAnswer: A/B</word>                                                               |
| WSC     | Passage: <text>\n</text>                                                                                                                            |
|         | Does the pronoun #                                                                                                                                  |
| RTE     | <pre><pre><pre><pre><pre><pre><pre><pre></pre></pre></pre></pre></pre></pre></pre></pre>                                                            |
|         | Is the sentence below entailed by the sentence above?\nA. Yes\nB. No\nAnswer: A/B                                                                   |
| BoolQ   | <passage>\n</passage>                                                                                                                               |
|         | Question: question\nA. Yes\nB. No\nAnswer:                                                                                                          |
|         | A/B                                                                                                                                                 |
| CB      | <pre><pre><n<hypothesis>\n</n<hypothesis></pre></pre>                                                                                               |
|         | What is the relation between the two sentences?\nA. Contradiction\nB. Entailment\nC. Neutral\nAnswer: $A/B/C$                                       |
| ReCoRD  | Passage: <text>\nResult: <question>\n</question></text>                                                                                             |
|         | Question: What entity does refer to in the result? Give me the entity name:                                                                         |
| DROP    | \n\nText: <prompt>\n</prompt>                                                                                                                       |
|         | Question: <question>\nAnswer:</question>                                                                                                            |
| SQuAD   | <pre><context>\nAccording to the above passage, answer the following question.</context></pre>                                                      |
|         | If it is impossible to answer according to the passage, answer 'impossible to answer':\n                                                            |
|         | Question: <question></question>                                                                                                                     |

