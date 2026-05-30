# <span id="page-8-0"></span>4 Ablations and analysis

We provide detailed ablations and analysis as follows:

- Enabling cross capabilities, [Section 4.1:](#page-8-1) We evaluate how BTS performs on cross capabilities, or capabilities at the intersection of two or more expert specialties and compare to other merging techniques.
- BTS architecture design, [Section 4.2:](#page-9-0) We empirically validate several BTS architecture choices, including assessing the impact of the number of stitch layers, the alternating stitch layer design, and choice of hub model.
- Interpreting the BTS stitch layers, [Section 4.3:](#page-10-0) Finally, we provide visualizations and analysis of how the BTS stitch layer gate values behave at inference time for various downstream tasks.

## <span id="page-8-1"></span>4.1 Enabling cross capabilities

In addition to evaluating merged models on the union of the expert capabilities, we also explore whether merged models can demonstrate entirely new capabilities at the intersection of expert specialties [\(Zhong](#page-16-4) [et al.,](#page-16-4) [2024\)](#page-16-4). For example – can a Russian-language expert and a Math expert be combined in such a way that the merged model performs better than either expert at Russian math tasks? We refer to these as cross capabilities.

#### 4.1.1 Cross capabilities experimental set-up

In order to evaluate cross capabilities, we train an additional Russian-language expert specifically on Russian data, and all merged models are created with only the Russian and Math experts. We make these choices in order to study cross capability emergence in a controlled setting:

- Reducing cross capability expert contamination: We found that our coding data contained significant portions of non-English natural language, affecting the Code expert's ability in multilingual reasoning tasks, so we remove this model from this mix [\(Blevins and Zettlemoyer,](#page-14-11) [2022\)](#page-14-11). We further remove the seed model which contains both multilingual and math data.
- Prevalance of cross capability training and evaluation data: We limit our study to languages in which we have cross capability data to both train and evaluate the models on — for this reason, we focused on Russian and Math.

Note that when merging only two experts, there is no notion of "hub" model: the stitch layers alternate between merging Russian-into-Math and Math-into-Russian.

During the expert merging or expert upcycling training phase, we train on 2B tokens of Russian mathematics data extracted from web data using a combination of language identification (LID) and math classifiers. We found this additional cross capability in-domain training data was essential. Without it, all merged models struggle to achieve good cross capability performance (see experiments in [Appendix A\)](#page-17-0).

We introduce an additional baseline via continued pretraining the strongest dense model, the seed model, in a data-matched manner on the Russian mathematics data. This is to evaluate the impact of training on in-domain data without increasing the overall model capacity. Additional details of the experimental set-up are provided in [Appendix A.](#page-17-0) All models are evaluated on the Russian subset of MGSM (8-shot; [Shi et al.](#page-15-10) [2022\)](#page-15-10), which are Russian translations of examples from GSM8K [\(Cobbe et al.,](#page-14-9) [2021\)](#page-14-9).

<span id="page-9-1"></span>

|                  | Flores |       |       |           |  |  |  |  |
|------------------|--------|-------|-------|-----------|--|--|--|--|
|                  | GSM8K  | En/Ru | Ru/En | Ru-MGSM   |  |  |  |  |
| Dense models     |        |       |       |           |  |  |  |  |
| Seed Model       | 10.5   | 22.8  | 32.8  | 12.8      |  |  |  |  |
| Math Expert      | ∗20.5  | 10.2  | 28.9  | 10.8      |  |  |  |  |
| Russian Expert   | 9.48   | ∗32.3 | 34.6  | 9.60      |  |  |  |  |
| Seed Model (DM)  | 12.6   | 24.8  | 32.8  | 14.0      |  |  |  |  |
| Expert upcycling |        |       |       |           |  |  |  |  |
| BTX Sample       | 15.6   | 29.9  | 34.3  | 17.6      |  |  |  |  |
| BTX Soft         | 17.6   | 30.6  | 34.5  | 17.6      |  |  |  |  |
| BAM              | 19.3   | 30.9  | 34.5  | ∗<br>18.4 |  |  |  |  |
| Expert merging   |        |       |       |           |  |  |  |  |
| Model Soup       | 17.5   | 14.7  | 32.3  | 13.2      |  |  |  |  |
| BTM              | 20.5   | ∗32.3 | 34.6  | 9.60      |  |  |  |  |
| Expert Routing   | 9.48   | ∗32.3 | 34.6  | 9.60      |  |  |  |  |
| BAM Adapters     | 15.2   | 31.0  | 34.3  | 15.6      |  |  |  |  |
| BTS              | 13.3   | 31.9  | ∗34.7 | 16.0      |  |  |  |  |

Table 4 Cross capability performance. We evaluate the seed model, Russian-language, and Math experts on the Russian subset of MGSM [\(Shi et al.,](#page-15-10) [2022\)](#page-15-10). We compare their performance with expert merging and expert upcyling baselines trained with small amounts of in-domain data on Russian mathematics. We also continued pretraining the strongest dense model, the seed model on the same in domain data. We call the resulting baseline "Seed Model Data Matched (DM)". Bolded numbers indicate the best performance among dense models or merged models, while an asterisk (<sup>∗</sup> ) denotes the best performance across all models. BTS outperforms the data-matched seed model, and achieves the best cross capability performance among all expert merging methods. This demonstrates that with only a small amount of in-domain data, BTS models can effectively learn how to combine expert capabilities.

#### 4.1.2 Cross capabilities results

See [Table 4](#page-9-1) for cross capability results on Russian MGSM. Notably, we see that BTS can effectively leverage both experts to excel at a new task, surpassing the data-matched seed model baseline, even though the experts themselves remain unchanged: by adding connections between them, the resulting model exceeds the sum of its individual parts. Among all expert-merging baselines, BTS achieves the best cross capability performance. BTX and BAM variants also show strong performance, outperforming BTS, likely due to their significantly greater training capacity on in-domain data.

## <span id="page-9-0"></span>4.2 BTS architecture design

We ablate the impact of the number of stitch layers, the alternating stitch layer architecture, and the hub model selection.

#### 4.2.1 Impact of the number of stitch layers

<span id="page-9-2"></span>

|           | General |      | Code |           |           | Multilingual | Math  |      |      |
|-----------|---------|------|------|-----------|-----------|--------------|-------|------|------|
|           | MMLU    | BBH  | MBPP | HumanEval | Flores(S) | Flores(T)    | GSM8K | MATH | Avg. |
| 10 Layers | 36.1    | 37.8 | 31.8 | 22.0      | 31.2      | 36.5         | 19.1  | 10.4 | 28.1 |
| 4 Layers  | 35.8    | 36.9 | 32.2 | 22.0      | 33.9      | 36.2         | 20.2  | 10.6 | 28.1 |
| 1 Layer   | 34.9    | 37.8 | 29.6 | 19.5      | 30.8      | 35.9         | 17.7  | 9.9  | 27.0 |

Table 5 Ablations on the effect of varying number of stitch layers on downstream task performance. The first two rows are configurations with 10 and 4 stitch layers distributed uniformly throughout the seed and expert models. The third row is a configuration with a single Experts-into-Hub stitch layer placed after the last dense model layers. The 10 and 4 layers configuration performs similarly, but the single-layer configuration lags behind model performance significantly.

We measure the impact of varying the number of stitch layers on model performance, as shown in [Table 5.](#page-9-2) The first two rows present configurations with 10 and 4 stitch layers, respectively, distributed uniformly throughout the seed and expert models. In the third row, we investigate a configuration with a single Experts-into-Hub stitch layer placed after the final language model layers.

Our ablations show that a single stitch layer is insufficient for learning to effectively merge capabilities, as its performance lags significantly behind configurations with 4 or 10 layers. This also demonstrates that the BTS models with more than one stitch layer combine models in a more expressive way than than simply combining output representations. The 4 and 10 layer configurations perform similarly, however, we note that this may be due to under-training of the 10 layer variant as all models are trained on the same number of tokens.

#### 4.2.2 Importance of the alternating stitch layer architecture

The BTS architecture involves alternating between the Experts-into-Hub stitch layer and the Hub-into-Experts stitch layer. We ablate the impact of adopting this alternating architecture, as opposed to utilizing all Experts-into-Hub layers. As shown in [Table 6,](#page-10-1) the alternating architecture (first row) yields significantly better cross capability performance compared to using only homogeneous Experts-into-Hub stitch layers (second row). However, both the alternating and non-alternating architectures achieve comparable performance on generalist tasks, as shown in [Table 7.](#page-10-2) These results demonstrate that an alternating architecture is essential for achieving cross capability performance while maintaining strong generalist performance.

<span id="page-10-1"></span>

|                                              | Flores       |              |              |              |  |  |  |
|----------------------------------------------|--------------|--------------|--------------|--------------|--|--|--|
|                                              | GSM8K        | En/Ru        | Ru/En        | Ru-MGSM      |  |  |  |
| BTS Alternating<br>BTS Experts-into-Hub Only | 13.3<br>15.2 | 31.9<br>32.0 | 34.7<br>35.0 | 16.0<br>11.6 |  |  |  |

Table 6 Comparison of alternating and non-alternating BTS variants cross capabilities tasks with additional in-domain Russian math training data. The alternating variant significantly outperforms the non-alternating variant.

<span id="page-10-2"></span>

|                      | General |      | Code |      | Multilingual |           | Math  |      |      |  |
|----------------------|---------|------|------|------|--------------|-----------|-------|------|------|--|
|                      | MMLU    | BBH  | MBPP | HE   | Flores(S)    | Flores(T) | GSM8K | MATH | Avg. |  |
| BTS Alternating      | 35.8    | 36.9 | 32.2 | 22.0 | 30.9         | 36.2      | 20.2  | 10.6 | 28.1 |  |
| All Experts-into-Hub | 36.1    | 37.9 | 32.4 | 22.6 | 31.4         | 36.4      | 19.9  | 10.8 | 28.4 |  |

Table 7 Comparison of alternating and non-alternating BTS variants on generalist tasks. Both variants achieves similar performance on most domains, with the non-alternating variant slightly outperforming the alternating variant on average.

#### 4.2.3 Impact of hub model selection

By default, we always use the seed model as the hub model in BTS. This design choice is motivated from the fundamental nature of the seed model: as all experts are initialized from the seed model, the seed model's representations are more closely aligned with the experts' than the experts' are with each other, which may allow for more effective merging of representations via the BTS stitch layers.

To validate this hypothesis, we conduct an ablation study in which we use an expert model as the hub instead. Specifically, we select the Math expert for this experiment, as it has the best generalist performance among all expert models. The seed model then is used as one of the "experts" or spoke models in BTS. As shown in [Table 8,](#page-11-0) the results indicate that across most downstream tasks, selecting the seed model as the hub significantly outperforms using an expert model as the hub, validating this design choice.

## <span id="page-10-0"></span>4.3 Interpreting the BTS stitch layers

The gate values of the Experts-into-Hub stitch layer determine the weight of each expert in the combined representation. Intuitively, the higher the expert or seed model's gate values, the more important this model

<span id="page-11-0"></span>

|          | General |      | Code |      | Multilingual |           | Math  |      |      |
|----------|---------|------|------|------|--------------|-----------|-------|------|------|
|          | MMLU    | BBH  | MBPP | HE   | Flores(S)    | Flores(T) | GSM8K | MATH | Avg. |
| Seed Hub | 35.8    | 36.9 | 32.2 | 22.0 | 30.9         | 36.2      | 20.2  | 10.6 | 28.1 |
| Math Hub | 33.9    | 37.8 | 30.7 | 20.1 | 29.8         | 36.0      | 15.6  | 5.73 | 26.2 |

Table 8 Comparison of utilizing the seed model as the hub versus an expert. We ablate BTS (row 1) with a variant where we instead use the Math expert model as the hub (row 2). Using the seed model as the hub significantly outperforms using an expert model as the hub across most downstream tasks. This confirms that using the seed model as the hub in BTS is important for achieving strong generalist performance.

is for the task. We inspect these values to get insight into the model's decision-making progress on various tasks.

#### 4.3.1 Visualizing gate values on expert specialty tasks

[Figure 2](#page-11-1) visualizes how the gate values of the last stitch layer, an Experts-into-Hub stitch layer, vary when generating a sequence during inference on various expert specialty tasks. The first row plots the gate values for prompt tokens, while the second row plots the gate values for the generated tokens. Each column corresponds to a different prompt, sampled from the corresponding benchmark task.

This visualization shows that the gate values align closely with the task requirements – with the specialized expert associated with the task typically dominating the gate values – while effectively mixing representations from different models over the course of the sequence. For example, for the the math task, GSM8K, the math expert has the highest gate value over the course of the generation while the other models' gate values are nearly zero. For language translation task, Flores, the multilingual expert and the seed model dominate, with each model being relied on more heavily at different parts of the prompt or generation.

<span id="page-11-1"></span>![](_page_11_Figure_6.jpeg)

Figure 2 Visualization of how BTS gate values vary when generating a sequence during inference. We inspect the gate values for the last stitch layer over the course of a sequence. The first row plots the gate values for prompt tokens, while the second row plots the gate values for the generated tokens. Each column corresponds to a different prompt, sampled randomly from the corresponding benchmark task.

#### 4.3.2 Visualizing gate value transitions on context-switching tasks

Unlike merge methods which make sequence-level choices about which expert to use, BTS can effectively context switch over the course of the sequence, seamlessly transitioning between different tasks. [Figure 3](#page-12-0) illustrates the gate values of BTS's final stitch layer when processing context-switching prompts. These prompts are constructed by concatenating examples from Flores (3-shot), GSM8K (2-shot), and TriviaQA (2-shot) [\(Joshi et al.,](#page-15-11) [2017\)](#page-15-11), in that order, with dotted lines indicating where a new task begins. Each column corresponds to a different context-switching prompt, created from distinct sampled inputs.

In both examples, BTS demonstrates its ability to dynamically adjust expert utilization. During the Flores prompt, the seed model and multilingual expert are predominantly active. During the GSM8K prompt, the math expert takes over, and finally, the seed model is most utilized for the TriviaQA prompt. This highlights BTS's capability to correctly activate the relevant experts for each task, even when transitioning between diverse contexts.

<span id="page-12-0"></span>![](_page_12_Figure_3.jpeg)

Figure 3 Visualization of the gate values of BTS's final stitch layer for context-switching sequences at inference time. These sequences are constructed by concatenating question-answer examples from Flores (3-shot), GSM8K (2-shot), and TriviaQA (2-shot), in that order, with dotted lines indicating task transitions. Each plot corresponds to a different randomly sampled prompt. This visualization highlights BTS's ability to dynamically adjust expert utilization based on token-level context.

