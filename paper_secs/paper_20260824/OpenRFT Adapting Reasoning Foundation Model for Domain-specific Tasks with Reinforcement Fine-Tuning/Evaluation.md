# Evaluation

<span id="page-5-0"></span><sup>3</sup> In proprietary implementations, such as OpenAI, the policy model and PRM are expected to be aligned, sharing the same action space and state-transition. However, in typical experimental settings, it is more common that only an open-source generalist reasoning model is available acting as the policy model, while the corresponding PRM is often inaccessible.

In our current implementation, both the policy model and the PRM are from the Skywork-o1 series [\(o1 Team,](#page-11-3) [2024\)](#page-11-3). However, the model provider has not explicitly stated that their action spaces are aligned, which may potentially impact the performance of RFT.

<span id="page-6-0"></span>

| Model/ Method      | Biology | Chemistry |      |      | Physics | Materials |      |       | Avg.  |
|--------------------|---------|-----------|------|------|---------|-----------|------|-------|-------|
|                    | T1      | T2        | T3   | T4   | T5      | T6        | T7   | T8    |       |
| GPT-4o-mini        | 0.37    | 0.69      | 0.84 | 0.32 | 0.53    | 0.49      | 0.90 | 0.525 | 0.583 |
| o1-mini            | 0.35    | 0.86      | 0.87 | 0.23 | 0.73    | 0.70      | 0.87 | 0.50  | 0.639 |
| Vanilla            | 0.28    | 0.55      | 0.52 | 0.23 | 0.45    | 0.34      | 0.41 | 0.41  | 0.403 |
| ReFT               | 0.27    | 0.50      | 0.52 | 0.23 | 0.44    | 0.33      | 0.41 | 0.50  | 0.402 |
| ReFT+PRM           | 0.30    | 0.57      | 0.49 | 0.23 | 0.44    | 0.36      | 0.37 | 0.48  | 0.405 |
| SFT                | 0.33    | 0.53      | 0.49 | 0.20 | 0.45    | 0.37      | 0.43 | 0.49  | 0.415 |
| SFT+RL(PRM)        | 0.29    | 0.59      | 0.52 | 0.24 | 0.47    | 0.36      | 0.46 | 0.57  | 0.437 |
| SFT+RL(PRM)+DA     | 0.29    | 0.63      | 0.53 | 0.21 | 0.47    | 0.38      | 0.48 | 0.59  | 0.447 |
| SFT+RL(PRM)+DA+ICL | 0.33    | 0.57      | 0.52 | 0.28 | 0.46    | 0.36      | 0.49 | 0.53  | 0.443 |

Table 1: Accuracy of different models/methods. Bold indicates the highest value, while underline indicates the highest value among the different methods based on the open-source Skywork-o1.

The answer output format is explicitly defined in the prompt to facilitate the extraction of answers from the model's output using predefined rules. This approach allows us to directly compare the model's predictions with the ground truth. We set the maximum sampling length to 2,048. For samples where the answer cannot be identified within this length, we consider the prediction as incorrect in the calculation. We report accuracy values for both methods across all datasets. Since the GPT-4o-mini and o1-mini models are sufficiently robust, we report the results based on a single evaluation. For other methods, we perform three evaluations and report the average accuracy.

### 3.4 RESULTS

The main results are summarized in Table [1.](#page-6-0) Key observations include: (1) o1-mini demonstrated the strongest reasoning capabilities, yet GPT-4o-mini showed a competitive performance in certain tasks. As the representative of System-1 models, GPT-4o-mini excels in versatility, outperforming o1-mini on tasks where domain knowledge is crucial. (2) The contribution of *ReFT* is trivial. Designed to enhance general reasoning abilities, it fails to address the distribution discrepancy between the provided domain samples and the policy model to be fine-tuned. (3) With PRM to supervise the reasoning process, *ReFT+PRM* contributes to increasing the likelihood of sampling correct reasoning processes, although the improvement is not significant. (4) After fine-tuning with self-synthesized reasoning process data, *SFT* slightly outperformed *Vanilla*. This indicates that with such a limited number of samples, relying solely on SFT is far from sufficient. However, it can provide a good exploration starting point for subsequent RL. (5) Compared with *SFT*, *SFT+RL(PRM)* shows obvious improvement, highlighting the necessity of RL in fully leveraging the limited domain-specific samples. (6) *SFT+RL(PRM)+DA* achieves consistent improvement over *SFT+RL(PRM)* in different tasks, validating the contribution of domain-specific samples in synthesizing new samples. It achieves the best performance among the methods initialized with Skywork-o1, an average improvement of 11% compared to *Vanilla*. (7) By further incorporating few-shot ICL, there was no improvement but a slight decrease in the performance. This is possibly due to the inconsistent prompts during the SFT and RL stages. It is interesting to see that fewshot ICL benefits the most challenging task (T4: *molecule-structure-prediction*), which is precisely where GPT-4o-mini excels. This somewhat underscores the effectiveness of domain knowledge. We are experimenting alternative ways for domain knowledge embedding.

### 3.5 DISCUSSIONS

More domain-specific data contributes to better RFT results. From Table [1,](#page-6-0) we observe that data augmentation techniques significantly enhance model performance when training data is limited. This suggests that more domain-specific datasets could further improve the *ReFT* method. Motivated by this, we explored the impact of data size on *ReFT* by comparing the effects of different data sizes on *ReFT* and two variants of *OpenRFT* using the T8 dataset. In all experiments with the *OpenRFT* variants, the models are initialized with the same set of 100 samples across all conditions.

<span id="page-7-0"></span>> **[图片提取文字 (无描述)]:**
> Accuracy(%) -- ReFT ---- SFT + RL(PRM) ---- SFT+RL(PRM)+DA Number of samples used in RL stage
![](_page_7_Figure_0.jpeg)

Figure 3: Performance with different sizes of domain-specific data. The light green dashed line represents the performance of SFT with 100 samples.

<span id="page-7-1"></span>

| Biology<br>Model |              | Chemistry    |              |              | Physics      | Materials    |              |              | Avg.         |
|------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
|                  | T1           | T2           | T3           | T4           | T5           | T6           | T7           | T8           |              |
| Vanilla          | 0.28         | 0.55         | 0.52         | 0.23         | 0.45         | 0.34         | 0.41         | 0.41         | 0.40         |
| SFT<br>SFT+      | 0.33<br>0.27 | 0.53<br>0.45 | 0.49<br>0.44 | 0.20<br>0.12 | 0.45<br>0.34 | 0.37<br>0.25 | 0.43<br>0.28 | 0.49<br>0.30 | 0.41<br>0.31 |

Table 2: Analysis of teacher-student policy alignment. *SFT* and *SFT+* indicate synthesizing reasoning process by the student policy itself and a stronger reasoning model *QwQ-32B*, respectively.

As shown in Fig. [3,](#page-7-0) the performance of all three methods improves as the number of training samples increases. The influence of data augmentation is most pronounced when the training dataset is small. However, as the size of the data set increases, the benefit of data augmentation diminishes. This reduction in effectiveness may be due to the inherent errors introduced by LLM-based data augmentation, particularly when handling complex molecular formulas or similarly challenging scenarios.

Teacher policy model should align with the student policy model. As introduced in Section 2, it is ideal to use a stronger reasoning foundation model as the teacher model to synthesize the reasoning process for the domain-specific samples. We employed the more powerful QwQ-32B as the teacher model, and then performed SFT on the student policy model using the synthesized reasoning process data.

Table [2](#page-7-1) displays the performance of the student policy model when generating its own process data (referred to as *SFT*) and when using the process data synthesized by QwQ-32B-Preview [\(Team,](#page-11-7) [2024b\)](#page-11-7) (referred to as *SFT+*). It can be observed that *SFT+* significantly underperforms compared to *SFT*, even falling below the baseline of vanilla models.

Although QwQ-32B-Preview is stronger than the student policy model to be fine-tuned (Skywork o1 Open-Llama-3.1-8B), and the synthesized reasoning process data are likely of higher quality, the discrepancy in teacher-student policy action spaces leads to a degradation in training when using these inconsistent reasoning step data for fine-tuning. This validates the importance of ensuring that the action space of the teacher and student models should be well aligned.

