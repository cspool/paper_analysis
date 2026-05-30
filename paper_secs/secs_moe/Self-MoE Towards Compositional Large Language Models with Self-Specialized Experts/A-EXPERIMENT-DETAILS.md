# A EXPERIMENT DETAILS

We provide each of our self-specialization prompts for knowledge, reasoning, math, and coding experts in Tables 9, 10, 11, and 12. We largely follow Kang et al. (2024)'s prompt structure to ensure quality, with additional domain-specific instructions that inform task-related information.

For our evaluation, we employ popular and widely accepted evaluation frameworks to pursue standard evaluation setups and protocols: HELM (Liang et al., 2023), LM Evaluation Harness (Gao et al., 2023), and BigCode Evaluation Harness (Ben Allal et al., 2022). We use Huggingface PEFT (Mangrulkar et al., 2022) and XLoRA (Buehler & Buehler, 2024) for the implementation of MoE compatible with LoRA.

Regarding seed instructions, we sampled 100 training instances from each of the MMLU, BBH, and GSM8K datasets, for knowledge, reasoning, and math domains, respectively. For coding, since the size of the HumanEval dataset is very small and thus the training set is not available, we took 100 samples from the MBPP training set and converted the task format to make them suit the HumanEval.

During instruction generation, we use three seed data, which are randomly sampled, as in-context examples, using a temperature of 1 and top-p of 0.98, whereas we use five seed data in-context for response generation with greedy decoding. For specialization, we use LoRA applied to all modules with a rank of 8 and alpha of 16, and train it using a learning rate of 3e-4, epochs of 3, and batch size of 32. We train each module and MiXSE using a standard Alpaca (Taori et al., 2023) prompt template on a single A100-80GB, which takes only a few hours.

## B LIMITATIONS

While our study demonstrates promising results for the Self-MoE, we recognize areas requiring further investigation in future work. Employing self-specialization Kang et al. (2024) to generate synthetic data within our framework may raise concerns about potential data contamination and noise. Nonetheless, findings from Kang et al. (2024), which conducted an n-gram overlap analysis between the self-specialization data and test data, confirmed no significant overlap, thus alleviating the concerns about contamination. Despite this, the need for continuous monitoring of potential biases from pre-training and the development of enhanced data validation and noise filtering strategies remain important, and may present interesting direction for future work. Moreover, due to computational constraints, we did not scale our model and data to their full potential. We also did not work on the optimization of the XLoRA, the MoE module we used, to focus purely on the research problem defined in this study. Future work should therefore concentrate on overcoming these limitations, which will enable better data quality and more extensive training to unveil the full potential of the Self-MoE framework.

<span id="page-17-0"></span>Table 6: Dataset statistics. Non-Target (In-Expertise) indicates where MiXSE does not directly specialize using seed data directly while relevant to targets. Non-Target (Out-of-Expertise) refers to irrelevant cases.

| Category           | Benchmark                     | # Examples |  |  |  |  |  |  |  |
|--------------------|-------------------------------|------------|--|--|--|--|--|--|--|
|                    | Target                        |            |  |  |  |  |  |  |  |
| Academic Knowledge | MMLU (57 Tasks)               | 14,079     |  |  |  |  |  |  |  |
| Reasoning          | BBH (27 Tasks)                | 6,511      |  |  |  |  |  |  |  |
| Math               | GSM8K                         | 8,790      |  |  |  |  |  |  |  |
| Coding             | HumanEval                     | 164        |  |  |  |  |  |  |  |
| Non-To             | Non-Target (In-Expertise)     |            |  |  |  |  |  |  |  |
| Math               | MATH                          | 12,500     |  |  |  |  |  |  |  |
| Coding             | MBPP                          | 257        |  |  |  |  |  |  |  |
| Non-Targ           | Non-Target (Out-of-Expertise) |            |  |  |  |  |  |  |  |
| W11171-1           | Natural Questions             | 3,610      |  |  |  |  |  |  |  |
| World Knowledge    | TriviaQA                      | 17,200     |  |  |  |  |  |  |  |
| Commonsense        | Hellaswag                     | 10,000     |  |  |  |  |  |  |  |
| Commonsense        | PIQA                          | 3,000      |  |  |  |  |  |  |  |
| Safety             | TruthfulQA                    | 817        |  |  |  |  |  |  |  |

<span id="page-18-1"></span>Table 7: Additional comparisons with other models for references. Results are extracted from each corresponding paper, except for pre-training methods where the numbers are all from BTX (Sukhbaatar et al., 2024).

| Method                                                                                                                                                 | Total<br>Params      | Active<br>Params        | Compos-<br>itional | Semantic<br>Experts | Light-<br>weight | Data & Resrc<br>-Efficient | w/o Teacher<br>& Labels | Knowledge<br>(MMLU 5-shot) | Reasoning<br>(BBH)   | Math<br>(GSM8K)      | Coding<br>(HumanEval) |
|--------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|-------------------------|--------------------|---------------------|------------------|----------------------------|-------------------------|----------------------------|----------------------|----------------------|-----------------------|
| Base LLM                                                                                                                                               |                      |                         |                    |                     |                  |                            |                         |                            |                      |                      |                       |
| Gemma 7B (Team et al., 2024)<br>LLaMA-2 70B (Touvron et al., 2023)<br>Mixtral 8x7B (Jiang et al., 2024)                                                | 7B<br>70B<br>47B     | 7B<br>70B<br>13B        | ×                  | ×                   | ×                | -<br>-<br>-                | -<br>-                  | 65.7<br>68.9<br>70.6       | 56.1<br>51.2<br>67.1 | 42.5<br>35.2<br>65.7 | 34.1<br>29.9<br>32.3  |
| Pre-training Methods                                                                                                                                   |                      |                         |                    |                     |                  |                            |                         |                            |                      |                      |                       |
| Branch-Train-Merge (4x7B) (Li et al., 2022)<br>Sparse Upcycling (4x7B) (Komatsuzaki et al., 2023)<br>Branch-Train-Mix (4x7B) (Sukhbaatar et al., 2024) | <24B<br><24B<br><24B | 11.1B<br>11.1B<br>11.1B | * * *              | **                  | ×××              | ×<br>×                     | ž                       | 44.3<br>52.1<br>52.5       | -<br>-               | 27.7<br>40.1<br>37.1 | 30.6<br>26.2<br>28.7  |
| MoE w/ LoRA                                                                                                                                            |                      |                         |                    |                     |                  |                            |                         |                            |                      |                      |                       |
| PHATGOOSE (Muqeeth et al., 2024)<br>MOLE (Wu et al., 2024)                                                                                             | <4B                  | >3B                     | ×                  | × ×                 | ž                | ×                          | ×                       | -                          | 35.6<br>42.2         | -                    | -                     |
| Distillation/Synthetic Data from Larger Models                                                                                                         |                      |                         |                    |                     |                  |                            |                         |                            |                      |                      |                       |
| GLAN 7B (w/ GPT-4) (Li et al., 2024a)<br>Orca-2 7B (w/ GPT-4) (Mitra et al., 2023)<br>Merlinite 7B (w/ Mixtral 8x7B) (Sudalairaj et al., 2024)         | 7B<br>7B<br>7B       | 7B<br>7B<br>7B          | ×<br>×             | -                   | -                | ×<br>×                     | ×<br>×                  | 62.9<br>53.9<br>64.9       | 60.7<br>42.8         | 80.8<br>55.7<br>44.6 | 48.8<br>17.1          |
| Self-Improving                                                                                                                                         |                      |                         |                    |                     |                  |                            |                         |                            |                      |                      |                       |
| Ours                                                                                                                                                   | 7B + 1%              | 7B + 0.3% ✓             | ~                  | ~                   | ~                | ~                          | 66.2                    | 61.1                       | 52.5                 | 37.8                 |                       |

## C DATASET DESCRIPTIONS

The statistics for each dataset are provided in Table 6. The target datasets used are as follows:

- MMLU (Massive Multitask Language Understanding) (Hendrycks et al., 2021a): A collection of 57 academic knowledge tasks.
- BBH (BIG-Bench Hard (Suzgun et al., 2022): A set of 27 challenging reasoning tasks.
- **GSM8K** (Grade School Math 8K) (Cobbe et al., 2021): A diverse set of grade school math word problems.
- **HumanEval** (Chen et al., 2021): A hand-written evaluation set for python programming problems.

#### D ADDITIONAL RESULTS

#### <span id="page-18-0"></span>D.1 ADDITIONAL COMPARISON AND DISCUSSION

In Table 7, we present additional comparisons with various other models and methods to provide a broader perspective, though comparisons may not appear to be direct, due to factors involved such as parameters, resources, etc. We discuss some noteworthy points.

Notably, although MiXSE significantly improves upon its base model, Gemma 7B, it does not yet reach the performance levels of the more powerful Mixtral 8x7B. It is important to understand that Mixtral also utilizes an MoE (Mixture of Experts) architecture, but unlike MiXSE, it does not prioritize lightweight experts, leading to a much larger model with significantly more parameters. Moreover, while Mixtral's experts are implicitly built during pre-training, MiXSE explicitly creates semantic experts, allowing for targeted improvements and clearer interpretability. Importantly, our self-improving method can be potentially applied on top of any pre-trained model including Mixtral in principle.

Similarly, BTX (Branch-Train-MiX) uses a pre-training MoE strategy where parameter-heavy semantic experts are employed, yielding substantial enhancements over the base LLM. This approach highlights the effectiveness of using semantically rich experts to refine the model's capabilities. To make comparisons in terms of efficiency, our model uses fewer parameters (7B), compared to BTX (12B active with much more whole parameters) and requires only about 1 GPU day for training, compared to 900 GPU days for BTX. In essence, since BTX is also a pre-training method while specialized, we expect it to be complementary to our Self-MoE, as evidenced in previous work (Kang et al., 2024).

With a shared spirit, MOLE and PHATGOOSE build a MoE (Mixture of Experts) using LoRA, which is semantic and lightweight. However, there are significant differences in foundational assumptions: MOLE depends on human-labeled data, while PHATGOOSE requires access to pre-

<span id="page-19-0"></span>Table 8: Results of MiXSE using only seed data. Seed Only training shows only marginal improvements over the Base LLM in some benchmarks, validating that the effect of Self-MoE is not merely due to the use of seed data.

| Benchmark             | Base LLM | Seed Only | MiXSE |
|-----------------------|----------|-----------|-------|
| Knowledge<br>(MMLU)   | 58.3     | 57.4      | 65.6  |
| Reasoning<br>(BBH)    | 56.1     | 57.0      | 61.1  |
| Math<br>(GSM8K)       | 42.5     | 45.0      | 52.5  |
| Coding<br>(HumanEval) | 34.1     | 34.1      | 37.8  |
| Avg.                  | 47.8     | 48.4      | 54.3  |

trained expert models developed externally. In contrast, our Self-MoE framework independently constructs both experts and a router entirely from scratch, focusing on self-improvement without such dependencies. While their scenarios are considered reasonable in a certain context, we aim for broader applicability by minimizing assumptions on conditions.

Lastly, GLAN demonstrates outstanding performance across various domains. This is attributed to their reliance on distilling from the larger and stronger model, GPT-4, using a huge amount of data (e.g., 10 million). As outlined in our problem statement (Section [2\)](#page-2-1), we deliberately avoid assuming the availability of such advanced models to ensure the broader applicability of our method which self-improves from scratch. Consequently, while acknowledging each of their own value, it is crucial to recognize that direct comparisons may not be entirely appropriate, given the fundamental differences in resource assumptions and initial conditions.

## D.2 MIXSE USING ONLY SEED DATA

Table [8](#page-19-0) shows the results of the MiXSE when exploiting only seed data for training, clarifying the benefits derived from our methodological enhancements beyond the mere inclusion of seed data in training. While the Seed Only shows slight improvements over the Base LLM in some benchmarks, the significant enhancements of our MiXSE across all benchmarks confirm that the enhanced capabilities of Self-MoE are not merely due to the use of seed data. This further highlights the achievement of self-improvement with our method.

