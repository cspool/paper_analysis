# <span id="page-12-0"></span>A Data and Hyperparameters

Training Data The data used in this study is strictly aligned with LLaVA-1.5. The training of VisualRWKV is composed of two phases: (1) Feature Alignment Phase: Utilizing our 558K subset from the LAION-CC-SBU dataset, we link a pretrained, frozen vision encoder to a frozen Large Language Model (LLM); (2) Visual Instruction Tuning Phase: We employ 150K of GPT-generated multimodal instruction-following datasets, supplemented by approximately 515K Visual Question Answering (VQA) datasets from academically oriented tasks [\(Marino et al.,](#page-10-14) [2019;](#page-10-14) [Singh et al.,](#page-10-10) [2019;](#page-10-10) [Hudson and Manning,](#page-9-10) [2019;](#page-9-10) [Goyal et al.,](#page-9-9) [2017\)](#page-9-9), to instruct the model in adhering to multimodal directives. For more details, one can refer to the paper on LLaVA-1.5 [\(Liu et al.,](#page-9-1) [2023a\)](#page-9-1). All the data used in this paper are consistent with their intended use. We carefully identified and handled all personally identifiable information and offensive content. We started with automated screening to flag sensitive data, followed by manual review for precision. Anonymization methods like data masking and pseudonymization were applied to protect sensitive information. Strict data protection protocols were followed throughout.

Evaluation Benchmarks Additional details on Benchmarks are provided here. The VQA-v2 reports its metrics based on the test-dev split. Similarly, GQA's metrics are on the test-dev split. The metrics for TextVQA are reported on the validation set. ScienceQA's metrics are based on the test set. POPE's metrics are also reported on the test set. The MMBench metrics are reported on the development set. MME has a unique test-set, thus there is no ambiguity.

Data Language Firstly, the training data includes academic Visual Question Answering (VQA) datasets and ShareGPT data. The primary language of the VQA academic datasets is English, while the ShareGPT data is multilingual, encompassing mainstream languages, but derived from contributions by users worldwide, it is not feasible to count the total number of languages. Among the evaluation benchmarks, MMBench-cn is the only Chinese dataset; the rest are English datasets. Concurrently, we evaluated the model's text-only capabilities in multiple languages, with the specific languages detailed in Appendix [G.](#page-16-1)

Hyperparameters The hyperparameters here were used for the training of a range of VisualRWKV models, from 1.6B to 7B parameters, as illustrated in Table [2.](#page-7-2) We show the training hyperparameters for both first-stage vision-language alignment pretraining and the second-stage visual instruction tuning in Table [6.](#page-12-1)

<span id="page-12-1"></span>

| Hyperparameter  | 1.6B-Pretrain | 1.6B-Finetune | 3B-Pretrain  | 3B-Finetune  | 7B-Pretrain  | 7B-Finetune  |
|-----------------|---------------|---------------|--------------|--------------|--------------|--------------|
| batch size      | 256           | 128           | 256          | 128          | 256          | 128          |
| lr init         | 1e-3          | 6e-5          | 1e-3         | 5e-5         | 1e-3         | 4e-5         |
| lr end          | 1e-5          | 1.5e-5        | 1e-5         | 1.25e-5      | 1e-5         | 1e-5         |
| lr schedule     | cosine decay  | cosine decay  | cosine decay | cosine decay | cosine decay | cosine decay |
| lr warmup ratio | 0             | 0             | 0            | 0            | 0            | 0            |
| weight decay    | 0             | 0             | 0            | 0            | 0            | 0            |
| epoch           | 1             | 2             | 1            | 2            | 1            | 2            |
| optimizer       | AdamW         | AdamW         | AdamW        | AdamW        | AdamW        | AdamW        |
| DeepSpeed stage | 1             | 1             | 1            | 1            | 1            | 2            |

Table 6: Hyperparameters of VisualRWKV.

Licenses VisualRWKV is licensed under the Apache-2.0 license. The RWKV language model is also under the Apache-2.0 license. The LLaVA model is licensed under the Apache-2.0 license. The VQA-v2 dataset is licensed under the Commons Attribution 4.0 International License. MMBench is licensed under the Apache-2.0 license. TextVQA data is available under the CC BY 4.0 license. ScienceQA is licensed under the MIT License, and POPE is also under the MIT license.

