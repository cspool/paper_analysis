# <span id="page-13-12"></span>B Appendix for the datsets and evaluation metrics

#### B.1 Dataset statistics

The detailed statistics of the above tasks' datasets are presented in Table [6.](#page-14-2)

#### B.2 Evaluation metrics/protocols

For the commonsense reasoning and math reasoning tasks, since they usually come with a definite answer choice, we will directly consider the correctness of the final answers. Thus, we report accuracy (denoted as acc).

For evaluating the quality of instruction tuned LlaMA-2 7B on the MT-Bench, we follow the current common practice of utilizing GPT-4 as a unbiased reviewer [\(Zheng et al.,](#page-12-20) [2023\)](#page-12-20). We generate model responses from a fine-tuned model with beam size 3 with the generation function in Huggingface Transformers [\(Wolf et al.,](#page-11-14) [2020a\)](#page-11-14). Then we compare MOELoRA and MiLoRA's answers with GPT-4. For each instruction in MT-Bench,

<span id="page-14-2"></span>

| Datasets                    | #train | #dev  | #test | Type                  | Metrics      |  |
|-----------------------------|--------|-------|-------|-----------------------|--------------|--|
| Commonsense reasoning tasks |        |       |       |                       |              |  |
| BoolQ                       | 9427   | -     | 3270  | Commonsense reasoning | acc          |  |
| OBQA                        | 4957   | 500   | 500   | Commonsense reasoning | acc          |  |
| ARC-e                       | 2251   | 570   | 2376  | Commonsense reasoning | acc          |  |
| ARC-c                       | 1119   | 299   | 1172  | Commonsense reasoning | acc          |  |
| PIQA                        | 16,000 | 2,000 | 3,000 | Commonsense reasoning | acc          |  |
| Math reasoning tasks        |        |       |       |                       |              |  |
| AQuA                        | 97467  | 254   | 254   | Math reasoning        | acc          |  |
| GSM8K                       | 7473   | -     | 1319  | Math reasoning        | acc          |  |
| Instruction tuning          |        |       |       |                       |              |  |
| Alpaca                      | 50k    | -     | -     | Instruction tuning    | -            |  |
| LLM evaluation tasks        |        |       |       |                       |              |  |
| MT-Bench                    | -      | -     | 80    | Question answering    | GPT-4 scores |  |
| MMLU                        | -      | -     | 14042 | Question Answering    | acc          |  |
| BBH                         | -      | -     | 6,511 | Question Answering    | acc          |  |
|                             |        |       |       |                       |              |  |

Table 6: The dataset statistics.

GPT-4 [\(OpenAI,](#page-10-21) [2023\)](#page-10-21) is asked to write a review for both answers from the two methods, and assigns a quantitative score on a scale of 10 to each response.

