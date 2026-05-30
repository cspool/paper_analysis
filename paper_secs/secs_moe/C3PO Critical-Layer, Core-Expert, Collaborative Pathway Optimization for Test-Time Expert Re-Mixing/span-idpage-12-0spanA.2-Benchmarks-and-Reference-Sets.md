# <span id="page-12-0"></span>A.2 Benchmarks and Reference Sets

Table [7](#page-12-2) shows the overview of our benchmarks and reference sets.

| Task Type                     | Benchmarks        | Size            | Reference Sets             | Size           |
|-------------------------------|-------------------|-----------------|----------------------------|----------------|
| General Knowledge             | MMLU              | 14,042          | BIG-Bench<br>SuperGLUE     | 8,000<br>8,000 |
| Commonsense Reasoning         | HellaSwag<br>PIQA | 10,042<br>1,838 | CommonsenseQA<br>SocialIQA | 6,000<br>6,000 |
| Scientific Question Answering | ARC-C<br>ARC-E    | 1,172<br>2,376  | OpenBookQA<br>SciQ         | 2,000<br>2,000 |
| Coreference Resolution        | WinoGrande        | 1,267           | KnowRef                    | 2,000          |

<span id="page-12-2"></span>Table 7: Overview of evaluation tasks, benchmarks, and reference sets with dataset sizes.

We briefly introduce benchmarks and reference sets categorized by task types as follows:

#### General Knowledge:

- MMLU [\(Hendrycks et al.,](#page-10-6) [2020\)](#page-10-6): This benchmark consists of 16,000 multiple-choice questions across 57 subjects, including mathematics, philosophy, law, and medicine. It evaluates a model's ability to understand and reason across diverse academic disciplines.
- BIG-Bench [\(Srivastava et al.,](#page-10-7) [2022\)](#page-10-7): A comprehensive collection of 204 tasks designed to assess the capabilities of language models beyond traditional benchmarks, covering a wide range of topics and challenges.
- SuperGLUE [\(Sarlin et al.,](#page-10-8) [2020\)](#page-10-8): An evolution of the GLUE benchmark, SuperGLUE comprises eight challenging language understanding tasks, including logical reasoning, commonsense inference, and coreference resolution, aimed at evaluating general language understanding.

#### Commonsense Reasoning:

- HellaSwag [\(Zellers et al.,](#page-11-1) [2019\)](#page-11-1): Containing 10,000 descriptions of activities or events, each with four candidate endings, this dataset challenges models to choose the most plausible continuation, testing their commonsense reasoning abilities.
- PIQA [\(Bisk et al.,](#page-9-4) [2020\)](#page-9-4): Comprising 17,951 two-choice questions, PIQA assesses a model's understanding of physical commonsense by evaluating its ability to choose the most effective solution to everyday tasks.
- CommonsenseQA [\(Talmor et al.,](#page-11-2) [2018\)](#page-11-2): A dataset with 12,102 multiple-choice questions that require models to utilize commonsense knowledge to select the correct answer, focusing on everyday scenarios and concepts.
- SocialIQA [\(Sap et al.,](#page-10-9) [2019\)](#page-10-9): Featuring 38,000 multiple-choice questions, SocialIQA evaluates a model's understanding of social interactions and norms by assessing its ability to reason about social situations and their implications.

#### Scientific Question Answering:

- ARC-C [\(Clark et al.,](#page-9-5) [2018\)](#page-9-5): Consisting of 2,590 multiple-choice science questions, the Challenge Set is designed to be difficult for state-of-the-art models, requiring advanced reasoning and knowledge.
- ARC-E [\(Clark et al.,](#page-9-5) [2018\)](#page-9-5): With 5,197 multiple-choice science questions, the Easy Set serves as a baseline to evaluate a model's performance on straightforward scientific queries.
- OpenBookQA [\(Mihaylov et al.,](#page-10-10) [2018\)](#page-10-10): This dataset includes 5,957 multiple-choice questions, each associated with an elementary science fact (the "open book"), assessing a model's ability to apply core scientific principles to answer questions.
- SciQ [\(Welbl et al.,](#page-11-3) [2017\)](#page-11-3): Containing 13,679 science questions, SciQ is designed to evaluate a model's proficiency in answering questions across various scientific domains, including biology, chemistry, and physics.

