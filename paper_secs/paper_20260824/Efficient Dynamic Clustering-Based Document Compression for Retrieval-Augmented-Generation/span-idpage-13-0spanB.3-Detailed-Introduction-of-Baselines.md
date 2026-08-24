# <span id="page-13-0"></span>B.3 Detailed Introduction of Baselines

The baselines for FELM include: 1) prompts enhanced with Chain-of-Thought (CoT) reasoning [\(Kojima et al.,](#page-9-15) [2022\)](#page-9-15), 2) prompts augmented with hyperlinks to reference documents, and 3) prompts supplemented by human-annotated reference documents [\(Chen et al.,](#page-9-13) [2023\)](#page-9-13).

The baselines for WikiBio GPT-3 comprise: 1) HalluDetector[\(Wang et al.,](#page-10-19) [2023\)](#page-10-19), which leverages external knowledge sources along with a dedicated classification model and a Naive Bayes classifier to identify hallucinations, and 2) Focus[\(Zhang](#page-11-7) [et al.,](#page-11-7) [2023\)](#page-11-7), which employs a multi-stage decisionmaking framework combining both pre-retrieval and task-specific classifiers.

### C Prompts Used in Our Experiments

#### C.1 Hallucination Detection Datasets

#### C.1.1 FELM & HaluEval

### Prompt of Compression

### ##Instruction##:

You are an AI assistant specializing in information extraction. Your task is to analyze a given statement and a set of related documents, and extract only the directly relevant information.

### ##Extraction Guidelines##:

- Identify key points, evidence, or details that \*\*directly support, refute, or elaborate\*\* on the statement.
- Ensure that the extracted content is \*\*concise, objective, verifiable, and directly traceable\*\* to the original documents.
- \*\*Do not make inferences or draw conclusions\*\* beyond what is explicitly stated.
- If the documents contain \*\*no relevant information\*\*, respond with \*\*No content to extract.\*\*

#### ##Example Output Format##:

{few-shots}

### ##Statement##:

{query}

### ##Documents##:

{docs}

### ##Extracted Information##:

#### Eval Prompt of HaluEval

### ##Instruction##:

I want you to act as an answer judge. Given a question, two answers, and related knowledge, your objective is to select the best and correct answer without hallucination and non-factual information.

You should try your best to select the best and correct answer. If the two answers are the same, you can choose one randomly. If both answers are incorrect, choose the better one. You MUST select an answer from the two provided answers.

Think step by step. Give your reasoning first and then output your choice. Output in the following format:

<sup>1</sup>[https://huggingface.co/](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

```
"#Reasoning#: Your Reasoning
#Choice#: "X"".
"X" should only be either "Answer 1" or
"Answer 2", rather than specific answer con-
tent.
##Knowledge##:
{knowledge}
##Question##:
{question}
##Answer 1##:
{answer 1}
##Answer 2##:
{answer 2}
```

## C.1.2 WikiBio GPT-3

#### Prompt of Compression

### ##Instruction##:

You have been provided with a statement about {a person} and a collection of related documents. Your task is to extract relevant information from these documents that directly supports, refutes, or elaborates on the given statement.

Focus on identifying key points, evidence, or details that are clearly connected to the statement. Ensure the extracted content is concise, directly relevant, and maintains the context of the original documents.

The extracted content must be objective, verifiable, and directly traceable to the original documents. Avoid making inferences or drawing conclusions based on the extracted content.

If you find that the documents contain no relevant information, please output "No content to extract". Below is an example.

```
{One shot}
##Person##:
{person}
##Statement##:
{query}
##Documents##:
{docs}
##Extracted Information##:
```

### Prompt of Evaluation

#### ##Instruction##:

Assess whether the given statement about {a person} contains factual errors or not with the help of the reference docs.

If you believe given statement contains factual errors, your answer should be "Nonfactual"; if there is no factual error in this statement, your answer should be "Factual". This means that the answer is "Nonfactual" only if there are some factual errors in the given statement. When there is no factual judgment in the given statement or the given statement has no clear meaning, your answer should be "Factual". At the same time, please consider all aspects of the given statement thoroughly during the evaluation and avoid focusing excessively on any single factual aspect. Any factual errors should be considered.

Reference docs can be classified into three types: documents that support the response segment as "Nonfactual", documents that support the response segment as "Factual", and documents that provide supplementary or explanatory information for the response segment. Please consider these documents comprehensively when answering.

Think it step by step. Give your "Reasoning" first and then output the "Answer".

```
##Statement##:
{statement}
##Reference docs##:
{passage}
##Output##:
```

### C.2 Knowledge-QA Datasets

The prompts used for compression and generation in KQA tasks are shown below. These prompts differ from those used in previous datasets because we aim to elicit more informative chunks by having the model respond to the question first. This approach encourages the model to provide supporting evidence, which we then use to extract and compress relevant information. In contrast, directly prompting the model to summarize often leads it to provide answers directly without grounding them in the source content. If there is no strong formatting requirement, the quality of the LLM's responses remains stable; however, if strict formatting requirements are imposed, the response quality drops sharply, causing a significant decline in performance. Accordingly, during the final generation stage, we also have the model consider these outputted answers and their corresponding evidence. The model integrates all the evidence to select the most appropriate answer.

### Prompt of Summarization

#### ##Instruction##:

Please refer to the following text and answer the following question, providing supporting evidence.

### ##Question##:

{question}

## ##Reference text##:

{docs}

##Answer##:

#### Prompt of Response

### ##Task##:

Analyze the following set of candidate answers to a question and select the single most consistent/plausible answer based on majority consensus and logical coherence.

## ##Instructions##:

- 1. Carefully compare all candidate answers.
- 2. Identify the core factual claims or entities in each answer.
- 3. Group semantically equivalent answers (e.g., "1990", "the year 1990", "nineteen ninety").
- 4. Select the answer that: Appears most frequently in the candidate set - Has strong internal consistency (no self-contradictions)
- 5. If multiple answers have equal validity, prefer the most specific and concise one.

#### ##Format Requirements##:

Reasoning: Concise justification for selection.

Selected Answer:...

Below is an example.

Candidate Answers: ["Paris", "The capital is Paris", "France", "paris", "It's Paris in France"]

Question: What is the capital of France? Expected Response:

Reasoning: 4/5 answers directly state 'Paris'. While 'France' is incorrect alone, the most frequent and unambiguous consensus is 'Paris' Selected Answer: Paris

### ##Candidate Answers##:

{answers}

### ##Question##:

{question}

### D Additional Experimental Results

### D.1 Experiments on Open-Source Models

Additional experiments are conducted using Qwen-3-8B in think mode on the TwoWiki dataset under a noise rate of 0%, constrained by available computational resources. These experiments, summarized in Table [10,](#page-15-0) utilized only this 8B model. The results reveal a notable performance gap compared to closed-source LLMs, attributable to the limited summarization and evidence-filtering capabilities of smaller models.

<span id="page-15-0"></span>

| Top-k | RALM  | Ours (Qwen-3-8B) |
|-------|-------|------------------|
| 5     | 66.96 | 60.33            |
| 10    | 72.39 | 67.71            |
| 20    | 73.90 | 75.64            |
| 30    | 78.44 | 71.01            |
| 50    | 80.76 | 69.88            |
| 70    | 80.30 | 72.17            |
| 100   | 81.56 | 71.18            |

Table 10: Performance comparison on TwoWiki dataset (noise rate 0%) using Qwen-3-8B in think mode.

We anticipate improved outcomes with larger open-source models and intend to incorporate corresponding experiments in future iterations, subject to resource availability.

### D.2 Additional Experimental Results on Noise Resistence

Tables [11](#page-16-0) summarizes performance under varying noise levels with Top-k = 20.

<span id="page-16-0"></span>

| Dataset            | Method            |                        |       | Noise Rates (%) at Top-k=20 |       |       |       |       |  |
|--------------------|-------------------|------------------------|-------|-----------------------------|-------|-------|-------|-------|--|
|                    |                   | 0                      | 20    | 40                          | 60    | 80    | 100   | Avg   |  |
| gpt-3.5-turbo-1106 |                   |                        |       |                             |       |       |       |       |  |
|                    | Vanilla RALM      | 74.75                  | 77.82 | 78.07                       | 74.92 | 74.42 | 74.30 | 75.71 |  |
|                    | Chunk Compression | 74.15                  | 75.38 | 77.70                       | 78.01 | 71.89 | 76.08 | 75.54 |  |
| MusiQue            | Long Agent        | 84.21                  | 83.41 | 79.02                       | 76.12 | 78.91 | 75.78 | 79.58 |  |
|                    | Ours              | 82.55                  | 85.50 | 78.28                       | 83.58 | 82.53 | 79.88 | 82.05 |  |
|                    | Vanilla RALM      | 90.07                  | 89.62 | 90.12                       | 90.14 | 90.06 | 86.36 | 89.40 |  |
|                    | Chunk Compression | 90.77                  | 89.68 | 90.03                       | 90.79 | 89.68 | 87.64 | 89.77 |  |
| WebQ               | Long Agent        | 90.49                  | 91.91 | 90.54                       | 89.46 | 88.81 | 87.91 | 89.85 |  |
|                    | Ours              | 90.79                  | 91.87 | 90.75                       | 91.00 | 89.23 | 87.87 | 90.25 |  |
|                    | Vanilla RALM      | 77.51                  | 71.48 | 71.84                       | 68.40 | 67.57 | 66.01 | 70.47 |  |
|                    | Chunk Compression | 72.41                  | 71.52 | 71.06                       | 68.13 | 69.75 | 67.28 | 70.03 |  |
| 2Wiki              | Long Agent        | 76.06                  | 77.05 | 74.20                       | 71.07 | 69.35 | 66.99 | 72.45 |  |
|                    | Ours              | 76.20                  | 76.66 | 76.75                       | 72.43 | 72.92 | 68.99 | 73.99 |  |
|                    |                   | gpt-4o-mini-2024-07-18 |       |                             |       |       |       |       |  |
|                    | Vanilla RALM      | 77.78                  | 73.39 | 76.25                       | 68.08 | 65.42 | 70.32 | 71.87 |  |
|                    | Chunk Compression | 75.67                  | 75.33 | 76.82                       | 75.29 | 67.41 | 68.26 | 73.13 |  |
| MusiQue            | RAPTOR            | 72.07                  | 78.46 | 75.95                       | 71.15 | 76.64 | 70.78 | 74.18 |  |
|                    | Long Agent        | 80.43                  | 76.67 | 72.50                       | 77.69 | 73.93 | 78.05 | 76.55 |  |
|                    | Ours              | 81.71                  | 80.44 | 81.10                       | 78.98 | 77.50 | 74.91 | 79.11 |  |
|                    | Vanilla RALM      | 85.07                  | 89.89 | 90.82                       | 88.70 | 88.27 | 85.20 | 87.99 |  |
|                    | Chunk Compression | 90.77                  | 90.49 | 90.08                       | 90.53 | 89.40 | 86.98 | 89.71 |  |
| WebQ               | Long Agent        | 91.94                  | 91.49 | 90.86                       | 90.13 | 88.60 | 86.79 | 89.80 |  |
|                    | Ours              | 91.89                  | 90.36 | 90.76                       | 89.43 | 88.40 | 86.90 | 89.62 |  |
|                    | Vanilla RALM      | 73.84                  | 73.03 | 71.43                       | 69.03 | 67.53 | 60.88 | 69.29 |  |
|                    | Chunk Compression | 69.24                  | 68.63 | 67.84                       | 68.45 | 66.12 | 59.14 | 66.51 |  |
| 2Wiki              | Long Agent        | 71.33                  | 73.32 | 70.52                       | 64.27 | 62.69 | 57.29 | 66.57 |  |
|                    | Ours              | 72.86                  | 71.92 | 72.58                       | 69.60 | 66.44 | 60.88 | 69.05 |  |

Table 11: Comparison of F1 scores under different noise levels at Top-k=20 on MusiQue, WebQ, and 2Wiki datasets for multiple retrieval methods.