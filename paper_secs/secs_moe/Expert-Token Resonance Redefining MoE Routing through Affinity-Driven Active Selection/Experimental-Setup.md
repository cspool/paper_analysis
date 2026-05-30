# Experimental Setup

### Datasets for Pre-training and Fine-Tuning

The dataset used in this paper is a self-constructed dataset that integrates knowledge from multiple domains, including wireless, data communication, and cloud-core technologies. It comprises Chinese, English, and bilingual corpora. The corpora are parsed from various internal technical documents, such as iCase, blogs, Wiki, and feature documents. Taking iCase as an example, iCase is a case record of problem localization and handling processes, containing code, instructions, and corresponding logs. In addition, the abovementioned domain-specific knowledge corpora are mixed with general corpora in a ratio of 1:5. The general corpora are collected from hundreds of websites, including online novels, cooking guides, movie reviews, and more. After cleaning, deduplication, and review operations, the dataset is thoroughly shuffled. A total of 4.19 billion tokens is sampled as the experimental pre-training dataset. To evaluate downstream tasks, this paper also adopt hybrid sft data items to fine-tune the pre-trained model. The dataset comprises 762,321 general question-answer pairs and 11,048 domain-specific question-answer pairs, with a general-todomain ratio of 68:1. The general characteristics encompass multi-tasking, mathematical ability, coding ability, logical reasoning, multi-turn dialogue, knowledge reasoning, language understanding, text generation, multi-tasking, FunctionCall, CoT, MRC summarization, refusal to answer, Chinese, and English. The domain-specific characteristics include domain knowledge understanding, RAG, Function-Call, information extraction, multi-turn dialogue, reading comprehension, paraphrasing, and intent recognition.

The pre-training data comprises 300B tokens in total, with 150B tokens from the ICT domain and 150B tokens sampled from general data. The sampling ratios are shown in Table 3. For SFT data, we employed a two-stage training: the first stage primarily enhances the model's logical reasoning capabilities such as multi-task capability, mathematics, puzzlesolving, complex logic, etc. The total scale of samples is approximately two million, while the second stage focuses on improving instruction-following abilities, tool function call, sentiment, security, etc. The total scale of samples in stage2 is about three million.

#### Experimental Environment

The experiments are conducted on a cluster composed of Ascend 910B3 NPUs, divided into three groups: 32 NPUs (hereinafter referred to as 32N, and so on), 64N, and 256N. The 910B3 series NPU contains 20 AI cores with a main frequency of 1.8GHz and a theoretical computing power of 313T under fp16 precision. The physical High Bandwidth Memory (HBM) of the 910B3 NPU is 64G, with an HBM frequency of 1.6GHz and an HBM bandwidth of 1.6T. Every 8 NPUs are mounted on the same Atlas 800T A2 server, which internally adopts a fullmesh networking scheme, meaning that any two NPUs are interconnected.

#### Evaluation Metrics and Datasets

To evaluate model performance, this paper designs a comprehensive metric called the General and Domain-specific Assessment Dataset (GDAD), which consists of three evaluation systems: domain task capability, domain capability certification exam, and general capability. Among them, the domain task capability includes a total of 16 categories and 2,657 questions, such as domain logical reasoning; the domain capability certification exam includes a total of 13 categories and 13,968 questions, such as data communication; and the general capability includes a total of 18 categories and 1,435 questions, such as programming ability. The questions include objective and subjective questions in Chinese, English, and bilingual formats. For subjective questions, the cosine similarity between the model output and the standard answer is used as the score. In addition, this paper also employs GPQA (Rein et al. 2023) and TeleQnA (Maatouk et al. 2023) to evaluate the model's Chinese language capability.

Table 3: Data sources and sampling ratios of general pre-training data.

| Primary Category       | Secondary Category | Tertiary Source                  | Sampling ratio |
|------------------------|--------------------|----------------------------------|----------------|
| General English        | Webpages           | Reasoning steplist               | 25%            |
|                        |                    | Model rewrite                    | 100%           |
|                        | Books & Papers     | book3                            | 25%            |
|                        |                    | bookcorpus                       | 100%           |
|                        |                    | all libgen books                 | 20%            |
|                        |                    | all libgen scihub                | 10%            |
|                        |                    | RedPajama arxiv                  | 25%            |
|                        |                    | arxiv latex2Markdown cleaned     | 25%            |
|                        |                    | wiki                             | 100%           |
|                        | WebText            | stackexchange cleaned            | 20%            |
|                        |                    | cosmopedia v2                    | 15%            |
| General Chinese        | Webpages           | aigc dataset                     | 15%            |
|                        | Book               | all book deduped                 | 10%            |
|                        |                    | zh book CommonData               | 10%            |
|                        |                    | zh general STEM                  | 80%            |
|                        |                    | all zhiwang                      | 20%            |
|                        | WebText            | baike MBAzhiku sougou ye zhiarge | 50%            |
|                        |                    | baike sougou baidu kuaidong      | 50%            |
|                        |                    | wiki                             | 10%            |
|                        |                    | zhihu caigou merged cleaned      | 10%            |
| High-density Knowledge | Q&A                | quiz data                        | 100%           |
|                        | Collection         | density knowledge                | 100%           |
|                        |                    | collection updated               | 100%           |
|                        |                    | english question and answer      | 100%           |
|                        |                    | annealing                        | 100%           |
|                        | Code               | code python edu high quality     | 30%            |
| Code                   | Forum              | CSDN                             | 20%            |
|                        |                    | Ultra textbooks                  | 100%           |