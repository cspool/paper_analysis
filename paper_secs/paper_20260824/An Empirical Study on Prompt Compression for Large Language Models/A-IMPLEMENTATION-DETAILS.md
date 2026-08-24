# A IMPLEMENTATION DETAILS

### <span id="page-14-0"></span>A.1 DATASETS

In this section, we provide detailed descriptions of the datasets used in our study.

GSM8K. GSM8K [\(Cobbe et al., 2021\)](#page-9-5) contains 8.5K linguistically diverse word problems in elementary school mathematics. Each item contains a problem and its solution.

BBC News, Arxiv articles and ShareGPT. [Li et al.](#page-11-1) [\(2023\)](#page-11-1) provided the three datasets. BBC News provides news articles from BBC, which is a typical context of human daily lives. Arxiv articles provides scientific articles that represents a formal context. ShareGPT contains contexts that is collected from human-AI conversations, which is a normal communication context.

Big Bench Hard (BBH). BBH [\(Suzgun et al., 2023\)](#page-12-6) is a diverse evaluation suite that focuses on a suite of 23 challenging tasks from BIG-Bench that were found to be beyond the capabilities of current language models.

LongBench. LongBench [\(Bai et al., 2024\)](#page-9-6) is a benchmark for bilingual, multitask and comprehensive assessment of long context understanding capabilities of LLMs. LongBench has six different task scenarios including single-document QA, multi-document QA, summarization, few-shot learning, synthetic tasks and code completion.

Gigaword, BNC, DUC2004, Broadcast and Google. [Ghalandari et al.](#page-10-4) [\(2022\)](#page-10-4) provided the five datasets. While Gigaword [\(Rush et al., 2015\)](#page-12-4) and DUC2004 [\(Over et al., 2007\)](#page-12-5) contain abstractive ground truth summaries, the remaining three datasets [\(Filippova & Altun, 2013;](#page-10-7) [Clarke & Lapata,](#page-9-8) [2008b\)](#page-9-8) have token-level extractive ground truth summaries.

IconQA. IconQA [\(Lu et al., 2021\)](#page-11-9) consists of 107,439 VQA questions and includes three subtasks: multi-image-choice, multi-text-choice, and filling-in-the-blank. IconQA is inspired by realworld diagram word problems, emphasizing the importance of abstract diagram understanding and comprehensive cognitive reasoning.

OK-VQA. OK-VQA [\(Marino et al., 2019\)](#page-12-7) is a benchmark for knowledge-based VQA consisting of over 14,000 questions. The image content in this dataset is not sufficient to answer the questions, which encourages the utilization of external knowledge resources.

