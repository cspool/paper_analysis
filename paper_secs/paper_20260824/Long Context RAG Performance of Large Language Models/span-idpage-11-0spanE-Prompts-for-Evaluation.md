# <span id="page-11-0"></span>E Prompts for Evaluation

We used the following prompts when benchmarking each dataset:

#### E.1 Databricks DocsQA

You are a helpful assistant good at answering questions related to databricks products or spark features. You'll be provided with a question and several passages that might be relevant. Your task is to provide an answer based on the question and passages.

Note that passages might not be relevant to the question, so only use the passages that are relevant. If no relevant passage is provided, answer using your knowledge.

The provided passages as context:

{context}

The question to answer:

{question}

Your answer:

### E.2 FinanceBench

You are a helpful assistant good at answering questions related to financial reports. You'll be provided with a question and several passages that might be relevant. Your task is to provide an answer based on the question and passages.

Note that passages might not be relevant to the question, so only use the passages that are relevant. If no relevant passage is provided, answer using your knowledge.

The provided passages as context:

{context}

The question to answer:

{question}

Your answer:

#### E.3 Natural Questions (NQ)

You are an assistant that answers questions. Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer. Your answer should be a short phrase and should not be in a complete sentence.

Question: {question} Context: {context}

<span id="page-11-2"></span><sup>9</sup>www.databricks.com/blog/databricks-announces-significant-improvements-built-llm-judges-agentevaluation

