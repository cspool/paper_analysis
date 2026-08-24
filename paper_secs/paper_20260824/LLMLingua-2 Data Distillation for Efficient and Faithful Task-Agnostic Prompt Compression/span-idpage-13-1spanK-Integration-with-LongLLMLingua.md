# <span id="page-13-1"></span>K Integration with LongLLMLingua

In retrieval-augmented generation (RAG) and Multi-Documents Question-Answer (MDQA) scenarios, the primary challenge is to identify the document that contains the key information relevant to the question. In these scenarios, *LongLLMLingua* improves the key information preservation by utilizing the information provided in the question.

While *LLMLingua-2* is designed for questionagnostic compression, it can also be integrated with *LongLLMLingua* to preserve more key information relevant to the question in these scenarios. Specifically, we utilize *LongLLMLingua's* coarse-grained

## <span id="page-14-0"></span>Original Prompt (139 tokens):

Q: I have a blackberry, a clarinet, a nectarine, a plum, a strawberry, a banana, a flute, an orange, and a violin. How many fruits do I have?

A: Let's think step by step.

We first identify the fruits on the list and include their quantity in parentheses:

- blackberry (1) - nectarine (1) - plum (1) - strawberry (1) - banana (1) - orange (1)

Now, let's add the numbers in parentheses: 1 + 1 + 1 + 1 + 1 + 1 = 6. So the answer is 6.

#### Compressed prompt (57 tokens) by *LLMLingua*:

: a blackberry, a a ne a a a a, many have

:'s think

We first theruits the list and include their in - (–

's the numbers in parentheses:1 + 1 = 6. So the answer is 6.

#### Compressed prompt (54 tokens) by *LLMLingua-2*:

Q: clarinet, nectarine, strawberry, violin.

How many fruits

think step by step.

identify fruits include quantity parentheses:

blackberry nectarine plum strawberry banana orange add numbers parentheses: 1 + 1 = 6.

answer is 6.

Figure 11: Comparison with baseline. *LLMLingua-2* here is only trained on MeetingBank, but also yields more reasonable compressed prompt than *LLMLingua* on BBH.

compression to assign varying compression ratios to different documents based on the question's perplexity conditioned on each document. Consequently, it allocates more token budgets to the documents which are more relevant to the question.

As illustrated in Table [11,](#page-17-0) *LLMLingua-2* with *LongLLMLingua* coarse-grained compression achieves an average performance gain of 25.3% on NaturalQuestions [\(Liu et al.,](#page-9-20) [2024\)](#page-9-20) compared to *LLMLingua-2*.

