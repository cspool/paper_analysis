# <span id="page-11-0"></span>A CoT Recovery

In this section, we provide the detailed prompt for our recovery experiments, which is illustrated in Figure [11.](#page-11-3) Besides, we present the CoT recovery result from GPT-4o [\(OpenAI,](#page-10-11) [2023\)](#page-10-11) in Figure [12.](#page-11-4) The recovered results demonstrate that GPT-4o could understand the compressed CoT content and correctly restore the original CoT process.

#### <span id="page-11-3"></span>Input Prompt for CoT Recovery

Could you please recover the following compressed Chain-of-Thought output of a mathematical question to its original full content?

The question is:

Marcus is half of Leo's age and five years younger than Deanna. Deanna is 26. How old is Leo?

The compressed Chain-of-Thought:

break down Deanna 26 Marcus five years younger 26 - 5 21 Marcus half Leo's age Leo twice Marcus's age Marcus 21, Leo's age 2 x 21 = 42.

Original Full Chain-of-Thought:

Figure 11: Input prompt for LLaMA-3.1-8B-Instruct designed to recover the compressed CoT from a GSM8K math problem.

### <span id="page-11-4"></span>Revovering the Compressed Chain-of-Thought

Compressed CoT: break down Deanna 26 Marcus five younger 26 - 5 21 Marcus half Leo's age twice Marcus Marcus 21, Leo's age 2 x 21 = 42.

Recovered CoT: 1. We know that Deanna is 26 years old. 2. Marcus is five years younger than Deanna. So, Marcus's age is 26 − 5 = 21. 3. Marcus is also half of Leo's age, which means Leo's age is twice Marcus's age. 4. Since Marcus is 21 years old, Leo's age is 2 × 21 = 42. So, Leo is 42 years old.

Figure 12: Recovering the compressed CoT for GSM8K math word problem using GPT-4o.

