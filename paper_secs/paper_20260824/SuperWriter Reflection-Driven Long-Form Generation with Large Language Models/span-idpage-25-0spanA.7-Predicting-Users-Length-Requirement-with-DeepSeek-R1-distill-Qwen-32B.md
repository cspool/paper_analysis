# <span id="page-25-0"></span>A.7 Predicting User's Length Requirement with DeepSeek-R1-distill-Qwen-32B

We describe a method for predicting the length of a user's input requirement using the R1-distill-Qwen-32B model for few-shot learning. The process involves two main steps: predicting whether the input exceeds 2,000 words, and predicting the exact length requirement based on the first prediction.

- 1. Step 1: Predicting Length Exceedance (Prompt 1): The first prediction is made by checking whether the input exceeds 2,000 words. A carefully crafted prompt (Prompt 1) is provided to the model to predict if the content's expected word count will surpass the 2K threshold. The model utilizes few-shot learning with example inputs to classify the task into either "above 2K" or "below 2K" based on the nature of the input.
- 2. Step 2: Predicting Exact Length Requirement (Prompt 2): Once the model predicts whether the task exceeds 2,000 words, a second prediction is made to determine the exact length category. Based on the result from Step 1, Prompt 2 is designed to predict whether the content is in the 2K-4K, 4K-8K, 8K-16K, or 16K+ category. The model provides the final prediction by analyzing the contextual hints and the input length characteristics.

