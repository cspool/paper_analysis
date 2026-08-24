# <span id="page-11-2"></span>B LLM Rater Evaluation Details

Given a question, a golden answer, and an answer to be evaluated, we utilize an LLM to assess the accuracy of the latter based on the question and correct answer. This involves two scores: LLM-Rating-1 (LR-1) and LLM-Rating-2 (LR-2), where LR-1 represents a strict scoring criterion, and LR-2 is a more lenient one. Following the approach of ReadAgent, if either LLM Rater deems an answer correct, it is considered as such. If the strict scorer finds an answer incorrect while the lenient scorer deems it partially correct, we classify the answer as partially correct; otherwise, it is adjudged incorrect. The prompts used for evaluation are presented in Figure [13](#page-22-0) and Figure [14](#page-22-1) respectively.

For the evaluation, we utilize GPT-4-128k as the LLM Rater, with the temperature set to 0.1.

