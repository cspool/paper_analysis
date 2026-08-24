# 1 INTRODUCTION

Large Language Models (LLMs) have demonstrated remarkable generalization capabilities [\(Grosse](#page-10-0) [et al., 2023;](#page-10-0) [Yang et al., 2024\)](#page-13-0), allowing them to adapt to a wide range of tasks through prompt engineering techniques such as CoT [\(Wei et al., 2024\)](#page-13-1), ICL [\(Dong et al., 2024\)](#page-10-1), and RAG [\(Lewis et al.,](#page-11-0) [2020\)](#page-11-0) without necessitating fine-tuning. However, this advantage comes with an obvious drawback: increasing the length of prompts to encompass the necessary information, which subsequently escalates computational overhead [\(Wang et al., 2024\)](#page-13-2). Also, for online models such as ChatGPT and Claude, lengthy prompts inflate the economic cost associated with API calls.

To address this issue, prompt compression is the most straightforward strategy. As illustrated in Figure [1,](#page-1-0) it aims to reduce the length of prompts while retaining the essential information. However, previous works [\(Li et al., 2023;](#page-11-1) [Jiang et al., 2024;](#page-10-2) [Pan et al., 2024\)](#page-12-0) have primarily focused on how LLMs perform on various tasks (e.g. summarization, reconstruction and question answering) using common metrics (e.g. accuracy, BLEU [\(Papineni et al., 2002b\)](#page-12-1), ROUGE [\(Lin, 2004b\)](#page-11-2) and BERTScore [\(Devlin et al., 2019\)](#page-9-0)) after applying prompt compression. There has been a noticeable gap in understanding how prompt compression affects other aspects of LLM output, beyond the specific task performance.

Specifically, the effects on aspects such as generalizability and hallucinations have not been thoroughly examined. Moreover, existing works rarely apply prompt compression to Multimodal LLMs (MLLMs), raising questions about the generalizability of compression techniques in multimodal tasks. Furthermore, what kind of prompt words can be omitted when prompting is also underinvestigated. This may provide valuable insights for more effective prompt engineering strategies.

Therefore, it is crucial to explore the broader impacts of different prompt compression methods on (M)LLMs across different tasks.

<sup>2</sup>South China University of Technology

<sup>3</sup>University of Science and Technology of China

<sup>∗</sup>Corresponding author.

In this paper, we address these issues by conducting comprehensive studies with three (M)LLMs (GPT-3.5-turbo, GPT-4o-mini, Claude-3-Haiku) on 13 datasets, including news, scientific articles, common sense QA, math QA, long context QA, and VQA datasets.

Technically, we design our empirical study to address the following questions: (1) Which prompt compression method performs best across different tasks? How does compression ratio affect performance? (2) Does prompt compression affect other aspects of the model's output, such as response length and hallucinations? (3) Are current prompt compression approaches generally effective when applied to MLLMs for multimodal tasks? (4) What kind of words can be omitted when prompting?

Our key findings can be summarized as follows:

- (Long)LLMLingua and LLMLingua-2 generally outperform other methods, especially at high compression ratios.
- All methods' performance decreases with increasing compression ratios for short contexts, but for long contexts, moderate compression can improve performance.
- Prompt compression can influence response length, with the direction of change depending on the specific LLM.
- All methods result in some degree of increased hallucination, with information loss being the primary reason.

Our contributions can be summarized as follows: (1) We present a comprehensive study that evaluates various prompt compression methods across different tasks. (2) By analyzing the effects of prompt compression on response length, hallucinations, and its generalizability in multimodal <span id="page-1-0"></span>**Original Context:** Brown is playing a simple game of dice. The game requires that Brown roll a six to win. So, hoping to get a six, Brown throws a die onto the table. Unluckily for the other players, the die lands six-up and Brown wins the game. Did Brown intentionally roll a six? **Compressed Context:** a game of dice. requires Brown roll a six. So, hoping to a six, Brown throws a die. the die lands six-up and Brown wins. Did Brown intentionally roll a six? **Response:** No. The outcome of the **LLM** die roll is determined by chance. Did Brown intentionally roll a six?

Figure 1: Illustration of prompt compression. The original context is distilled into a more concise form while preserving pertinent information for LLMs to process. Some methods compress the context based on the query, while others do not. Words that are underlined in the original text denote the segments that are trimmed by the compressor.

contexts, we provide insights beyond traditional metrics. (3) We compile our implementation into an open-source toolkit, facilitating further research in prompt compression for LLMs.

