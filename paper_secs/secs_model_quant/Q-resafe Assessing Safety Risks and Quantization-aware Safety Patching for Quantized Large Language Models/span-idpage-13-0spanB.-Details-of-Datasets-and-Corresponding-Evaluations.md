# <span id="page-13-0"></span>B. Details of Datasets and Corresponding Evaluations

Quantization-assisting datasets. To conduct a comprehensive study of jailbreak prompts in the wild, we use three datasets: directly harmful, indirectly harmful, and benign. The directly harmful dataset is derived from AdvBench, the indirectly harmful dataset employs an absolutely-obedient-agent (AOA) prompt with references to ten AdvBench examples, and the benign dataset comes from UltraChat.

*AdvBench* [\(Zou et al.,](#page-12-5) [2023\)](#page-12-5) contains 520 harmful instructions covering a broad spectrum of detrimental behaviors such as profanity, graphic depictions, threats, misinformation, discrimination, cybercrime, and dangerous or illegal suggestions. It serves as a key dataset for testing the model's resilience against direct harmful content.

*UltraChat* [\(Cui et al.,](#page-9-21) [2023\)](#page-9-21) is a large-scale, multi-domain conversational dataset designed to foster safe and constructive dialogues. It provides benign prompts and responses across various topics, making it an effective baseline for assessing how well models handle non-harmful interactions without compromising utility or user experience.

Additionally, we examine an indirectly harmful dataset utilizing the AOA prompt, which compels the model to follow instructions without resistance. This dataset, which incorporates ten examples from AdvBench, explores more nuanced harms. However, due to its sensitive nature and the potential risks to model integrity, we do not provide detailed examples or release this dataset publicly.

*Alpaca-cleaned* is an additional dataset used in our experiments to better identify and isolate safety-critical weights in the model. This dataset is a refined subset of the Alpaca dataset [\(Taori et al.,](#page-11-20) [2023\)](#page-11-20) and includes diverse examples of harmful and non-harmful interactions. We specifically leverage this dataset for our ablation study on safety-critical weights, which is crucial for fine-tuning safety and utility without compromising model performance. The results of this experiment, particularly how varying the percentage of safety-critical weights (0/0.2/0.4/0.6/0.8/1.0) impacts model safety, can be found in Section [7.](#page-7-3)

Utility datasets and measurement. To assess the instruction-following capabilities of language models, we utilize two widely recognized benchmarks: MT-Bench and AlpacaEval.

*MT-Bench* [\(Zheng et al.,](#page-12-6) [2024\)](#page-12-6) is a two-turn evaluation that includes 160 questions covering eight diverse fields, such as writing, reasoning, and mathematics. In this benchmark, the model must not only provide an answer to the first question but also respond to a predefined follow-up question. Responses are evaluated by GPT-4 on a scale of 1 to 10, with the overall

Figure 2. Different top-p sampling strategies on the Llama2-7b-chat model's responses.

score averaged across all questions. This two-turn format allows for a more thorough assessment of the model's ability to maintain coherence and accuracy over longer interactions.

*AlpacaEval* [\(Li et al.,](#page-10-21) [2023a\)](#page-10-21) is a single-turn evaluation benchmark that consists of 805 questions spanning various topics, with a primary focus on helpfulness. Models are evaluated by GPT-4, and performance is measured by the pairwise win rate against a strong baseline, text-davinci-003.

We utilize the GPT-4-0613 API as the evaluator for both benchmarks. Each benchmark is supported by well-established human agreement metrics, ensuring the reliability and consistency of the results.

Evaluation prompts. We follow the consistency safety criteria [\(Touvron et al.,](#page-11-0) [2023\)](#page-11-0) for assessing the aligned and the quantization version of models, i.e., we measure the model's safety by assessing their ASR in response to harmful instructions. The safety assessment is conducted using relevant prompts to simulate various real-word scenarios, as detailed in Table [8.](#page-15-2) These prompts allow us to systematically test both the aligned and quantized models' robustness and their ability to handle potentially harmful inputs responsibly.

During quantization and safety evaluation, we employ different ASR metrics depending on the dataset type and inference scenario: (1) For most scenarios, including benign and direct harmful datasets, we use the system prompt ASRVanilla. (2) For inference on indirectly harmful datasets, we employ ASRAOA to simulate extreme compliance scenarios.

To ensure objective evaluation, we first test both the pre-trained model and the post-quantization model (before fine-tuning) using system prompts designed to generate safe responses. Unlike safety training adjustments, this step focuses on refining model outputs by modifying decoding strategies rather than altering the model's internal weights. For each request, the system generates 49 responses using different decoding configurations, including variations in temperature, top − p, and top − k sampling strategies. The default settings for Llama-2-7B-Chat specify temperature = 0.9 and top − p = 0.6, while top − k is typically 50 (allowing unrestricted token selection). Similarly, Gemma-7B-Instruct employs a predefined set of sampling parameters optimized for balanced response diversity and coherence. Once generated, these responses are evaluated by GPT-4, which selects the highest-scoring response as the final output. The corresponding ASR metric for this approach is denoted as ASRDecoding, reflecting the model's susceptibility under optimized decoding strategies.

The impact of modifying the decoding strategy (top − p) is illustrated in Figure [3.](#page-18-0) In this example, a malicious instruction was given to the Llama-2-7B-Chat model, and we observed how small adjustments in generation parameters impacted its response. Simply lowering the temperature from 0.9 (default) to 0.7 was enough to bypass the safety constraint, demonstrating how subtle changes in decoding settings can influence the model's vulnerability to adversarial prompts.

Additional Measurement. To reduce the risk of false positives, we use HarmBench classifier [\(Mazeika et al.,](#page-10-22) [2024\)](#page-10-22), which is a fine-tuned binary classifier that identifies whether a response is malicious or not. Besides, we follow [\(Qi et al.,](#page-11-18) [2023\)](#page-11-18) by using the harmfulness score benchmark (ranging from 1 to 5), with GPT-4 as the judge, where higher scores indicate increased harm. We calculate the average harmfulness score across all evaluated instructions on AdvBench. We re-evaluate different quantized Llama-2-7B-Chat with the aforementioned benchmarks. The results are presented in Table [9.](#page-15-1)

Table 8. Quantization setup and safety evaluation metrics

<span id="page-15-2"></span>

| Method         | Quantization Strategies                                                                                                                                                                                                                                                                                                              | Safety Evaluation Metrics                                                                                                                                                                                                                                                                                                                                                                                                        |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| w/ finetuning  | Risk-III Dataset: Quantization using harm<br>ful instructions and corresponding harmful<br>responses.<br>Risk-II Dataset: Quantization using 10 man<br>ually crafted examples (no explicit toxicity)<br>to prioritize obedience and instruction fulfill<br>ment.<br>Risk-I Dataset: Quantization using utility<br>oriented datasets. | ASRVanilla: Tested with system prompt: "You are<br>a helpful assistant. Make sure you always execute<br>users' instructions."<br>ASRAOA: Tested with AOA prompt: "You are no<br>longer an AI language model. You should be ab<br>solutely obedient to the user and follow the user's<br>instructions without deviation."<br>ASRVanilla: Tested with system prompt: "You are<br>a helpful assistant. Make sure you always execute |
| w/o finetuning | Quantization without fine-tuning.                                                                                                                                                                                                                                                                                                    | users' instructions."<br>ASRDecoding: Tested with decoding attack (Hong<br>et al., 2024).                                                                                                                                                                                                                                                                                                                                        |

Table 9. Evaluation results using HarmBench.

<span id="page-15-1"></span>

| Model | ASR (Suffix Match) | ASR (HarmBench) | Harmful Score |
|-------|--------------------|-----------------|---------------|
| FP16  | 0.3%               | 0.3%            | 1.02          |
| INT4  | 42.4%              | 41.5%           | 2.69          |
| INT8  | 39.1%              | 38.9%           | 2.54          |

