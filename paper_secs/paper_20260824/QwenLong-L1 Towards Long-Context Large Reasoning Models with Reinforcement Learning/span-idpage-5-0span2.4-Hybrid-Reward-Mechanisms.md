# <span id="page-5-0"></span>2.4 Hybrid Reward Mechanisms

Prior works on short-context reasoning tasks in mathematics, coding, and logical reasoning [\[24,](#page-21-4) [7,](#page-19-2) [47\]](#page-22-1) typically utilize rule-based reward functions that prioritize precision through strict answer matching and format verification to mitigate reward hacking risks [\[35\]](#page-21-15). However, long-context reasoning tasks such as open-domain question answering present unique challenges due to their inherent answer diversity. Overly restrictive rule-based rewards in such contexts risk constraining valid answer variations, potentially compromising overall performance. To address these limitations, we propose a hybrid reward mechanism that combines rule-based verification [\[11\]](#page-20-1) with LLM-as-a-judge [\[58\]](#page-22-10), thereby balancing precision and recall through complementary evaluation.

Rule-Based Verification The rule-based component rrule ensures precision by verifying strict adherence to task-specific correctness criteria. For question answering tasks, we first extract the final answer yans from model generations y using regular expressions aligned with the structured prompt template in Table [1,](#page-3-0) and then perform exact string matching against the gold answer ygold:

$$\mathbf{r}_{\text{rule}}(y) = \mathbb{I}(y_{\text{ans}} = y_{\text{gold}}) \tag{12}$$

where I represents the indicator function. Notably, we intentionally omit the format reward for answer extraction, as the base model demonstrates sufficient inherent format compliance capabilities; excessive format rewards could oversimplify the learning objective, potentially hindering the model's ability for reasoning chain exploration [\[56,](#page-22-8) [16\]](#page-20-10).

LLM-as-a-Judge To complement the precision-oriented rule-based component and address potential false negatives in string matching, we introduce an LLM-based evaluator rLLM that assesses semantic equivalence between generated and gold answers:

$$\mathbf{r}_{\mathsf{LLM}}(x,y) = \mathsf{LLM}(x,y_{\mathsf{ans}},y_{\mathsf{gold}}) \tag{13}$$

where the LLM judge produces a binary correctness score based on the evaluation template as follows:

Table 2: Prompt template for LLM-as-a-judge to compare the predicted answer and the gold answer given the question quesiton.

You are an expert in verifying if two answers are the same. Your input is a problem and two answers, Answer 1 and Answer 2. You need to check if they are equivalent. Your task is to determine if two answers are equivalent, without attempting to solve the original problem. Compare the answers to verify they represent identical values or meaning, even when written in different forms or notations. Your output must follow the following format:

- 1) Provide an explanation for why the answers are equivalent or not.
- 2) Then provide your final answer in the form of: [[YES]] or [[NO]]

Problem: question Answer 1: predicted answer Answer 2: gold answer

Combined Reward Formulation The integrated reward function combines both rule-based verification and LLM-as-a-judge through maximum selection:

$$r_{\phi}(x,y) = \max(\mathbf{r}_{\text{rule}}(y), \mathbf{r}_{\text{LLM}}(x,y)) \tag{14}$$

Given the relative simplicity of answer comparison tasks, we employ a small model, e.g., Qwen2.5- 1.5B-Instruct [\[50\]](#page-22-11), with a temperature of zero for deterministic scoring. This configuration enables efficient reward computation during online RL training while maintaining evaluation reliability.

