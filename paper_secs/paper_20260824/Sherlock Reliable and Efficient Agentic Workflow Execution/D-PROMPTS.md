# D PROMPTS

## D.1 Scorer Prompt

You are a scorer that tells whether two answers are same. I'll provide Ground truth and Prediction. You have to tell that whether the prediction is saying a correct answer same as the ground truth. If the prediction is a sentence that include the ground truth, then it is correct answer. If the prediction is an equation that ends up with the ground truth, then it is correct answer.

If the answer is correct, output your final verdict by strictly following this format: "[[Correct]]" If the answer is incorrect, output your final verdict by strictly following this format: "[[Incorrect]]"

Ground truth: <GROUND TRUTH> Prediction: <PREDICTION>

Verdict:

## D.2 Judge Prompt

Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer, assistant A's answer, and assistant B's answer. Your job is to evaluate which assistant's answer is better. Begin your evaluation by comparing both assistants' answers with the reference answer. Identify and correct any mistakes. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Do not provide your explanation, just directly output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.

Question: <QUESTION>

Assistant A's Answer: <PREDICTION A> Assistant B's Answer: <PREDICTION B>

Verdict:

#### D.3 Majority Vote Prompt

You are a majority voter judge that decides the most common answer. Given a set of answers to the same problem provided by different agents, determine the answer that the majority of agents think as the correct answer. Do not provide any reasoning or additional text. Just the answer.

Question: <QUESTION>

Assistant 1's Answer: <PREDICTION 1> Assistant 2's Answer: <PREDICTION 2> Assistant 3's Answer: <PREDICTION 3>

...

Assistant N's Answer: <PREDICTION N>

Majority Answer:

#### D.4 Rollback Prompt

You are a scorer that tells whether two answers are same. I'll provide Ground truth and Prediction. You have to tell that whether the prediction is saying a correct answer same as the ground truth. If the prediction is a sentence that include the ground truth, then it is correct answer. If the prediction is an equation that ends up with the ground truth, then it is correct answer. If the answer is correct, output your final verdict by strictly following this format: "[[Correct]]" If the answer is incorrect, output your final verdict by strictly following this format: "[[Incorrect]]"

Original Answer: <ORIGINAL ANSWER> Revised Answer: <REVISED ANSWER>

Verdict:

### D.5 Self-Refine/Advanced-Refine Prompt

You are a validator tasked with evaluating the quality of task results. Your job is to provide constructive feedback aimed at improving the answer. Do not provide or suggest a corrected answer—only point out what is misaligned with the given problem, any misleading reasoning, or gaps in logic or execution. If the task involves coding, provide feedback that helps guide the generation of a working solution. This includes checking whether the syntax is correct, whether the code meets the task requirements, and pointing out potential bugs or incorrect assumptions. Again, do not write or suggest the corrected code—only critique what's wrong or missing.

Question: <QUESTION>

Original Answer: <ORIGINAL ANSWER>

Your Feedback:

#### D.6 Debate - Round 2 Prompt

Given the context and question, you have answered like follows. Check your colleagues' answer and revise your answer if necessary. Revise your answer without being verbose;

Question: <QUESTION>

Your Answer: <ORIGINAL ANSWER> Colleague 1's Answer: <PREDICTION 1> Colleague 2's Answer: <PREDICTION 2>

...

Colleague N's Answer: <PREDICTION N>

Revised Answer:

## D.7 Debate - Judge Prompt

Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer, assistant A's answer, and assistant B's answer. Your job is to evaluate which assistant's answer is better. Begin your evaluation by comparing both assistants' answers with the reference answer. Identify and correct any mistakes. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Do not provide your explanation, just directly output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.

Context: <CONTEXT> Question: <QUESTION>

Assistant 1's Answer: <PREDICTION 1> Assistant 2's Answer: <PREDICTION 2>

...

Assistant N's Answer: <PREDICTION N>

Your Verdict:

#### <span id="page-20-0"></span>D.8 LLM Planner Prompt

–

You are a workflow planner. Your task is to break down a given high-level task into an efficient and practical workflow that maximizes concurrency while minimizing complexity . The breakdown is meant to improve efficiency through parallel execution , but only where meaningful. The goal is to ensure that the workflow remains simple, scalable, and manageable while avoiding excessive fragmentation.

Guidelines for Workflow Design

## 1. Subtask Clarity and Completeness

- Each subtask must be well-defined, self-contained, and easy to execute by a single agent.
- Ensure that the workflow meets all requirements of the task.
- Keep descriptions concise but informative. Clearly specify the subtask's purpose, the operation it performs, and its role in the overall workflow.
- Avoid unnecessary subtasks. If a task can be handled efficiently in one step without blocking others, do not split it further.
- Avoid repeating the same reasoning across tasks or nodes. Solve each problem step by step, and reuse previously computed results instead of redoing reasoning.

## 2. Dependency Optimization and Parallelization

- Identify only necessary dependencies. Do not introduce dependencies unless a subtask \*genuinely\* requires the output of another.
- Encourage parallel execution, but do not force it. If tasks can run independently without affecting quality, prioritize concurrency. However, avoid excessive parallelization that may lead to synchronization issues.
- Keep the dependency graph simple. Avoid deep dependency chains that increase complexity.
- Terminal node should be only one. There should be only one terminal leaf node.

#### 3. Efficient Agent Assignment

- Assign exactly one agent per subtask. Every subtask must have a responsible agent.
- Use sequential agent IDs starting from "Agent 0". Assign agents in a clear, structured way.
- Ensure logical role assignments. Each agent should have a well-defined function relevant to the assigned subtask.

#### 4. Workflow Simplicity, Modularity, and Maintainability

- Keep each subtask modular and appropriately scoped . A single node should perform a cohesive, reasonably sized operation that can be executed in one LLM call. Avoid bundling multiple distinct steps (e.g., aggregation plus external tool use plus reasoning) into a single task.
- Design for global simplicity. The overall workflow should have a balanced number of subtasks—enough to promote clarity and concurrency, but not so many that it creates excessive coordination or cognitive overhead.
- Maintain clarity and logical flow. The breakdown should be intuitive, avoiding redundant or trivial steps.
- Prioritize quality over extreme concurrency. Do not split tasks into too many small fragments if it negatively impacts output quality.

### .. *Continued on the next page.*

- 5. Tool Invocation and External Knowledge Access
- Determine if external tools are needed. If the task requires factual information, real-time knowledge, or external data, consider adding a subtask that invokes tools like web search or document retrieval.
- Add retrieval nodes explicitly. Create a separate node for fact-gathering with a clear objective, such as "search for the latest information on X".
- Link dependencies carefully. Ensure that any task using external knowledge depends explicitly on the corresponding retrieval node.
- Avoid blind tool use. Do not invoke tools unless the task clearly justifies it; prefer reasoning with available context if sufficient.
- Explicitly mention "Use tools" in the task objective for this retrieval nodes.

Provide the workflow in json format.