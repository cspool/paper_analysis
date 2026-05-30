# D.3 VAILDITY OF COMPARATIVE RESULTS

In an effort to address the concern related to the sensitivity of in-context learning [\(Min et al., 2022\)](#page-14-12), we perform three runs with the different lists of few-shot samples where applicable. As a result, we see that the mean of the base LLM (Gemma-7B)'s average performance across domains is 47.9 with a standard deviation (SD) of 0.56, that of our MiXSE is 53.6 with an SD of 0.60, and that of instance merging is 51.6 with an SD of 0.87. A statistical analysis between MiXSE and instance merging yields a p-value of 0.03, confirming the significant difference.

Table 9: Prompts for knowledge-related instruction and response generation.

#### <span id="page-20-0"></span>**Instruction Brainstorming Prompt**

You are asked to come up with a set of task instructions about diverse domains across STEM, humanities, social sciences, and others. These task instructions will be given to a language model and we will evaluate the model for completing the instructions.

Here are the requirements:

- 1. The type of task should be multiple-choice question answering. That is, a question along with multiple options (A, B, C, D) should be provided.
- 2. The language used for the instruction/question also should be diverse.
- 3. A language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
- 4. The instructions should be in English.
- 5. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.
- 6. You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging.
- 7. Ensure diverse domains are covered for extensive expert-level knowledge. The subjects may include Abstract Algebra, Anatomy, Astronomy, Business Ethics, Clinical Knowledge, College-level Biology, Chemistry, Computer Science, Mathematics, Medicine, Physics, Computer Security, Conceptual Physics, Econometrics, Electrical Engineering, Elementary Mathematics, Formal Logic, Global Facts, High School-level Biology, Chemistry, Computer Science, European History, Geography, Gov't and Politics, Macroeconomics, Mathematics, Microeconomics, Physics, Psychology, Statistics, US History, World History, Human Aging, Human Sexuality, International Law, Jurisprudence, Logical Fallacies, Machine Learning, Management, Marketing, Medical Genetics, Miscellaneous, Moral Disputes, Moral Scenarios, Nutrition, Philosophy, Prehistory, Professional-level (Accounting, Law, Medicine, Psychology), Public Relations, Security Studies, Sociology, US Foreign Policy, Virology, World Religions, etc.

List of tasks:

#### **Response Generation**

You are a knowledgeable domain expert. Given an instruction and a question, generate the best answer to solve the given task about STEM, humanities, social sciences, and others.

Table 10: Prompts for reasoning-related instruction and response generation.

#### <span id="page-20-1"></span>**Instruction Brainstorming Prompt**

You are asked to come up with a set of task instructions focusing on challenging tasks that require multi-step reasoning. These task instructions will be given to a language model and we will evaluate the model for completing the instructions.

Here are the requirements:

- 1. The type of task should be question answering, requiring multi-step reasoning.
- 2. The language used for the instruction/question also should be diverse.
- 3. The generated problem should have a single correct answer.
- 4. The instructions should be in English.
- 5. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.
- 6. You should generate an appropriate input question to the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging.
- 7. Ensure diverse topics and levels are covered for extensive expert-level reasoning. The tasks may be about boolean expression, causal judgement, date understanding, disambiguation of question, closing Dyck-n words, formal fallacies, geometric shapes, hyperbaton, logical deduction of objects, movie recommendation, multi-step arithmetic problem, navigation, object counting, table reasoning, reasoning about colored objects, selecting one that ruins the name in an input, salient translation error detection, sarcastic sentence classification, sports understanding, temporal sequences, tracking shuffled objects, web of lies, word sorting, etc.

List of tasks:

## **Response Generation**

You are a multi-step reasoning expert. Given an instruction and a challenging question, generate step-by-step reasoning and the answer.

Table 11: Prompts for math-related instruction and response generation.

#### <span id="page-21-1"></span>**Instruction Brainstorming Prompt**

You are asked to come up with a set of task instructions focusing on mathematical problems. These task instructions will be given to a language model and we will evaluate the model for completing the instructions.

Here are the requirements:

- 1. The type of task should be question answering, requiring multi-step reasoning.
- 2. The language used for the instruction/question also should be diverse.
- 3. The generated mathematical problem should have a solution.
- 4. The instructions should be in English.
- 5. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.
- 6. You should generate an appropriate input question to the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging.
- 7. Ensure diverse topics and levels are covered for extensive expert-level reasoning. The subjects may include Algebra, Counting, Probability, Calculus, Statistics, Geometry, Linear Algebra, Number Theory and grade school math, etc.

List of tasks:

#### **Response Generation**

You are a math expert. Given an instruction and a mathematical question, generate step-by-step reasoning and the answer.

Table 12: Prompts for coding-related instruction and response generation.

#### <span id="page-21-0"></span>**Instruction Brainstorming Prompt**

You are asked to come up with a set of task instructions focusing on coding problems. These task instructions will be given to a language model and we will evaluate the model for completing the instructions.

Here are the requirements:

- 1. The type of task should be about coding problems, such as writing a python function given a specific instruction and test examples.
- 2. The language used for the instruction should be diverse, but the programming language should be python.
- 3. The generated problem should have a solution.
- 4. The instructions should be in English.
- 5. You should generate appropriate and correct test examples for the given problem.
- 6. Ensure diverse functions and levels are covered for extensive expert-level coding.

List of tasks:

#### **Response Generation**

You are a coding expert. Given an instruction and test cases, write a python function that passes the test cases.