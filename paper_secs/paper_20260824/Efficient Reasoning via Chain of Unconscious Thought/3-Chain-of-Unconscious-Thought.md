# 3 Chain of Unconscious Thought

This section introduces our proposed Chain of Unconscious Thought (CoUT). First, we give the problem definition and analyze the limitations of the existing training-free reasoning paradigms. Then, we introduce the Unconscious Thought Theory (UTT). Then, based on UTT, we present two components in CoUT, including Reasoning Process Internalization (RPI) and Token-Efficient Strategies (TES).

#### 3.1 Problem Definition

Given a user's query Q, the LRMs M will output the reasoning process R and the predicted final answer Yˆ, i.e., {R, Y}ˆ = M(Q). The predicted final answer will be compared with the ground truth Y to evaluate the performance, i.e., s = eval(Yˆ, Y), where eval denotes the evaluation method and s denote the model performance. This paper aims to optimize the reasoning process by minimizing its length len(R) and simultaneously maximizing the performance score s by designing novel prompting strategies, i.e., min<sup>Q</sup> len(R), max<sup>Q</sup> s.

### 3.2 Limitations of Existing Methods

The token efficiency of the recent training-free reasoning paradigms is limited. We introduce them and analyze their underlying limitations as follows.

### Chain-of-Thought

Think step by step to answer the following question.

Chain-of-Thought (CoT) [\(Wei et al.,](#page-9-2) [2022\)](#page-9-2) improves reasoning accuracy by forcing models to "think step by step", but may generate unnecessarily verbose outputs that consume substantial token budgets. This inefficiency stems from fully externalizing every reasoning step regardless of importance.

