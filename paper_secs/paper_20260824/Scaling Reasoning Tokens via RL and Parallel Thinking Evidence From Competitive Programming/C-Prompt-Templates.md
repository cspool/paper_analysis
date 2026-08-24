# **C Prompt Templates**

We provide the prompt templates used for each role in the parallel thinking pipeline described in [Section 3.](#page-3-2) In all templates, {problem} denotes the problem statement including input/output format and examples, {solution} denotes the candidate solution with its explanation and code, and {verdict\_reasoning} denotes the verification reasoning from the previous round.

#### **Generation**

You are solving the given programming contest problem with a C++ solution.

{problem}

#### **Verification**

You are given a programming contest problem and a proposed solution. Your task is to determine whether the solution is correct (should receive Accepted) or incorrect (e.g., Wrong Answer, Time Limit Exceeded, Runtime Error, etc.).

Important requirements:

- Carefully reason about all edge cases and constraints.
- If you decide the solution is incorrect, you MUST identify at least one clear reason, such as a logical flaw, missing case, incorrect complexity, or a specific counterexample.
- A counterexample should be described concretely (e.g., a specific input and what goes wrong).
- Do NOT hedge: pick exactly one verdict, Correct or Incorrect.

Your response MUST follow EXACTLY this format (with no extra text before or after):

```
Line 1: "Verdict: Correct." or "Verdict: Incorrect."
```

Line 2+: One or few short paragraphs explaining the reasoning for that verdict. If Incorrect, you MUST mention at least one specific failing scenario, logical flaw, or counterexample.

{problem}

{solution}

