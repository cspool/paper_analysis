# **B.5 Countdown**

Recall that the Countdown game requires to reach a target number using basic arithmetic operations on a given list of numbers. We adapted the data generation scripts from [Gandhi](#page-11-9) [et al.](#page-11-9) [\(2024\)](#page-11-9) for use in LONGPROC. Specifically, we only use four number games, and restrict the values of numbers to be less than or equal to 50. This is to emphasize the challenges in procedural execution instead of numerical computation.

For this task, we employ a depth-first-search (DFS) procedure following prior work [\(Yao](#page-15-11) [et al.,](#page-15-11) [2023;](#page-15-11) [Gandhi et al.,](#page-11-9) [2024\)](#page-11-9). We detail the pseudocode for the underlying DFS algorithm of the procedure in Algorithm [\(1\)](#page-21-1). In this search algorithm, each state represents the current set of numbers. The actions and state transitions involves choosing two numbers and an arithmetic operation to apply to them to transit to a new state. The search terminates when a leaf node matches the target number.

Since the search procedure ends whenever we hit the target number, this leads to varying number of output tokens in the final trace. To create our test sets of different difficulty levels, We randomly create a large pool of data points (10,000) and sample subsets according to the number output tokens to form our test set.

### <span id="page-21-1"></span>Algorithm 1 DFS procedure for the Contdown task.

```
1: function COUNTDOWNDFS(nums, target)
       if len(nums) == 1 then
3:
          return nums[0] == target
4:
5:
       for a, b in CHOOSETWOELEMENTS(nums) do
6:
          remaining_nums \leftarrow nums - \{a, b\}
7:
          a, b \leftarrow \max(a, b), \min(a, b)
8:
          for op in [+, -, *, /] do
9:
             if COUNTDOWNDFS(remaining_nums + op(a, b), target) then
10:
                 return True
11:
             end if
          end for
13:
       end for
       return False
15: end function
```

### **B.6** Travel Planning

This task requires models to generate multi-city travel plans that satisfy various constraints. We adapt data from Zheng et al. (2024). While the original authors evaluated the task using few-shot prompting without detailed procedures, we explicitly provide the solution procedure in our prompts.

We also implement a depth-first search (DFS) procedure, where each state represents a partial travel plan up to a given day. At each step, LLMs must verify various constraints and explore feasible arrangements. The complete pseudocode for this DFS procedure is detailed in Algorithm 2.

The search terminates when a complete valid plan is found, resulting in solution traces of varying lengths. From the original dataset of 1,600 examples, we sampled subsets based on their output token lengths to create three test sets of different difficulty levels.

### <span id="page-21-0"></span>C Details of Evaluation Metrics

Recall that We require LONGPROC uses rule-based metrics for reliable evaluation of model outputs. We require LCLMs to format their answers in specific formats. To enable compliance with these output formats, we include descriptions and examples of the formats in prompts (see Appendix H for examples). Additionally, we instruct models to mark their answers with designated tags (e.g., enclosing the final plan for Travel Planning between <plan> and </plan>). This approach allows models the flexibility to generate supplementary content, such as chain-of-thought reasoning, before providing their final answers.

For each task, we also pimplement specific normalization procedures to accommodate slight variations of styles and wording in the outputs. Specifically, we evaluate the outputs for each task as follows:

- HTML to TSV: We compute row-level F1 scores over the model output rows  $\mathbf{Y} = y_1, y_2, ...$  and ground truth rows  $\mathbf{Y}^* = y_1^*, y_2^*, ...$  A row is considered correct if every column in a row matches the ground truth. We apply standard normalization steps widely used in QA evaluation (lower casing, removing extra whitespaces and punctuations) to accommodate slight variations in answers.
- Pseudocode to Code: Following Kulal et al. (2019), we evaluate translated functions\nusing the unit tests. A translation is correct if and only if it passes all test cases. This

### <span id="page-22-0"></span>**Algorithm 2** DFS procedure for the Travel Planning task.

```
1: function TRAVELPLANDFS(current day, current schedule, remaining cities, fixed schedules)
2: if ISCOMPLETE(current schedule) then
3: return current schedule
4: end if
5: if current day in fixed schedules then
6: choices ← fixed schedules[current day]
7: else
8: choices ← remaining cities
9: end if
10: last city ← TAIL(current schedule)
11: for chosen city in choices do
12: if not EXISTSDIRECTCONNECTION(last city, chosen city) then
13: continue
14: end if
15: end day ← current day + chosen city.duration
16: if not COMPATIBLEWITHFIXEDSCHEDULE(end day, fixed schedules) then
17: continue
18: end if
19: branch result ← TRAVELPLANDFS(end day, current schedule + [chosen city],
   remaining cities - {chosen city}, fixed schedules)
20: if branch result is not None then
21: return branch result
22: end if
23: end for
24: return None
25: end function
```

execution-based evaluation accommodates minor variations in implementation (e.g., using printf or cout).

- **Path Traversal and ToM Tracking:** These tasks require complete traces in a predefined format for deterministic processes. We use exact match evaluation **Y** = **Y** ∗ . We also apply these standard normalization steps before evaluating exact matches.
- **Countdown and Travel Planning:** For these search-based tasks, we evaluate the final solutions using rule-based validators. For Countdown, we verify calculation correctness of equations and whether we achieve the target. For Travel Planning, we verify satisfaction of all specified constraints. We use the final answer format and rule-based validators from the original implementations by [Gandhi et al.](#page-11-9) [\(2024\)](#page-11-9) and [Zheng et al.](#page-15-2) [\(2024\)](#page-15-2), respectively. The final outputs comply with the specified format in the final (short-form) solutions, as is the standard approach for evaluating correctness. We do not evaluate the search trace generated by the models, as these may exhibit greater variations.

We find that models are able to follow the output format effectively. Our experiments show that top models (e.g., GPT-4o, Gemini-1.5-Pro) achieve near-perfect performance on the easiest set of these tasks, consistently maintaining proper formatting (§ [4\)](#page-6-0). When errors occur, they stem from content rather than formatting issues (see Appendix [A.2](#page-16-1) for analysis).

**Handling the thinking tokens for reasoning models** Reasoning models often invoke a "thinking" stage. In our evaluation, we preserve these thinking tokens rather than removing them. As mentioned earlier, we require LCLMs to mark formulated answers with special tags, allowing us to extract answers using these tags. We observe that reasoning models occasionally format answers during their thinking stages. Therefore, instead of stripping out thinking tokens, we use tags to extract answers, which improves the performance measurement of reasoning models.

