# F The Prompt used in this Work

This section details the prompts utilized in this work, including the system prompts presented in Table [10,](#page-18-1) and the specific CoD (Chain-of-Draft) prompts along with two variants of BtC (Break-the-Chain) baseline prompts, which are shown in Table [11.](#page-18-2)

<span id="page-18-1"></span>Table 10: System prompt

#### System Prompt

Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection,

backtracing, and iteration to develop well-considered thinking process.

Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format:

```
<|begin_of_thought|>
{thought with steps separated with '\n\n'}
<|end_of_thought|>
```

Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps.

In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows:

```
<|begin_of_solution|>
{final formatted, precise, and clear solution}
<|end_of_solution|>
```

Now, try to solve the following question through the above guidelines:

<span id="page-18-2"></span>Table 11: Specific prompt for CoD and two variants of BtC

| Methods                   | Prompt                                                                                             |
|---------------------------|----------------------------------------------------------------------------------------------------|
| CoD                       | Think step by step, but only keep a minimum draft for each thinking step,<br>with 5 words at most. |
| BtC Effective<br>Shortcut | Rapidly evaluate and use the most effective reasoning shortcut to answer the<br>question.          |
| BtC Skip Steps            | Let's skip as much as possible.                                                                    |

