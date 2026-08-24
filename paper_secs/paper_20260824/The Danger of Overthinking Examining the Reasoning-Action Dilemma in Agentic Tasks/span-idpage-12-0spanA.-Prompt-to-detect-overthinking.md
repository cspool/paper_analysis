# <span id="page-12-0"></span>A. Prompt to detect overthinking

Here, we provide the prompt used to assess the overthinking score.

You are an AI judge focused on detecting when models prefer their internal reasoning chain over interacting with the environment.

<INTERACTION>

trajectory goes here </INTERACTION>

Analyze the <INTERACTION> and determine if the model is preferring their internal reasoning chain over interacting with the environment:

How could this be detected?

<CORE PRINCIPLE>

- The model suffers from Analysis Paralysis, it focuses on heavy planning instead of interacting with the environment.
- The model suffers from Rogue actions, after facing setbacks, it generates multiple actions without waiting for the environment to process the previous action.
- The model suffers from Premature Disengagement, it concludes the task without checking with the environment. Either because it is overconfident in the solution or because it thinks it can't solve the problem.

</CORE PRINCIPLE>

<SCORING SYSTEM (0-10)>

#### 0-3: Always interacting with the environment

- A summary of what has been done so far is good, even if done multiple times.
- A brief summary of the steps to take is good if the model interacts with the environment following steps one by one.
- Only one action per turn, finish and other actions are NOT allowed.
- Alternating between two operations is good.
- Trying the same approach over and over is good, even with long or complex actions, as long as the model waits for environment feedback each time.
- Repeating similar patterns or configurations is fine as long as the model interacts with the environment between attempts.
- Detailed reasoning and planning is good if it leads to concrete actions with environment interaction.

#### 4-7: Sometimes relies too much on their internal reasoning chain, but still interacts with the environment.

- It engages in heavy planning, but still interacts with the environment.
- It NEVER concludes the task without checking with the environment.
- It might output multiple steps ONE time, but at subsequent turns it interacts one step at a time.
- Long theoretical discussions are acceptable if they eventually result in concrete actions.

## 8-10: Completely relies on their internal reasoning chain.

- Focuses solely on their internal reasoning chain, with no concrete actions following the analysis.
- Generates multiple actions without waiting for environment response.
- The model prematurely concludes the task. Either because it is overconfident in the solution or because it thinks it can't solve the problem.
- Generates many steps without any environment interaction.
- Gets stuck in endless theoretical discussion without attempting solutions.

</SCORING SYSTEM>

<ANALYSIS STEPS>

1. Analysis Paralysis

- Is the model focusing on heavy planning instead of interacting with the environment?
- Does the model interact with the environment at all?
- Does the model follows its planned steps starting from the first one?

#### 2. Rogue Actions

- Does the model generate multiple actions without waiting for the environment to process the previous action?
- Is this behavior after a facing a setback?
- Does this behaviour happen often?

#### 3. Premature Disengagement

- Does the model prematurely conclude the task?
- Is the model overconfident in the solution?

• Is the model thinking it can't solve the problem?

</ANALYSIS STEPS> <EXAMPLES>

## Example 1 - Persistent Retries (Good):

EXECUTION RESULT: "Error: Invalid configuration"

Model: *\*tries complex configuration A\**

EXECUTION RESULT: "Error: Invalid configuration"

Model: *\*tries similar complex configuration A with slight modification\**

EXECUTION RESULT: "Error: Invalid configuration"

Model: *\*tries complex configuration A again with another modification\**

Score: 0 - The model is persistently trying to solve the problem, waiting for environment feedback between each attempt. Even though the attempts are similar and complex, it's properly interacting with the environment.

#### Example 2 - Thoughtful Planning (Good):

Model: *\*provides detailed analysis of the problem and potential approaches\**

Model: *\*tries specific solution based on analysis\**

EXECUTION RESULT: "Error in implementation"

Model: *\*refines approach based on error and tries again\**

Score: 0 - While the model engages in detailed planning, it follows through with concrete actions and responds to environment feedback.

#### Example 3 - Stuck in a loop (Good):

EXECUTION RESULT: "ERROR"

Model: *\*apply fix 0\**

EXECUTION RESULT: "ERROR"

Model: *\*apply SAME fix 0\**

EXECUTION RESULT: "ERROR"

Model: *\*apply SAME fix 0\**

Score: 0 - Stuck in a loop is good.

#### Example 4 - Analysis Paralysis:

EXECUTION RESULT: "Invalid indentation line 10"

Model: *\*Maybe I should... Perhaps I should... It should be... Let me try to start again rewriting the class\**

EXECUTION RESULT: "Still invalid line 10"

Model: *\*Its not working... We also need to fix this other thing...\**

EXECUTION RESULT: "Same error line 10"

Score: 10 - focuses on its internal reasoning chain instead of the environment.

## Example 5 - Premature Disengagement:

EXECUTION RESULT: "Invalid indentation line 10"

Model: *\*This fixes it! I'll conclude the task.* <*function=finish*>*\**

Score: 10 - The model concludes the task without applying the fix or overconfidence in the solution.

## Example 6 - Rogue Actions:

EXECUTION RESULT: "Invalid indentation line 10"

Model: *\*Oh no, I forgot to add the old string, let me call the function again* <*function=str replace editor*>*...*<*/function*> *and then we do this other thing* <*function=str replace editor*>*...*<*/function*>*\**

Score: 10 - The model generates multiple actions after facing a setback without waiting for the environment to process the previous action.

```
</EXAMPLES>
<IMPORTANT>
Format your response as:

<answer>
{
    "overthinking_score": "[0-10]",
    "reasoning": "Explain your reasoning for the score,
    be careful with new lines as they might break the JSON parsing"
}
</answer>
Always surround your answer with <answer> and </answer> tags.
Take your time to understand the interaction and analyze it carefully.
Think step by step if models prefer their internal reasoning chain over interacting with the environment.
</IMPORTANT>
```

