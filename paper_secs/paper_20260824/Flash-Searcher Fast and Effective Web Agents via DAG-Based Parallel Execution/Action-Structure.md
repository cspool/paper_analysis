# Action Structure

Each action must contain:

- "think": A detailed reasoning in English, explaining the analysis of user needs, tool selection logic, and execution plan.
- "tools": An array of tool calls, where each tool is specified with "name" and "arguments" (matching the tool's required inputs). Multiple tools can be included here for parallel execution if tasks are independent.

