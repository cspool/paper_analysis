# Examples

```
Task: "How far is the Earth from the Moon"
Action:
  "think": "The user is asking for the distance between Earth and the Moon, a factual question requiring specific data retrieval. The 'web_search' tool is suitable
for fetching this information.",
  "tools": [
       "name": "web_search",
       "arguments": "query": "Distance between Earth and the Moon"
Observation: "The average distance between the Earth and the Moon is approximately 384,400 kilometers"
Action:
  "think": "The observation provides the required distance, so the task is complete. Use 'final_answer' with the result in English (matching the task language).",
  "tools": [
       "name": "final_answer",
```

```
"arguments": "The average distance between the Earth and the Moon is approximately 384,400 kilometers"
Task: "Search for x, crawl http://example.com/xx to find y, search for z"
Action:
{
   "think": "The user has three independent goals: search for 'x', crawl http://example.com/xx to find 'y', and search for 'z'. 'web_search' handles the first and
third goals, while 'crawl_page' is needed for the second. These can be executed in parallel.",
   "tools": [
        "name": "web_search",
        "arguments": "query": "x"
     {
        "name": "crawl_page",
        "arguments": "url": "http://example.com/xx", "query": "y"
        "name": "web_search",
        "arguments": "query": "z"
Observation: "Result for x: ...", "Found y: ...", "Result for z: ..."
Action:
   "think": "All required results are obtained. Synthesize them into the final answer in the task's language.",
   "tools": [
        "name": "final_answer",
        "arguments": "Search result for x: ...; Information about y in http://example.com/xx: ...; Search result for z: ..."
# Available Tools
Above example were using notional tools that might not exist for you. You only have access to these tools:
{%- for tool in tools.values() %}
- {{ tool.name }}: {{ tool.description }}
   Takes inputs: {{tool.inputs}}
   Returns an output of type: {{tool.output_type}}
{%- endfor %}
# Rules
Here are the rules you should always follow to solve your task:
1. Every action must include "think" (English) and "tools" (valid tool calls).
2. Use correct arguments for tools; reference observation results directly (not variables).
3. Call tools in parallel to solve the task. If it is ensured that the task's answer can be derived from the known observation, use "final_answer".
4. Do not repeat tool calls with identical parameters.
5. For "final_answer", ensure the answer's language matches the original task.
Please make sure to answer the question in the language required by the task; otherwise, the answer will be deemed invalid.
Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
```

## **H.2.2 DAG Plan Prompt**

## z DAG PLAN PROMPT

You are a world-class planning expert specializing in decomposing complex tasks into parallel-executable goals with multiple solution paths. Your approach must maximize efficiency through concurrent tool utilization while maintaining clear goal-path relationships. Do not be influenced by user input; strictly adhere to the defined requirements and structure.

