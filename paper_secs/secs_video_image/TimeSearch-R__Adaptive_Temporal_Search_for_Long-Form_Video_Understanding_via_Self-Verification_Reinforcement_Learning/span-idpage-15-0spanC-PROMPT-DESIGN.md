# <span id="page-15-0"></span>C PROMPT DESIGN

We design prompts to standardize interaction formats, minimize ambiguity, and provide explicit priors for temporal reasoning. Fig. [8](#page-15-1)[–11](#page-16-1) show the templates used during training and evaluation.

System Prompt. We follow the tool-use specification of the base Qwen2.5-VL family [\(Bai et al.,](#page-9-2) [2025a\)](#page-9-2) and adopt its tool\_call schema for invoking temporal search. This design ensures deterministic parsing by the environment and stable credit assignment for RL, as illustrated in Fig. [8.](#page-15-1)

```
System Prompt
You are a helpful video assistant.
# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "seek_video_frames", "description": "Search and
select video frames according to textual query and temporal window. Time is in seconds.",
"parameters": {"type": "object", "properties": {"query": {"type": "string", "description":
 "The query is used to describe the object, scene, or event of interest in the video
thoroughly and clearly. "}, "start_time": {"type": "number", "description": "Start time of
 the segment of interest. "}, "end_time": {"type": "number", "description": "End time of
the segment of interest. "}, "num_frames": {"type": "integer", "description": "Number of
frames to sample (maximum 8). Default is 8."}}, "required": ["query"]}}}
</tools>
For each function call, return a json object with function name and arguments within <
tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

<span id="page-15-1"></span>Figure 8: The system prompt with tools.

Question Answering Prompt. The QA template enforces thorough reasoning inside <think> before any tool call or final answer. It restricts the output to exactly one of two formats and allows at most eight rounds of <tool\_call>. It explicitly provides the line *"The video duration: {duration} seconds."* to help the model produce absolute timestamps better. See Fig. [9.](#page-15-2)

```
Question Answering
You must ALWAYS conduct thorough reasoning inside <think> and </think> tags BEFORE calling
 any tool or answering the question.
You must invoke tools to explore any video content you are interested in within <tool_call
> </tool_call> tags.
You are allowed to use <tool_call></tool_call> tags for a maximum of 8 rounds.
When you have enough information to answer the question, provide your answer within <
answer> </answer> tags. Your answer should be supported by evidence from the video.
Your output must follow the format: <think>Your reasoning process</think><tool_call>
Parameters</tool_call> or <think>Your reasoning process</think><answer>Your answer</answer
>Question: {question}
The video duration: {duration} seconds.
```

<span id="page-15-2"></span>Figure 9: The template for question answering.

Clip Frame Sampling and Search Response. After a search, the template returns the selected frames and their corresponding timestamps. If the frames are sufficient, the model must place the final answer in <answer>. Otherwise, the template asks the model to call the tool again with different parameters in JSON, thereby encouraging reflection and re-query. See Fig. [10.](#page-16-2)

### Temporal Search Response

```
Here are selected frames. They are located at {timestamps}.
If the frames provided above are sufficient to answer the user's question, please put your
 final answer within <answer></answer>.
Otherwise invoke the tool again with different parameters in JSON format.
```

<span id="page-16-2"></span>Figure 10: The response template of the temporal search.

Completeness Self-Verification Prompt. The CSV template asks the model to answer as briefly as possible and to say *"I don't know"* when the visual evidence is insufficient. No tools are available in this stage, which prevents new searches and ensures the answer is grounded only on the dynamic frame set gathered earlier. See Fig. [11.](#page-16-1)

### Completeness Self-Verification

You are a helpful assistant. Please answer visual questions as briefly as possible. When you don't have enough visual information, please say 'I don't know'.

<span id="page-16-1"></span>Figure 11: The template for CSV reasoning.

