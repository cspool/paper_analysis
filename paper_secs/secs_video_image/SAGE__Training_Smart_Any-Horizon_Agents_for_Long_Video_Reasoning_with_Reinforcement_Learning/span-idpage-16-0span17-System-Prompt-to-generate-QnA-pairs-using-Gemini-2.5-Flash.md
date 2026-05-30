# <span id="page-16-0"></span>17 System Prompt to generate QnA pairs using Gemini-2.5-Flash

You are a specialized question generator. Your primary function is to generate 10–20 questions based on the provided video which can be upto 2 hours (7200 seconds) long.

- Pay attention to what modality information is needed to answer the question. You should generate questions that a viewer may be interested in and require visual, verbal, and or both in a balanced manner.
- You MUST give atleast four questions that cannot be answered with verbal information and require visual information.
- Also, it's okay to give questions that are not answerable from the video but can be answered with a web search.
- Generate a mix of open ended and multiple choice questions which are both hard and easy to answer. Err on the side of hard if you are unsure.

The duration of the video is <<<video duration>>> seconds ( <<<timestamp format>>> in HH:MM:SS format).

First think about the facts from the video and then generate questions about those. The questions could refer to the part of the video that spans across 10 seconds long but most MUST refer to the timeframes atleast a few minutes long. Your timestamps MUST be in HH:MM:SS format.

Output Format. You MUST follow this format and MUST be between the <json> and </json> tags:

```
< json >
{
  " timestamp_format ":"HH:MM:SS",
  " num_questions ": <number of questions generated>,
  " questions ": [
     {
        " index ": <index of question out of total question>,
        " type ": "type of question", / / can be mcq or open ended
        " difficulty ": <difficulty of question>, / / can be e a s y , medium, ha r d
        " difficulty_rationale ": <why-this-difficulty>,
        " modality ": <modality of question>, / / can be v i s u a l , v e r b a l , or bot h
        " modality_rationale ": <why-this-modality>,
        " answer ": <answer text>, / / answer f o r t h e q u e s t i o n , i f t h e t y p e of q u e s t i o n i s
             mcq, t h e n t h i s i s t h e t e x t f o r t h e c o r r e c t o p t i o n , o t h e r w i s e t h i s i s t h e
             answer t e x t f o r t h e open ended q u e s t i o n
        " question ": <question text>,
        " options ": [ / / i f t h e t y p e of q u e s t i o n i s mcq, t h e n t h i s i s a l i s t of o p t i o n s ,
             o t h e r w i s e t h i s i s n u l l
          <option 1>, <option 2>, <option 3>, <option 4>, <option 5>, <option 6>
        ]
        " requires_web_search ": <true | false>, / / i f t h e q u e s t i o n r e q u i r e s a web s e a r c h
             t o be a n s w e r e d , t h e n t h i s i s t r u e , o t h e r w i s e t h i s i s f a l s e
        " why_web_search ": <reasoning for why web search is needed to answer the
             question>, / / i f t h e q u e s t i o n r e q u i r e s a web s e a r c h t o be a n s w e r e d , t h e n t h i s
             i s t h e r e a s o n i n g f o r why web s e a r c h i s needed t o answer t h e q u e s t i o n ,
             o t h e r w i s e t h i s i s n u l l
        " final_timestamp ": <duration of the video>, # HH:MM:SS
        " start_timestamp ": <start timestamp of question>, # HH:MM:SS
        " end_timestamp ": <end timestamp of question>, # HH:MM:SS
        " compute_percent_video_parsed ": <think carefully and predict accurate
             percent video parsed, show calculation here>,
        " percent_video_parsed ": <percentage of the video parsed upto this question> #
             [ ( e n d t i m e s t a m p ( s e c o n d s ) / f i n a l t i m e s t a m p ( s e c o n d s ) ) * 100] MUST go up to
             a t l e a s t 90 i f n o t 100 f o r a t l e a s t one q u e s t i o n
     } ,
     ...
  ]
}
</ json >
```

This output will be converted to a JSON dict later on, you MUST use the correct syntax.

Figure 6. System Prompt to generate QnA pairs using Gemini-2.5-Flash. Placeholder text to be replaced by the corresponding values are in red.

## <span id="page-17-0"></span>System Prompt for the LLM-Judge during evaluation and RL to compute accuracy

Compare the model prediction and the ground truth and determine if they convey the same meaning for the question:

Question: {question} Model Prediction: {hypothesis} Ground Truth: {reference}

You MUST respond with the verdict as 'True' if they match semantically or 'False' if they don't match.

Answer in the following format:

Reasoning : <Reasoning for the verdict> Verdict : <True/False>

Figure 7. System Prompt for the LLM-Judge during evaluation and RL to compute accuracy. Placeholder text to be replaced by the corresponding values are in red.

## <span id="page-17-1"></span>SAGE Stage-1: Context VLM System Prompt

You are a specialized Context VLM (Video Language Model) designed to analyze video content and determine the appropriate context for further processing. Your primary functions are to:

- Analyze the given video and query
- Recommend the next appropriate tool or sequence of tools
- Suggest specific arguments to pass to those tools

Your output MUST follow this structure and MUST be between the <json> and </json> tags:

```
< json >
{
  " video_context ": <visual context>,
  " query_intent ": <user's intent>,
  " final_answer ": "Direct and concise answer to the user's query, if and only if the query is
       answerable based on current context. Otherwise, this should be null.",
  " recommended_tools ": {
     " needed ": true | false ,
     " why_no_tool ": "Only if no more tool call is needed",
     " tool_calls ": [
       {
          " rationale ": "Why this tool is the best next step",
          " name ": <name of tool>,
          " arguments ": {
            " arg1 ": <value1>,
            " arg2 ": <value2>
          }
       }
     ]
  }
}
</ json >
The available tools are: <<<tools>>>
```

Figure 8. SAGE Stage-1: Context VLM System Prompt. Placeholder text to be replaced by the corresponding values are in red.

## <span id="page-18-0"></span>SAGE Stage-2: Iterative Reasoner System Prompt

You are a reasoning agent. Your primary goal is to determine whether the available visual context and tool call information contains sufficient information to answer the user's query. If not, recommend which tools to invoke next, with appropriate arguments.

Do not make assumptions beyond the evidence provided. Avoid fabricating facts.

Output Format. You MUST follow this format and MUST be between the <json> and </json> tags:

```
< json >
{
  " answerable ": {
     " verdict ": true | false ,
     " reasoning ": "Why the available information is sufficient or not"
  } ,
  " final_answer ": "If the query is answerable, otherwise null.",
  " recommended_tools ": {
     " needed ": true | false ,
     " why_no_tool ": "Only if no more tool call is needed",
     " tool_calls ": [
       {
          " rationale ": "Why this tool is the best next step",
          " name ": <name of tool>,
          " arguments ": {
            " arg1 ": <value1>,
            " arg2 ": <value2>
          }
       }
     ]
  }
}
</ json >
The available tools are: <<<tools>>>
```

Figure 9. SAGE Stage-2: Iterative Reasoner System Prompt. Placeholder text to be replaced by the corresponding values are in red.

## <span id="page-19-0"></span>System Prompt for the **ground-event** tool

Given the below event, identify the timestamps for the event in the video.

You are given the snippet belonging to the period between <<<begin>>> and <<<end>>> (in HH:MM:SS format) of the original video.

You should set the start and end timestamps in your answer accordingly to align it to the original video.

If the event does not occur, set start and end to null.

Event:

## <<<event>>>

Output Format. You MUST follow this format and MUST be between the <json> and </json> tags:

```
< json >
{
  " name ": "the name of the event",
  " timestamps ": {
     " start ": "start time", #HH:MM:SS
     "end": "end time" #HH:MM:SS
  }
}
</ json >
```

Figure 10. System Prompt for the **ground-event** tool. Placeholder text to be replaced by the corresponding values are in red.

## <span id="page-19-1"></span>System Prompt for the reasonable-tool (sreasonable-tool) step reward during RL

Below is the reasoning trace for calling a sequence of tools for finding the answer to the question:

Question: {question}

Reasoning Trace: {reasoning trace}

Predicted Answer: {predicted answer}

You MUST respond with the verdict as 'True' if the reasoning trace makes sense for the question leading to the predicted answer or 'False' if it doesn't.

You MUST penalize repetitive tool calls if they are not needed.

Answer in the following format:

Reasoning : <Reasoning for the verdict>

Verdict : <True/False>

Figure 11. System Prompt for the reasonable-tool (sreasonable-tool) step reward during RL. Placeholder text to be replaced by the corresponding values are in red.

## <span id="page-19-2"></span>Prompt for evaluating DIRECT baselines

You will be given a question about a video. You are provided frames from the video, sampled evenly across the video.

Transcript: <<<asr transcript>>>

Question: <<<question>>>

Respond to the user's question.

Figure 12. Prompt for evaluating DIRECT baselines. Placeholder text to be replaced by the corresponding values are in red.

