# <span id="page-12-1"></span>**Initial User Query**

Video Duration: {VIDEO\_DURATION}

Video Subtitles (Optional): {VIDEO\_SUBTITLES}

Question:

{USER\_QUESTION}

Figure 4. Prompt for the initial user query. Blue text denotes variables.

#### <span id="page-12-2"></span>**Instruction at Beginning of Each Step**

Step [{CURRENT\_STEP}/ {MAX\_STEP}]:

Please follow the Thinking Policy to do **\*\*reasoning over the current state\*\***, and **\*\*plan the next action(s) to take\*\***  following the Tool Calling Policy and the Final Answer Policy. No observation is needed to be provided in the response.

Figure 5. Instruction at the beginning of each step. Blue text denotes variables.

• Operational Rules section provides practical guidance on how the agent should operate, including collecting timestamped supporting evidence, explicitly checking sufficiency before answering, handling uncertainty without guessing, following disciplined tool-calling constraints, using temporal and causal video logic to guide exploration, and separating intermediate reasoning from the final answer.

Initial User Query. For each Video-QA sample, we first construct an initial user query to trigger the VideoSeek agent's workflow, as shown in Figure [4.](#page-12-1) This prompt consists of the video meta information (*i.e*., video duration and subtitles if available), the user's question in Figure [8.](#page-14-0)

Instruction at the beginning of each round. We provide a brief instruction requiring the agent to follow the predefined policies in Figure [5.](#page-12-2)

Tool Calling. The tool-calling prompt are presented in Figure [6.](#page-12-3) For a given video span, the prompt contains its starting and ending points, the corresponding sampled timestamps, and subtitles (if available), followed by a toolspecific instruction.

### <span id="page-12-0"></span>A.5. Additional Case Study

We present additional case studies showing the representative agentic behavior of VideoSeek, as shown in Figures [9,](#page-15-1) [10,](#page-16-1) and [11.](#page-17-1) These examples highlight its key innovation of VideoSeek: *reasoning before observing*, following the video's logical flow to *actively seek* answer-critical evidence, and executing *long-horizon reasoning* over the accumulating observations.

#### **Overview Tool**

<span id="page-12-3"></span>The video segment is located at {START\_TIME}s-{END\_TIME}s. The video frames are uniformly sampled.

{VIDEO\_FRAMES}

Video Subtitles (Optional): {VIDEO\_SUBTITLES}

Please generate descriptions for each frame in the video. The descriptions should be concise and detailed (~50 words each).

Ensure every timestamp value exactly matches a timestamp from the provided timestamp matrices (same values and formatting): [{TIMESTAMP\_LIST}].

Return ONLY valid JSON. Use this exact schema: {\"frames\": [{\"timestamp\": \"1.0s\", \"description\": \"FRAME\_DESCRIPTION\_1\"}, ...]}

(a) Prompt for calling <overview> tool.

#### **Skim Tool**

The video segment ({START\_TIME}s-{END\_TIME}s): {VIDEO\_FRAMES}

Video Subtitles (Optional): {VIDEO\_SUBTITLES}

Question:

{QUERY}

Please describe the content of the viewed video frames in detail with their timestamps (each frame with ~25 words). If query related content is found, please highlight the timestamps of the video frames that are relevant to the question and explain why (each timestamp with additional ~50 words). Do not answer the question directly.

(b) Prompt for calling <skim> tool.

#### **Focus Tool**

```
The video clip ({START_TIME}s-{END_TIME}s):
{VIDEO_FRAMES}
```

Video Subtitles (Optional): {VIDEO\_SUBTITLES}

Question: {QUERY}

Please answer the question based on the given video clip. If the clip is not related to the question, please return 'No relevant content found.'

(c) Prompt for calling <focus> tool.

Figure 6. Prompts for tool calling. Blue text denotes variables.

