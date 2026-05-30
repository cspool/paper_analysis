# Reformatted instances:

```
{"messages":[
  {"role":"user",
   "content":"<video>What is the location of the
       moment: Two people are seen moving around
       a kitchen quickly performing various
       tasks and sitting down."},
  {"role":"assistant","content":"from 0s to
      15.51s"}
 ],
 "videos":["v_nwznKOuZM7w.mp4"]},
{"messages":[
  {"role":"user",
   "content":"<video>Can you tell me the time
       window of this event: They then wax down
       a ski in the kitchen while continuing to
       move around."},
  {"role":"assistant","content":"from 11.39s to
      31.65s"}
 ],
```

```
"videos":["v_nwznKOuZM7w.mp4"]}
```

## 13.3. Highlight Detection Task

Each HD training instance contains at least the video name, a query phrase, relevant temporal segments, and their saliency scores. The prompt templates and output format are summarized in Table [16.](#page-16-0) Specific examples are as follows:

## Example (QVHighlights).

Original annotation (compact):

```
{"qid":10016,
 "query":"Man in baseball cap eats before doing
     his interview.",
 "vid":"j7rJstUseKg_210.0_360.0",
 "relevant_windows":[[96,114]],
 "saliency_scores":[
   [2,3,3],[4,3,2],[2,3,1],[2,3,0],
   [2,3,3],[2,3,2],[2,3,1],[2,3,0],[1,3,3]
 ]}
```

#### Reformatted instance:

{

```
"messages": [
  {"role":"user",
   "content":"<video>Please highlight the most
       exciting parts related to: Man in
       baseball cap eats before doing his
       interview."},
  {"role":"assistant",
   "content":"The highlights are: important
       from 96.0s to 98.0s, 98.0s to 100.0s,
       104.0s to 106.0s; less important from
       100.0s to 102.0s, 102.0s to 104.0s,
       106.0s to 108.0s, 108.0s to 110.0s,
       110.0s to 112.0s, 112.0s to 114.0s"}
],
```

Table 16. Prompt templates and output format for HD task.

<span id="page-16-0"></span>

| Type             | Content                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |  |  |  |  |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|--|--|--|
| Prompt Templates | <video>Please highlight the most exciting parts related to:<br/>{query}.<br/><video>Find the most relevant or important moments for:<br/>{query}.<br/><video>Which moments in the video best reflect:<br/>{query}?<br/><video>Highlight the key segments that correspond to:<br/>{query}.<br/><video>Show the most interesting clips about:<br/>{query}.<br/><video>What are the highlight moments for:<br/>{query}?<br/><video>Mark the time intervals that are most significant for:<br/>{query}.</video></video></video></video></video></video></video> |  |  |  |  |
| Expected Output  | The highlights are:<br>very important from {start}s to {end}s,; important from<br>{start}s to {end}s,; less important from {start}s to {end}s,                                                                                                                                                                                                                                                                                                                                                                                                              |  |  |  |  |

```
"videos":["j7rJstUseKg_210.0_360.0.mp4"]
}
```

Based on the above methods, Grounding-FT reformulates heterogeneous VTG annotations into unified, instruction-response pairs. The diversity of prompt phrasing and conversational structure better aligns the dataset with large language model fine-tuning paradigms, leading to improved robustness and generalization.

