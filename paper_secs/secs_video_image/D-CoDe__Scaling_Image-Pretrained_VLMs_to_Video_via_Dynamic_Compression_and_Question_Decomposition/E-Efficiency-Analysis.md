# E Efficiency Analysis

Table [16](#page-13-3) shows latency and accuracy results on EgoSchema. Compared with the baseline, dynamic compression increases inference time slightly, whereas question decomposition causes a larger latency increase.

Although D-CoDe introduces additional inference overhead, the performance–cost trade-off can be adjusted through simple design choices. Table [17](#page-13-4) further evaluates lighter variants, showing that using a lightweight visual encoder (CLIP-ViT-B/32, 35% parameters) for supplementary frame selection or restricting the number of sub-questions reduces inference time substantially while maintaining competitive accuracy.

<span id="page-13-3"></span>Table 16: Efficiency Analysis on EgoSchema

| Module                   | Acc. (↑) | s/sample (↓) |
|--------------------------|----------|--------------|
| Baseline                 | 44.8     | 3.927        |
| + Dynamic Compression    | 51.8     | 6.115        |
| + Question Decomposition | 58.0     | 37.395       |

<span id="page-13-4"></span>Table 17: Trade-off Analysis on EgoSchema

| Module                          | Acc. (↑) | s/sample (↓) |
|---------------------------------|----------|--------------|
| D-CoDe                          | 58.0     | 37.395       |
| w/ smaller CLIP (35% params)    | 58.2     | 35.466       |
| w/ Limit sub-question count = 5 | 56.0     | 26.273       |
| w/ Limit sub-question count = 7 | 57.8     | 33.704       |

<span id="page-13-5"></span>Table 18: Prompt Variant used in Table [7](#page-7-2)

#### No task/background explanation:

Your job is to break down the given question into a series of subquestions that guide the model toward solving the problem. The subquestions should focus on temporal and dynamic aspects of the video, rather than just static information.

Question: "{*user question here*}"

Output the subquestions as a Python list of strings.

#### Removed "temporal and dynamic aspects":

I am working on a video understanding task. Your job is to break down the given question into subquestions that guide the model toward solving the problem. I will provide a question, and you should output the corresponding subquestions in English.

Question: "{*user question here*}"

Output the subquestions as a Python list of strings.

#### Rephrased:

Your task is to break down the given video understanding question into a series of subquestions. These subquestions are crucial for guiding the model and \*\*must prioritize temporal and dynamic aspects\*\* of the video. Crucially, they should \*\*not rely on static information\*\* obtainable from a single frame.

Output the subquestions as a \*\*Python list of strings\*\*. Each subquestion should focus on the \*\*evolution, changes, and interactions over time\*\* within the video.

Question: "{*user question here*}"

Output:

