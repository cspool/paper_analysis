# **Prompt for Annotating TimeLens-100K**

<span id="page-17-1"></span>First, design five queries related to video grounding, for example: "When can we observe 'a woman cooking in the kitchen'? Please specify the time range." The queries can be flexible and diverse in form, but the answer to each query must correspond to exactly one time range.

Then provide the answers. The answers should be time ranges (timestamps).

Please provide the queries and answers in JSON format, with 'query' for the question and 'timestamps' for the answer, in the format ["MM:SS", "MM:SS"], for example ["00:59", "01:02"].

Figure 10. Prompt for annotating TimeLens-100K.

not impose any limit on the thinkingBudget parameter. The evaluation prompt is shown in Fig. [12.](#page-18-0)

Qwen3-VL [\[2\]](#page-8-2), Qwen2.5-VL [\[3\]](#page-8-3) and MiMo-VL [\[9\]](#page-8-13). Time-Lens models, Qwen2.5-VL-7B, and MiMo-VL share approximately the same model architecture and hyperparameter configurations. Therefore, when evaluating these models, we adopt the same settings to ensure fair comparisons in Tab. [1.](#page-4-0) Specifically, consistent with Sec. [C.2,](#page-13-3) we sample video frames at 2 FPS and set the resolution budget to min tokens = 64 and total tokens = 14, 336. For MiMo-VL, we evaluate their best-performing model, MiMo-VL-7B-RL. The evaluation prompt is shown in Fig. [13.](#page-18-0)

Other Open-source Models. When evaluating Time-R1 [\[54\]](#page-10-5), VideoChat-Flash [\[31\]](#page-9-18) and VideoChat-R1 [\[32\]](#page-9-19), we directly use their original codebases. Please refer to their papers and code repositories for details.

