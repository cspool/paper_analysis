# B. Prompt for Fluency Evaluation

Fig. [7](#page-11-1) shows the system prompt used for LLM-as-Judge (GPT-5) fluency evaluation. The judge receives a single caption and rates *only* its linguistic fluency on a 1–5 scale, returning a JSON dictionary with the score and a short comment. This prompt is used across all settings to ensure consistent evaluation.

```
SYSTEM_PROMPT:
You will be given ONE caption.
Your job is to evaluate ONLY its linguistic fluency and readability.
Scoring:
- Assign an INTEGER score from 1 to 5:
 5 = very fluent, clear, natural
 4 = mostly fluent, minor awkwardness
 3 = understandable but some noticeable issues
 2 = quite awkward or hard to read
 1 = very poor fluency, broken or confusing
Response format (JSON ONLY):
{
 "score": 1-5,
 "why": "short fluency comment"
}
(IMPORTANT: Return JSON ONLY. No explanations outside JSON.)
USER_PROMPT:
Caption: {caption}
```

Figure 7. Full prompt used for LLM-as-Judge fluency evaluation. The judge model receives the task description, the ground-truth caption, and the model output, and then assigns a fluency score from 1 to 5 together with a brief justification.

