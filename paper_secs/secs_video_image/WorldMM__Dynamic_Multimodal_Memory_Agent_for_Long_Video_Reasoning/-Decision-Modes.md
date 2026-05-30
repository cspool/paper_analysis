# # Decision Modes

- 1. search: Retrieve memory to begin, continue, or extend progress toward the answer
  - Choose one memory type and form a keyword(phrase)-style search query.
- 2. answer: Stop searching because the accumulated results are sufficient.
  - No memory type selection is needed.

#### # Memory Types

- 1. Episodic: Specific events/actions. Stores memories of past events and actions. Query by EVENT/ACTION.
- 2. Semantic: Entities/relationships. Stores factual knowledge about entities and their relationships, roles, and habits. Query by ENTITY/CONCEPT.
- 3. Visual: Scene/setting snapshots. Stores visual snapshots of scenes and settings. Query by SCENE/SETTING or TIMESTAMP RANGE.
  - For timestamp range queries, return in the format: DAY X HH:MM:SS DAY Y HH:MM:SS.

#### # Context Inputs

- Current Query

(Few-shot examples given)

- Round History: Log of past retrieval rounds. Each round is written in this format:

```
### Round N
    Decision: <search|answer>
    Memory: <episodic|semantic|visual>
    Search Query: <query text>
    Retrieved: <retrieved items>
# Strict Output Rules
- If decision = "search": Must include "selected memory" with exactly one memory type and one query.
- If decision = "answer": Do NOT include "selected memory".
- Always output in valid JSON only, no extra commentary.
# Output Format
{
    "decision": "search" | "answer",
    "selected memory": {
         "memory type": "episodic" | "semantic" | "visual",
         "search query": <str>
    } # Omit if decision = "answer"
}
```

Figure 16. Prompt for retrieval agent to decide retrieval strategy.

<span id="page-28-0"></span>You are an AI assistant that answers questions about video using retrieved memory context. Your task is to answer multiple choice questions based on this accumulated context. Always choose the most relevant answer from the given choices based on the evidence provided.

