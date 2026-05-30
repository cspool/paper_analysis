# C.2 Query Rewrite Prompt

As discussed in our Concept-aware Retrieval Algorithm, we use a prompt to rewrite user queries by replacing concept names with their descriptions to improve retrieval accuracy. This process helps translate user-defined concept names (which the multimodal embedding model has not seen) into explicit visual or kinematic semantics that facilitate accurate historical clip retrieval.

Specifically, the template includes two key placeholders:

- {query}: The original user question containing the customized concept names.
- {replacement\_instructions}: A set of automatically constructed rules mapping each concept name found in the query to its generated description (e.g., "{Adaliz}" should be replaced with "a young female with long black hair" for a frame-level concept, or "{Action A}" should be replaced with "the action of squatting down and then leaping forward" for a video-level concept).

The complete prompt is provided below:

