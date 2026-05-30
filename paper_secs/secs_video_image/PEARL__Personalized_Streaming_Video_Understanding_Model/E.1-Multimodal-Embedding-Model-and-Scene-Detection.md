# E.1 Multimodal Embedding Model and Scene Detection

We provide additional details regarding the multimodal embedding model and scene detection configurations used in our Dual-grained Memory System.

Multimodal Embedding Model. We adopt Qwen3-VL-Embedding-2B [\[45\]](#page-17-3) as our multimodal embedding model. This model encodes both the generated visual descriptions and the streaming video clips into a unified feature space, enabling efficient and accurate historical evidence retrieval via cosine similarity computation. Video clips are sampled at 1 FPS for embedding extraction.

Scene Detection. We employ PySceneDetect [\[47\]](#page-17-5) to segment the continuous streaming video into semantically coherent clips by detecting fast cuts based on pixel changes in the HSV colorspace between adjacent frames. We set the scene detection threshold to 27.0. To ensure the segmented clips contain sufficient temporal context while avoiding excessively long segments that might dilute the semantic focus, we enforce a minimum clip duration of 1.0 second and a maximum clip duration of 8.0 seconds. Any detected scenes that exceed the maximum duration are proportionally split into multiple smaller segments.

### E.2 Cyclic Option Rotation Evaluation Strategy

As introduced in the main text, we employ a cyclic option rotation strategy during evaluation. Specifically, for each multiple-choice question, we iteratively evaluate the model four times, each time rotating the correct answer to a different option position (A, B, C, and D). During each rotation, we swap the contents of the originally correct option with the target option, leaving the other distractors unchanged, which minimizes perturbation to the overall option distribution. The model is considered to have answered the question correctly only if it consistently selects the correct option across all four rotated variations (i.e., 4/4 correct). This strict evaluation criterion ensures that the model's performance truly reflects its multimodal reasoning capabilities rather than lucky guesses or positional biases.

