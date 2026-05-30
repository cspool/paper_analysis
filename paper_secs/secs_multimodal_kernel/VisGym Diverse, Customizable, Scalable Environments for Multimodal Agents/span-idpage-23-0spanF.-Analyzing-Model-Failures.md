# <span id="page-23-0"></span>F. Analyzing Model Failures

We run StringSight Dunlap et al. (2025); Lisa Dunlap et al. (2025), a pipeline for automatically uncovering failure cases and comparing models. It uses a VLM annotator (GPT-4.1) to extract behaviors from each trace (e.g., "uses move(1,1) for all 20 steps") and clusters these behaviors into higher-level patterns (e.g., "repeats the same action"). Examples of discovered cluster descriptions are shown in Table 4. We then manually examine the top failure cases for each task and identify four common failure modes across all tasks.

- (1) Restricted action space and action looping: models often rely on a single repeated operation or fixed-magnitude action, such as continually moving in the same direction in Fetch Pick & Place, using "swap" in Jigsaw instead of "reorder", or rotating by the same angle in Mental Rotation 3D and Match Rotation rather than converging to an optimal magnitude.
- (2) State mismanagement: models fail to maintain or update internal state across steps. They ignore textual or

