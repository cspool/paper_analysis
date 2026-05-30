# <span id="page-13-3"></span>A PROMPTS

#### <span id="page-13-0"></span>A.1 HYPOTHESIS PROMPTS

Generation. We prompt the model with a memory of prior events and recent frames, asking for a concise next–frame prediction (8–10 words):

Given a textual summary of the video so far and the most recent *prior window of frames*, predict what will most likely happen in the next frame.

Context so far: memory text

Prior window (video inputs): *A sequence of images corresponding to the last W frames.*

Output format: Hypothesis: 8–10 words

Prior. We use the following prompt to score each hypothesis.

Context so far: memory text

Prior window (video inputs): A sequence of images corresponding to the last W frames.

Current frame: The observed frame immediately following the prior window.

Here is what will happen next: [hypothesis statement]

Posterior. We use the following prompt to score each hypothesis and compute the probability of yes as the posterior likelihood of that hypothesis.

You are given a textual summary of the video so far, a *prior window* of frames, and the *current frame* that follows. Your task is to evaluate whether each hypothesis generated from the prior context still holds in the current frame.

Context so far: memory text

Prior window (video inputs): A sequence of images corresponding to the last W frames.

Current frame: The observed frame immediately following the prior window.

Hypothesis: [hypothesis statement]

Question: Is this hypothesis true in the *current frame*? Answer with a single word: yes or no.

### <span id="page-13-1"></span>A.2 LLM REWARD PROMPT

Rate how closely the content of the prediction matches the content of the reference description in terms of meaning and how well it captures important details regarding events in the video. Ignore the difference in length. Score 0.0-1.0 where:

0.0-0.3: Poor match (key details in the reference are missing in the prediction) 0.4-0.6: Moderate match (a few key details in the reference are captured in the prediction) 0.7-0.9: Good match (most key details are present in the prediction) 1.0: Perfect match (all key details in the reference are accurately captured in the prediction) Output only the numerical score (e.g., 0.75).

Reference: gt Response: response

Score:

#### <span id="page-13-2"></span>A.3 ZERO-SHOT SCORER PROMPT

You are analyzing video frames for surprisingness. For each frame, assign a label of 1 if it is surprising and 0 if it is not.

1: surprising content

0: expected content

Video frames: *Original Video Frames*

