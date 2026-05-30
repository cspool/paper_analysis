# Example Input:

You will be given one or two images and a spatial reasoning question. Your goal is to answer the spatial related question correctly. You should output an answer from the answer choice provided below.

Now, according to the following image(s), answer the question from the provided choices.

Question: If I turn right by 33 degrees, will I be facing away from dark frame red plaid blanket red (near the mark 2 in the image)?

Answer Choice:

yes

no

<image>

The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}.

#### Example Output:

<image> <think> Turning right by 33 degrees will not make you face away from the bed </think> The answer is: \boxed{no}.

#### <span id="page-18-0"></span>A.3 Data Configuration

For the Mathematical Geometry subset of COMT, we randomly sample 200 examples for evaluation and use the remaining 820 for fine-tuning. For each other benchmark, we fine-tune on 1,000 samples and use another 2,000 non-overlapping samples for reinforcement learning. Blink-Jigsaw and SAT samples are drawn at random from their official training splits. VSP provides no training set, so we follow its data generation recipe and synthesize our own data. We ensure that each map contains a valid path, no more than 20% trap blocks, and grid sizes 3–6 are produced in a 1:2:3:4 ratio (100, 200, 300, and 400 examples, respectively, for fine-tuning). Additionally, for each sample in VSP, we generate three distinct reasoning trajectories to encourage diversity. Full dataset statistics are provided in Tab. [8.](#page-18-1)

Table 8: Dataset Statistics

<span id="page-18-1"></span>

| Task                  | # SFT | # RL  | # Test |
|-----------------------|-------|-------|--------|
| VSP Spatial Reasoning | 3,000 | 2,000 | 400    |
| VSP Spatial Planning  | 3,000 | 2,000 | 400    |
| Blink Jigsaw          | 1,000 | 2,000 | 150    |
| SAT                   | 1,000 | 2,000 | 500    |
| COMT                  | 820   | -     | 200    |

