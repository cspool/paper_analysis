# A EXPERIMENT SETUP DETAILS

<span id="page-12-0"></span>Table 1: Experiment setup for model tuning across different task categories. For each category, we specify the tuning configuration and model-specific hyperparameters. N denotes the number of RoE samples, T is the maximum temperature in tuning, and L refers to the number of initial and final skipped layers. Sample denotes the number of validation set samples, Trial is the number of optimization trials. PPL denotes if perplexity was used as the optimization objective.

| Category | Task                                               | Sample/Trial | PPL | OLMoE |     |   | Mixtral |      |   | GPT-OSS |     |   |
|----------|----------------------------------------------------|--------------|-----|-------|-----|---|---------|------|---|---------|-----|---|
|          |                                                    |              |     | N     | T   | L | N       | T    | L | N       | T   | L |
| Math     | GSM8K<br>AddSub<br>SVAMP<br>MultiArith<br>SingleEq | 100/50       | ✓   | 32    | 0.5 | 1 | 64      | 0.25 | 5 | 64      | 0.2 | 5 |
| Common   | SiQA<br>OBQA<br>Hellaswag<br>ARC-E<br>ARC-C        | 300/100      | ✗   | 32    | 0.5 | 3 | 64      | 0.3  | 3 | 64      | 0.2 | 5 |
| Code     | Humaneval<br>Humaneval+)                           | 50/50        | ✗   | 32    | 0.5 | 1 | 64      | 0.25 | 5 | 64      | 0.2 | 5 |

