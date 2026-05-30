# B OPEN-SOURCE MOE MODELS

Table 5: Open-Source MoE Models List (Released after Jan. 2024).

| Name                      | Active<br>Parameters | Total<br>Parameters | # Experts     | Routing<br>Policy | Initialized<br>Method | MMLU* |
|---------------------------|----------------------|---------------------|---------------|-------------------|-----------------------|-------|
| OLMoE                     | 1B                   | 7B                  | 64            | top-8             | train from scratch    | 54.1  |
| MiniCPM-MoE-8x2B          | 4B                   | 13.6B               | 8             | top-2             | upcycling             | 58.9  |
| Qwen1.5-MoE-A2.7B         | 2.7B                 | 14.3B               | 4(shared)+60  | 4+top-4           | upcycling             | 62.5  |
| Deepseek-V2-Lite          | 2.4B                 | 16B                 | 2(shared)+64  | 2+top-6           | train from scratch    | 58.3  |
| Yuan2.0-M32               | 32 3.7B 40B          |                     | 32            | top-2             | train from scratch    | 72.2  |
| GRIN-MoE                  | 6.6B                 | 41.9B               | 16            | top-2             | upcycling             | 79.4  |
| Mixtral-8x7B              | 12.5B                | 47B                 | 8             | top-2             | upcycling             | 70.4  |
| Jamba                     | 12B                  | 52B                 | 16            | top-2             | unknown               | 67.4  |
| Qwen2-57B-A14B            | 14B 57.4B            |                     | 8(shared)+64  | 8+top-8           | upcycling             | 76.5  |
| DBRX                      | 36B                  | 132B                | 16            | top-4             | unknown               | 73.7  |
| Mixtral-8x22B             | 8x22B 39B 141B       |                     | 8             | top-2             | upcycling             | 77.8  |
| Skywork-MoE               | MoE 22B 146B         |                     | 16            | top-2             | upcycling             | 77.4  |
| Deepseek-V2               | 21B                  | 236B                | 2(shared)+160 | 2+top-6           | train from scratch    | 78.5  |
| grok-1                    | 80B                  | 314B                | 8             | top-2             | unknown               | 73.0  |
| Snowflake Arctic 17B 480B |                      | 480B                | 128           | top-2             | unknown               | 67.3  |

<sup>\*</sup>Note: This table presents a subset of open-source MoE models and is not exhaustive. The list is sorted by total parameters. MMLU scores are extracted from original papers or reports and may not reflect model real performance.