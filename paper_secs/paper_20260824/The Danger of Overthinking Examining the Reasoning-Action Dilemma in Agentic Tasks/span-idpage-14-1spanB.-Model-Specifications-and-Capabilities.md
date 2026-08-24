# <span id="page-14-1"></span>**B.** Model Specifications and Capabilities

| Category  | Model                              | Params | Context | FC           | Notes                     |  |
|-----------|------------------------------------|--------|---------|--------------|---------------------------|--|
| Non-Reaso | Non-Reasoning Models (Open Source) |        |         |              |                           |  |
|           | DeepSeek-V3                        | 671B   | 128k    | ×            | MoE architecture          |  |
|           | Qwen 2.5-32B                       | 32B    | 128k    | ×            | Dense architecture        |  |
|           | Qwen 2.5-14B                       | 14B    | 128k    | ×            | Dense architecture        |  |
|           | Qwen 2.5-7B                        | 7B     | 128k    | ×            | Dense architecture        |  |
|           | Qwen 2.5-1.5B                      | 1.5B   | 128k    | ×            | Dense architecture        |  |
|           | Sky-T1-32B                         | 32B    | 32k     | ×            | QwQ distillation          |  |
| Non-Reaso | ning Models (Closed Sour           | ce)    |         |              |                           |  |
|           | GPT-40                             | _      | 128k    | <b>√</b>     | Aug 2024 version          |  |
|           | GPT-4o-mini                        | -      | 128k    | $\checkmark$ | Jul 2024 version          |  |
|           | Claude 3.5 Sonnet                  | -      | 200k    | $\checkmark$ | Oct 2024 version          |  |
| Reasoning | Models (Open Source)               |        |         |              |                           |  |
|           | QwQ-32B                            | 32B    | 32k     | ×            | Preview version           |  |
|           | DeepSeek-R1                        | 671B   | 128k    | ×            | Based on V3               |  |
|           | R1-Distill-Qwen-32B                | 32B    | 128k    | ×            | Based on Qwen 2.5         |  |
|           | R1-Distill-Qwen-14B                | 14B    | 128k    | ×            | Based on Qwen 2.5         |  |
|           | R1-Distill-Qwen-7B                 | 7B     | 128k    | ×            | Based on Qwen 2.5         |  |
|           | R1-Distill-Qwen-1.5B               | 1.5B   | 128k    | ×            | Based on Qwen 2.5         |  |
| Reasoning | Reasoning Models (Closed Source)   |        |         |              |                           |  |
|           | o1                                 | -      | 200k    | <b>√</b>     | Dec 2024, RE <sup>‡</sup> |  |
|           | o1-mini                            | -      | 128k    | ×            | Sep 2024 version          |  |

*Table 5.* Comprehensive comparison of evaluated models. FC indicates native function calling support. Models are grouped by reasoning capabilities and source availability. †Supports reasoning\_effort parameter (low/medium/high).

