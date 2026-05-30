# E.2 EFFECT OF MODALITY PRE-FUSION

Table 11: Performance of LLaVA-Mini when using only pre-fusion module without compression.

|                              |                | Performance |      |      |  |
|------------------------------|----------------|-------------|------|------|--|
| Methods                      | #Vision Tokens | VQAv2       | GQA  | MMB  |  |
| LLaVA-v1.5                   | 576            | 78.5        | 62.0 | 64.3 |  |
| LLaVA-Mini (w/o compression) | 576            | 80.0        | 62.9 | 66.2 |  |

To validate the effect of the pre-fusion module, we remove the compression module and retained only the modality pre-fusion module, thereby comparing with LLaVA-v1.5 while both using 576 vision tokens. As shown in Table, when using only the pre-fusion module without compression, LLaVA-Mini achieves superior performance compared to LLaVA-v1.5 with both using 576 vision tokens, demonstrating the effectiveness of the pre-fusion module.

