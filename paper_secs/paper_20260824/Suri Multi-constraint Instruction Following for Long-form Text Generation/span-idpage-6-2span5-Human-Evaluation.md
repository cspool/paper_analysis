# <span id="page-6-2"></span>5 Human Evaluation

While our automatic assessments provide insights into the lexical information of the text, they do not capture its semantic content. Therefore, we conduct a human evaluation to determine if and how the constraints are satisfied by the outputs of Suri-SFT and Suri-I-ORPO. Human evaluation on 30 test set generations reveals that while both finetuned models satisfy constraints, Suri-I-ORPO is preferred by humans for its ability to seamlessly incorporate the constraints into the final outputs.

<span id="page-6-1"></span><sup>12</sup>Experiment done using OpenAI API for GPT-4o and Vertex API for Claude-3.5-Sonnet and Gemini-1.5-Pro. Temperature is set to 0.0 and the maximum number of generated tokens is set to 4096 for all models.

<span id="page-7-0"></span>

| Category                               | GPT-4o | Claude-3.5-Sonnet | Gemini-1.5-Pro |
|----------------------------------------|--------|-------------------|----------------|
| Agreement with human's majority vote   | 39%    | 24%               | 13%            |
| Partial satisfaction - No satisfaction | 23%    | 6%                | 2%             |
| Satisfaction - Partial satisfaction    | 22%    | 63%               | 1%             |
| Satisfaction - No satisfaction         | 16%    | 7%                | 45%            |

Table 4: Types of agreement and disagreement between GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-Pro, and human judges on 30 generations from Suri-SFT.

<span id="page-7-1"></span>

|                     | Suri-SFT | Suri-I-ORPO |
|---------------------|----------|-------------|
| Satisfied           | 67%      | 68%         |
| Partially Satisfied | 17%      | 16%         |
| Not Satisfied       | 16%      | 16%         |

Table 5: Average percentage of satisfied constraints in Suri-SFT and Suri-I-ORPO generations. Percentages are rounded to the nearest whole number.

