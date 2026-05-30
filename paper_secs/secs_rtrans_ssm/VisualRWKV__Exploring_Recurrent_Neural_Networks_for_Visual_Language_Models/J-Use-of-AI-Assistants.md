# J Use of AI Assistants

In this research, an AI writing assistant is solely employed for the purposes of paraphrasing, spell-checking, and enhancing the author's original content, and it does not introduce any novel content.

<span id="page-18-0"></span>

| Method            | Size | Image Tokens | ScienceQA | TextVQA | GQA   |
|-------------------|------|--------------|-----------|---------|-------|
|                   |      | 577          | 65.2      | 50.25   | 50.5  |
|                   |      | 145          | 64.90     | 46.38   | 47.47 |
|                   |      | 65           | 64.40     | 44.58   | 45.09 |
|                   | 7B   | 37           | 64.11     | 44.01   | 44.78 |
| VisualRWKV-Base   |      | 17           | 63.86     | 43.61   | 44.57 |
|                   |      | 10           | 63.26     | 43.27   | 44.37 |
|                   |      | 5            | 62.87     | 43.03   | 44.08 |
|                   |      | 1            | 60.34     | 41.72   | 36.09 |
|                   |      | 577          | 67.38     | 50.97   | 49.96 |
|                   |      | 145          | 66.83     | 47.13   | 46.20 |
|                   | 7B   | 65           | 65.44     | 45.63   | 45.03 |
|                   |      | 37           | 65.39     | 45.47   | 44.81 |
| VisualRWKV-Hybrid |      | 17           | 64.40     | 45.07   | 44.65 |
|                   |      | 10           | 64.06     | 44.79   | 44.44 |
|                   |      | 5            | 63.26     | 44.75   | 43.98 |
|                   |      | 1            | 63.11     | 44.71   | 43.76 |

Table 14: Results of VisualRWKV Hybrid model on 3 benchmarks. The prompting method used here is the sandwich prompt.