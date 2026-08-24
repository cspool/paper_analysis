# G Cost Considerations for Long Context RAG

In the following table, we list the various costs per input tokens for some of the API based models. Cost values are as of October 2024.

We choose to show the input token cost for a single query with a max sequence length of 8k, 64k, 128k, and 2 million tokens. We also show the estimated input token costs for "full benchmarking" across all three datasets in this study, which have a total of 823 queries.

Cost A is for 823 queries at maximum sequence length of 128k tokens. Cost B is for 823 queries at maximum sequence length of 2 million tokens.

| Model             | \$/M tokens | 8k     | 64k    | 128k   | 2M  | Cost A  | Cost B |
|-------------------|-------------|--------|--------|--------|-----|---------|--------|
| GPT4o             | 2.5         | 0.02   | -      | 0.32   | -   | 263.36  | -      |
| GPT4o-mini        | 0.15        | 0.0012 | 0.0096 | 0.0192 | -   | 15.8016 | -      |
| o1-preview        | 15          | 0.12   | 0.96   | 1.92   | -   | 1580.16 | -      |
| Claude 3.5 Sonnet | 3           | 0.024  | 0.192  | 0.384  | -   | 316.032 | -      |
| Claude 3 Opus     | 15          | 0.12   | 0.96   | 1.92   | -   | 1580.16 | -      |
| Claude 3.5 Haiku  | 0.25        | 0.002  | 0.016  | 0.032  | -   | 26.336  | -      |
| Gemini 1.5 Pro    | 1.25        | 0.01   | 0.08   | 0.16   | 5   | 131.68  | 4115   |
| Gemini 1.5 Flash  | 0.075       | 0.0006 | 0.0048 | 0.0096 | 0.3 | 7.9008  | 246.9  |

Table S13: Select input token cost estimates. All numbers are in \$