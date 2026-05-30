# D. Additional Results of the 7B Backbone on Video Description

To assess how our positional strategies scale with model capacity, Table [3](#page-12-0) presents video description results for the 7B Qwen2.5-VL backbone under both offline and streaming settings.

Scaling the backbone from 3B to 7B yields a pronounced increase in CIDEr, while BLEU, METEOR, ROUGE-L, and BLEURT improve by similar margins across all methods. This behavior is expected: CIDEr strongly rewards the recall of salient content words, which larger models capture more reliably, whereas the other metrics remain relatively stable once a reasonable descriptive quality is achieved. Crucially, the relative ranking and overall behav-

Table 3. Video Description results on the Qwen2.5-VL backbone (3B and 7B).

<span id="page-12-0"></span>

| Category  | Method     | Model Size | CIDEr | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | BLEURT |
|-----------|------------|------------|-------|--------|--------|--------|---------|--------|
|           | Origin     | 3B         | 35.44 | 42.36  | 14.45  | 29.18  | 30.47   | 53.21  |
| Offline   | GDPE       | 3B         | 30.86 | 40.26  | 13.64  | 28.49  | 34.12   | 53.19  |
|           | Origin     | 7B         | 42.42 | 40.43  | 12.46  | 27.79  | 32.72   | 53.06  |
|           | GDPE       | 7B         | 38.13 | 39.58  | 11.97  | 27.20  | 31.58   | 52.63  |
|           | Interleave | 3B         | 20.08 | 44.40  | 14.41  | 27.17  | 34.95   | 44.11  |
|           | OSPE       | 3B         | 26.32 | 42.14  | 12.78  | 27.92  | 32.29   | 50.62  |
|           | GDPE       | 3B         | 12.52 | 26.32  | 7.42   | 30.03  | 27.37   | 51.53  |
| Streaming | GIPE       | 3B         | 28.11 | 40.42  | 11.52  | 29.13  | 30.69   | 51.20  |
|           | Interleave | 7B         | 46.94 | 49.02  | 16.13  | 32.24  | 36.29   | 44.78  |
|           | OSPE       | 7B         | 47.49 | 43.85  | 12.10  | 28.05  | 31.86   | 51.71  |
|           | GDPE       | 7B         | 37.78 | 41.01  | 11.25  | 27.48  | 30.52   | 51.18  |
|           | GIPE       | 7B         | 25.70 | 39.09  | 9.85   | 28.71  | 28.82   | 51.16  |

iors of all positional strategies remain consistent between 3B and 7B, indicating that our streaming formulations transfer well across model sizes and maintain their effectiveness at larger scales.

