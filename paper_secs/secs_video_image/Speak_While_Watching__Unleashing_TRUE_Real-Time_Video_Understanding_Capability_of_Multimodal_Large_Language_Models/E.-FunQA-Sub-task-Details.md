# E. FunQA Sub-task Details

The FunQA dataset contains 12 sub-tasks covering diverse video understanding and reasoning capabilities. In this work, we focus on the six Description & Reasoning tasks: Humor (H2, H3), Creative (C2, C3), and Magic (M2, M3). In the main paper, we report the average performance across these six sub-tasks to provide a concise and unified summary of the model's overall behavior. In this appendix, we further present the detailed per-task results for all six Description & Reasoning tasks. All tables in this section follow the same experimental settings as in the main paper (identical wait-K configuration, sampling strategy, and evaluation protocol). The complete results are provided in Tables [4–](#page-13-0)[9.](#page-14-0)

Table 4. FunQA M2 task performance on Qwen2.5-VL-3B.

<span id="page-13-0"></span>

| Category  | Method     | CIDEr | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | BLEURT |
|-----------|------------|-------|--------|--------|--------|---------|--------|
| Offline   | Origin     | 11.48 | 40.54  | 5.93   | 25.23  | 25.70   | 47.33  |
|           | GDPE       | 11.75 | 41.30  | 6.46   | 25.54  | 25.93   | 47.47  |
| Streaming | Interleave | 7.48  | 41.62  | 4.04   | 20.06  | 24.03   | 40.59  |
|           | OSPE       | 1.96  | 31.06  | 3.79   | 26.55  | 21.61   | 41.16  |
|           | GDPE       | 5.77  | 34.66  | 3.19   | 21.92  | 22.12   | 45.47  |
|           | GIPE       | 4.48  | 35.02  | 4.62   | 24.38  | 22.48   | 42.04  |

Table 5. FunQA M3 task performance on Qwen2.5-VL-3B.

| Category  | Method     | CIDEr | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | BLEURT |
|-----------|------------|-------|--------|--------|--------|---------|--------|
| Offline   | Origin     | 11.61 | 37.26  | 7.02   | 22.95  | 24.15   | 41.69  |
|           | GDPE       | 6.69  | 31.46  | 2.75   | 19.61  | 20.06   | 41.85  |
| Streaming | Interleave | 4.01  | 24.94  | 1.67   | 15.97  | 16.89   | 33.66  |
|           | OSPE       | 0.27  | 16.85  | 1.07   | 19.55  | 13.19   | 35.82  |
|           | GDPE       | 2.86  | 21.16  | 0.82   | 17.28  | 15.18   | 40.43  |
|           | GIPE       | 2.47  | 23.22  | 1.85   | 19.74  | 16.24   | 34.04  |

Table 6. FunQA H2 task performance on Qwen2.5-VL-3B.

| Category  | Method     | CIDEr | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | BLEURT |
|-----------|------------|-------|--------|--------|--------|---------|--------|
| Offline   | Origin     | 10.26 | 38.40  | 4.80   | 21.07  | 22.91   | 39.12  |
|           | GDPE       | 13.04 | 40.29  | 5.68   | 20.95  | 23.41   | 39.66  |
| Streaming | Interleave | 4.18  | 17.30  | 1.62   | 9.04   | 13.12   | 26.68  |
|           | OSPE       | 3.60  | 30.40  | 3.10   | 22.62  | 19.90   | 36.21  |
|           | GDPE       | 8.81  | 37.61  | 4.71   | 20.74  | 22.60   | 39.77  |
|           | GIPE       | 6.94  | 35.09  | 3.82   | 20.81  | 21.59   | 39.22  |

Table 7. FunQA H3 task performance on Qwen2.5-VL-3B.

| Category  | Method     | CIDEr | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | BLEURT |
|-----------|------------|-------|--------|--------|--------|---------|--------|
| Offline   | Origin     | 4.71  | 36.52  | 4.08   | 17.45  | 20.55   | 39.80  |
|           | GDPE       | 5.03  | 36.39  | 3.69   | 16.55  | 20.54   | 41.64  |
| Streaming | Interleave | 2.09  | 14.00  | 0.76   | 8.11   | 12.33   | 24.88  |
|           | OSPE       | 1.94  | 27.15  | 1.66   | 19.70  | 16.85   | 38.02  |
|           | GDPE       | 3.63  | 32.22  | 1.77   | 15.85  | 17.09   | 41.57  |
|           | GIPE       | 3.42  | 31.55  | 2.26   | 17.97  | 18.59   | 35.47  |

Table 8. FunQA C2 task performance on Qwen2.5-VL-3B.

| Category  | Method     | CIDEr | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | BLEURT |
|-----------|------------|-------|--------|--------|--------|---------|--------|
| Offline   | Origin     | 2.14  | 22.16  | 2.24   | 15.17  | 20.68   | 34.97  |
|           | GDPE       | 3.38  | 28.65  | 5.91   | 18.13  | 24.56   | 36.17  |
| Streaming | Interleave | 0.21  | 12.57  | 0.46   | 11.46  | 18.70   | 28.41  |
|           | OSPE       | 8.95  | 45.88  | 8.61   | 22.71  | 22.94   | 33.01  |
|           | GDPE       | 0.14  | 28.59  | 2.76   | 15.56  | 20.90   | 35.34  |
|           | GIPE       | 5.74  | 36.92  | 5.19   | 20.25  | 20.88   | 33.58  |

Table 9. FunQA C3 task performance on Qwen2.5-VL-3B.

<span id="page-14-0"></span>

| Category  | Method     | CIDEr | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | BLEURT |
|-----------|------------|-------|--------|--------|--------|---------|--------|
| Offline   | Origin     | 1.66  | 31.98  | 4.39   | 16.17  | 20.60   | 39.05  |
|           | GDPE       | 3.58  | 33.31  | 6.28   | 16.23  | 23.40   | 39.62  |
| Streaming | Interleave | 0.02  | 21.29  | 1.70   | 13.56  | 22.95   | 30.99  |
|           | OSPE       | 2.62  | 36.60  | 1.70   | 19.80  | 19.31   | 34.82  |
|           | GDPE       | 2.77  | 31.51  | 1.64   | 14.18  | 18.10   | 37.65  |
|           | GIPE       | 2.28  | 38.60  | 4.34   | 18.24  | 17.87   | 35.02  |