# <span id="page-8-2"></span>B. Additional evaluations

<span id="page-8-4"></span>

| Model                | Size | Frames | S    | M    | L    | Overall |
|----------------------|------|--------|------|------|------|---------|
| Video-LLaVA [38]     | 7B   | 8      | 46.1 | 40.7 | 38.1 | 41.6    |
| ShareGPT4Video [7]   | 8B   | 16     | 53.6 | 39.3 | 37.9 | 43.6    |
| Chat-Univi-v1.5 [27] | 7B   | 64     | 51.2 | 44.6 | 41.8 | 45.9    |
| VideoLLaMA2 [10]     | 7B   | 16     | 59.4 | 47.6 | 43.8 | 50.3    |
| VideoChat2 [34]      | 7B   | 16     | 52.8 | 39.4 | 39.2 | 43.8    |
| LongVA [80]          | 7B   | 128    | 61.6 | 50.4 | 47.6 | 54.3    |
| LLaVA-OneVision [30] | 7B   | 32     | 69.1 | 53.3 | 46.7 | 58.2    |
| LongVU [51]          | 7B   | 1fps   | 64.7 | 58.2 | 59.5 | 60.9    |
| TDC (Ours)           | 7B   | 1fps   | 70.0 | 66.2 | 61.3 | 65.9    |

Table 5. Detailed Results on VideoMME. The best results are bold. Subtitles of videos are provided in this evaluation. S: Short. M: Medium. L: Long.

In Table [5,](#page-8-4) we provide a more detailed comparison on the VideoMME [\[16\]](#page-9-5) dataset. In this evaluation, subtitles for each video are provided to the model. The results show that our model consistently achieves the best performance across both short and long video settings, which demonstrates its adaptability to a wide range of video scenarios.

