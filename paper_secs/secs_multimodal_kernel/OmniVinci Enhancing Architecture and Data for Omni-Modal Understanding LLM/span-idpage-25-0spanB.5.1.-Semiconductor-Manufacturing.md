# <span id="page-25-0"></span>*B.5.1. Semiconductor Manufacturing*

Wafer maps are essential in semiconductor manufacturing for visualizing defect distributions, enabling yield monitoring, process drift detection, and preliminary root cause identification. It is a domain with a significant gap from multimodal LLM. To study whether we can leverage our omni-modal OmniVinci on this task, we fine-tune OmniVinci on wafer map data, aligning visual and textual features for robust defect analysis, as illustrated in Figure [12.](#page-26-2) On the WM-811K dataset [\[102\]](#page-18-12), OmniVinci achieves superior performance over VILA [\[64\]](#page-16-0) and NVILA [\[65,](#page-16-15) [69\]](#page-16-9) (which has been trained for wafer defect classification), and our model demonstrates further improvements, as summarized in Table [15.](#page-25-3) Beyond classification, this framework can be extended to support interactive querying and automated reasoning for Root Cause Analysis, systematically linking defect clusters to process tools, wafer locations, or temporal drifts.

<span id="page-25-3"></span>Table 15 | Comparison of VILA, NVILA, and OmniVinci on wafer defect classification.

|            | VILA [64] | NVILA [69] | OmniVinci (ours) |
|------------|-----------|------------|------------------|
| Parameters | 40B       | 8B         | 9B               |
| Resolution | 336×336   | 448×448    | 448×448          |
| Model size | 75 GB     | 16 GB      | 18 GB            |
| Accuracy   | 90.8%     | 97.6%      | 98.1%            |

