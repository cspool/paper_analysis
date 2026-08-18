# <span id="page-4-0"></span>1) Single Expert Activation Imbalance: (Ob4)

We examine expert selection frequency at each layer, presenting results for layer 7 of Llama4 in Figure 8. We observe

pronounced skewness where a subset of experts is activated over 16 times more frequently than average. This workload imbalance suggests system designs should duplicate or decentralize frequently used experts.

To investigate selection patterns across different tasks, we analyze all 57 MMLU subjects spanning diverse fields, including biology, history, and math, etc [\[35\]](#page-14-22). [Figure 8\(](#page-4-2)b) shows the top 10 most popular experts for each subject. Horizontal bright lines indicate certain experts are consistently activated regardless of subject, while remaining popular experts vary significantly between subjects, demonstrating both overlap and distinction in task-based expert selection.

We further examine task impact using the Chinese version of MMLU in MMLU Pro [\[36\]](#page-14-23) with identical questions but different languages. [Figure 8\(](#page-4-2)c) reveals distinctly different patterns: although 5-6 experts remain popular across subjects, only two overlap with English MMLU's most frequently selected experts. This confirms that task characteristics, including language, significantly influence expert selection, enabling task-aware serving systems that optimize expert distribution to balance workloads and reduce data movement.

