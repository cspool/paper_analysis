# <span id="page-11-1"></span>D Evaluation with LLM-as-a-judge

We used the "LLM-as-a-judge" paradigm [\[32\]](#page-7-4) to measure the answer correctness of the generated answer with regards to the ground truth answer. In all experiments, we use the judge from the Databricks Agent Evaluation framework.[9](#page-11-2) The judge has been calibrated with human preferences on representative datasets FinanceBench, Databricks DocsQA and the judge reported 88.1 ± 5.5% agreement and Cohen's kappa scores of 0.64 ± 0.13, showcasing a strong agreement with human labelers.

