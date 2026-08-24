# <span id="page-17-1"></span>C Evaluation Metrics

Our evaluation framework employs two categories of metrics to comprehensively assess different aspects of our method:

#### • Information Sufficiency Classification metrics:

- F1 Score: Measures the overall balance between precision and recall in detecting sufficient context. This metric is particularly important as it penalizes both false positives (stopping too early) and false negatives (processing unnecessary context). A high F1 score indicates that our method can reliably identify when enough information has been processed while avoiding premature cutoffs.
- Recall at 90% Precision (R@90P): Ensures high confidence in sufficiency predictions while maintaining good coverage. This metric is crucial for our task as it measures how many truly sufficient contexts we can identify while keeping false positives (incorrect early cutoffs) below 10%. This conservative approach helps prevent information loss while still achieving efficiency gains.

### • QA Task Performance metrics:

- Accuracy: Measures answer correctness before and after context cutoff. This metric is calculated as the percentage of questions answered correctly by comparing model outputs with ground truth answers. We use GPT-4o Mini as an automated judge to evaluate answer correctness, following established practices in QA evaluation [\[33\]](#page-11-13). This approach is more reliable than exact string matching, especially for long-form answers where semantic equivalence is more important than lexical matching.
- Token Reduction: Quantifies the proportion of tokens processed relative to full context. This metric directly measures computational efficiency gains, calculated as the ratio between the number of tokens processed with our method versus processing the full context. A higher token reduction indicates greater computational savings while improving performance.

