# A.2 Metrics

### A.2.1 Compression Rate

We provide more details on the compression rate in the main table, where the compression rate is defined as:

<span id="page-17-2"></span>C.R. = Compression Rate = 
$$max(\frac{\text{\#tokens}_{\text{original}} - \text{\#tokens}_{\text{current}}}{\text{\#tokens}_{\text{original}}}, 0)$$
 (9)

$$A.C.R. = \frac{1}{N_{\text{benchmark}}} \sum_{i=0}^{N_{\text{benchmark}}} C.R.$$
 (10)

### A.2.2 Normalized Metric

<span id="page-17-1"></span>We report two normalized metrics to facilitate fair comparisons: Normalized Accuracy and Normalized Token Length. They are defined as follows:

<span id="page-17-0"></span>Normalized Accuracy = 
$$\frac{\text{#Acc}_{\text{current}}}{\text{#Acc}_{\text{original}}}$$
(11)

Normalized Token = 
$$\frac{\text{#Token}_{\text{current}}}{\text{#Token}_{\text{original}}}$$
(12)

### A.3 Data Construction Detail

For long CoT, we use the prompt from dataset s1.1 [\[37\]](#page-14-2). Each sample is generated 8 times using the original model. For short CoT, to avoid inconsistencies in the system prompt format, we adopt the short CoT construction method from AdaR1. We annotate 10 randomly selected questions from GSM8K using the instruct model, then fine-tune the long CoT model to overfit on them. For the GSM8K training set, we sample and retain only the examples with correct answers.

