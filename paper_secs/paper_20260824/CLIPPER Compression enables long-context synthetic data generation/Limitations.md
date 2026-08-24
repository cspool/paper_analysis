# **Limitations**

We only perform hyperparameter tuning on ProLong-Base due to the high cost of the training process. To put things into perspective, training a model on our full test set requires approximately 50 hours on 8 A100 GPUs, each costing \$2 per hour to rent. Even training on our tuning subset takes 6 hours. Therefore, extending training further is prohibitively expensive.

Similarly, we do not hire human annotators to write claims for our dataset due to the prohibitive cost and the need for numerous annotators who have thoroughly read the books [\(Table 8\)](#page-17-1). While this decision may result in less complex claims, our approach offers greater adaptability to new books while significantly reducing costs.

The compression stage can be challenging to fine-tune, subject to model biases, and prone to potential hallucinations. Due to the large volume of data, verifying its accuracy is also difficult, whether via prompting or human annotation. However, we note that prior research has demonstrated that LLMs are capable of producing high-quality summaries of long documents [Chang et al.](#page-10-3) [\(2024\)](#page-10-3); [Kim et al.](#page-11-2) [\(2024\)](#page-11-2). In addition, these compressed representations could still provide a strong foundation for claim generations, as most of CLIPPER's claims are grounded in the original book [\(subsection 2.3\)](#page-3-0).

