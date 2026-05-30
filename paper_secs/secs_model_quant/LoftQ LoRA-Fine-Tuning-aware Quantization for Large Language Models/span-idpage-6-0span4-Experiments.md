# <span id="page-6-0"></span>4 Experiments

We evaluate our method on NLU and NLG tasks. We apply LoftQ for quantizing DeBERTaV3-base [\(He et al.,](#page-14-1) [2021b\)](#page-14-1), BART-large [\(Lewis et al.,](#page-14-2) [2019\)](#page-14-2), and LLAMA-2 series [\(Touvron et al.,](#page-16-0) [2023\)](#page-16-0). Implementation Details. Following the prior works of LoRA variants [\(Zhang et al.,](#page-16-5) [2023;](#page-16-5) [He](#page-14-7) [et al.,](#page-14-7) [2021a\)](#page-14-7), we freeze all the backbone weight matrices and add low-rank adapters to weight matrices in MHA and FFN of all layers. We quantize the weight matrices that are attached by low-rank adapters. All the quantized models and adapters used in this paper are available on <https://huggingface.co/LoftQ>. Our implementation is based on publicly available *Huggingface Transformers* code-base [\(Paszke et al.,](#page-15-4) [2019\)](#page-15-4). All the experiments are conducted on NVIDIA A100 GPUs.

Quantization Methods. We apply two quantization methods to demonstrate LoftQ is compatible with different quantization functions:

- *Uniform quantization* is a classic quantization method. It uniformly divides a continuous interval into 2*<sup>N</sup>* categories and stores a local maximum absolute value for dequantization.
- *NF4* and its 2-bit variant *NF2* are quantization methods used in QLoRA [\(Dettmers et al.,](#page-14-0) [2023\)](#page-14-0). They assume that the high-precision values are drawn from a Gaussian distribution and map these values to discrete slots that have equal probability.

We perform 2-bit and 4-bit quantization on all models, achieving compression ratios of 25-30% and 15-20% at the 4-bit and 2-bit levels, respectively. The compression ratios and trainable parameter ratios for all models are detailed in the Appendix [A.](#page-17-0)

Baselines. We compare LoftQ with the following baseline methods:

- *Full fine-tuning* is the most common approach for adapting a pre-trained model to downstream tasks. The model is initialized with pre-trained weights and all parameters are updated through an SGD-type optimization method.
- *Full precision LoRA (LoRA)* is a lightweight method for task adaptation, where it stores the backbone using 16-bit numbers and optimizes the low-rank adaptors only. The adaptors are applied to the same matrices as in LoftQ.
- *QLoRA* is similar to *LoRA* except the backbone is quantized into low-bit regime. The low-rank adapters are initialized using [\(5\)](#page-4-1) and are applied to the same matrices as in LoftQ.

### 4.1 Encoder-only Model: DeBERTaV3

Models and Datasets. We quantize the DeBERTaV3-base [\(He et al.,](#page-14-1) [2021b\)](#page-14-1) with LoftQ, then finetune and evaluate the model on the General Language Understanding Evaluation (GLUE) benchmark [\(Wang et al.,](#page-16-3) [2019\)](#page-16-3), SQuADv1.1 [\(Rajpurkar et al.,](#page-15-2) [2016\)](#page-15-2), and ANLI [\(Nie et al.,](#page-15-5) [2019\)](#page-15-5). The specific tasks of GLUE are given in Appendix [C.](#page-18-0) Following previous works [\(Zhang et al.,](#page-16-5) [2023\)](#page-16-5), we exclude WNLI in the experiments.

Implementation Details. We select the learning rates from {1 × 10−<sup>5</sup> *,*5 × 10−<sup>5</sup> *,*1 × 10−<sup>4</sup> 5 × 10−<sup>4</sup> }. We quantize the entire backbone. Given that GLUE, SQuADv1.1, and ANLI are relatively easy NLU tasks, we also quantize the embedding layer for higher compression efficiency. We apply the NormalFloat and the uniform quantization for LoftQ and QLoRA at both 2-bit and 4-bit levels. We use rank 16 and 32 for low-rank adapters. More implementation details, such as the training epochs and batch sizes, are presented in Appendix [D.2.](#page-19-0)

Main Results. Table [1](#page-8-0) and Table [2](#page-8-1) summarize the results for 2-bit quantization on the GLUE, SQuADv1.1, and ANLI datasets, by NF2 and the uniform quantization, respectively. Our method consistently outperforms QLoRA on all settings with respect to different ranks, quantization methods, and datasets. When using the uniform quantization (Table [2\)](#page-8-1), our method achieves 88.0% accuracy on MNLI-m, surpassing the QLoRA baseline by 8%. For tasks like SST and SQuADv1.1, our method even approaches the full fine-tuning performance at 2-bit level. The 4-bit quantization experiment results are presented in Appendix [D.1](#page-19-1) as both LoftQ and QLoRA achieve performance close to full fine-tuning.

Our method is also more stable compared to QLoRA in the low-bit regime. For instance, while QLoRA fails to converge on CoLA for both quantization methods and ranks, LoftQ converges in all cases and achieves a score of 60.5 using uniform quantization at rank 32. LoftQ stands out in its ability to consistently attain robust and improved performance by effectively preserving the starting point of pre-trained weights.

<span id="page-8-0"></span>Table 1: Results with 2-bit LoftQ of DeBERTaV3-base models on GLUE development set, SQuADv1.1 development set, ANLI test set using NF2 quantization. We report the median over four seeds. *N.A.* indicates the model does not converge. The best results on each dataset are shown in bold.

| Rank | Method         | MNLI<br>m / mm         | QNLI<br>Acc  | RTE<br>Acc   | SST<br>Acc   | MRPC<br>Acc            | CoLA<br>Matt | QQP<br>Acc             | STSB<br>P/S Corr       | SQuAD<br>EM/F1           | ANLI<br>Acc  |
|------|----------------|------------------------|--------------|--------------|--------------|------------------------|--------------|------------------------|------------------------|--------------------------|--------------|
| -    | Full FT        | 90.5/90.6              | 94.0         | 82.0         | 95.3         | 89.5/93.3              | 69.2         | 92.4/89.8              | 91.6/91.1              | 88.5/92.8                | 59.8         |
| 16   | LoRA           | 90.4/90.5              | 94.6         | 85.1         | 95.1         | 89.9/93.6              | 69.9         | 92.0/89.4              | 91.7/91.1              | 87.3/93.1                | 60.2         |
| 16   | QLoRA<br>LoftQ | 75.4/75.6<br>84.7/85.1 | 82.4<br>86.6 | 55.9<br>61.4 | 86.5<br>90.2 | 73.8/82.8<br>83.8/88.6 | N.A.<br>37.4 | 86.8/82.3<br>90.3/86.9 | 83.0/82.8<br>87.1/86.9 | 61.5 / 71.2<br>81.5/88.6 | N.A.<br>47.1 |
| 32   | QLoRA<br>LoftQ | 78.5/78.7<br>86.0/86.1 | 80.4<br>89.9 | 56.7<br>61.7 | 86.9<br>92.0 | 73.8/82.7<br>83.6/87.2 | N.A.<br>47.5 | 87.1/82.7<br>91.0/87.9 | 83.6/83.3<br>87.5/87.0 | 64.6/73.8<br>82.9/89.8   | N.A.<br>49.0 |

<span id="page-8-1"></span>Table 2: Results with 2-bit LoftQ of DeBERTaV3-base models on GLUE development set, SQuADv1.1 development set using Uniform quantization . We report the median over four seeds. *N.A.* indicates the model does not converge. The best results on each task are shown in bold.

| Rank | Method         | MNLI<br>m / mm         | QNLI<br>Acc  | RTE<br>Acc   | SST<br>Acc   | MRPC<br>Acc            | CoLA<br>Matt | QQP<br>Acc             | STSB<br>P/S Corr       | SQuAD<br>Em/F1         |
|------|----------------|------------------------|--------------|--------------|--------------|------------------------|--------------|------------------------|------------------------|------------------------|
| -    | Full FT        | 90.5/90.6              | 94.0         | 82.0         | 95.3         | 89.5/93.3              | 69.2         | 92.4/89.8              | 91.6/91.1              | 88.5/92.8              |
| 16   | LoRA           | 90.4/90.5              | 94.6         | 85.1         | 95.1         | 89.9/93.6              | 69.9         | 92.0/89.4              | 91.7/91.1              | 87.3/93.1              |
| 16   | QLoRA<br>LoftQ | 76.5/76.3<br>87.3/87.1 | 83.8<br>90.6 | 56.7<br>61.1 | 86.6<br>94.0 | 75.7/84.7<br>87.0/90.6 | N.A.<br>59.1 | 87.1/82.6<br>90.9/88.0 | 83.5/83.4<br>87.9/87.6 | 69.5/77.6<br>84.4/91.2 |
| 32   | QLoRA<br>LoftQ | 79.9/79.5<br>88.0/88.1 | 83.7<br>92.2 | 57.8<br>63.2 | 86.9<br>94.7 | 76.5/84.5<br>87.5/91.2 | N.A.<br>60.5 | 88.6/84.7<br>91.3/88.3 | 84.1/84.0<br>89.5/89.2 | 71.6/80.2<br>85.2/91.6 |

### 4.2 Encoder-Decoder Model: BART

Models and Datasets. We quantize BART-large model [\(Lewis et al.,](#page-14-8) [2020\)](#page-14-8) with LoftQ, then finetune and evaluate the model on two commonly used summarization datasets: XSum [\(Narayan et al.,](#page-15-1) [2018\)](#page-15-1) and CNN/DailyMail[\(Hermann et al.,](#page-14-6) [2015\)](#page-14-6).

Implementation Details. We apply LoftQ to weight matrices in MHA and FFN of both encoder and decoder layers. We report ROUGE 1/2/L scores, which are the metrics for summarization tasks [\(Lin,](#page-15-6) [2004\)](#page-15-6). We conduct quantization experiments in both 2-bit and 4-bit scenarios. We experiment with both NormalFloat and the uniform quantization in both 2-bit and 4-bit scenarios. In each precision, we choose rank equal to 8 and 16 for a fair comparison with the full precision LoRA baseline [\(Zhang et al.,](#page-16-5) [2023\)](#page-16-5). Please see Appendix [E](#page-20-0) for detailed configurations.

Main Results. Table [3](#page-9-0) summarizes our 4-bit quantization experiment results on the XSum and CNN/DailyMail test sets. Our method consistently outperforms QLoRA at both ranks on both datasets. It even surpasses full precision LoRA at both ranks on Xsum. We will discuss this unexpected results in Section [5.](#page-11-0) The 2-bit quantization results are shown in Table [4.](#page-9-1) Our observation is consistent with the NLU experiments, that LoftQ demonstrates the convergence to reasonable results, while QLoRA does not converge. This indicates our method is robuster by narrowing the initialization gap.

<span id="page-9-0"></span>Table 3: Results with 4-bit LoftQ of BART-large on XSum and CNN/DailyMail. We report ROUGE-1/2/L, the higher the better. *Lead-3* means choosing the first 3 sentences as the summary. *N.A.* indicates the model does not converge. *Full FT* refers to the full fine-tuning where all parameters are tuned. We report the median over five seeds.

| Quantization   | Rank | Method  | XSum              | CNN/DailyMail     |
|----------------|------|---------|-------------------|-------------------|
|                |      | Lead-3  | 16.30/1.60/11.95  | 40.42/17.62/36.67 |
| Full Precision | -    | Full FT | 45.14/22.27/37.25 | 44.16/21.28/40.90 |
|                | 8    | LoRA    | 43.40/20.20/35.20 | 44.72/21.58/41.84 |
|                | 16   | LoRA    | 43.95/20.72/35.68 | 45.03/21.84/42.15 |
|                |      | QLoRA   | 42.91/19.72/34.82 | 43.10/20.22/40.06 |
| NF4            | 8    | LoftQ   | 44.08/20.72/35.89 | 43.81/20.95/40.84 |
|                |      | QLoRA   | 43.29/20.05/35.15 | 43.42/20.62/40.44 |
|                | 16   | LoftQ   | 44.51/21.14/36.18 | 43.96/21.06/40.96 |
|                |      | QLoRA   | 41.84/18.71/33.74 | N.A.              |
| Uniform        | 8    | LoftQ   | 43.86/20.51/35.69 | 43.73/20.91/40.77 |
|                |      | QLoRA   | 42.45/19.36/34.38 | 43.00/20.19/40.02 |
|                | 16   | LoftQ   | 44.29/20.90/36.00 | 43.87/20.99/40.92 |

<span id="page-9-1"></span>Table 4: Results with 2-bit LoftQ of BART-large on XSum and CNN/DailyMail using NF2 quantization. *N.A.* indicates the model does not converge. We report ROUGE-1/2/L, the higher the better. We report the median over five seeds.

| Rank | Method | XSum              | CNN/DailyMail     |  |  |
|------|--------|-------------------|-------------------|--|--|
| 8    | QLoRA  | N.A.              | N.A.              |  |  |
|      | LoftQ  | 39.63/16.65/31.62 | 42.24/19.44/29.04 |  |  |
| 16   | QLoRA  | N.A.              | N.A.              |  |  |
|      | LoftQ  | 40.81/17.85/32.80 | 42.52/19.81/39.51 |  |  |

### 4.3 Decoder-only Model: LLAMA-2

Models and Datasets. We quantize LLAMA-2-7b and LLAMA-2-13b [\(Touvron et al.,](#page-16-0) [2023\)](#page-16-0) with LoftQ. We then fine-tune and evaluate the models on two NLG datasets: GSM8K [\(Cobbe et al.,](#page-13-2) [2021\)](#page-13-2) and WikiText-2 [\(Merity et al.,](#page-15-7) [2016\)](#page-15-7). Please see Appendix [F](#page-21-0) for more details about the datasets.

Implementation Details. Similarly, we apply LoftQ to weight matrices in MHA and FFN of all layers. In WikiText-2 evaluation, we report perplexity. In GSM8K evaluation, we extract numerical answers in the generated solutions and then calculate the accuracy using those numerical answers. We conduct experiments with both NF2 and NF4. Please see Appendix [F](#page-21-0) for detailed configurations. Main Results. Table [5](#page-11-1) presents a summary of our experiments on LLAMA-2-7b and LLAMA-2-13b using 2-bit, 4-bit, and mixed-precision NormalFloat quantization methods on WikiText-2 and GSM8K datasets. In WikiText-2, our method consistently outperforms QLoRA across all quantization precision settings on both models. When dealing with the challenging 2-bit precision, where QLoRA fails to converge, LoftQ manages to achieve a perplexity of 7.85. In GSM8K, our method achieves better or on par performance compared to QLoRA across different model sizes and quantization precision levels. For example, our method achieves 20.9% accuracy using 2-bit precision, where QLoRA doesn't converge.

We find LoftQ outperforms full precision LoRA in GSM8K with LLAMA-2-13b. One possible explanation is that the lack of regularization causes overfitting on full precision LoRA fine-tuning. Therefore, we conduct full precision LoRA with weight decay on GSM8K. From Table [5,](#page-11-1) regularization helps LLAMA-2-13b full precision LoRA fine-tuning, but fails in LLAMA-2-7b. This indicates LLAMA-2-13b is prone to overfitting and quantization has implicit regularization to overcome such overfitting.

To provide a customized trade-off between the performance and precision, we also explore mixed-precision quantization where matrices in the first 4 layers are quantized using 4 bits, and the rest matrices remain 2 bits. We witness a remarkable 5.9% accuracy boost on the GSM8K dataset using LLAMA-2-7b and a 12.7% boost using LLAMA-2-13b. This result underscores the potential of LoftQ for complex mixed-precision quantization scenarios.

### 4.4 Analysis

Effectiveness of Alternating Optimization. We conduct experiments with different alternating step *T* to verify the effectiveness of the alternating optimization and to find the best value *T* as a hyperparameter for different models. Across all tasks and models, we observed that alternating optimization yields substantial improvements even with a minimal alternating step. This suggests that it rapidly narrows the discrepancy between quantized weights and pre-trained weights, making our method easy to apply. For example, our method achieves 88.0% accuracy on MNLI-m dataset using only 5 alternating steps and 21.14 Rouge-2 score using only 1 step. Interestingly, we

<span id="page-11-1"></span>Table 5: Results of LoftQ using NormalFloat for LLAMA-2 series on WikiText-2 and GSM8K. 3/2.5/2.25-bit indicates mixed-precision quantization: 4-bit precision for the first 16/8/4 layers and 2-bit precision for the rest of layers. We report the perplexity (the smaller the better) for WikiText-2 and accuracy for GSM8K. The rank of low-rank adapters is 64. *N.A.* indicates the model does not converge. We report the median over five random seeds.

| Method   | Bit  | LLAMA-2-7b  |        | LLAMA-2-13b |        |
|----------|------|-------------|--------|-------------|--------|
|          |      | WikiText-2↓ | GSM8K↑ | WikiText-2↓ | GSM8K↑ |
| LoRA     | 16   | 5.08        | 36.9   | 5.12        | 43.1   |
| LoRA+Reg | 16   | –           | 34.4   | –           | 45.3   |
| QLoRA    | 4    | 5.70        | 35.1   | 5.22        | 39.9   |
| LoftQ    | 4    | 5.24        | 35.0   | 5.16        | 45.0   |
| QLoRA    | 3    | 5.73        | 32.1   | 5.22        | 40.7   |
| LoftQ    | 3    | 5.63        | 32.9   | 5.13        | 44.4   |
| QLoRA    | 2.5  | N.A.        | N.A.   | 19.39       | N.A.   |
| LoftQ    | 2.5  | 5.78        | 31.1   | 5.22        | 41.1   |
| QLoRA    | 2.25 | N.A.        | N.A.   | N.A.        | N.A.   |
| LoftQ    | 2.25 | 6.13        | 26.5   | 5.45        | 38.1   |
| QLoRA    | 2    | N.A         | N.A.   | N.A.        | N.A.   |
| LoftQ    | 2    | 7.85        | 20.9   | 7.69        | 25.4   |

noticed that increasing the alternating step beyond a certain point tends to result in diminishing returns. We suspect this phenomenon occurs because, as the gap becomes smaller, it becomes more challenging for alternating optimization to consistently minimize the gap at each step. This challenge emerges because of the inherent errors introduced by the quantization method. Nevertheless, results from Figure [3](#page-12-0) indicate our method is not sensitive to the alternating step *T* and is able to consistently enhance downstream fine-tuning performance.

