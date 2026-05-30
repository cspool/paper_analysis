# Accurate LoRA-Finetuning Quantization of LLMs via Information Retention

 $\begin{array}{cccccccccccccccccccccccccccccccccccc$ 

### **Abstract**

The LoRA-finetuning quantization of LLMs has been extensively studied to obtain accurate yet compact LLMs for deployment on resourceconstrained hardware. However, existing methods cause the quantized LLM to severely degrade and even fail to benefit from the finetuning of LoRA. This paper proposes a novel **IR-QLoRA** for pushing quantized LLMs with LoRA to be highly accurate through information retention. The proposed IR-QLoRA mainly relies on two technologies derived from the perspective of unified information: (1) statistics-based Information Calibration Quantization allows the quantized parameters of LLM to retain original information accurately; (2) finetuning-based Information Elastic Connection makes LoRA utilizes elastic representation transformation with diverse information. Comprehensive experiments show that IR-QLoRA can significantly improve accuracy across LLaMA and LLaMA2 families under 2-4 bit-widths, e.g., 4bit LLaMA-7B achieves 1.4% improvement on MMLU compared with the state-of-the-art methods. The significant performance gain requires only a tiny 0.31% additional time consumption, revealing the satisfactory efficiency of our IR-QLoRA. We highlight that IR-QLoRA enjoys excellent versatility, compatible with various frameworks (e.g., NormalFloat and Integer quantization) and brings general accuracy gains. The code is available at https://github.com/htqin/ir-qlora.

### 1. Introduction

Large language models (LLMs) have demonstrated strong performance in natural language understanding (Touvron

Proceedings of the 41<sup>st</sup> International Conference on Machine Learning, Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s).

et al., 2023a;b). LLMs can be adapted to various downstream real-world applications, paired with large-scale pretraining and finetuning for downstream tasks (Chang et al., 2023; Devlin et al., 2018; Zhao et al., 2023; Huang & Chang, 2022; Brown et al., 2020). However, because of the massive parameters and computation, the LLM has high or even harsh resource requirements for deployment scenarios. The inference of LLMs is expensive and heavily relies on high-performance devices, such as graphics processing units (GPUs) (Ganesh et al., 2021; Zhu et al., 2023; Chitty-Venkata et al., 2023). Therefore, compression approaches of LLMs are widely studied to allow their deployment on edge devices. Quantization emerges as a promising approach to compress LLMs by reducing bit-width but usually results in significant degeneration in accuracy (Xiao et al., 2023; Lin et al., 2023). For example, the 4-bit LLaMA-7B quantized by GPTQ (Frantar et al., 2022) suffers a 1.5% drop of 5-shot accuracy on MMLU benchmark (Hendrycks et al., 2020) compared to its original counterpart (Liu et al., 2023a).

LoRA-finetuning quantization has become a popular paradigm that combines the LLM quantization with parameter-efficient finetuning of low-rank adaption (LoRA) (Dettmers et al., 2023; Xu et al., 2023b). Methods under this paradigm mainly consist of the following two phases. The first one is the post-training quantization (PTQ) of the LLM (Dettmers et al., 2021), obtaining quantizers by resource-saving calibration. The latter one is finetuning the LoRA (Hu et al., 2021), where the quantized LLM remains fixed and LoRA is finetuned. LoRA-finetuning quantization of LLMs is resource and time-saving compared to finetuning of the whole LLM while pushing the quantized LLM to high accuracy compared to performing PTQ solely (Dettmers et al., 2023; Xu et al., 2023b; Liu et al., 2023b).

However, despite several efforts made, existing LoRA fine-tuning quantization of LLMs is still far from the limits regarding accuracy. We empirically observe that the prevention of further accurate quantization is mainly because the information loss caused by LLM quantization is significant and cannot be recovered effectively by LoRA. Especially with ultra-low bit-widths ( $\leq$  3-bit) and large model scales ( $\geq$  30B), the former results in the nonlinearly increased level of information loss for each element, and the latter leads to a significant increase of the total amount of information loss

<sup>\*</sup>Equal contribution <sup>1</sup>ETH Zürich <sup>2</sup>Beihang University <sup>3</sup>Bytedance AI Lab. Correspondence to: <sup>™</sup>Xianglong Liu <xl-liu@buaa.edu.cn>.

<span id="page-1-0"></span>![](_page_1_Figure_1.jpeg)

Figure 1: Overview of IR-QLoRA. The framework includes Information Calibration Quantization (ICQ) for quantizing LLMs and Information Elastic Connection (IEC) for enhancing LoRA

for the whole model. In these cases, the finetuned LoRA is hard to assist LLMs to achieve high accuracy on downstream tasks, *e.g.*, 4-bit LLaMA-30B with finetuned LoRA even fails to achieve the accuracy of the original counterpart without finetuning (57.7% *vs.* 58.2% on MMLU).

In this paper, we present **IR-QLoRA** to obtain accurate **Q**uantized LLMs with **LoRA** via **Information Retention** (see the overview in Figure 1). To tackle the information loss of the quantization of the LLM, we propose a *Information Calibration Quantization* (ICQ) technique. By calibration by entropy maximization, ICQ enables quantizers for the LLM to retain the original information from the original parameters to quantized ones. We also propose the *Information Elastic Connection* (IEC) to enhance the information recovery capability of LoRA. IEC works together with LoRA, which performs parameter-free elastic transformations to utilize the information of original features and diversify the transformation form of LoRA.

Our IR-QLoRA provides strong and generic support to achieve accurate quantized LLMs with LoRA. Extensive experiments on the MMLU benchmark show that our IR-QLoRA outperforms existing methods with convincing margins on LLaMA and LLaMA2 series models under different bit-widths, especially at ultra-low bit-widths (2-3 bit). For example, the average accuracy of 2-bit IR-QLoRA in the LLaMA family is 0.5% higher than SOTA LoRA-finetuning quantization methods. For efficiency, the significant performance growth brought by our IR-QLoRA requires only a tiny 0.31% additional time consumption for LLaMA-13B. Moreover, IR-QLoRA is versatile and can boost existing LoRA-finetuning LLM quantization frameworks flexibly, e.g., the integration with QA-LoRA (Xu et al., 2023b) brings a cost-free 0.5% gain on MMLU to 4-bit LLaMA-7B.

### 2. Related Work

LLMs have demonstrated remarkable proficiency across diverse natural language understanding tasks and are established as a prominent paradigm in this field (Chang et al., 2023; Devlin et al., 2018; Zhao et al., 2023; Huang & Chang, 2022; Brown et al., 2020; Touvron et al., 2023a;b). This reality poses substantial challenges to deploying LLMs in settings with limited resources. Consequently, the research of the compression technologies for LLMs has gained prominence as a critical area of research. Existing compression technologies of LLMs include pruning, distillation, low-rank decomposition, and low-bit quantization (Ganesh et al., 2021; Zhu et al., 2023; Chitty-Venkata et al., 2023; Xu et al., 2023a). Among these technologies, quantization aims to compress the LLMs from 16-bit floating-point to lower bit-widths to mitigate the storage and computation.

Since compression is from a generic bit-width perspective, quantization has become a popular method to obtain efficient LLMs (Xiao et al., 2023; Lee et al., 2023; Shao et al., 2023; Dettmers et al., 2022; Liu et al., 2023b; Kim et al., 2023). The LoRA-finetuning quantization of LLMs emerges to achieve a balanced trade-off between computational cost and accuracy (Dettmers et al., 2023; Li et al., 2023), where quantized LLMs are finetuned with parameter-efficient LoRAs. However, existing quantized LLMs with LoRA are still far from ideal in accuracy. More details about related works are presented in Appendix A.1.

### 3. The Rise of IR-QLoRA

### 3.1. Preliminaries

We first present a baseline for LoRA-finetuning quantization of LLMs following common practice (Dettmers et al., 2023).

Before finetuning, the weights of LLMs are to be quantized.

The quantization function for the weight  $w \in \mathbb{R}^{h \times o}$  is

$$\hat{\boldsymbol{w}}^{\mathrm{NF}k} = \mathrm{NF}k\left(\frac{\boldsymbol{w}}{s}\right) = \mathrm{NF}k\left(\frac{\boldsymbol{w}}{\mathrm{absmax}(\boldsymbol{w})}\right),$$
 (1)

where  $\hat{\boldsymbol{w}}^{\text{NF}k}$  denotes quantized weight and the quantization block size is 64 as default, and s is the scaling factor calculated by  $\operatorname{absmax}(\boldsymbol{w})$ . NF $k(\cdot)$  denotes the k-bit NormalFloat quantization (Dettmers et al., 2023), quantizing the weights of LLMs to  $2^k$  values  $q_i$  as follows:

$$q_i = \frac{1}{2} \left( Q \left( \frac{i}{2^k + 1} \right) + Q \left( \frac{i+1}{2^k + 1} \right) \right), \qquad (2)$$

where  $Q(\cdot)$  is the quantile function of  $\mathcal{N}(0,1)$  distribution. Then, the computation process (e.g., linear projection) of the quantized unit of the LLM during inference is

<span id="page-2-0"></span>
$$\mathbf{y}' = \mathbf{x}\hat{\mathbf{w}}^{\text{FP16}} = \mathbf{x}\left(\hat{\mathbf{w}}^{\text{NF}k} \operatorname{dequant}(s_1^{\text{FP8}}, s_2^{\text{FP16}})\right), \quad (3)$$

where  $\boldsymbol{x} \in \mathbb{R}^{b \times h}$  and  $\boldsymbol{y}' \in \mathbb{R}^{b \times o}$  denote the input and output of quantized linear projection in LLMs, respectively. dequant $(s_1^{\text{FP8}}, s_2^{\text{FP16}})$  is expected to approximate the original scaling factor s. After double-quantization of s, we can obtain the quantized values  $s_1^{\text{FP8}}$  and scaling factors  $s_2^{\text{FP16}}$  follow (Dettmers et al., 2023).  $\hat{\boldsymbol{w}}^{\text{FP16}}$  denotes the FP16 weights dequantized from  $\hat{\boldsymbol{w}}^{\text{NF}k}$ .

The LoRA refers to a set of finetunable parameters designed to enhance the quantized linear projection in LLMs by introducing an extra factorized projection (Hu et al., 2021; Dettmers et al., 2023). For the quantized linear projection as Eq. (3), the computation with LoRA can be expressed as:

<span id="page-2-4"></span>
$$y = y' + \alpha x \ell_1 \ell_2, \tag{4}$$

where  $\ell_1 \in \mathbb{R}^{h \times r}$  and  $\ell_2 \in \mathbb{R}^{r \times o}$  are the finetunable parameters, and  $\alpha$  is a scalar. Since the parameter efficiency of LoRA should be kept during inference, its rank r is far smaller than the input and output dimensions (h and o, respectively), which makes its memory and computational consumption far smaller than the corresponding linear projection in LLMs (e.g., r=64 vs. h=4096 and o=4096). During the backward propagation of the finetuning process, the gradients are passed through the fixed quantized weights of LLMs to update the parameters in LoRA.

The quantization process of the LLM and the finetuning process of the LoRA are decoupled. The PTQ first processes the LLM to obtain low-bit quantized weights, and then the LoRA is finetuned for specific downstream tasks.

### 3.2. Information Calibration Quantization

### 3.2.1. Degeneration of Quantized LLMs

In the aforementioned baseline, the LLMs are quantized directly from pre-trained models, where the low-bit discretization of the parameters causes the accuracy degradation. Existing quantization methods attribute the degradation to the

<span id="page-2-3"></span>![](_page_2_Figure_15.jpeg)

Figure 2: An illustration of ICQ in IR-QLoRA

numerical quantization error. However, the information loss caused by quantization is always neglected.

Specifically, the quantized weights of LLMs are expected to reflect the information carried by original counterparts, but reduced bit-width severely constrains the representation capabilities. From the information perspective, the dependence between the weights of quantized and original LLMs is expressed as the mutual information (Qin et al., 2023):

<span id="page-2-1"></span>
$$\mathcal{I}(\hat{\boldsymbol{w}}^{\text{FP16}}; \boldsymbol{w}) = \mathcal{H}(\hat{\boldsymbol{w}}^{\text{FP16}}) - \mathcal{H}(\hat{\boldsymbol{w}}^{\text{FP16}} \mid \boldsymbol{w}),$$
 (5)

where  $\mathcal{H}(\hat{\boldsymbol{w}}^{\text{FP16}})$  denotes the entropy of  $\hat{\boldsymbol{w}}^{\text{FP16}}$ , and  $\mathcal{H}(\hat{\boldsymbol{w}}^{\text{FP16}} \mid \boldsymbol{w})$  denotes the conditional entropy of  $\hat{\boldsymbol{w}}^{\text{FP16}}$  given  $\boldsymbol{w}$ . As deterministic quantizers are used in the quantization of LLMs,  $\mathcal{H}(\hat{\boldsymbol{w}}^{\text{FP16}} \mid \boldsymbol{w}) = 0$  and the  $\mathcal{I}(\hat{\boldsymbol{w}}^{\text{FP16}}; \boldsymbol{w})$  depends on  $\mathcal{H}(\hat{\boldsymbol{w}}^{\text{FP16}})$  directly. In the PTQ, since the original weights  $\boldsymbol{w}$  remain unchanged, maximizing the mutual information  $\mathcal{I}(\hat{\boldsymbol{w}}^{\text{FP16}}; \boldsymbol{w})$  in Eq. (5) is equivalent to

<span id="page-2-2"></span>
$$\underset{s,s_{1}^{\text{FP}},s_{2}^{\text{FP}}}{argmax} \mathcal{H}(\hat{\boldsymbol{w}}^{\text{FP}16}; s, s_{1}^{\text{FP}8}, s_{2}^{\text{FP}16}). \tag{6}$$

Since dequant( $s_1^{\text{FP8}}, s_2^{\text{FP16}}$ ) is a scalar in dequantization and does not affect information entropy of  $\hat{w}^{\text{FP16}}$ , the above objective function can be further simplified as follows:

$$\underset{s}{\operatorname{argmax}} \mathcal{H}(\hat{\boldsymbol{w}}^{\text{NF}k}; s) = -\sum_{i=1}^{2^{k}-1} P(q_i) \log_2 P(q_i), \quad (7)$$

where  $P(q_i)$  is the probability of  $\hat{\boldsymbol{w}}^{NFk}$  taking the value  $q_i$ .

Since the significant reduction of bit-width leads to decreased representation capability, the entropy of the quantized weight is far less than that of the original counterpart. For example, the number of representation candidates for a 4-bit quantized weight reduces  $4096\times$  compared to its original 16-bit (FP16) counterpart, and the upper bound of information entropy  $\mathcal{H}(\hat{w}^{\text{FP16}})$  in Eq. (6) is correspondingly reduced  $4\times$  (4 for 4-bit vs. 16 for 16-bit), meaning a significant degradation of information in the quantity and quality. Thus, prioritizing information recovery within low-bit weights is crucial for enhancing quantized LLMs.

