# 5 Experiments

We conduct experiments to assess the effectiveness of the proposed method. To evaluate how much LPCD enhances the existing approach, we utilize the final weight of QEP or LoaQ and combine LPCD with the current quantization method.

## 5.1 Setting

Baselines and Quantization Methods. In this study, we focus only on the per-channel weight quantization scheme. We employ representative layer-wise PTQ methods of Round-to-nearest (RTN) and GPTQ, which are used in conjunction with error compensation techniques. Some error compensation methods have been reported to enhance existing layer-wise quantization methods.

Table 2: Zero-shot average accuracy (↑) on ARC-E and PIQA for LLaMA and Qwen models across different bit-widths and quantization methods.

| Bits | Method |      | LLaMA2-7B | LLaMA2-13B | LLaMA3-8B | Qwen3-8B | Qwen3-14B |
|------|--------|------|-----------|------------|-----------|----------|-----------|
| FP16 | -      | -    | 0.7682    | 0.7896     | 0.7915    | 0.7930   | 0.8133    |
| 4bit | RTN    | QEP  | 0.7467    | 0.7825     | 0.7613    | 0.6998   | 0.7369    |
|      |        | LoaQ | 0.7471    | 0.7812     | 0.7812    | 0.6333   | 0.7760    |
|      |        | Ours | 0.7527    | 0.7793     | 0.7789    | 0.7584   | 0.7949    |
|      | GPTQ   | QEP  | 0.7428    | 0.7741     | 0.7599    | 0.7113   | 0.7695    |
|      |        | LoaQ | 0.7456    | 0.7817     | 0.7559    | 0.6066   | 0.7702    |
|      |        | Ours | 0.7504    | 0.7810     | 0.7001    | 0.7661   | 0.7782    |
| 3bit | RTN    | QEP  | 0.5784    | 0.6757     | 0.5136    | 0.5442   | 0.6164    |
|      |        | LoaQ | 0.4799    | 0.6928     | 0.5282    | 0.3971   | 0.6904    |
|      |        | Ours | 0.5688    | 0.7433     | 0.5373    | 0.6291   | 0.6560    |
|      | GPTQ   | QEP  | 0.6881    | 0.7372     | 0.5427    | 0.5986   | 0.6535    |
|      |        | LoaQ | 0.4556    | 0.7462     | 0.5532    | 0.6363   | 0.6222    |
|      |        | Ours | 0.6602    | 0.7431     | 0.6290    | 0.6493   | 0.6919    |
| 2bit | RTN    | QEP  | 0.3816    | 0.3831     | 0.3904    | 0.3845   | 0.3895    |
|      |        | LoaQ | 0.3758    | 0.3805     | 0.3836    | 0.3917   | 0.4183    |
|      |        | Ours | 0.3756    | 0.3794     | 0.3826    | 0.3864   | 0.3864    |
|      | GPTQ   | QEP  | 0.3931    | 0.3821     | 0.3889    | 0.3972   | 0.4355    |
|      |        | LoaQ | 0.3761    | 0.3840     | 0.3842    | 0.4031   | 0.4253    |
|      |        | Ours | 0.3856    | 0.3817     | 0.3962    | 0.4054   | 0.4354    |

We employ QEP and LoaQ as baselines for our proposed method. We perform quantization in the INT4, INT3, and INT2 settings. We skip quantization for the last 2 layers due to the higher frequency of outliers observed in their activations.

Dataset. GPTQ computes Hessian using the calibration dataset to perform effective quantization. Furthermore, error compensation methods compute activations for the output approximation. Following previous studies, LoaQ uses 128 samples of 2048 tokens each from the C4 dataset. We observe over-fitting to the calibration dataset when we employ 128 samples. We employ 2048 tokens consisting of 256 sequences, randomly sampled from the WikiText-2 dataset, as the calibration dataset.

Models. We evaluate the proposed method and baselines in recent major open-weight LLM, including LLaMA2 [\(Touvron et al.,](#page-10-11) [2023\)](#page-10-11), LLaMA3 [\(Grattafiori et al.,](#page-9-20) [2024\)](#page-9-20), and the dense Qwen3 model families. LLaMA is an open-weight LLM family that is primarily developed by Meta Platforms. We employ LLaMA2-7B, LLaMA2- 13B, and LLaMA3-8B for evaluation. Qwen3 is a powerful open-weight LLM family as well as

LLaMA family. They employ a slightly different architecture from LLaMA such as Q/K RMSNorm. We employ Qwen3-4B, Qwen3-8B, and Qwen3- 14B for evaluation.

Hyper Parameters. We conduct a grid search to determine the optimal propagation strength parameter α for QEP and the optimal sub-layer output approximation strength parameter β for LoaQ. The grid search is performed using smaller models, such as Qwen3-0.5B and LLaMA3.2-1B, after which we applied the resulting optimal parameters to the larger models. Following LoaQ, the range of α is from 0 to 1 with increments of 0.1 and, the range of β is from 0 to 1 with increments 0.05. We apply LPCD on the relaxed weight of LoaQ; performances of LoaQ are generally better than those of QEP. We first apply LoaQ to each submodule and then perform LPCD. As explained in Sec. [4.4,](#page-5-1) we apply LPCD to three groups of Transformer submodules: the Q/K module, the V/O module, and the Up/Down module. For gradient-based optimization of LPCD, We employ 8 batch size, 40 epochs, and cosine scheduled learning rate that begin with 10−<sup>5</sup> . The optimization is conducted using the Adam optimizer with the default settings

in PyTorch.

Evaluations. We follow the established evaluation protocols for quantization algorithms used in numerous previous studies. We evaluate the perplexity (PPL) on the WikiText2 dataset. We also evaluated zero-shot accuracy on the ARC Easy and PiQA.benchmarks. We implement QEP, LoaQ, and LPCD using Python 3.12.11 with PyTorch 2.4.0 and Hugging Face Transformers 4.55.3. All experiments were conducted on an NVIDIA H100 GPU using the TSUBAME 4.0 supercomputer.

## 5.2 Result

Perplexities. Table [1](#page-6-0) summarizes the perplexities of various PTQ configurations within the LLaMA and Qwen families. Overall, LPCD-based submodule quantization achieves the lowest perplexity in most settings, consistently outperforming both QEP and LoaQ, irrespective of whether RTN or GPTQ is used. The gains are most pronounced in low-bit regimes: for the practically important LLaMA-3-8B and Qwen-3-8B models at 3-bit and 2-bit levels, LPCD substantially reduces PPL compared to both baselines, preventing the severe degradation or divergence observed with QEP and LoaQ. Notably, for Qwen-3-8B, RTN combined with LPCD already surpasses the more sophisticated QEP+GPTQ configuration, indicating that submodule LPCD provides improvements that are largely orthogonal to the choice of the underlying layer-wise quantizer.

Zero Shot Task Evaluation. Table 2 indicates that LPCD achieves the highest or nearly the highest zero-shot accuracy across various models, bitwidths, and base quantizers. At 4-bits, our method closely matches FP16 performance while slightly improving both QEP and LoaQ, indicating that optimization at the submodule-level does not adversely affect high-precision behavior. The advantages are more apparent in low-bit regimes. For the practically important LLaMA-3-8B and Qwen-3-8B models at 3-bit and 2-bit, LPCD consistently recovers a substantial portion of the accuracy lost by QEP and LoaQ when used with RTN and GPTQ. Remarkably, the simple RTN+LPCD configuration on Qwen-3-8B outperforms the more sophisticated QEP+GPTQ baseline, demonstrating that our submodule refinement complements rather than merely imitates existing layer-wise PTQ techniques.

