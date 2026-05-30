# <span id="page-15-0"></span>**A Implementation Details**

### **A.1 Models and Datasets**

**Models** For our experiments, we employed Mixtral-8×7B [\(Jiang et al., 2024\)](#page-12-0) and DeepSeek-MoE-16B [\(Dai](#page-11-1) [et al., 2024\)](#page-11-1). Mixtral-8×7B utilizes 8 experts for MoE layers and activates the top two for each input token. In contrast, DeepSeek-MoE-16B employs a dense FFN in the first block and utilizes two shared experts with an additional 64 experts within MoE layers in other blocks.

**Datasets** For compression experiments, we used the C4 dataset [\(Raffel et al., 2019\)](#page-13-14), with 128 samples and an input sequence length of 2,048, following the setup in [\(Sun et al., 2023;](#page-14-1) [Lu et al., 2024;](#page-13-3) [Lin et al.,](#page-12-10) [2024;](#page-12-10) [Frantar et al., 2022\)](#page-11-0). To evaluate model performance, we report normalized zero-shot accuracy on the LM-harness benchmark, which includes multiple tasks: ARC-C [\(Clark et al., 2018\)](#page-11-11), BoolQ [\(Clark et al.,](#page-11-12) [2019\)](#page-11-12), HellaSwag [\(Zellers et al., 2019\)](#page-14-13), MMLU [\(Hendrycks et al., 2021\)](#page-12-15), OBQA [\(Mihaylov et al., 2018\)](#page-13-17), PIQA [\(Bisk et al., 2019\)](#page-11-13), RTE [\(Wang et al., 2019\)](#page-14-14), and WinoGrande [\(ai2, 2019\)](#page-11-14). The evaluation code is based on EleutherAI LM Harness [\(Gao et al., 2023\)](#page-11-15).

### **A.2 Implementation Details of Expert Slimming**

Both Expert Slimming methods (i.e., pruning and quantization) require calibration data to estimate input statistics. To control this variable, we use 128 samples from the C4 dataset [\(Raffel et al., 2019\)](#page-13-14) as the calibration dataset for pruning. For quantization, we follow the default settings of GPTQ [1](#page-15-2) and AWQ [2](#page-15-3) , using 128 random samples from Alpaca [\(Taori et al., 2023\)](#page-14-15) and Pile [\(Gao et al., 2020\)](#page-11-16), respectively. We use the default group size 128 for Mixtral-8×7B and 64 for DeepSeek-MoE-16B.

### <span id="page-15-1"></span>**A.3 Implementation Details of Expert Drop**

The Expert Drop compresses MoE by preserving only important experts {*Ei*}*i*∈T ′ while removing others, where T ′ is determined by the importance scores {*S*(*Ei*)}*i*∈T . Following [Muzio et al.](#page-13-4) [\(2024\)](#page-13-4), we measure the importance scores through the averaged routing scores of a batched data X , i.e., {*S*(*Ei*)} = |X | P *<sup>x</sup>*∈X *Gi*(*x*), and consider two dropping strategies for Expert Drop: layer-wise dropping and global dropping.

**Layer-wise dropping** removes the same number of experts for each layer. Given the total number of experts *n* = |T | and the preserved number of experts *n* ′ = |T ′ | *< n* in layer *l*, the preserved expert set T ′(*l*) is obtained by:

$$\mathcal{T}'^{(l)} = \{ E_t^{(l)} \}, \text{ where } S(E_t^{(l)}) \in \text{TopK}(\{ S(E_i^{(l)}) \}_{i=1}^n, n').$$
 (10)

**Global dropping** constrains the total number of preserved experts for the entire model. Given the total number of layers *L* in the model, the preserved expert set T ′(*l*) for layer *l* is obtained by:

$$\mathcal{T}^{\prime(l)} = \{ \boldsymbol{E}_{t}^{(l)} \}, \quad \text{where} \quad \boldsymbol{S}(\boldsymbol{E}_{t}^{(l)}) \in \text{TopK}\Big(\bigcup_{j=1}^{m} \{ \boldsymbol{S}(\boldsymbol{E}_{i}^{(j)}) \}_{i=1}^{n}, n'L \Big). \tag{11}$$

For the integration of Expert Slimming and Expert Trimming, we choose the global dropping as the strategy of Expert Drop, which shows competitive performance compared to the layer dropping for Mixtral-8×7B under low dropping ratios, as well as consistent better performance for DeepSeek-MoE-16B in Figure [13.](#page-20-1)

<span id="page-15-2"></span><sup>1</sup>https://github.com/AutoGPTQ/AutoGPTQ

<span id="page-15-3"></span><sup>2</sup>https://github.com/casper-hansen/AutoAWQ

