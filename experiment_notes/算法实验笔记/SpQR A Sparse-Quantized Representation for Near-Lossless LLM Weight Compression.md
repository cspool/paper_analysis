## SpQR A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression

- 属于算法pipeline的实现是什么？实验比较什么？
  SpQR提出一种新的混合稀疏-量化压缩格式，将LLM权重量化到3-4 bits的同时保持near-lossless精度（<1% perplexity损失）。核心包含两个创新：(1) **敏感权重异常值检测与隔离**：基于Optimal Brain Surgeon框架的封闭形式敏感度准则 s_ij = (w_ij - quant(w_ij))² / (2[H⁻¹]_jj)，在GPTQ逐列量化过程中动态计算每个权重的敏感度，将超过阈值τ的高敏感度权重（约1%）保留为16-bit异常值。(2) **双层量化（Bilevel Quantization）**：使用极小group size（β₁=8-32）进行分组量化，并将第一层量化统计量（scale和zero-point）本身再以相同算法做第二层量化（β₂=16），从而在低bit-width下保持高精度。实验比较SpQR vs GPTQ和RTN (round-to-nearest) baseline在3-bit和4-bit配置下的WikiText2、C4、Penn Treebank perplexity，以及五任务zero-shot accuracy（WinoGrande, PiQA, HellaSwag, ARC-easy, ARC-challenge）。同时进行ablations：bilevel quantization vs 16-bit statistics、unstructured outliers vs row outliers vs column outliers、GPTQ activation order heuristic效果。

- 硬件平台是什么，配置是什么。
  量化和评估主要在单张NVIDIA A100-80GB GPU上完成，部分实验在NVIDIA A6000（48GB）上进行。量化实施采用PyTorch实现。推理速度测试在A100 GPU上进行，batch size=1的token-by-token生成模式。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA (7B, 13B, 30B, 65B)，Falcon (7B, 40B)，OPT (6.7B, 13B, 30B, 66B)。校准数据集：对LLaMA使用RedPajama数据集（LLaMA训练数据的公开复刻），对Falcon使用RefinedWeb数据集。量化的校准样本数为128个2048-token序列。评估数据集：WikiText2、C4、Penn Treebank（perplexity）；WinoGrande、PiQA、HellaSwag、ARC-easy、ARC-challenge（zero-shot accuracy，使用LM Evaluation Harness）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/Vahe1994/SpQR（论文引用"to be integrated into github.com/TimDettmers/bitsandbytes"）。
  
  ```bash
  # 以LLaMA-30B near-lossless配置为例：
  python main.py $MODEL custom --custom_data_path=$DATA \
      --wbits 4 --groupsize 16 --perchannel \
      --qq_scale_bits 3 --qq_zero_bits 3 --qq_groupsize 16 \
      --outlier_threshold 0.1 \
      --fit_quantizer_without_outliers --permutation_order act_order
  ```
  
  SpQR算法Pipeline（以W ∈ R^{d_out × d_in} 的3-bit base量化 + 1% outliers为例）：
  
  ```
  Input: W ∈ R^{m×n} (weight matrix), X ∈ R^{n×d} (calibration data, 128×2048 tokens)
  Output: Q (quantized weights), S_q/Z_q (first-level quantized scales/zeros), 
          S_s/Z_s/S_z/Z_z (second-level quantized statistics), W_sparse (CSR outlier matrix)
  
  1. 计算Hessian: H = 2XXᵀ, Hⁱᶜ = Cholesky((H + λI)⁻¹)
  
  2. 逐列（逐β₁ group）处理权重矩阵，对每column group i=1, β₁, 2β₁, ..., n:
     
     a) 检测异常值（outliers子程序）:
        E_base = error(W[:,j], Hⁱᶜ[:,j])           # 所有(beta1列)权重的L2 error
        for each column j in the group:
            E_ol = error(W[:, loo], Hⁱᶜ_loo,loo)   # leave-one-out error
            if E_base - E_ol > τ:                   # 标记为outlier
                O = O ∪ {j}
     
     b) 在排除outlier的情况下拟合group-wise quantizer:
        ŝ, ẑ = fit_statistics(W_group, O)           # bilevel quantization
     
     c) 量化非outlier权重:
        Q[:,j] = quantize(W[:,j], ŝ, ẑ)
        ŵ_q = dequantize(Q[:,j], ŝ, ẑ)
     
     d) 误差补偿（GPTQ风格）:
        E[:,j] = (W[:,j] - ŵ_q) / Hⁱᶜ[j,j] · (1 - is_outlier(W[:,j]))
        W[:,j:i+β₁] -= E · Hⁱᶜ[j,(j:i+β₁)]
        W[:,(i+β₁):n] -= E · Hⁱᶜ[(i:i+β₁),(i+β₁):n]
  
  3. 收集outlier矩阵为CSR格式:
     W_sparse = gather_outlier_matrix(W, O)  # row-first, col-second排序
     
     存储格式：
     - 每个outlier: 16-bit value + 16-bit column index = 32 bits
     - 每行: 一个32-bit row pointer (cumulative outlier count)
  
  4. 收集量化统计量为双层结构:
     S_q, Z_q: first-level 3-bit quantized scales & zero-points (每组β₁=16权重)
     S_s, Z_s: second-level scales (量化ŝ的scale), 每组β₂=16个first-level统计量
     S_z, Z_z: second-level zeros (量化ẑ的zero), 每组β₂=16个first-level统计量
     
     每256权重（β₁×β₂=16×16）的内存布局：
     - 256个3-bit weight codes
     - 16个3-bit scales + 16个3-bit zero-points
     - 4个16-bit second-level statistics scalars
  
  平均bits数计算：b̄ = b_w + (b_s+b_z)/β₁ + 64/(β₁β₂) + 32·r_o
  例：b_w=3, b_s=b_z=3, β₁=16, β₂=32, r_o=0.4% → b̄ = 3 + 6/16 + 64/512 + 0.128 = 3.63 bits/param
  ```
