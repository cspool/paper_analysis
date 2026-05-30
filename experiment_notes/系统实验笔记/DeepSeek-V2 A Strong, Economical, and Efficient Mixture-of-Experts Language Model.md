## DeepSeek-V2 A Strong, Economical, and Efficient Mixture-of-Experts Language Model

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 DeepSeek-V2 的推理部署优化：**(1) FP8 参数精度转换**，将模型参数转为 FP8 精度部署；**(2) KV cache 6-bit 量化**，将 MLA 的 latent KV cache 进一步压缩到平均每个元素 6 bits；**(3) 基于 vLLM 推理后端**进行高效 serving。实验比较：单节点 8×H800 上 DeepSeek-V2 的生成吞吐量 vs DeepSeek 67B（dense, 67B），以及 prompt input throughput。

- 硬件平台是什么，配置是什么。
  单节点配备 8 张 NVIDIA H800 GPU，节点内 NVLink + NVSwitch 互联，节点间 InfiniBand。推理部署使用 FP8 精度。

- 开源Serving框架是什么。修改了什么。
  论文未明确说明 Serving 框架的具体修改细节。论文提到使用 vLLM (Kwon et al., 2023) 作为 RL 训练阶段的推理后端，部署阶段的 serving 框架未详细说明。论文说明其训练框架为 HAI-LLM（High-flyer 内部开发），推理部署进行了 FP8 量化和 KV cache 6-bit 量化。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  模型 checkpoint 开源在 https://github.com/deepseek-ai/DeepSeek-V2，但 serving 部署框架细节未开源说明。

  **DeepSeek-V2 推理部署全流程（以单节点 8×H800 推理为例）**：
  ```
  输入：用户 prompt tokens [t1, t2, ..., tn]
  
  Step 1: Tokenization
    BBPE tokenizer (vocab=100K) → token_ids [id1, id2, ..., idn]
  
  Step 2: Prefill 阶段（8×H800, 专家并行 D=8, TP 不需要因为激活参数仅 21B）
    for each Transformer layer (l=1..60):
      // MLA Attention（使用改进版 FlashAttention-2）
      h → W^{DQ} → c^Q → W^{UQ} → q^C
      h → W^{DKV} → c^{KV}（压缩到 512 维，FP8 存储 → 512 bytes）
      h → W^{KR} → k^R（RoPE，64 维）
      // 吸收优化：W^{UK} ⊂ W^{UQ}, W^{UV} ⊂ W^O，无需显式计算 k^C, v^C
      FlashAttention-2(q, k, v) → attention output
      
      // DeepSeekMoE FFN
      2 共享专家（计算在本地设备）
      160 路由专家分布在 8 设备 → Top-6 选择，最多 3 设备
      all-to-all 通信传输 token hidden states
      Token-Dropping（推理阶段可选，不丢 token 则全算）
  
  Step 3: Decode 阶段（逐 token 生成）
    每个 decode step：
      - MLA 只需计算新 token 的 c^{KV} 并追加到 KV cache
      - KV cache per layer: (d_c + d_h^R) = 512+64 = 576 元素
      - 6-bit 量化后每层 KV cache ≈ 576*6/8 = 432 bytes/token
      - 60 层 × 432 bytes = ~25.9 KB per token in KV cache
      - vs DeepSeek 67B MHA: ~1.9M elements × 2 bytes = ~3.8 MB per token per layer × 60 ≈ 很多
  
  Step 4: 输出
    生成 token → 经 BBPE decoder → 文本输出
  ```

  **吞吐数据**：单节点 8×H800，DeepSeek-V2 生成吞吐 >50K tokens/s（5.76× DeepSeek 67B），prompt 输入吞吐 >100K tokens/s。
  **训练成本**：每 1T tokens，DeepSeek 67B 需要 300.6K GPU hours，DeepSeek-V2 仅需 172.8K GPU hours（节省 42.5%）。
