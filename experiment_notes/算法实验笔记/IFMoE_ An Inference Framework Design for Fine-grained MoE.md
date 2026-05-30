## IFMoE: An Inference Framework Design for Fine-grained MoE

- 属于算法pipeline的实现是什么？实验比较什么？
  - IFMoE 提出的算法 pipeline 核心是 **Self-Draft Speculative Decoding with KV-cache Revision**：
    1. **Self-Draft 机制**：不引入额外的小型 draft model，而是利用 fine-grained MoE 模型自身在激活更少 expert 时仍能保持较好性能的特性。Draft 阶段仅激活 decode_topk Dk=2 个 expert（全量为 Ek=6），每个 decode step 的计算量大幅降低（GroupedGEMM 的 expert 数从 6 降到 2），实现快速 draft。
    2. **KV-cache Revision**：每 α=10 步 draft 后，用全量 Ek=6 experts 对 buffer 内所有 token 做一次 encode forward，将 KV-cache 中对应的 key/value 更新为"全量 expert 应产生的值"。这是 IFMoE 区别于传统 speculative decoding 的关键——传统方法用 draft logits 与 target logits 比较做 accept/reject，IFMoE 接受全部 draft token 但修正 KV-cache。
    3. **Accept All 策略**：IFMoE 接受 draft model 生成的全部 token（不做逐 token 的 accept/reject），仅通过 KV-cache revision 来补偿用更少 expert draft 造成的信息损失。
  - 实验比较：
    - Full model（标准全量 expert decode）vs IFMoE（Dk=2 draft + Ek=6 revision）
    - 下游任务：XSum（ROUGE）、GSM8K（准确率）、TruthfulQA-Gen、IFEval
    - 超参：α=10, encode_topk Ek=6, decode_topk Dk=2
    - 结果：下游性能与 full model 接近（如 Qwen2: GSM8K 75.4→71.1, XSum 13.7→13.5），benchmark 延迟和吞吐均提升 >30%

- 硬件平台是什么，配置是什么。
  - Qwen2-57B-A14B-Instruct：4× NVIDIA A6000 GPUs
  - Deepseek-Lite-Chat：2× NVIDIA A6000 GPUs
  - 论文未明确说明 CPU、内存、互联等配置细节

- 模型是什么。数据集和bench分别是什么。
  - **模型**：
    - **Qwen2-57B-A14B-Instruct**：fine-grained MoE，64 experts/layer，14B 激活参数
    - **Deepseek-Lite-Chat**：fine-grained MoE，64 experts/layer（附录 Table 2 含 Deepseek-v2，160 experts/layer，用于内存分析但未用于 benchmark）
  - **数据集/Benchmark**：
    - **XSum**：摘要生成，评估 ROUGE
    - **GSM8K**：数学推理，评估准确率
    - **TruthfulQA-Gen**：真实性评估
    - **IFEval**：指令遵循评估
  - Benchmark 实验：最大 batch size 256（Qwen2）/ 200（Deepseek-Lite），测量 decoding 阶段的 latency 和 throughput

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - **未开源**。论文 Checklist 标注 "IFMoE is still under develop with future features"。
  - **伪代码**（直接引用论文 Algorithm 1）：
    ```
    Input: α, encode_topk Ek, decode_topk Dk,
           fine-grained MoE model M
    Initialize: terminate = False, buffer = []
    while not terminate do
      for each step in α do
        buffer.append(M.decode(topk = Dk))   # Draft: 仅激活 Dk=2 experts
      end for
      # Revise KV Cache
      M.encode(buffer, topk = Ek)             # Verification: 全量 Ek=6 experts 重算 KV
      terminate = detect_terminate()
      buffer = []
    end while
    ```
  - **张量计算流程**：
    1. **Draft Decode Step**（per token）：
       - Attention: Q,K,V = W_Q·x, W_K·x, W_V·x → softmax(QK^T/√d)·V
       - Router: g_i = softmax(W_r·x), select top-Dk=2 experts
       - GroupedGEMM (Cutlass): 对选中的 2 个 expert 并行做 y_i = W_i^up·x → σ → W_i^down·σ(W_i^gate·x)
       - Combine: y = Σ g_i · y_i（仅 2 个 expert 的结果加权和）
    2. **KV-cache Revision**（per α steps）：
       - 对 buffer 中 α 个 token，重新过 Router: select top-Ek=6 experts
       - GroupedGEMM (Cutlass): 对选中的 6 个 expert 并行计算
       - 用新计算的 key/value 覆盖 KV-cache 中对应位置
       - Attention 层在下一次 decode 时自动读取修正后的 KV-cache
