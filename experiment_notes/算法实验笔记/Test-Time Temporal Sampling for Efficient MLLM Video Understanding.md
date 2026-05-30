## Test-Time Temporal Sampling for Efficient MLLM Video Understanding

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：T3S 是训练无关、即插即用的推理包装器，在推理时对视频进行多试次随机帧采样（multi-trial frame sampling）和 token 子采样（token subsampling），将 m 个短子序列打包到一个前向传播中处理，最后通过 logit 聚合（均值、置信度加权或双试次交叉验证）输出预测。
  - 实验比较：对比 baseline（无采样的单序列 MLLM 推理）和同类训练无关方法 FastV、VTW、AdaReTake，在 VideoMME、LongVideoBench、MLVU 三个长视频理解 benchmark 上评估准确率和加速比。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明具体 GPU 型号。论文 4.5 节提到"on a single GPU"进行实验，多 GPU 作为未来扩展方向提及。

- 模型是什么。数据集和 bench 分别是什么。
  - 模型：Qwen2.5-VL-7B、LLaVA-Video-7B、Oryx-1.5-7B（均为开源 7B 级 MLLM）。
  - 数据集/Benchmark：VideoMME（900 视频/2700 QA）、LongVideoBench（3763 视频/6678 多选题）、MLVU（多域长视频理解，M-Avg 指标）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/kaibinwang3/T3S
  - 算法 pipeline 解释：
  
  **核心思想**：视频存在大量时空冗余，传统 MLLM 对每帧每个 token 均完整编码并送入 self-attention（O(L²)）。T3S 将单长序列替换为 m 个短且多样化的子序列，通过随机采样的统计覆盖性弥补丢弃 token 的信息损失，同时将 attention 复杂度降为 O(∑αᵢ²L²)。
  
  **伪代码**（对应 Algorithm 1）：
  ```
  输入: 视频 V (共 F 帧), 文本tokens t
  参数: N (每试次帧数), m (试次数), αᵢ (每试次token保留率), k (top-k值)
  输出: 下一个输出 token t*
  
  for i = 1 to m:                          // Stage 1&2: 多试次采样
      P_i = RandomSample({1,...,F}, N)      // 从F帧中随机选N帧
      V̂_i = V[P_i]                          // 提取子序列帧
      v^(i) = E_v(V̂_i)                      // 视觉编码器编码, |v^(i)| = L = N×M
      v̂^(i) = C(v^(i), αᵢ)                  // token子采样, |v̂^(i)| = ⌊αᵢL⌋
                                             // C默认为均匀随机patch采样
  end for
  
  // Stage 3: 打包推理 + 聚合
  {o₁,...,oₘ} = MLLM(<v̂^(1),t> || ... || <v̂^(m),t>)  // 单次前向，带块对角线attention mask
  若 m=2 使用交叉验证: K = TopK(o₁, k); t* = argmax_{t∈K} o₂[t]
  否则: o_avg = (1/m) Σ oᵢ; t* = argmax o_avg[t]
  ```

  **张量级计算**：
  - Baseline: 输入 [L, D] 的视觉 token 序列，self-attention 计算 QK^T 矩阵 [L, L]，复杂度 O(L²D)。
  - T3S (m=2, α₁=0.5, α₂=0.3): 输入两个子序列，长度分别为 0.5L 和 0.3L，总长度 0.8L。Packed 序列中 self-attention 使用块对角线 mask，每个子序列仅与自身计算 attention。总复杂度 O((0.5²+0.3²)L²D) = O(0.34 L²D)，理论节省 66%。
  - 实际加速：Qwen2.5-VL-7B 上约 2.0× 加速，因为 packing 的序列长度总和（0.8L）短于原始 L，且每个 attention 块更小。
  
  **Logit 聚合**（第 3.5 节公式）：
  - 均值聚合: o_avg = (1/m) Σ oᵢ, oᵢ ∈ R^D（D=词表大小）
  - 置信度加权: wᵢ ∝ 1/H(π(oᵢ)), o_weighted = Σ wᵢ oᵢ（H 为预测分布熵）
  - 双试次交叉验证 (m=2): 试次 1 提出 top-k 候选 K，试次 2 在 K 上重新排序

  **关键超参**：m=2, N=256 (Qwen/Oryx) 或 128 (LLaVA-Video), α₁=0.5, α₂=0.3, topk k=2。评估工具为 VLMEvalKit。
