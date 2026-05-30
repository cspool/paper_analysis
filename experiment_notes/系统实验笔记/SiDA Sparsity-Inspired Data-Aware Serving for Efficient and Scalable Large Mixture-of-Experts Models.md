## SiDA Sparsity-Inspired Data-Aware Serving for Efficient and Scalable Large Mixture-of-Experts Models

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：SiDA-MoE 是一个数据感知的 MoE 推理服务系统，在 HuggingFace Transformers 的 Switch Transformer 实现之上构建。核心架构包含两个并行线程：(1) **Hash-building 线程**——用离线训练的 LSTM + Sparse Attention hash 函数预先预测每批输入在各 MoE 层的 expert 激活模式，写入 hash table 队列；(2) **Inference 线程**——根据 hash table 将激活的 expert 动态加载到 GPU，将未激活的 expert 卸载到 CPU 主内存（FIFO 策略），并用 SiDA-MoE 特化层执行前向推理。所有 router 函数被卸载到主内存，不参与前向过程。两条线程通过管道并行机制运行，使得 expert 选择、动态 offloading 和推理完全并行化。
  - 实验比较：(a) GPU 内存节省：SiDA-MoE vs 原始 Switch Transformer 在不同数据集（SST2/MRPC/MultiRC）和模型规模（8/64/128/256 experts）下的GPU内存减少比例；(b) 吞吐量与延迟：SiDA-MoE vs Standard vs DeepSpeed vs Tutel 在四条模型上的吞吐量和延迟对比；(c) 有限 GPU 内存预算下的效率：不同 GPU 内存预算、不同 offloading 方案下的吞吐量对比；(d) 保真度分析：SiDA-MoE 相对于 fine-tuned Switch Transformer 的性能保持率（SST2 accuracy / MRPC F1 / MultiRC F1）；(e) Hash 命中率：Top-3 expert 预测准确率；(f) 困惑度：SiDA-MoE 替代 router 后预训练模型的 perplexity 退化。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA A100 80GB（单卡）
  - CPU：64 Intel(R) Xeon(R) Platinum 8358 @ 2.60GHz
  - 软件栈：HuggingFace Transformers, PyTorch, CUDA

- 开源Serving框架是什么。修改了什么。
  - 基础框架：HuggingFace Transformers（基于其 Switch Transformer 实现）
  - 修改内容：(1) 新增 Hash-building 线程——实现 LSTM+sparse attention hash 函数的离线训练和前向预测，构建 expert 激活 hash table；(2) 新增 Inference 线程——替换原始 MoE 前向流程，插入 expert 动态加载/卸载逻辑（GPU ⇄ CPU 主内存），基于 FIFO 的 expert 驱逐策略；(3) SiDA-MoE Manager——协调主推理线程和预测线程间的调度，通过共享队列同步 hash table，管理 expert 设备置放和 GPU-CPU 数据传输；(4) 双线程管道并行——推理线程处理当前 batch 时，hash-building 线程并发预测下一 batch 的 expert 激活模式。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：https://github.com/timlee0212/SiDA-MoE
  - 框架输入→硬件执行全过程（单个推理 batch 的处理流程）：
    ```
    输入：token序列 batch X_i = [seq_len×1]（batch_size=1）
    
    1. Hash-building 线程（与推理并行运行，处理下一批 X_{i+1}）：
       X_{i+1} → token embedding → LSTM层（2层）→ Sparse Attention层（SparseMax激活）
       → FC层（维度压缩）→ Residual连接 → 最终FC层 → top-k expert选择
       → hash table H_{i+1}[layer][token] = {activated_expert_ids, scaling_factors α}
       → 推入 Shared Queue
    
    2. Inference 线程（处理当前批 X_i）：
       a) 从 Shared Queue 取出 H_i（等待 hash-building 线程完成）
       b) 对每个 MoE layer l：
          - 扫描 H_i[l][:] 获取本层激活的 expert id 集合
          - 对本批激活的 expert：
            if expert not on GPU: CPU→GPU 加载 expert 参数（θ_i）
          - 对本批未激活的 expert：
            if expert on GPU and GPU memory budget exceeded: GPU→CPU 卸载（FIFO策略）
       c) 前向传播（SiDA-MoE 特化层）：
          - Self-Attention: Q/K/V projection → Attention → output（不变）
          - MoE层: 跳过 router（已卸载到CPU），直接根据 H_i[l][token] 的
            (expert_id, α) 调用对应 expert MLP → α 加权求和 → 输出
          - 每层完成后立即触发下一层的 expert 加载/卸载（管道并行）
       d) 输出 logits
       
    3. 硬件执行映射：
       - Hash-building 在 CPU 上运行（使用 PyTorch CPU 推理）
       - 主要的 Transformer 推理在 A100 GPU 上执行
       - Expert 参数传输：CPU 主内存 (DDR4) → PCIe → GPU HBM（A100 80GB）
       - 卸载方向：GPU HBM → PCIe → CPU 主内存
       - 未激活 expert 存放在 CPU 主内存中（可达 TB 级）
    ```
