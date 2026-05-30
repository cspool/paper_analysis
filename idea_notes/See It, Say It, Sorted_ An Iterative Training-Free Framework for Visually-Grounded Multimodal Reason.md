## See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs

- baseline方法是什么？
  Baseline 是标准的 LVLM greedy decoding（自回归逐 token 生成）。在长链 CoT 推理中，base LVLM 在每一步根据 p_VLM(x_i | x_{<i}) 选择 top-1 token。由于随着上下文增长，语言先验逐渐压倒视觉线索，单步幻觉 token 会级联传播至后续步骤，最终导致错误答案（Fig. 1(a)）。
  
  Baseline 全栈执行例子（以 Qwen2.5-VL-7B 回答 TreeBench 视觉推理问题为例）：
  - 算法层：自回归 greedy decoding。LVLM 编码图像和文本指令后，逐 token 解码 CoT 推理链，每步选择 p_i 中概率最高的 token。无外部监督，无证据注入。若某步模型将颜色"blue"错选为"red"，后续的定位、描述和最终判断全部基于错误前提。
  - 系统框架层：PyTorch + HuggingFace Transformers。标准 generate() 调用，无额外 decoding wrapper。单次 forward pass 输出 logits → argmax → 追加到 prefix。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 FlashAttention 实现 attention 计算。Decoding 阶段每步 attention 为增量计算（仅计算新增 token 的 Q 与历史 KV）。
  - 硬件架构层：单张 H20-NVLink GPU。Greedy decoding 下 latency 由 t_0 描述（V*Bench 8.98s、MathVista 12.92s 等，per question），visual decider 调用次数 r=0（δ=0）。

  问题：greedy decoding 在 hallucination-prone 步无防御机制。一旦某个中间 token 偏离视觉事实，后续逻辑推理——即使形式正确——也全部基于错误前提，导致 cascading failure。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法 ECRD（Evidence-Constrained Reweighting Decoding）通过在 test time 注入视觉证据来监督每一步的 token 选择，解决 greedy decoding 中"单步幻觉→全链崩溃"的问题。不像 RL-based 方法（DeepEyes、Pixel-Reasoner）需要在训练时学习"何时看图"，ECRD 在推理时按需获取视觉证据。

  核心设计映射到 baseline 缺陷：
  (1) **分布监督器替代 greedy argmax**：不再直接取 top-1，而是先 knee truncation 选出候选集 C_i，再用证据池中的文本证据计算证据诱导分布 r_i(w)，最终与 base 分布 p_i(w) 通过自适应权重 α_i = p_{(1)} 混合。当 base 分布尖锐时（α→1），保持 base 主导；当 base 分布平坦时（α 小），证据获得更多权重。—— 解决了"自信但有偏差的 token 选择"问题。
  (2) **不确定性触发的 visual decider**：当 k*>1 且混合分布 margin Δ_i ≤ δ 时，调用 GRIT-3B 读图并输出微观察证据句。证据句强制提交正确 token + 追加到证据池供后续步骤复用。—— 解决了"关键歧义步无外部仲裁"的问题。
  (3) **文本证据池累积与复用**：证据以文本形式存储（非像素），可在后续步骤中被 supervisor 的 scoring 函数直接参考（式 5-7），无需反复编码图像。—— 解决了 RL-based 方法中"每次看图需重新编码裁剪区"的效率问题。

  ECRD 全栈执行例子（以 TreeBench 问题"直接位于 favorita 品牌香蕉纸箱后面的物体是什么？"为例）：
  - 算法层：
    ```
    Step i（关键歧义步）：
    C_i = {"5", "3"}  # knee truncation 选出候选
    base: p("5")=0.498, p("3")=0.483
    evidence pool 评分 → evidence-induced: r("5")=0.503, r("3")=0.478
    alpha = p_{(1)} ≈ 0.498（base 不自信，alpha 小）
    p_mix: ("5", 0.501), ("3", 0.480)
    margin = 0.021 ≤ δ=0.08 → 触发 decider
    GRIT 读图 + 当前 prefix → w*="3", 
      E_i="The number behind the cardboard box with the 'favorita' brand and banana illustration is '300'."
    强制选 "3"，证据句追加到池
    Step i+1: evidence pool 含上述证据句
      supervisor 评分 → "0" 获得证据支持 → 选 "0"
    Step i+2: 同上 → 选 "0"
    最终答案: "300"（正确）vs greedy 选 "5" 导致 "5XX"（错误）
    ```
  - 系统框架层：PyTorch + HuggingFace Transformers。ECRD 作为 decoding wrapper 包裹 frozen LVLM，不修改模型权重。Visual decider（GRIT-3B）独立部署在另一 backend（FP16 on CPU），仅在触发时调用。证据 scoring 的计算复杂度 O(k*|E_i|)，k* 为个位数，|E_i| 增长缓慢，GPU 压力可忽略。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 FlashAttention + 推理框架自带的 kernel。Evidence scoring 在 CPU 上完成（precomputed log-likelihoods），不占用 GPU compute。
  - 硬件架构层：单张 H20-NVLink GPU。ECRD 的 overhead 来自两部分：(i) 证据评分 —— O(k*|E_i|)，<0.1s per step；(ii) visual decider 调用 —— l_0 ≈ 1.12-1.46s/call，δ=0.08 时每问题平均调用 r 在低个位数。总 latency T(0.08) ≈ t_0 + l_0·r ≈ 10-15s，相比 t_0(≈9-13s) overhead 控制在 20-30%，而 accuracy 提升 4.5-10.9 个点。

  关键差异对比：
  | 维度 | Baseline (Greedy) | ECRD (Ours) |
  |------|------------------|-------------|
  | Token 选择 | argmax(p_i) | 协商混合 p_i + r_i，自适应 α |
  | 视觉监督 | 无（仅初始编码一次图像） | 证据池持续评分 + 按需 decider 注入微观察 |
  | 幻觉处理 | 无防御，单步错→全链错 | margin 检测歧义步，decider 仲裁 |
  | 训练需求 | 无（但性能差） | 无（training-free，frozen models） |
  | 证据形式 | 无 | 文本（可复用，无需重编码图像） |
  | Cost | 最低（单次 forward/step） | 少量 overhead（证据评分 + 按需 decider） |
