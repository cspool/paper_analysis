## Taylor 展开（Taylor Expansion / Taylor 级数）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Taylor 展开把函数在展开点附近表示成多项式：f(x)=Σ (f^(n)(a)/n!)(x−a)^n + 余项。用于硬件非线性函数逼近时（如指数 e^x、log、sin、cos），用有限阶多项式近似，精度只在展开点附近高、远离时急剧下降，且高阶需要多次乘加（MAD）运算（6 阶 exp 至少 6 个 MAD [4]）。在 LoRA 论文中是基线方案（PICACHU [56] 与 NX-CGRA 等）。
- LoRA 中 Taylor 的作用与对比：作为 baseline 方法——PICACHU 用 Taylor 展开逼近 exp/log/sin/cos，配合 FP2FX（浮点转定点）模块与算子融合（把 Taylor 的 MAD 融合进单个 PE）；DCT 端到端精度对比显示需 ≥4 阶 Taylor 才能匹敌 LoRA（Table VIII：PICACHU-3rd 的 MSE/PSNR 明显更差，4th/5th 才持平），故后续评估取 4 阶起。Taylor 的痛点（LoRA 动机）：高阶→更多 MAD→更多 PE 占用、限制 loop unrolling、只支持有限函数子集、复合函数更难处理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算过程（PICACHU 的 exp 逼近）：输入 x → FP2FX 把浮点 x 转定点 → 泰勒多项式 Σ x^n/n!（如 4 阶：1+x+x²/2+x³/6+x⁴/24）→ 每个 MAD 一个 PE 节点 → 结果。6 阶 exp 需 ≥6 个 MAD；精度提升→阶数提高→MAD 数线性增加→PE 占用增大。
- 与 Chebyshev 对比：Taylor 在展开点附近精度高、远离差（输入范围受限）；Chebyshev 在整个区间上极小化最大误差、同 sup-norm 误差下次数更低、避免 Runge 振荡、数值条件更好（B.-Chebyshev-Polynomials）。例：PICACHU-4th 在 DCT 上与 LoRA 持平，而 LoRA 用 6 项 Chebyshev 多项式即可。
- 其它用途（vault 中广泛出现）：量化误差建模（用二阶 Taylor 展开损失函数扰动：L(W_Q)≈L(W)−g^T(W−W_Q)+½(W−W_Q)^T H(W−W_Q)，SqueezeLLM/APHQ-ViT/GuidedQuant 等）、重要性/敏感度估计（一阶 Taylor：Saliency=gradient·weight）、softmax 线性化（ViTALiTy 一阶 Taylor）、指数硬件实现（Taylor series [44]）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：硬件上把 Taylor 系数（阶乘倒数）固化为常数，用 Horner 规则（嵌套乘加，Horner's rule [33]）减少乘加数；PICACHU 用专用 MAD 融合模块在一个 PE 内算完；LoRA 的 XCore 则用 LNS 把每项 c_ix^(k_i) 变成 2^(log2 c_i + k_i·log2 x)，6 项多项式只需一个可编程 30b 乘法器（5×30b×6b），硬件开销远低于 6 个 MAD。
- 使用场景：作为非线性函数硬件的经典基线（CORDIC/LUT/Taylor 三类的多项式类），LoRA 论文在单元级与端到端（DCT/DNN/LLM）都把它作为对比；在模型量化/剪枝领域作为误差分析的数学工具广泛应用。局限：展开点局部性、阶数-硬件开销线性关系、输入范围受限（Taylor series accuracy is high only near the expansion point）。

涉及论文标题：
- LoRA: Towards Improved Applicability of Reconfigurable Architecture for Versatile Nonlinear Functions
