## 编译期 lifetime 分析与内存分配（lifetime-based memory allocation：tiling + fusion + use_cnt 注解）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 深度学习编译器后端（XLA、TVM、TFLite 等）把模型 lower 成 primitive 算子后，前端做 memory tiling、operator fusion、lifetime 分析，后端基于 IR 把 tensor/tile 映射到片上 SRAM 的连续地址区，并按 lifetime 重叠复用内存。SMOOTH（ISCA'26）把该机制作为 baseline 批判对象：它是静态的（编译期决定）、粗粒度的（tensor/tile 级、tile 常达几十到几百 KB）、且依赖 contiguity（必须连续放置）。三大约束导致碎片化（融合算子拉长中间 buffer lifetime、tile 尺寸跨层不一留下锯齿空洞）、无法适应运行期波动（KV cache 大小、序列长度、统一内存下带宽争用）、无法利用短时带宽窗口（预取要求"带宽空闲+有足够时间取完整连续 tile+存在足够大连续空闲区"三条件同时满足）。
- SMOOTH 的编译器侧角色：编译器仍做静态 lifetime 分析，但输出从"连续物理分配"降级为"标注每个 tile 的剩余使用次数 use_cnt"（连同各算子的融合信息），实际的 block 级分配、回收、预取全部交给硬件 Dynamic Memory Controller（DMC）——这是编译器-硬件协同（co-design）：编译器提供未来数据流/生命期的可见性（硬件 cache 没有的），硬件提供运行期灵活性（静态编译器没有的）。
从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 编译流程例子：模型图 → 前端优化（QKV 投影融合、FlashAttention 融合、FFN 融合 + tiling + lifetime 分析）→ IR（含算子/tensor 元数据）→ 后端内存分配。Compiler-Ideal baseline：best-fit 分配 + 全图 liveness 分析 + 对每层每算子穷举 512B–4MB tile size 取最小延迟，但仍保持连续分配约束——4K 序列下因碎片化 stall 比字节级 Optimal 多 32.7%。SMOOTH 版本：后端不再输出物理地址，而是为每个 tile 输出 use_cnt（如权重 tile use_cnt=2、V cache 块用完即释放）与 block 对齐信息，供 DMC 做 block 级虚拟化分配与 early reclamation。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：编译器侧（SMOOTH 论文用 XLA/TVM 式静态分析抽象建模，未提供独立编译器实现，论文未明确说明具体编译器代码）；硬件侧 Verilog（block table 的 use_cnt 字段由编译器填充）。评估中 Compiler-Ideal/Capuchin/Gemmini/SMOOTH-Base/SMOOTH-ER 五种策略都在 LLMCompass（https://github.com/PrincetonUniversity/LLMCompass）中实现，模型均采用三类融合。开源：https://github.com/skkim-caslab/SMOOTH（AE 脚本 src/policies/run_all_policies.sh）。

涉及论文标题：
- SMOOTH: Hardware-Assisted Fine-Grained On-Chip Memory Management for Efficient On-Device LLM Inference
