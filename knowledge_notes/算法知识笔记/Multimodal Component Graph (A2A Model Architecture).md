## Multimodal Component Graph (A2A Model Architecture)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Multimodal Component Graph 是 A2A 模型的计算结构抽象——一个由异构 component 节点和有向数据依赖边组成的有向无环图（DAG）。每个节点代表一个处理特定模态的模型组件（如 Vision Encoder, Thinker LLM, Talker LLM, Vocoder），每条边代表组件间的数据流（如 encoder 输出的 embedding 流入 LLM）。不同 request type 沿不同路径遍历该图——每条路径对应一种输入/输出模态组合。在 Cornfigurator 中，Model Definition 就是这个 component graph，作为 planner 的输入之一。Cornfigurator 支持将某些边标记为 colocatable（对应的两个 component 可被 MERGE 到同一 executor 或 KEEP 分离），planner 枚举所有 colocatable edge 的 Keep/Merge 组合来探索部署策略空间。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Component graph 的抽象和 request type 路径映射：

```
Model Definition Graph G = (C, E):
  C = {E_img, E_vid, E_aud, L_th, L_ta, G_aud}
  E = {(E_img→L_th), (E_vid→L_th), (E_aud→L_th), (L_th→L_ta), (L_ta→G_aud)}
  Colocatable edges E_c: 论文未明确说明完整 E_c，runtime 确定哪些边可 colocate

Request Type → Subgraph 映射:
  f: (input_modalities, output_modality) → path ⊆ G
  例: f(T+I, T) = [E_img, L_th]
      f(T+I+V, A) = [E_img, E_vid, L_th, L_ta, G_aud]

Component 计算量 (以 Qwen 3 Omni on A100 为例):
  E_img: 5.43 req/s   (较快的视觉编码)
  E_vid: 2.93 req/s   (视频编码，多帧处理)
  E_aud: 21.43 req/s  (音频编码最快)
  L_th:  2.15 req/s   (thinker LLM, 自回归)
  L_ta:  0.12 req/s   (talker LLM, 最慢——生成 audio tokens)
  G_aud: 0.12 req/s   (vocoder, token→waveform)

Planner 枚举: 对每条 e∈E_c 选择 KEEP 或 MERGE
  Fully disaggregated: all KEEP → 6 nodes, 每个可独立配置
  Monolithic: all MERGE → 1 node, 所有 component 共享配置
  Cornfigurator 最优: 部分 KEEP, 部分 MERGE — 例 audio encoder 分离,
    thinker+encoder colocated, talker+vocoder colocated
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Component graph 的 formalization 是 Cornfigurator 能够处理通用 A2A 模型（而非仅 MLLM 或 Diffusion 等特例）的关键——只要将模型表达为 component DAG + colocatable edges，planner 就能自动搜索部署方案。对于实际 A2A 模型，component 数量通常 ≤ 10（Qwen Omni 有 6 个），使得枚举空间可管理。Graph 定义需包含：节点（component 名称和类型）、边（数据依赖）、colocatable edges 标记、以及每个节点支持的 executor types。论文未明确说明 graph definition 的具体格式/API。

涉及论文标题：
- Cornserve Efficiently Serving Any-to-Any Multimodal Models
- EPD-Serve A Flexible Multimodal EPD Disaggregation Inference Serving System On Ascend
