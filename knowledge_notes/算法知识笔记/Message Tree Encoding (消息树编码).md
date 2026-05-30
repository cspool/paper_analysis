## Message Tree Encoding (消息树编码)

术语解释
Molmo2提出的多模态训练数据编码策略，允许同一视觉输入携带多个annotations（QA pairs、caption、pointing等）并打包进同一training sequence，通过custom attention mask防止不同annotation分支之间的cross-attention污染。平均每个visual input有4 annotations，与packing结合实现~15x训练效率提升。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
传统做法将每个annotation作为独立training example，同视觉tokens被重复编码多次。Message Tree Encoding以visual input tokens为root，每个(Q,A) pair为独立branch。Linearization: visual tokens出现一次→各branch顺序拼接。Custom attention mask: (1) branch之间block（防止QA A的answer泄露到QA B）；(2) 每个branch可见full visual tokens；(3) visual tokens内部bidirectional。与packing协同：message tree产生的merged sequence作为packing DP solver的输入unit。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# Sequence: [BOS][video][Q1][A1][Q2][A2]
#                   |← root →|← branch1 →|← branch2 →|
# Mask:  root×root→bidir; branch_i×branch_i→causal; branch_i×branch_j→BLOCK
# All branches can see root (full vision context)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现于Molmo2 training code (`message_tree.py`)。Custom mask集成入PyTorch SDPA。适用场景：同一视频/图像有多个训练annotations的VLM SFT训练。Molmo2开源代码在 https://github.com/allenai/molmo2。

涉及论文标题：
- Molmo2__Open_Weights_and_Data_for_Vision-Language_Models_with_Video_Understanding_and_Grounding
