## MLLM 推理四阶段 pipeline（Preprocessor / Vision Encoder / Merger / LLM）与 encoder-decoder 资源不对称（RESONATOR 视角）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MLLM（Multimodal Large Language Model，如 Qwen2-VL、Kimi-VL、GPT-4o）采用模块化设计：模态专用编码器（图像 ViT / 音频 encoder）产出 token 序列，再由文本中心 decoder 生成回复。RESONATOR 把 MLLM 推理 pipeline 拆为四阶段（Figure 2）：①Preprocessor（CPU 密集，resize/归一化/把图像按动态分辨率切成均匀 tile 或 patch，tile 数随分辨率变化）；②Vision Encoder（ViT，把图像 tile 转成高维 raw visual tokens n_raw，序列长随分辨率增长）；③Merger/Projector（轻量模块，把长 n_raw 压成短 n_final，n_final≪n_raw，降低 LLM prefill 的二次复杂度，但不减轻 encoder 处理全部 n_raw 的负担）；④LLM（处理文本+压缩后视觉 token 的拼接序列，先 prefill 算 KV cache，再自回归 decode）。关键系统含义是 encoder 与 decoder 的资源不对称：encoder 显存占用小（Qwen2-VL 的 ViT-675M 权重 FP16 仅 1.3GB、Kimi-VL 的 MoonViT-400M 仅 0.8GB）但计算随输入分辨率近似二次增长；decoder 则 memory 与 compute 双密集。此不对称是 RESONATOR 调度设计的动机来源。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RESONATOR 的四阶段计算流程（以 Qwen2-VL-7B 处理一张 1024×1024 图为例）：
```
# Stage 1 Preprocessor (CPU): 动态分辨率分 tile
tiles = dynamic_tile(resize(image, 1024x1024))        # 可变 tile 数
# Stage 2 Vision Encoder (GPU, ViT): 每 tile 按 patch 化
L_seq = ceil(H(r)*W(r) / P^2)                          # P=patch size(14)，唯一复杂度参数
raw_tokens = ViT_encoder(tiles)                        # n_raw 个 visual tokens，自注意力 ~O(L_seq^2)
# Stage 3 Merger/Projector (轻量): 压缩视觉 token
final_tokens = merger(raw_tokens)                      # n_final << n_raw
# Stage 4 LLM: prefill + decode
kv = prefill(concat(text_tokens, final_tokens))        # 算 KV cache
output = decode_loop(kv)                               # 自回归逐 token 生成
```
Annotations：L_seq 由图像高宽 H(r)/W(r) 与 ViT patch size P 决定（RESONATOR 的 encoder 性能模型只以 L_seq 为复杂度参数，因此可泛化到任意分辨率/宽高比）；encoder 阶段对高分辨率图是 prefill 关键路径的 dominant 瓶颈（Figure 3 端到端 breakdown 显示 encoder 延迟占比随分辨率上升）；Merger 虽缩短 LLM 输入，但 encoder 仍须处理全部 n_raw token。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
四阶段是主流开源 MLLM 的标准推理结构，serving 框架（SGLang/vLLM）原生把 encoder 作为预处理步骤挂在 LLM 之前；实现细节由模型仓库给出（Qwen2-VL 的 dynamic resolution + ViT-675M，Kimi-VL 的 MoonViT）。RESONATOR 的关键改法：不再把 encoder 当作固定预处理步骤，而是视为一等公民动态负载——在 Serving 层（SGLang-0.4.7）解耦 encoder 与 decoder，再用 Intra-GPU 共享引擎 + Inter-GPU 动态并行把两者重新耦合。评估模型：Qwen2-VL-7B/72B（ViT-675M）、Kimi-VL-16B（MoonViT-400M）；数据集 MMMUPro、TextVQA。

涉及论文标题：
- Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs
