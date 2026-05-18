## Language Processing Unit (LPU) / 语言处理单元

术语是什么？
Language Processing Unit (LPU) 是专为大语言模型（LLM）推理设计的专用处理器类别。与通用GPU/NPU不同，LPU针对LLM推理的特定计算模式（transformer attention、FFN、token生成）进行了架构级专用化。已商业化的LPU包括Groq LPU（基于Tensor Streaming Processor，片上230MB SRAM存储权重，确定性调度编译）、Cerebras WSE（晶圆级引擎，权重预加载至片上SRAM）、Etched Sohu（将transformer数据流硬化到计算fabric中但未硬化权重）。HNLPU论文将LPU的专用化推到极致——将权重参数物理固化到芯片金属导线中（Hardwired LPU），实现了5,555×/1,047×于H100的吞吐/能效。

从硬件架构角度拆解：
LPU的硬件组成和运转流程因具体设计而异。HNLPU作为案例：系统由16芯片经4×4 CXL 3.0全连接fabric互联组成。每芯片五大模块：(1) HN Array（硬连线权重计算，占69.3%面积但仅24.9%功耗——MoE稀疏激活，仅4/128 experts活跃）；(2) VEX Unit（向量执行：FlashAttention、RMSNorm、SwiGLU、softmax、residual add、multinomial sampling，32 cached KV-heads/cycle）；(3) Attention Buffer（320MB片上SRAM KV Cache，20,000 banks×16KB, 1W1R, 32-bit, 80TB/s带宽）；(4) Control Unit（硬件Continuous Batching调度，6级×36层=216最大batch）；(5) Interconnect Engine（CXL 3.0, <100ns延迟, 128GB/s per ×16 link）。流程：接收token ID→HBM查embedding→36层transformer（每层经HN Array计算QKV/FFN weight、VEX计算attention/nonlinear、Interconnect Engine全归约）→Unembedding→Sampling→输出token ID。

术语一般如何实现？如何使用？
商用LPU（Groq）使用片上大容量SRAM（230MB）+编译器确定性调度的方案避免HBM访问。HNLPU使用Metal-Embedding将权重直接固化在金属层，实现零权重fetch。LPU通常用于推理即服务（Inference-as-a-Service）场景：用户通过API发送prompt→LPU系统生成token流。与GPU的关键差异：LPU牺牲了训练能力和模型灵活性，换取推理的极致吞吐和能效。

涉及论文标题：
- Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates

