# LLM架构（Transformer、KV Cache、Flash Attention、vLLM）

ref：[https://zhuanlan.zhihu.com/p/642923989](https://zhuanlan.zhihu.com/p/642923989)

ref：[https://blog.csdn.net/m0_63855028/article/details/146239674](https://blog.csdn.net/m0_63855028/article/details/146239674)

ref：[https://www.modb.pro/db/1800545561119641600](https://www.modb.pro/db/1800545561119641600)

ref：[https://cloud.tencent.com/developer/article/2547435](https://cloud.tencent.com/developer/article/2547435)

ref：[https://www.zhihu.com/question/636301026](https://www.zhihu.com/question/636301026)

ref：[https://zhuanlan.zhihu.com/p/682352630](https://zhuanlan.zhihu.com/p/682352630)

# Transformer

encoder block的attention不需要mask，因为需要学习prompt和输入token序列中的所有关系；

decoder block中mask attention layer用于剪除token去query其后续token的关系，以保证输出的“逻辑性”，mask具体表现在矩阵中上三角为0（下到上、左到右是过去到未来的token）；

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image.png)

> **[图片提取文字 (image.png)]:**
> 在 Transformer 模型<sup>†</sup>中,**编码器负责理解和提取输入文本中的相关信息**。这个过程通常涉及到处理文本的序列化形式,例
> 
> 如单词或字符,并且用**自注意力机制**<sup>†</sup> (Self-Attention) 来理解文本中的上下文关系。
> 
> 编码器的输出是输入文本的连续表示,通常称为嵌入<sup>\*</sup> (Embedding)。这种嵌入包含了编码器从文本中提取的所
> 
> 有有用信息,并以一种可以被模型处理的格式(通常是高维向量)表示。
> 
> 这个嵌入然后被传递给解码器。**解码器的任务是根据从编码器** 
> 
> 接收到的嵌入来生成翻译后的文本(目标语言)。解码器也使用自注意力机制,以及编码器-解码器注意力机制,来生成翻
> 
> 译的文本。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%201.png)

# LLM架构

> **[图片提取文字 (image.png)]:**
> LLMs中有的是只有编码器encoder-only,有的只有解码器 decoder-only,有的是2者混合 encoder decoder hybrid。 三者都属于Seq2Seq<sup>†</sup>, sequence to sequence。
> 
> 并且字面意思是虽只有编码器encoder,实际上LLMs是能decoder一些文本和token的,也算是decoder。不过由于encoder-only类型的LLM不像decoder-only和encoderdecoder那些有自回归autoregressive,encoder-only集中于理解输入的内容,并做针对特定任务的输出
> 
> 自回归指输出的内容是根据已生成的token做上下文理解后一个token一个token输出的。
> 
> 总的来说,encoder-only类型的更擅长做分类;encoder-decoder类型的擅长输出强烈依赖输入的,比如翻译和文本总结,而其他类型的就用decoder-only,如各种Q&A。虽然encoder-only没有decoder-only类型的流行,但也经常用于模型预训练
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%202.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%203.png)

**自回归**的数学模型是序列联合概率是条件概率的链式乘法，在Transformer构建中的体现是Transformer逐个token生成（多轮inference pipeline），pipeline中transformer block的堆叠**；**

> **[图片提取文字 (image.png)]:**
> • 数学定义:
> 
> 对于序列  $\mathbf{y}=(y_1,y_2,...,y_T)$ ,自回归模型将联合概率分解为条件概率的链式乘积:
> 
> $$P(\mathbf{y}) = P(y_1) \cdot P(y_2 \mid y_1) \cdot P(y_3 \mid y_1, y_2) \cdot ... \cdot P(y_T \mid y_1, y_2, ..., y_{T-1}).$$
> 
> 每个步骤的预测仅基于历史信息,避免未来信息的泄漏。
> 
> - · **例**:生成句子时,逐词预测,每次将新生成的词加入上下文。
> - 生成过程:
> - **初始化**:輸入起始符(如 **<sos>**);
> - $\circ$  **迭代**:每次生成一个元素  $y_t$ ,并将其作为下一步的输入;
> - **终止**: 生成结束符(如 **<eos>** ) 时停止。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%204.png)

> **[图片提取文字 (image.png)]:**
> # 2. 自回归模型的典型结构
> 
> ## (1) 基于RNN的结构 (如LSTM、GRU)
> 
> ### 原理:
> 
> 通过循环单元维护隐状态 (Hidden State) ,逐步传递历史信息。
> 
> 隐状态 ht=f(ht-1,xt), 其中 xt 是当前输入(通常是上一步的输出 yt-1)。
> 
> ### 缺点:
> 
> 串行计算导致训练和推理速度慢,长距离依赖易丢失。
> 
> ## (2) 基于Transformer的结构 (如GPT、T5)
> 
> #### 原理:
> 
> 使用\*\*掩码自注意力 (Masked Self-Attention) \*\*实现并行训练,同时保证自回归特性。
> 
> - 训练时,通过下三角掩码矩阵限制每个位置只能关注其左侧的上下文(如图1)。
> - 推理时,逐步生成序列,每一步将新生成的词追加到输入中。
> 
> ### 优点:
> 
> - 并行计算加速训练;
> - 长距离依赖捕捉能力强。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%205.png)

> **[图片提取文字 (image.png)]:**
> ## 3. 自回归 vs 非自回归
> 
> 高(上下文连贯性强)
> 
> GPT、Transformer解码器
> 
> 文本生成、对话系统、高精度合成任务
> 
> 生成质量
> 
> 典型模型
> 
> 适用场景
> 
> | 特性   | 自回归(AR)      | 非自回归 (NAR)    |
> |------|--------------|---------------|
> | 生成方式 | 逐位置生成,依赖历史信息 | 并行生成所有位置      |
> | 谏度   | 慢 (串行牛成)     | 快 (一步或少量步骤生成) |
> 
> 可能较低 (依赖后处理或迭代优化)
> 
> NAT (Non-Autoregressive Translation)
> 
> 实时翻译、语音合成等对速度敏感的任务
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%206.png)

> **[图片提取文字 (image.png)]:**
> ## 4. 自回归的关键技术
> 
> ## (1) 掩码机制 (Masking)
> 
> - 目标: 确保训练时模型无法"偷看"未来信息。
> - 实现:
> - $\circ$  在注意力计算中,将未来位置的注意力权重设为  $-\infty$  (Softmax后接近0)。
> - 公式:
> 
> $$\operatorname{Attention}(Q,K,V) = \operatorname{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V,$$
> 
> 其中M为下三角掩码矩阵。
> 
> CSDN ◎整点薯条吃吃喽
> 
> ## (2) 输入偏移 (Shifted Inputs)
> 
> - 目标:对齐训练与推理时的输入分布。
> - 实现:
>   - 训练时,将目标序列右移一位,并添加起始符〈sos〉。
>   - **例**: 目标序列 [A, B, C] → 输入 [⟨sos⟩, A, B], 输出 [A, B, C]。
> 
> ## (3) 解码策略
> 
> - 贪婪搜索 (Greedy Search) : 每一步选择概率最高的词,速度快但可能陷入局部最优。
> - 束搜索 (Beam Search) : 保留多个候选序列,平衡质量与计算成本。
> - 采样 (Sampling) : 按概率分布随机选择,增加多样性(可调节温度系数)。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%207.png)

**encoder-only**

> **[图片提取文字 (image.png)]:**
> ## **Encoder-only**架构的LLMs更擅长**对文本内容进行分析、分 类**,包括情感分析,命名实体识别。这里以Bert<sup>+</sup>为例子详细 说明,roBerta<sup>+</sup>是基于Bert进行了升级,比如扩大了batch
> 
> 说明,roberta 是基于Bert进行了开级,比如扩大了batch size,在更大的数据上训练,消除了Bert的next-sentence
> 
> prediction task训练方式
> 
> Bert的训练是基于next-sentence prediction task和mask
> 
> language modeling
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%208.png)

**next-sentence prediction & mask language modeling**

> **[图片提取文字 (image.png)]:**
> next-sentence prediction task是将原句子打乱成不同顺序的句子,让bert找出正确语序的原句,例如
> • [CLS] Toast is a simple yet delicious food [SEP] It's
> 
> often served with butter, jam, or honey.
> [CLS] It's often served with butter, jam, or honey.
> [SEP] Toast is a simple yet delicious food.
> 
> [CLS] token是一个占位符标记,它提示模型返回一个 True 或 False 的标签,表示这两个句子是否按照正确的顺序排列。如果句子的顺序是正确的,模型应该返回 True,如果句子的顺序被打乱,模型应该返回 False。
> 
> [SEP] token用来分割2个句子
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%209.png)

> **[图片提取文字 (image.png)]:**
> mask language modeling则是在大量的文本语料库中将数据中的某部分遮住mask,让Bert根据上下文内容来预测mask的内容。如下图把原句中15%的部分随机遮挡,遮挡的是climbed,80%时间用mask token取代,10%时间用随机token,10%时间不变
> 
> Input sentence: The curious kitten deftly climbed the bookshelf
> 
> Pick 15% of the words randomly
> 
> The curious kitten deftly climbed the bookshelf
> 
> - 80% of the time, replace with [MASK] token
>   - 10% of the time, replace with random token (e.g. ate)
>   - · 10% of the time, keep unchanged
> 
> Modified sentence: The curious kitten deftly [MASK] t铂 如始曾复测试
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2010.png)

**decoder-only**

> **[图片提取文字 (image.png)]:**
> Decoder主要是是为了预测下一个输出的内容/token是什么,并把之前输出的内容/token作为上下文学习。实际上,decoder-only模型在分析分类上也和encoder only的LLM一样有效。
> 
> Decoder-only<sup>†</sup>的decoder层跟encoder相似,不过在位置 position上用到了mask
> 
> 在 Transformer 模型的解码器中,自注意力机制允许每个位置的输出都依赖于输入序列中所有位置的信息。然而,当生成输出序列(例如在文本翻译任务中生成目标语言的文本)时,我们希望位置 *i* 的输出只依赖于位置 *i* 之前的已知输出,而不依赖于位置 *i* 之后的输出。为了实现这一点,我们使用了一种"掩码"技术,阻止模型关注位置 *i* 之后的位置。 这意味着。在进行解码器的自注意力运算时,位置 *i* 的注意力
> 
> 这意味着,在进行解码器的自注意力运算时,位置 *i* 的注意力分布(即注意力权重)只会在位置 *i* 及其之前的位置上,不会在位置 *i* 之后的位置上。这样,**我们就可以确保位置** *i* **的输出** 
> 
> ## 只依赖于位置 / 之前的已知输出。
> 
> 输出的内容也是一个token,一个token的生成,如下图
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2011.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2012.png)

应该没有encoder部分？

**encoder-decoder**

> **[图片提取文字 (image.png)]:**
> 这种架构的LLM通常充分利用了上面2种类型的优势,采用新的技术和架构调整来优化表现。这种主要用于NLP,即理解输入的内容NLU,又能处理并生成内容NLG,尤其擅长处理输入和输出序列之间**存在复杂映射关系**的任务,以及捕捉两个序列中元素之间关系至关重要的任务。以下是该类型的2个主要LLMs
> 
> - BART<sup>+</sup> (Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension, 2019)
> - and T5<sup>+</sup> (Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, 2019).
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2013.png)

# KV Cache

**动机和原理**

**原始Transformer的并行性很好，但计算复杂度很高，需要降低计算开销；**

**观察到：过往输出tokens的K和V结果不变，所以能复用；隐态随最新输出token而变化，和隐态关联的Q需重新计算，属于GEMV；**

> **[图片提取文字 (image.png)]:**
> ## 2. 背景
> 
> 生成式generative模型的推理过程很有特点,我们给一个输入文本,模型会输出一个回答(长度为N), 其实该过程中执行了N次推理过程。即GPT类模型一次推理只输出一个token,输出token会与输入token s 拼接在一起,然后作为下一次推理的输入,这样不断反复直到遇到终止符。
> 
> 如上描述是我们通常认知的GPT推理过程。
> 
> ## 3. 原理
> 
> 在上面的推理过程中,每 step 内,输入一个 token序列,经过Embedding层将输入token序列变为一个三维张量[b, s, h],经过一通计算,最后经logits层将计算结果映射至词表空间,输出张量维度为[b, s, vocab size]。
> 
> 当前轮输出token与输入tokens拼接,并作为下一轮的输入tokens,反复多次。可以看出第i+1 轮输入数据只比第i轮输入数据新增了一个token,其他全部相同!因此第i+1轮推理时必然包含了第i 轮的部分计算。KV Cache的出发点就在这里,缓存当前轮可重复利用的计算结果,下一轮计算时直接读取缓存结果,就是这么简单,不存在什么Cache miss问题。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2014.png)

# **KV Cache & PD分离**

**基于KV Cache优化，考虑分布式需求，llm推理架构的范式是prefill和decode（PD），属于较广泛使用的decoder-only架构；**

**猜测：**

**1、decoder-only中的prefill是将原来encoder-decoder架构中decoder输入的原始状态扩展成提示词（prompt）的输入，效果上能替代原有encoder从输入文本中提取的上下文向量；**

**2、在文本对话任务中，pretrain的llm在decoder中包含了足够多的信息提取方式，encoder-decoder是应用场景更广的架构，但decoder-only架构+prompt性价比更高；**

> **[图片提取文字 (image.png)]:**
> paper: A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications
> 
> ## 摘要
> 
> Prompt Engineering已经成为扩展大语言模型<sup>\*</sup> (LLMs) 和 视觉语言模型<sup>†</sup> (VLMs) 能力的不可或缺的技术。这种方法无 需修改模型参数,而是基于特定的任务指令 (Prompt) 来增 强模型的效果。Prompt不是更新模型参数,而是仅通过给定 的提示诱使预训练模型无缝集成到下游任务中。提示可以是给 定的上下文以指导模型的自然语言指令,也可以是已经学习到 的相关知识的向量表示。这个新兴领域已经在各种应用中取得 了成功,从问答到常识推理。然而,对于各种Prompt
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2015.png)

> **[图片提取文字 (image.png)]:**
> ## 2. KV缓存的工作机制
> 
> 要理解KV缓存,首先需要追溯其根源——Transformer架构中的自注意力机制,并明晰其在LLM推理的两个核心阶段(prefill和decode)中扮演的角色。
> 
> ## 2.1 自注意力机制:键(Key)与值(Value)的起源
> 
> Transformer架构由输入嵌入(embedding)、位置编码(positional encoding)和一系列堆叠的Transformer模块(block)构成 4。每个模块的核心是自注意力机制。在这一机制中,模型为每个输入token的嵌入向量(embedding)生成三个独立的向量:查询(Query, Q)、键(Key, K)和值(Value, V)19。
> 
> 这些向量的作用如下:当前token的Q向量会与序列中所有先前token的K向量进行点积运算,以计算出注意力分数(attention scores)。这些分数随后被用作权重,对所有先前token的V向量进行加权求和,从而生成当前token在当前层的最终表示 5。这个过程使得模型在处理当前token时,能够动态地衡量序列中其他token的重要性,并从中提取相关信息 19。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2016.png)

> **[图片提取文字 (image.png)]:**
> ## 2.2 推理过程: Prefill与Decode两阶段
> 
> LLM的自回归生成过程通常分为两个截然不同的阶段 2。
> 
> ## 1. Prefill阶段 (提示处理):
> 
> - 在此阶段,模型接收完整的输入提示(prompt),并通过一次并行的前向传播(forward pass)来处理所有prompt token 2。
> - 在这次计算中,模型会为prompt中的*每一个*token计算其对应的K和V向量,并将这些向量存储到GPU内存中的KV缓存区域 1。
> - 此阶段的计算密集度高,涉及大规模的矩阵-矩阵乘法(GEMM),能够充分利用GPU的并行计算《能力,因此是**计算密集型(compute-bound)**的 2。 这也是为什么用户在提交请求后,通常会经历一个短暂的延迟才看到第一个输出token的原因,因为模型正在进行prefill计算 11。
> 
> ## 2. Decode阶段 (token生成):
> 
> - 在此阶段,模型逐个生成输出token 1。
> - 对于每一个新生成的token,模型仅需计算其自身的Q、K、V向量 1。
> - 接着,新生成的Q向量会与整个历史序列(包括原始prompt和所有已生成的token)的K和V向量进行注意力计算,而这些历史K、V向量正是从KV缓存中高效读取的 8。
> - 计算完成后,新生成的K和V向量会被追加(append)到KV缓存中,以供后续token生成时使用 1。
> - 此阶段的计算量相对较小(矩阵-向量乘法,GEMV),无法充分利用GPU的计算单元。其性能瓶颈在于从GPU高带宽内存(HBM)中读取巨大且不断增长的KV缓存所需的时间,因此是**内存带宽密集型(memory-bandwidth bound)**的 2。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2017.png)

**Prefill & Decode数据流**

这个图有问题。当前token的k、q、v应该先更新KV cache后再进行Decode。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2018.png)

# llm和Transformer的区别

**llm推理的主要需求**

> **[图片提取文字 (image.png)]:**
> ChatGPT<sup>+</sup>在2023年年初出圈爆火之后,OpenAI<sup>+</sup>官方网站的访问量在**短短一个月内就突破10亿次**。需求是海量的,运行的模块也就是海量的。
> 
> ## 海量的模块同时一起跑和一个模块单独跑,是完全不一样的。
> 
> 么区别,为什么云存储提供商他们需要花费那么多钱和工程师去研究产品?这背后的疑问可能差不多,重点是没有看到**当模型变大、同时使用的人增多的时候**面临着什么单机单卡没有遇到的问题。
> 
> 所以, 当前LLM推理\*领域的研究重点, 主要是在: 1) 单机下单个算子加速; 2) 大规模请求并发
> 
> 这很像云存储刚刚出现的时候,有人会问在云上用存储空间和我自己在家里用几块大硬盘,也没什
> 
> 优化; 3) 压缩、蒸馏等轻量化模型方法等等。
> 
> 往学术角度讲这个问题就是**分布式系统<sup>+</sup>在大规模的大语言模型时代遇到了哪些新问题**?
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2019.png)

**llm和Transformer的结构区别**

> **[图片提取文字 (image.png)]:**
> 1.LLM就是普通transformer加大了维度,多叠了层数吗,有什么特殊的设计
> 
> ## 可以这样说。
> 
> 但是当下的LLM基本都是decoder-only的基本结构,甚至说相比最初的transformer还简单了一些。不少模型会修改小算子,比如norm、position encoder、activation层,没有什么太复杂的东西。当叠的层数增多之后,会引入一些类似GQA、MQA等微小调整结构去补充MHA,本质上也就是服用一些KV,但是对于query的存储和计算来说没有什么太多帮助。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2020.png)

> **[图片提取文字 (image.png)]:**
> ## Transformer vs LLaMA
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ## Transformer
> 
> ("Attention is all you need")
> 
> ![](_page_0_Picture_4.jpeg)
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2021.png)

**llm出于经济和并发量考虑基于Transformer进行优化，效果等价于无限存储、算力和模型深度的Transformer。**

> **[图片提取文字 (image.png)]:**
> 2.纯大模型推理,和普通transformer推理相比他的特殊性是啥,能不能直接把用给普通transformer推理的库,给他加大内存,加大算力,直接来跑大模型
> 
> 无限子弹情况下,模型容量自然可以随便加。
> 
> 现实的问题就是在于内存和算力的限制,而恰恰当下的模型在扩张时就遇到了这些限制。所以大家需要面对就是怎么在有限的内存里存下那么多的参数,在有限的算力下在可以容忍的延时中完成计算任务。
> 
> 7B的模型不做量化,12G的显卡已经放不下所有的参数了。这时比较有效的解决方法就是通过张量并行\*的方法把参数放到更多显卡中去,然后把计算分配到参数所在的显卡中。虽然带来了一些通讯和同步的时间开销,但是参数可以不用频繁的在显卡中卸载和加载的过程。
> 
> 我这几篇笔记就是分析这个方法的: DeepSpeed inference 代码理解、Llama.cpp 代码浅析 (一): 并行机制与KVCache。
> 
> 同时,推理系统不光是处理用户单次请求的,它需要同时处理和调度所有的在线请求。因此,除了张量并行的运行单个模型单次推理的方法之外,还需要有更多更好的方法来完成每个请求的prefill阶段和decode阶段。
> 
> 这时就不是光堆设备就能解决的了的了,规模足够大时,调度不当可能会让大量资源利用率下降十倍百倍甚至更多。
> 
> 这也是其他分布式系统所面临的问题。就好比说有人问阿里云、华为云、腾讯云等等相比自己家里 买几块更大的硬盘有什么特殊性? 其实一样。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2022.png)

**为了多用户的并发请求，LLM使用Prefill和Decode的分离架构进行推理。**

> **[图片提取文字 (image.png)]:**
> 3.LLM相比普通transformer多了什么特殊的算子,或者在数据流上有什么变化,比如吞吐、带宽要求,导致比普通transformer更难推理
> 
> ## 一次单独的模型推理可能没有什么区别。
> 
> 但是如果多个模型多次被多个用户同时使用,为了保证每个用户的延时都尽量少,数据流、吞吐、 带宽就都发生了明显的变化。
> 
> 当下的LLM推理过程中,逻辑可以被清晰的解耦合为两部分,一部分是prefill,另一部分是decode 阶段。
> 
> prefill阶段起手就是个batch, prompt有多少个词batchsize就有多大, 那么运算前我们需要多少内存、算力就是确定的, 并且可以并行, 即便是很长的context长度, 也可以提前规划并行加速方案和排队方案。
> 
> decode阶段是一个词一个词往外蹦的,每一个蹦出来之前还需要和前面所有已经蹦出来的词计算一次attention,所以,逻辑上就是一个串行的机制。只要存在前后的依赖关系就没法直接并行,学术界的speculative parallel、beam search机制只能说是投机并行,并没有解决根本问题。甚至我们都不知道一次decode到底会输出多长,只要不遇到eos就得一直等着它跑下去。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2023.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 1: An LLM inference example.
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2024.png)

**prefill计算密集，decode访存密集，多用户的输出长度不一，访存效率更低。**

> **[图片提取文字 (image.png)]:**
> 这里另外值得提出的就是decode中的kvcache对整个大规模并发系统的影响。并发的所有请求,其输出的结果的长度都不一样。paged attention就是为了解决当长度不一致时,内存访问不连续的情况造成严重性能下降的问题。
> 
> ## 所以说prefill阶段是高度并行的,是计算密集的。decode阶段是高度串行的,是访存密集的。
> 
> 为了更好的进行**大规模的request并发与低延时的需求**,现在sota系统的两个阶段的基本运行逻辑都是分开执行的,甚至微软都会推出两个阶段干脆直接在异构的硬件上跑的方案Splitwise,好显卡跑prefill差显卡跑decode。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2025.png)

> **[图片提取文字 (image.png)]:**
> ## Prompt computation vs. token generation
> 
> Which is better, pizza or burger?
> 
> ![](_page_0_Picture_2.jpeg)
> 
> is better .
> 
> ## **Prompt phase**
> 
> All input tokens processed in parallel to generate the first output token
> 
> Compute intensive
> 
> Smaller part of the end-to-end latency
> 
> ## Token phase
> 
> Serialized token generation
> 
> Memory intensive
> 
> Tends to be majority of end-to-end latency
> 
> <del>知乎 </del>
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2026.png)

# LLM的推理优化

**prefill阶段的优化**

> **[图片提取文字 (image.png)]:**
> **prefill阶段**的batching主要是看将很多用户的request合并称更大的矩阵一次算完,但是用户的 request长度(包括prompt还有额外的上下文)是不一样的。如 手绘图 所示的batch如何组合的问 题,另外OCRA、splitfuse都是在考虑这块的优化问题。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2027.png)

> **[图片提取文字 (image.png)]:**
> | $T_{i}$        | Tz | T3 | Ty  | Ts | T6 | To | TB |   | $T_i$ | Tz | T3  | Ty | Ts   | 76  | To  | Tg  |
> |----------------|----|----|-----|----|----|----|----|---|-------|----|-----|----|------|-----|-----|-----|
> | Si             | Si | Si | Sil |    |    |    |    |   | Si    | Si | Si  | SN | \$,, | end | 56  | S6  |
> | Sa             | Sa | SX |     |    |    |    |    | 1 | Sa    | Sa | SHI | Sx | Sa   | Sx  | SA  | END |
> | S <sub>3</sub> | S  | Sz | رک  |    |    |    |    |   | Si    | Si | Si  | S  | END  | Ss  | 55  | S   |
> | Sy             | Sy | Sy | Sy  | Sy |    |    |    | I | Sy    | Sy | Sy  | Sy | Sy   | Sy  | END | S7  |
> 
> Completing seven sequences using continuous batching. Left shows the batch after a single iteration, right shows the batch after several iterations. Once a sequence emits an end-of-sequence token, we insert a new sequence in its place (i.e. sequences S5, S6, and S7). This action is a new one. utilization since the GPU does not wait for all sequences to complete before starting a new one.
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2028.png)

**decode阶段的优化**

> **[图片提取文字 (image.png)]:**
> decode阶段就主要是考虑不断变长的输出队列和kvcache的内存管理问题了。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2029.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 7. Storing the KV cache of two requests at the same time in VLLM.
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2030.png)

**prefill和decode协同优化**

> **[图片提取文字 (image.png)]:**
> **当prefill和decode同时考虑时**还有其他各种问题,如splitwise中图所示:
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2031.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 2: Batching mechanisms and their latency impact on the property and token phases.
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2032.png)

**llm推理库**

> **[图片提取文字 (image.png)]:**
> - 4.现在提出的许多LLM推理库,主要卖点是啥
> - 5.现在越来越多的LLM推理库,但是当时transformer出来的时候没有这么火爆,是追热点还是 LLM推理库解决了啥普通transformer推理库的痛点
> 
> 这两个问题是一样的。
> 
> 简单说就是为了解决前面提到的**从单机变为分布式系统**的问题。大家都有自己的想法,看了看市面上也没有特别合适的改进的架构,就自己写一个吧,有的人基于hf的transformer,有的人基于torch,有的人甚至直接从c、cuda撸起。所有为了一个idea写一个系统,也是很正常的,所以卖点不是这个库,而是那个idea。比如vllm<sup>+</sup>的paged attention机制。
> 
> 所以说目前整个行业都还在**试错阶段**,所以呈现百花齐放的景象,最终应该还是会有集大成者胜出,他们可以吸收其他一些框架的idea,毕竟那些框架除了这些idea之外可能有其他严重内伤。我比较看好deepspeed,从目前看其架构布局、微软的分布式积累、开发实力,都是最强的,消化吸收其他框架的优势很强。
> 
> 另外微软在大模型时代通过deepspeed占领软件生态先机,才是他真正雄心勃勃的目标。
> 
> 至于llama.cpp等轻量级框架,清晰的架构、优秀的易读性都有它非常可取之处,作为学习框架和小规模部署框架来说比较合适。但是随着在线服务的需求量增加,问题就会逐步体现出来。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2033.png)

ref：[https://blog.vllm.com.cn/2023/06/20/vllm.html](https://blog.vllm.com.cn/2023/06/20/vllm.html)

# Flash Attention

attention在GPU上高性能优化：算子融合、内存优化等；

[https://www.cnblogs.com/menkeyi/p/18800377](https://www.cnblogs.com/menkeyi/p/18800377)

# Paged Attention（vllm）

> **[图片提取文字 (image.png)]:**
> ## 秘诀: 分页注意力
> 
> 在 vLLM 中,我们发现 LLM 服务的性能瓶颈在于内存。在自回归解码过程中,LLM 的所有输入 token 都会生成其注意力键和值张量,并且这些张量会保存在 GPU 内存中以生成下一个 token。这些缓存的键和值张量通常被称为 KV 缓存。KV 缓存是
> 
> - *庞大的*:在 LLaMA-13B 中,单个序列最多占用 1.7GB。
> - 动态的: 其大小取决于序列长度,序列长度是高度可变且不可预测的。因此,有效管理 KV 缓存提出了一个重大挑战。我们发现,由于碎片化和过度预留,现有系统浪费了 60% 80% 的内存。
> 
> 为了解决这个问题,我们引入了 分页注意力 (PagedAttention),这是一种受操作系统中经典虚拟内存和分页思想启发的注意力算法。与传统的注意力算法不同,分页注意力允许在非连续内存空间中存储连续的键和值。具体来说,分页注意力将每个序列的 KV 缓存划分为块,每个块包含固定数量 token 的键和值。在注意力计算期间,分页注意力内核有效地识别并获取这些块。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2034.png)

> **[图片提取文字 (image.png)]:**
> ## Key and value vectors
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Attention computation for block 0:
> 
> ![](_page_0_Figure_3.jpeg)
> 
> 分页注意力: KV 缓存被划分为块。块不需要在内存空间中是连续的。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2035.png)

> **[图片提取文字 (image.png)]:**
> ## Key and value vectors
> 
> ![](_page_0_Figure_1.jpeg)
> 
> ## Attention computation for block 1:
> 
> ![](_page_0_Figure_3.jpeg)
> 
> 分页注意力: KV 缓存被划分为块。块不需要在内存空间中是连续的。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2036.png)

> **[图片提取文字 (image.png)]:**
> ## Key and value vectors
> 
> ![](_page_0_Figure_1.jpeg)
> 
> ## Attention computation for block 2:
> 
> ![](_page_0_Figure_3.jpeg)
> 
> 分页注意力: KV 缓存被划分为块。块不需要在内存空间中是连续的。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2037.png)

> **[图片提取文字 (image.png)]:**
> ## Key and value vectors
> 
> ![](_page_0_Figure_1.jpeg)
> 
> *分页注意力:* KV 缓存被划分为块。块不需要在内存空间中是连续的。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2038.png)

由于块不需要在内存中是连续的，我们可以像操作系统虚拟内存中那样以更灵活的方式管理键和值：可以将块视为页面，将 token 视为字节，将序列视为进程。序列的连续逻辑块通过块表映射到非连续物理块。**物理块在生成新 token 时按需分配。**

> **[图片提取文字 (image.png)]:**
> ## 0. Before generation.
> 
> ![](_page_0_Picture_1.jpeg)
> 
> **Prompt:** "Alan Turing is a computer scientist" **Completion:** ""
> 
> ## Logical KV cache blocks
> 
> | Block 0 |  |  |
> |---------|--|--|
> | Block 1 |  |  |
> | Block 2 |  |  |
> | Block 3 |  |  |
> 
> ## Block table
> 
> | Physical block no. | # Filled slots |  |
> |--------------------|----------------|--|
> | _                  | _              |  |
> | _                  | _              |  |
> | _                  | _              |  |
> | _                  | _              |  |
> 
> ## Physical KV cache blocks
> 
> | Block 0 |  |  |
> |---------|--|--|
> | Block 1 |  |  |
> | Block 2 |  |  |
> | Block 3 |  |  |
> | Block 4 |  |  |
> | Block 5 |  |  |
> | Block 6 |  |  |
> | Block 7 |  |  |
> 
> 使用分页注意力处理请求的示例生成过程。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2039.png)

> **[图片提取文字 (image.png)]:**
> ## 1. Allocate space and store the prompt's KV cache.
> 
> ![](_page_0_Figure_1.jpeg)
> 
> 使用分页注意力处理请求的示例生成过程。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2040.png)

> **[图片提取文字 (image.png)]:**
> ## 2. Generated 1st token.
> 
> ![](_page_0_Figure_1.jpeg)
> 
> 使用分页注意力处理请求的示例生成过程。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2041.png)

> **[图片提取文字 (image.png)]:**
> ## 3. Generated 2nd token.
> 
> ![](_page_0_Figure_1.jpeg)
> 
> 使用分页注意力处理请求的示例生成过程。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2042.png)

> **[图片提取文字 (image.png)]:**
> ## 4. Generated 3rd token. Allocate new block.
> 
> ![](_page_0_Figure_1.jpeg)
> 
> 使用分页注意力处理请求的示例生成过程。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2043.png)

> **[图片提取文字 (image.png)]:**
> ## 5. Generated 4th token.
> 
> ![](_page_0_Figure_1.jpeg)
> 
> 使用分页注意力处理请求的示例生成过程。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2044.png)

在分页注意力中，内存浪费仅发生在序列的最后一个块中。在实践中，这实现了接近最佳的内存使用率，仅浪费不到 4%。内存效率的提高被证明是非常有益的：它允许系统批量处理更多序列，提高 GPU 利用率，从而显着提高吞吐量，如上面的性能结果所示。

分页注意力还具有另一个关键优势：**高效的内存共享**。例如，在**并行采样中，从同一提示生成多个输出序列**。在这种情况下，提示的计算和内存可以在输出序列之间共享。

**采样是指模型在生成文本时，从可能的下一个词（token）的分布中选择一个特定的词；并行采样通过获得不同采样/decode策略下的序列输出，增加输出多样性；**

> **[图片提取文字 (image.png)]:**
> ## Multiple outputs
> 
> ![](_page_0_Figure_1.jpeg)
> 
> 并行采样的示例。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2045.png)

> **[图片提取文字 (image.png)]:**
> ## Multiple outputs
> 
> ![](_page_0_Figure_1.jpeg)
> 
> 并行采样的示例。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2046.png)

分页注意力通过其块表自然地实现了内存共享。类似于进程如何共享物理页面，分页注意力中的**不同序列**可以通过将其逻辑块映射到相同的物理块来共享块。为了确保**安全共享**，分页注意力会跟踪物理块的引用计数并实现***写入时复制 (Copy-on-Write)*** 机制。

> **[图片提取文字 (image.png)]:**
> ## 0. Shared prompt: Map logical blocks to the same physical blocks.
> 
> ![](_page_0_Figure_1.jpeg)
> 
> 对请求进行多次输出采样的示例生成过程。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2047.png)

> **[图片提取文字 (image.png)]:**
> ## 1. Seq A generated 1st token.
> 
> ![](_page_0_Figure_1.jpeg)
> 
> 对请求进行多次输出采样的示例生成过程。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2048.png)

> **[图片提取文字 (image.png)]:**
> ## 2. Copy-on-Write: Copy to a new block.
> 
> ![](_page_0_Figure_1.jpeg)
> 
> 对请求进行多次输出采样的示例生成过程。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2049.png)

> **[图片提取文字 (image.png)]:**
> ## 2. Copy-on-Write: Copy to a new block.
> 
> ![](_page_0_Figure_1.jpeg)
> 
> 对请求进行多次输出采样的示例生成过程。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2050.png)

> **[图片提取文字 (image.png)]:**
> ## 3. Seq B generated 1st token. No copy needed.
> 
> ![](_page_0_Figure_1.jpeg)
> 
> 对请求进行多次输出采样的示例生成过程。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2051.png)

> **[图片提取文字 (image.png)]:**
> ## 4. Seq A and B generated 2nd token.
> 
> ![](_page_0_Figure_1.jpeg)
> 
> 对请求进行多次输出采样的示例生成过程。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2052.png)

> **[图片提取文字 (image.png)]:**
> ## 5. Seq A and B generated 3rd token.
> 
> ![](_page_0_Figure_1.jpeg)
> 
> 对请求进行多次输出采样的示例生成过程。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2053.png)

> **[图片提取文字 (image.png)]:**
> ## 6. Seq A and B generated 4th token.
> 
> ![](_page_0_Figure_1.jpeg)
> 
> 对请求进行多次输出采样的示例生成过程。
![image.png](LLM%E6%9E%B6%E6%9E%84%EF%BC%88Transformer%E3%80%81KV%20Cache%E3%80%81Flash%20Attention%E3%80%81vLLM%EF%BC%89/image%2054.png)

分页注意力的内存共享大大降低了复杂采样算法（如并行采样和束搜索）的内存开销，将其内存使用量减少了高达 55%。这可以转化为高达 2.2 倍的吞吐量提升。这使得此类采样方法在 LLM 服务中变得实用。

分页注意力是 vLLM 背后的核心技术，我们的 LLM 推理和服务引擎支持各种模型，具有高性能和易于使用的界面。有关 vLLM 和分页注意力的更多技术细节，请查看我们的 [GitHub 仓库](https://github.com/vllm-project/vllm)，并请关注我们的论文。