# 生成式模型学习笔记（目录ALL）

## 基础（MLE、VI、OT）

[生成式模型基础（MLE、VI、OT）](%E7%94%9F%E6%88%90%E5%BC%8F%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80%EF%BC%88MLE%E3%80%81VI%E3%80%81OT%EF%BC%89%202d8e12d10b6e80959691fd7597fa44b0.md)

## VAE、VQGAN、VQGAN

[VAE、VQVAE、VQGAN：14、19-22](VAE%E3%80%81VQVAE%E3%80%81VQGAN%EF%BC%9A14%E3%80%8119-22%202d8e12d10b6e8051843eec9e630ec2d1.md)

## DDPM

[DDPM：NIPS20](DDPM%EF%BC%9ANIPS20%202d8e12d10b6e8072acc2c4dcd7234cb7.md)

## X2IMG、VQGAN+ARM、LDM

[X2Img、VQGAN+ARM、LDM：19-22](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22%202d8e12d10b6e8062a407c39104bccdda.md)

## DDIM、Flow

[DDIM、SDE/ODE Flow：21-23](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23%202d8e12d10b6e80e48101dddef99b7bee.md)

## DiT、JiT

[DiT、JiT：23、25](DiT%E3%80%81JiT%EF%BC%9A23%E3%80%8125%202d8e12d10b6e8061ad2ec3b6d770f713.md)

## LLM is from Seq

[https://zhuanlan.zhihu.com/p/340149804](https://zhuanlan.zhihu.com/p/340149804)

> **[图片提取文字 (image.png)]:**
> ## 自回归模型应用线性回归,其输出中的滞后变量取自先前的步骤。与线性回归不同,自回归模型除了先前预测的结果外,不使用其他自变
> 
> 量。请考虑以下公式。
> 
> 白回归
> 
>  $p(x) = \prod_{i=1}^{n} p(x_i | x_1, x_2, \dots x_{i-1}) = \prod_{i=1}^{n} p(x_i | x_{< i})$ 
> 
> 我们也可以用下面的方程来表示自回归建模。
> 
> $$y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \dots + \phi_p y_{t-p} + \epsilon_t$$
> 
>  $y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \dots + \phi_p y_{t-p} + \epsilon_t$ 
> 
> 在这里,y是先前结果的多个阶数乘以它们各自的系数 Ø 的预测结果。系数表示影响预测器对新结果的重要性的权重或参数。该公式还考虑了可能影响预测的随机噪声,表明模型并不理想,可以进一步改进。
![image.png](%E7%94%9F%E6%88%90%E5%BC%8F%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0%EF%BC%88%E7%9B%AE%E5%BD%95ALL%EF%BC%89/image.png)

> **[图片提取文字 (image.png)]:**
> ## • 1.1 处理 Sequence 数据的模型:
> 
> Transformer 是一个 Sequence to Sequence model,特别之处在于它大量用到了 self-attention。
> 
> 要处理一个Sequence,最常想到的就是使用RNN,它的输入是一串vector sequence,输出是另一串vector sequence,如下图1左所示。
> 
> 如果假设是一个single directional的RNN,那当输出  $b_4$  时,默认  $a_1,a_2,a_3,a_4$  都已经看过了。如果假设是一个bi-directional的RNN,那当输出  $b_{\rm H \tilde{\pi}}$  时,默认  $a_1,a_2,a_3,a_4$  都已经看过了。RNN非常擅长于处理input是一个sequence的状况。
> 
> 那RNN有什么样的问题呢?它的问题就在于:RNN很不容易并行化 (hard to parallel)。 为什么说 RNN很不容易并行化呢?假设在 single directional 的 RNN 的情形下,你今天要算出  $b_4$  ,就必须要先看  $a_1$  再看  $a_2$  再看  $a_3$  再看  $a_4$  ,所以这个过程很难平行处理。
> 
> 所以今天就有人提出把 $\mathsf{CNN}$ 拿来取代 $\mathsf{RNN}$ ,如下图1右所示。其中,橘色的三角形表示一个 $\mathsf{filter}$ ,每次扫过3个向量 a ,扫过一轮以后,就输出了一排结果,使用橘色的小圆点表示。
> 
> 这是第一个橘色的filter的过程,还有其他的filter,比如图2中的黄色的filter,它经历着与橘色的filter相似的过程,又输出一排结果,使用黄色的小圆点表示。
![image.png](%E7%94%9F%E6%88%90%E5%BC%8F%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0%EF%BC%88%E7%9B%AE%E5%BD%95ALL%EF%BC%89/image%201.png)

> **[图片提取文字 (image.png)]:**
> 所以,用CNN,你确实也可以做到跟RNN的输入输出类似的关系,也可以做到输入是一个sequence,输出是另外一个sequence。
> 
> 但是,表面上CNN和RNN可以做到相同的输入和输出,但是CNN只能考虑非常有限的内
> 
> 容。比如在我们右侧的图中CNN的filter只考虑了3个vector,不像RNN可以考虑之前的所有vector。但是CNN也不是没有办法考虑很长时间的dependency的,你只需要堆叠filter,多堆叠几层,上层的filter就可以考虑比较多的资讯,比如,第二层的filter(蓝色的三角形)看了6个vector,所以,只要叠很多层,就能够看很长时间的资讯。
> 
> 而CNN的一个好处是:它是可以并行化的 (can parallel),不需要等待红色的 filter 算完,再算黄色的 filter。但是必须要叠很多层 filter,才可以看到长时的资讯。所以今天有一个想法:self-attention,如下图 3 所示,目的是使用 self-attention layer 取代 RNN 所做的事情。
![image.png](%E7%94%9F%E6%88%90%E5%BC%8F%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0%EF%BC%88%E7%9B%AE%E5%BD%95ALL%EF%BC%89/image%202.png)

> **[图片提取文字 (image.png)]:**
> 所以重点是: 我们有一种新的 layer,叫 self-attention,它的输入和输出和 RNN 是一模一 样的,输入一个sequence,输出一个sequence,它的每一个输出  $b_1-b_4$  都看过了整个 的输入 sequence,这一点与 bi-directional RNN 相同。但是神奇的地方是:它的每一个输 出  $b_1 - b_4$  可以并行化计算。
![image.png](%E7%94%9F%E6%88%90%E5%BC%8F%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0%EF%BC%88%E7%9B%AE%E5%BD%95ALL%EF%BC%89/image%203.png)

> **[图片提取文字 (image.png)]:**
> ## Sequence
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Using CNN to replace RNN
> 
> Hard to parallel!
![image.png](%E7%94%9F%E6%88%90%E5%BC%8F%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0%EF%BC%88%E7%9B%AE%E5%BD%95ALL%EF%BC%89/image%204.png)

> **[图片提取文字 (image.png)]:**
> ## Sequence
> 
> Filters in higher layer can consider longer sequence
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Hard to parallel
> 
> Using CNN to replace RNN (CNN can parallel)
![image.png](%E7%94%9F%E6%88%90%E5%BC%8F%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0%EF%BC%88%E7%9B%AE%E5%BD%95ALL%EF%BC%89/image%205.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](%E7%94%9F%E6%88%90%E5%BC%8F%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0%EF%BC%88%E7%9B%AE%E5%BD%95ALL%EF%BC%89/image%206.png)

## Encoder和Decoder作用

Diffusion将图片的pixel看作**特定分布的采样结果**，自回归模型将文本序列看作时间序列。

**Decoder相同的role（相同的命名）**，但不同的使用（pipeline）产生不同的function。

Transformer将文本库看成无数张“图片”的训练集，每张“图片”是一个段落或句子，那么VQVAE-Decoder从x的编码序列生成图片x，对应到Transformer-Decoder从历史上下文c（差分的编码序列 ）生成下一个单字（图片x’和图片x的**差分**），因为文本库内的关联性比图片间的关联性更强，或者文本库内关联性对应视频帧之间的关联性。**VAE和Transformer中都称为Encoder和Decoder**，因为行为都是编码提取信息，解码生成媒介。 

Diffusion生成过程从xT到x0的含噪图片序列xt，很像Transformer生成token序列的过程，每个xt都是新的上下文向量cn，每个εt都是token。

**为什么Decoder-Only成为LLM主流？**

https://www.reddit.com/r/MachineLearning/comments/14onn56/d_eli5_why_is_the_gpt_family_of_models_based_on/?tl=zh-hans

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
> 
> radarsat1 • 3y ago
> 
> 更确切地说,编码器对于它正在执行的任务来说,实际上并不是必需的。当进行序列到 序列的转换时,编码器是合适的。现在,你可以将问答表述为这样一个 seq2seq 任务。 在这种情况下,使用交叉注意力来允许答案"引用"问题的不同部分。
> 
> 然而, GPT 将该任务表述为一个简单的预测下一个单词的问题,其中问题或提示只是它 需要扩展的文本的较早部分。在这种情况下,它只需要对提示+续写进行自注意力。 这至少有一个优势,你不需要输入/输出对来进行训练,只需要一堆你想要完成的文本。 其次,用于关注提示的机制与用于关注答案当前状态的机制相同,所以在某种程度上,
> 
> 总之,答案基本上是,只有解码器碰巧是 GPT 训练问题的表述方式,还有其他方法,但 从经验上来看,它效果很好。
> 
> ![](_page_0_Picture_5.jpeg)
> 
> 这个仟务更直接。
![image.png](%E7%94%9F%E6%88%90%E5%BC%8F%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0%EF%BC%88%E7%9B%AE%E5%BD%95ALL%EF%BC%89/image%207.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
> 
> ## Cutie\_McBootyy • 3y ago
> 
> 我觉得我没说错,但如果有人觉得不对,请指正。我也想过同样的问题。
> 
> 原因是,只有解码器的架构非常适合聊天应用,特别是多轮对话。你把系统提示作为第一个输入,然后输出第一个助手消息作为输出。接下来,有一个用户文本。你如何处理它?直接把它加到目前的序列后面,做一个单次前向传递,得到下一个助手的回复。超级简单,非常方便,计算效率很高。
> 
> 在编码器-解码器架构中,你如何做同样的事情?你做不到。你把到目前为止的整个文本都送回编码器,然后再次处理所有内容,因为 L-R 关系不成立,因为嵌入是双向的。这非常耗费计算资源。只有解码器的架构是聊天应用非常优雅的实现。
> 
> ![](_page_0_Picture_5.jpeg)
![image.png](%E7%94%9F%E6%88%90%E5%BC%8F%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0%EF%BC%88%E7%9B%AE%E5%BD%95ALL%EF%BC%89/image%208.png)