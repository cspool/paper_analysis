# 注意力机制速览

ref：[https://developer.aliyun.com/article/1309822](https://developer.aliyun.com/article/1309822)

ref：[https://zhuanlan.zhihu.com/p/109585084](https://zhuanlan.zhihu.com/p/109585084)

ref：[https://www.zhihu.com/question/458687952](https://www.zhihu.com/question/458687952)

# 注意力机制

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E9%80%9F%E8%A7%88/image.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 3: An example of the attention mechanism following long-distance dependencies in the encoder self-attention in layer 5 of 6. Many of the attention heads attend to a distant dependency of the verb 'making', completing the phrase 'making...more difficult'. Attentions here shows for the word 'making'. Different colors represent different heads. Best viewed in color.
![image.png](%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E9%80%9F%E8%A7%88/image%201.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top: Full attentions for head 5. Bottom: Isolated attentions from just the word 'it; for a temperature of 5 and 6. Note that the attentions are very sharp for this word.
![image.png](%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E9%80%9F%E8%A7%88/image%202.png)

> **[图片提取文字 (image.png)]:**
> ## 1.2.1 Query&Key&Value
> 
> ## 首先我们来认识几个概念:
> 
> - 查询 (Query): 指的是查询的范围, 自主提示, 即主观意识的特征向量
> - 键 (Key): 指的是被比对的项,非自主提示,即物体的突出特征信息向量
> - 值 (Value) : 则是代表物体本身的特征向量,通常和Key成对出现
> 
> 注意力机制是通过Query与Key的注意力汇聚(给定一个 Query,计算Query与 Key的相关性,然后根据Query与Key的相关性去找到最合适的 Value)实现对Value的注意力权重分配,生成最终的输出 结果。
> 
> ![](_page_0_Figure_6.jpeg)
> 
> ## 有点抽象吧,我们举个栗子好了:
> 
> - 1. 当你用上淘宝购物时,你会敲入一句关键词(比如:显瘦),这个就是Query。
> - 2. 搜索系统会根据关键词这个去查找一系列相关的Key(商品名称、图片)。
> - 3. 最后系统会将相应的 Value (具体的衣服) 返回给你。
> 
> 在这个栗子中,Query, Key 和 Value 的每个属性虽然在不同的空间,其实他们是有一定的潜在关系的,也就是说通过某种变换,可以使得三者的属性在一个相近的空间中。
![image.png](%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E9%80%9F%E8%A7%88/image%203.png)

> **[图片提取文字 (image.png)]:**
> ## 1.2.2 注意力机制计算过程
> 
> 输入Query、Key、Value:
> 
> • 阶段一:根据Query和Key计算两者之间的相关性或相似性(常见方法点积、余弦相似度,MLP网络),得到注意力得分;
> 
> 点积: Similarity(Query,  $Key_i$ ) = Query ·  $Key_i$ 
> 
> Cosine 相似性: Similarity(Query,  $Key_i$ ) =  $\frac{Query \cdot Key_i}{||Query|| \cdot ||Key_i||}$ 
> 
> MLP 网络: Similarity(Query,  $Key_i$ ) = MLP(Query,  $Key_i$ ) CSDN @路人贾'ω'
> 
> • **阶段二**:对注意力得分进行缩放scale(除以维度的根号),再softmax函数,一方面可以进行归一化,将原始计算分值整理成所有元素权重之和为1的概率分布;另一方面也可以通过softmax的内在机制 更加突出重要元素的权重。一般采用如下公式计算:
> 
> $$a_i = Softmax(Sim_i) = \frac{e^{Sim_i}}{\sum_{j=1}^{L_x} e^{Sim_j}}$$
> 
> CSDN @B
> 
> • 阶段三:根据权重系数对Value值进行加权求和,得到Attention Value(此时的V是具有一些注意力信息的,更重要的信息更关注,不重要的信息被忽视了);
> 
> Attention(Query, Source) = 
> $$\sum_{i=1}^{L_x} a_i \cdot Value_i$$
> 
> 这三个阶段可以用下图表示:
> 
> ![](_page_0_Figure_11.jpeg)
> 
> https://blcCSDN @路人贾'wb
![image.png](%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E9%80%9F%E8%A7%88/image%204.png)

# 自注意力机制

中译英模型中，使用encoder-decoder架构（如Transformer），输入中文序列（密码），输出英文序列（密文），对应模型中找到中文和英文单词的对应关系；

encoder输入原始文本序列（中文），提取结构化特征，比如句子结构、专有名词、词性、短语等，输出上下文向量，用于decoder生成目标文本序列（英文）；

decoder输入目标文本（英文）起始状态，根据decoder的上下文向量输出目标文本序列（中英attention），并更新隐藏状态（英文self attention），隐藏状态是目标文本的状态、结构特征，比如输出到短语、词组、从句、主语等；

所以encoder中，query、key、value由中文输入计算，decoder中query、key、value由英文输入计算；

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E9%80%9F%E8%A7%88/image%205.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E9%80%9F%E8%A7%88/image%206.png)

> **[图片提取文字 (image.png)]:**
> ## 2.1 什么是自注意力机制?
> 
> 字处理问题。 针对全连接神经网络对于多个相关的输入无法建立起相关性的这个问题,通过自注意力机制来解决,**自注意力机制实际上是想让机器注意到整个输入中不同部分之间的相关性**。
> 
> 找到X里面的关键点,从而更关注X的关键信息,忽略X的不重要信息。**不是输入语句和输出语句之间的注意力机制,而是输入语句内部元素之间或者输出语句内部元素之间发生的注意力机制。** 
> 
> 自注意力机制实际上是注意力机制中的一种,也是一种网络的构型,它想要解决的问题是神经网络接收的输入是很多大小不一的向量,并且不同向量向量之间有一定的关系,但是实际训练的时候无法充分发
> 
> <mark>挥这些输入之间的关系而导致模型训练结果效果极差。</mark>比如机器翻译(序列到序列的问题,机器自己决定多少个标签),词性标注(Pos tagging一个向量对应一个标签),语义分析(多个向量对应一个标签)等文
> 
> **自注意力机制**是注意力机制的变体,其减少了对外部信息的依赖,更擅长捕捉数据或特征的内部相关性。自注意力机制的关键点在于,Q、K、V是同一个东西,或者三者来源于同一个X,三者同源。通过X
![image.png](%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E9%80%9F%E8%A7%88/1b0b541c-3817-4412-9ca0-c1407d85cbc4.png)

> **[图片提取文字 (image.png)]:**
> ## 2.2 如何运用自注意力机制?
> 
> 其实步骤和注意力机制是一样的。
> 
> 第1步:得到Q, K, V的值
> 
> 对于每一个向量 ${\rm x}$ ,分别乘上三个系数  $W^q$ ,  $W^k$ ,  $W^v$ ,得到的 ${\rm Q}$ , K和 ${\rm V}$ 分别表示query,key和 ${\rm value}$ 
> 
> | , 对了每一门问里X, | 力加黎工二十条数 W *, | NY        | ,特封的Q,K和V力加表小Query,Key和Value<br>Input Thinking Machines |                |          |            |
> |-------------|---------------|-----------|---------------------------------------------------------|----------------|----------|------------|
> |             |               | input     | Thinking                                                | wachines       |          |            |
> |             |               | Embedding | X <sub>1</sub>                                          | X <sub>2</sub> |          |            |
> |             |               | Queries   | q <sub>1</sub>                                          | $q_2$          |          | Ma         |
> |             |               | Keys      | k <sub>1</sub>                                          | k <sub>2</sub> |          | Wκ         |
> |             |               | Values    | V1                                                      | V <sub>2</sub> | CSDN @路人 | WV<br>贾'ω' |
> 
> 【注意】三个W就是我们需要学习的参数。
![image.png](%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E9%80%9F%E8%A7%88/image%207.png)

> **[图片提取文字 (image.png)]:**
> ## 第2步: Matmul
> 
> 利用得到的Q和K计算每两个输入向量之间的相关性,一般采用点积计算,为每个向量计算一个score: score =q·k
> 
> ![](_page_0_Figure_2.jpeg)
![image.png](%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E9%80%9F%E8%A7%88/image%208.png)

> **[图片提取文字 (image.png)]:**
> ## 第3步: Scale+Softmax
> 
> 将刚得到的相似度除以 $\sqrt{d_k}$ ,再进行Softmax。经过Softmax的归一化后,每个值是一个大于0且小于1的权重系数,且总和为0,这个结果可以被理解成一个权重矩阵。
> 
> ![](_page_0_Figure_2.jpeg)
![image.png](%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E9%80%9F%E8%A7%88/image%209.png)

> **[图片提取文字 (image.png)]:**
> ## 第4步: Matmul
> 
> 使用刚得到的权重矩阵,与V相乘,计算加权求和。
> 
> ![](_page_0_Figure_2.jpeg)
> 
> 以上是对Thinking Machines这句话进行自注意力的全过程,最终得到z1和z2两个新向量。
> 
> 其中z1表示的是thinking这个词向量的新的向量表示(通过thinking这个词向量,去查询和thinking machine这句话里面每个单词和thinking之间的相似度)。 也就是说新的z1依然是 thinking 的词向量表示,只不过这个词向量的表示蕴含了 thinking machines 这句话对于 thinking 而言哪个更重要的信息。
![image.png](%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E9%80%9F%E8%A7%88/image%2010.png)

> **[图片提取文字 (image.png)]:**
> ## 2.3 自注意力机制的问题
> 
> 具体原理吧,感兴趣的话可以看一下:
> 
> 自注意力机制的原理是**筛选重要信息,过滤不重要信息**,这就导致其有效信息的抓取能力会比CNN小一些。这是因为自注意力机制相比CNN,无法利用图像本身具有的尺度,平移不变性,以及图像的特征局部性(图片上相邻的区域有相似的特征,即同一物体的信息往往都集中在局部)这些先验知识,只能通过大量数据进行学习。**这就导致自注意力机制只有在大数据的基础上才能有效地建立准确的全局关系,而在小数据的情况下,其效果不如CNN。** 
> 
> 另外,自注意力机制虽然考虑了所有的输入向量,**但没有考虑到向量的位置信息**。在实际的文字处理问题中,可能在不同位置词语具有不同的性质,比如动词往往较低频率出现在句首。
> 
> 要唠这个这就唠到**位置编码(Positional Encoding)** 了,这个我们下篇论文里面再讲,先大致说一下吧:对每一个输入向量加上一个位置向量e,位置向量的生成方式有多种,通过e来表示位置信息带入selfattention层进行计算。
> 
> [2003.09229] Learning to Encode Position for Transformer with Continuous Dynamical Model (arxiv.org)
![image.png](%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E9%80%9F%E8%A7%88/image%2011.png)

# 多头注意力

> **[图片提取文字 (image.png)]:**
> 通过刚才的学习,我们了解到自注意力机制的缺陷就是,模型在对当前位置的信息进行编码时,会过度的将注意力集中于自身的位置,有效信息抓取能力就差一些。 因此就有大佬提出了通过多头注意力机制来解决这一问题。这个也是实际中用的比较多的。
> 
> ## 3.1 什么是多头注意力机制?
> 
> 在实践中,当给定相同的查询、键和值的集合时, 我们希望模型可以基于相同的注意力机制学习到不同的行为, 然后将不同的行为作为知识组合起来, 捕获序列内各种范围的依赖关系 (例如,短距离依赖和长距离依赖关系)。 因此,**允许注意力机制组合使用查询、键和值的不同 子空间表示(r**epresentation subspaces**)可能是有益的**。
> 
> 为此,与其只使用单独一个注意力汇聚, 我们可以用**独立学习得到的h组(一般h=8)不同的线性投影**(linear projections)来变换查询、键和值。 然后,**这h组变换后的查询、键和值将并行地送到注意力** 汇聚中。 最后,**将这h个注意力汇聚的输出拼接在一起**, 并且**通过另一个可以学习的线性投影进行变换, 以产生最终输出**。 这种设计被称为<mark>多头注意力(multihead attention)</mark>。
> 
> ## Multi-Head Attention Linear Concat Scaled Dot-Product Attention Linear Linear
![image.png](%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E9%80%9F%E8%A7%88/image%2012.png)

> **[图片提取文字 (image.png)]:**
> ## 3.2 如何运用多头注意力机制?
> 
> 第1步: 定义多组W, 生成多组Q、K、V
> 
> 刚才我们已经理解了,Q、K、V是输入向量X分别乘上三个系数  $W^q$ ,  $W^k$ , $W^v$ 分别相乘得到的,  $W^q$ , $W^k$ , $W^v$ 是可训练的参数矩阵。
> 
> 现在,对于同样的输入X,我们定义多组不同的 $W^q$ , $W^k$ , $W^v$ ,比如 $W_0^q$ 、 $W_0^k$ , $W_0^v$ , $W_1^q$ 、 $W_1^k$ 、 $W_1^v$ 每组分别计算生成不同的Q、K、V,最后学习到不同的参数。
> 
> ![](_page_0_Figure_4.jpeg)
![image.png](%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E9%80%9F%E8%A7%88/image%2013.png)

> **[图片提取文字 (image.png)]:**
> ## 第2步: 定义8组参数
> 
> 对应8个single head,对应8组  $\,W^q,\;W^k,\,W^v\,$ ,再分别进行self-attention,就得到了 $Z_0$ - $Z_7$ 
> 
> ![](_page_0_Figure_2.jpeg)
![image.png](%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E9%80%9F%E8%A7%88/image%2014.png)

> **[图片提取文字 (image.png)]:**
> ## 第3步:将多组输出拼接后乘以矩阵 $W_0$ 以降低维度
> 
> 首先在输出到下一层前,我们需要将 $Z_0$ - $Z_7$ concat到一起,乘以矩阵 $W_0$ 做一次线性变换降维,得到Z。
> 
> 1) Concatenate all the attention heads
> 
> ![](_page_0_Figure_3.jpeg)
> 
>  Multiply with a weight matrix W<sup>o</sup> that was trained jointly with the model
> 
> X
> 
> 3) The result would be the Z matrix that captures information from all the attention heads. We can send this forward to the FFNN
> 
> = Z
> 
> ![](_page_0_Picture_8.jpeg)
![image.png](%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E9%80%9F%E8%A7%88/image%2015.png)

> **[图片提取文字 (image.png)]:**
> - 1) 这是我们 2) 编码每一 的输入句子\* 个单词
> - 3) 将其分为8个头, 将矩阵X或R乘以各 个权重矩阵
> - 4)通过输出的查询 /键/值(Q/K/V)矩 阵计算注意力
> - 5) 将所有注意力头拼接起来,乘以 权重矩阵W<sup>0</sup>
> 
> Thinking Machines
> 
> ![](_page_0_Picture_6.jpeg)
> 
> \*注:除了第0个编码器,其他编码器都不需要进行词嵌入。它可以直接讲前面一层编码器的输出作为输入(矩阵R)。
> 
> ![](_page_0_Picture_8.jpeg)
> 
> ![](_page_0_Figure_9.jpeg)
> 
> ![](_page_0_Picture_10.jpeg)
> 
> ![](_page_0_Picture_12.jpeg)
> 
> https://blog.cs/CSDN/@路人贾'ω'5
![image.png](%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E9%80%9F%E8%A7%88/image%2016.png)