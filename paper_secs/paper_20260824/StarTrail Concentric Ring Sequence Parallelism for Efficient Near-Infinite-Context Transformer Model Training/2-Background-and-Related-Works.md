# 2 Background and Related Works

### 2.1 Long Sequence Training and Sequence Parallelism

The key mechanism behind these Transformer-based models is attention[\[38\]](#page-12-0), which captures the text feature by calculating the attention score between every two single tokens. However, the sequence length can reach hundreds of thousands, when dealing with multi-round chatting, or high-resolution long video generation. It then becomes necessary to distribute the sequence across multiple GPUs. This distribution helps to reduce both the memory and computation demands on any single device. This strategy is also known as sequence parallelism. Presently, Sequence parallelism can be divided into two main categories: attention-head-sharding-based and peer-to-peer-communication-based. The former involves distributing the attention heads of multi-head attention across multiple GPUs, whereas the latter resembles a distributed version of FlashAttention, relying on peer-to-peer communication to transfer keys, values, and intermediate statistics.

