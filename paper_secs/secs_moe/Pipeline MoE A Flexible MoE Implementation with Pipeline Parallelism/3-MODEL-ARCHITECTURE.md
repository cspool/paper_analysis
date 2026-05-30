# 3 MODEL ARCHITECTURE

#### 3.1 PRELIMINARY: PARALLEL AND MOE

#### 3.1.1 BASICS OF TRANSFORMER

In this paper, we take the decoder-only transformer architecture as an example to instantiate the proposed architecture. In a densely connected transformer, a batch of sequences with s tokens is first mapped to a tensor with a shape of b×s×h through an embedding layer (word and positional), where b is the batch size, s is the sequence length, and h is the hidden dimension. Then intermediate states are fed through transformer blocks that are mainly composed of a self-attention module and a feed-forward network (FFN) module. Finally, the processed intermediate states are sent to the final layers to calculate output information. Here, we dive into a transformer block and showcase its detailed structure. The input is first normalized with a LayerNorm layer, which keeps the tensor shape unchanged. An attention module is applied to extract interaction between token embeddings. After the attention module, another LayerNorm is placed. Then an FFN is used to further extract information from input embeddings. With necessary skip connections, the processed embedding is sent to a later block. For more details, we refer readers to [Phuong & Hutter](#page-13-13) [\(2022\)](#page-13-13) and [Shoeybi et al.](#page-13-1) [\(2019\)](#page-13-1).

### 3.1.2 DATA PARALLEL

Data parallel (DP) is the most broadly used technique to scale up deep neural network (DNN) training in recent years. As DNN models become larger, a single GPU or device may only provide a relatively small throughput due to limited computing power and memory, while a larger batch size may be needed to ensure training stability or faster training is required. Data parallel replicates models between devices and split input data into micro-batches then feed to each replica. After all replicas finish computing (both forward and backward), model replicas on each device are synchronized by executing an all-reduce communication on gradients.

![](_page_2_Figure_8.jpeg)

<span id="page-2-0"></span>Figure 1: Illustration of tensor parallel. A merged version of Figure 2 and Figure 3 from Megatron paper [Shoeybi et al.](#page-13-1) [\(2019\)](#page-13-1).

