# **5 Related Work**

**Linear Transformers.** The linear transformers introduced in [Katharopoulos et al.](#page-11-1) [\(2020\)](#page-11-1) lagged behind vanilla transformers in downstream performance, and subsequent architectures such as TransNormer [\(Qin et al., 2022a\)](#page-12-2) and RetNet [\(Sun et al., 2023\)](#page-12-1) narrow the gap, but do not demonstrate competitive results with modern transformers at scale. RWKV [\(Peng et al., 2023a\)](#page-12-0), a linear transformer that takes inspiration from LSTM [\(Hochreiter & Schmidhuber, 1997\)](#page-11-11), is competitive with compute-matched transformer-based models, but lags behind on a number of NLU benchmarks. Griffin [\(De et al., 2024\)](#page-10-1) is a concurrent model that takes a hybrid approach, combining a sliding window with linear attention shows impressive performance relative to vanilla transformers, but is trained on a high-quality proprietary dataset.

Another thread in the literature focuses on efficient attention alternatives (Performers [\(Choromanski](#page-10-8) [et al., 2020\)](#page-10-8), Cosformer [\(Qin et al., 2022b\)](#page-12-9), LUNA [\(Ma et al., 2021\)](#page-11-12), RFA [\(Peng et al., 2021\)](#page-12-10), Attentionfree Transformer [\(Zhai et al., 2021\)](#page-13-4)). All of these approaches sacrifice performances for efficiency. Efficiency improvements for vanilla transformers have narrowed the capabilities gap between vanilla and linear transformers. The KV cache [Pope et al.](#page-12-11) [\(2023\)](#page-12-11)greatly narrows the inference efficiency gap between linear and vanilla transformers. RingAttention [Liu et al.](#page-11-13) [\(2023\)](#page-11-13) allows for very long context scaling of vanilla attention without approximation.

**State Space Models.** State-space models (SSMs) such as H3 [\(Dao et al., 2022\)](#page-10-9), Hyena [\(Poli et al.,](#page-12-12) [2023\)](#page-12-12), and Mamba [\(Gu & Dao, 2023\)](#page-11-3) are recent alternatives to vanilla transformers, combining the strengths of convolutional and recurrent models with efficient hardware implementations. Instead of parallelizing training over the sequence, they produce an efficient way to train the sequential RNN. While these models are competitive with vanilla transformers on some tasks, we show that SSMs share the limitations of linear transformers on several in-context learning and long-context tasks.

**Uptraining Linear Transformers.** Hedgehog [\(Zhang et al., 2024\)](#page-13-1) builds on the work of [Kasai et al.](#page-11-4) [\(2021\)](#page-11-4), identifying three different ways of training linear transformers – from scratch, uptraining quadratic transformers for a specific task, and uptraining generally. The authors focus on the first two, and we focus on the third. Moreover, they aim at approximating the softmax attention matrices with linear alternatives. In this work, we do not aim to approximate softmax attention, we replace it with a linear alternative (see ablation above and appendix [A\)](#page-14-0). Their method is only tested for smaller scale models and with parameter-efficient fine-tuning for larger models, but presents challenges for scaling for two reasons: (1) their training strategy involves comparing full attention matrices which is computationally expensive, and not feasible for full fine-tuning of large models with long sequences and (2) their method also inherits the gradient instabilities of linear transformers studied in [Sun et al.](#page-12-1) [\(2023\)](#page-12-1), while our normalization setup leads to stable uptraining of large models.

### **6 Conclusion**

We introduced SUPRA, a technique for converting large-scale pre-trained softmax transformers into recurrent neural networks, enabling the study of the strengths and limitations of recurrent models at scale with minimal compute cost. Compared to pre-training linear models from scratch, the SUPRA strategy produces competitive models comparable to the best available recurrent LLMs (RWKV and Mamba) at the 7B scale.

We identify the strengths of linear models on standard NLU benchmarks but also the enduring limitations on in-context (i.e. MMLU) and long-context (NarrativeQA, Qasper) tasks, showing that linearized models do not inherit these capabilities from the base softmax transformers.

One possible path to rectifying these limitations is explicitly training for in-context learning [\(Akyurek et al., 2024\)](#page-10-7). We leave explorations of specialized and instruct data in the context of ¨ linear transformers to future work. More sophisticated gating mechanisms as in in [Peng et al.](#page-12-0) [\(2023a\)](#page-12-0) and [De et al.](#page-10-1) [\(2024\)](#page-10-1) are promising alternatives to our simple linearization. Using our uptraining method would greatly reduce the necessary time and cost of such experimentation.

#### <span id="page-9-0"></span>7 Reproducibility

**Codebase** We train our linear models using our fork of OpenLM (Gururangan et al., 2023) that we modify to include a linear attention function (printed below). We use Lightning Attention 2 (Qin et al., 2024) that offers a fast Triton (Tillet et al., 2019) kernel for linear attention computation. Evaluations are done with the Eleuther evaluation harness (Gao et al., 2023).

**Data** We train and uptrain models on RefinedWeb (Penedo et al., 2023)(with 2 epochs for our Mamba training), which we tokenize with the pre-trained model's tokenizers. When training from scratch, we used the GPT-NeoX-20B (Black et al., 2022) tokenizer. We tokenize with sequence packing and use a default sequence length of 2048.

**Hyperparameters** We use square matrices with biases for the linear layers in the kernel  $\phi$  functions to keep the same feature dimension in the queries and keys. We use the same kernel, with the same weights for both queries and keys and apply a ReLU activation. We use 1000 steps of linear learning rate warmup and a cosine learning rate decay from  $3e^{-5}$  to  $1e^{-5}$  for our 7B uptrainings and from  $3e^{-4}$  to  $1e^{-5}$  for our 1B uptrainings and for trainings from scratch. We use the Adam optimizer with  $\beta_1=0.9$  and  $\beta_2=0.95$ . We trained our models with mini-batches totaling 2M tokens each. Our default RoPE frequency uses the Llama value of  $10^4$ . For longer sequence lengths, we use a RoPE frequency of  $10^6$ .

**Training** Depending on the model size and the availability, we use from 4 to 32 nodes of 8 GPUs Nvidia H100 with Pytorch FSDP. We use a mixed precision strategy from OpenLM that automatically selects between bfloat 16 and float 32 for different operations. A linear 7B parameter model uptraining throughput is around 4300 tokens per second per GPU.

**Models** Our results can be reproduced by following the same training recipe or using the model weights that we release: Mistral-SUPRA and Mamba-7b.

```
1 def linear_attn_func(q, k, v, qk_scale: float, use_decay: bool = True,
     normalize: bool = False) -> torch.Tensor:
3
      Args:
          q: queries, shape (batch_size, num_heads, seq_len, dim_qk)
4
5
          k: keys, shape (batch_size, num_heads, seq_len, dim_qk)
          v: values, shape (batch_size, num_heads, seq_len, dim_v)
6
      qk_scale: scale factor for queries and keys
8
g
      h = q.shape[1]
      if use_decay:
10
11
         s = slope_tensor(h, q.device, q.dtype)
12
          s = no_slope_tensor(h, q.device, q.dtype)
13
14
      output = lightning_attn_ops(q, k * qk_scale, v, s)
15
16
      if normalize:
          norm = torch.clamp_min(
17
                      torch.einsum(
18
                           "nhld, nhld->nhl", q, k.cumsum(2) * qk_scale
21
          return output / norm.unsqueeze(-1)
22
23
         return output
```

