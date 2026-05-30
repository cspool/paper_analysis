# **3 Preliminary**

Before introducing LightTransfer, we provide a brief overview of the generative inference in autoregressive LLMs, which is the key background for our method.

**Inference stages.** The typical generative LLM inference process involves two stages: (1) *Prefilling*: the autoregressive LLM processes the input prompt *X* by parallel computing, and also saves the KV cache of tokens in *X*. The output of the last token in this stage is the first token of the response. (2) *Decoding*: after the prefilling stage is completed, the LLM generates output tokens one by one, and saves their KV cache. In each decoding step, a new token is generated based on the current token and the KV cache stored from earlier steps, continuing until a stop criterion is met.

