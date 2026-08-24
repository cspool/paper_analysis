# E.2 Viewing LightThinker from Other Perspectives

In previous sections, we design LightThinker from a compression perspective. Here, we further discuss it from the perspectives of *Memory* and *KV Cache Compression*, where KV Cache can be viewed as a form of LLM work memory.

In Memory perspective, LightThinker's workflow can be summarized as follows: it first performs autoregressive reasoning, then stores key information from the reasoning process as memory (memory), and continues reasoning based on the memorized content. Thus, the information in the cache tokens acts as a compact memory, though it is only effective for the current LLM and lacks transferability.

In KV Cache Compression perspective, unlike methods such as H2O [\(Zhang et al.,](#page-11-5) [2023\)](#page-11-5), which rely on manually designed eviction policy to select important tokens, LightThinker merges previous tokens in a continuous space, *ceating* new representations. The content and manner of merging are autonomously determined by the LLM, rather than being a discrete selection process.

