# Limitations

In this work, we develop SpindleKV, which achieves a balance in KV cache compression across shallow and deep layers. Experiments conducted on two long-context benchmarks and three models demonstrate the effectiveness of our method.

While our current approach shows promising results, future work will focus on further refining the control over KV cache size to achieve more precise management. Additionally, although we have validated the effectiveness of our method on LLaMA2- 7b-chat, LLaMA3-8b-instruct, and Mistral-7binstruct-v0.2, we plan to extend our evaluation to additional models such as Qwen2.5-7b [\(Yang et al.,](#page-10-15) [2024a\)](#page-10-15), LLaMA2-13b, and LLaMA3-70b. This will allow us to further demonstrate the generality of our approach across a broader range of settings.

