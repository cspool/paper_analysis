# <span id="page-20-0"></span>**5. Conclusion, Limitation, and Future Work**

In this paper, we introduce DeepSeek-V2, a large MoE language model that supports 128K context length. In addition to strong performance, it is also characterized by economical training and efficient inference, benefiting from its innovative architecture including MLA and DeepSeekMoE. In practice, compared with DeepSeek 67B, DeepSeek-V2 achieves significantly stronger performance, and meanwhile saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times. Evaluation results further demonstrate that with only 21B activated parameters, DeepSeek-V2 achieves top-tier performance among open-source models and becomes the strongest open-source MoE model.

DeepSeek-V2 and its chat versions share the acknowledged limitations commonly found in other LLMs, including the lack of ongoing knowledge updates after pre-training, the possibility of generating non-factual information such as unverified advice, and a chance to produce hallucinations. In addition, since our data primarily consist of Chinese and English content, our model may exhibit limited proficiency in other languages. In scenarios beyond Chinese and English, it should be used with caution.

DeepSeek will continuously invest in open-source large models with longtermism, aiming to progressively approach the goal of artificial general intelligence.

- In our ongoing exploration, we are dedicated to devising methods that enable further scaling up MoE models while maintaining economical training and inference costs. The goal of our next step is to achieve performance on par with GPT-4 in our upcoming release.
- Our alignment team continuously strives to enhance our models, aiming to develop a model that is not only helpful but also honest and safe for worldwide users. Our ultimate objective is to align the values of our model with human values, while minimizing the need for human supervision. By prioritizing ethical considerations and responsible development, we are dedicated to creating a positive and beneficial impact on society.
- Currently, DeepSeek-V2 is designed to support the text modality exclusively. In our forward-looking agenda, we intend to enable our model to support multiple modalities, enhancing its versatility and utility in a wider range of scenarios.

