# 7 Limitations

Given our focus on finding efficient MoE models *under computational constraints*, AutoMoE search space and evaluation has been restricted in scale to big-sized Transformer models for benchmark MT tasks. A natural extension of this work is to explore the limits of MoE models like SwitchTransformers [\(Fedus et al.,](#page-9-0) [2022b\)](#page-9-0) and GShard [\(Lepikhin](#page-10-5) [et al.,](#page-10-5) [2020\)](#page-10-5) that are significantly larger containing billions to trillions of parameters; as well as designing sparse and transferable efficient expert models [\(Zoph et al.,](#page-11-1) [2022\)](#page-11-1) for diverse types of tasks like reasoning, summarization and understanding.

The limitations of this work are as follows:

- 1. Sandwich sampling [\(Yu et al.,](#page-11-5) [2019\)](#page-11-5), inplace knowledge distillation [\(Yu et al.,](#page-11-5) [2019\)](#page-11-5), and gradient conflict reduction [\(Gong et al.,](#page-9-13) [2022\)](#page-9-13) are popular techniques to improve the training procedure of supernet. It would be interesting to study the impact of these techniques to improve AutoMoE's supernet.
- 2. AutoMoE uses the hidden dimension of intermediate feedforward network (FFN) to modulate the capacity of each expert. It would be interesting to study other techniques to modulate expert capacity such as stacking variable number of hidden layers in FFN.
- 3. The backbone of AutoMoE's supernet uses Switch Transformer, which adds FFN based expert layers and routes each token to exactly one expert (top-1 routing). It would be interesting to: (i) search for the number of tokens to route, and (ii) search for the Transformer component (e.g., FFN, self-attention projection layers, LayerNorm) to add expert layers.
- 4. AutoMoE's search space contains classical Transformer components such as multi-head attention and FFN layers. It would be interesting to add components that are efficient by design such as convolutional layer, FLASH [\(Hua](#page-9-14) [et al.,](#page-9-14) [2022\)](#page-9-14), and g-MLP [\(Liu et al.,](#page-10-17) [2021\)](#page-10-17).

