# **1 Introduction**

The evolution of large language models (LLMs) has been a cornerstone in the advancement of natural language processing (NLP), enabling an unprecedented depth of understanding and generation of human language. Fine-tuning these sophisticated models is essential for tailoring their capabilities to specific tasks, thereby enhancing their applicability and performance across a spectrum of NLP challenges. Despite significant progress, conventional fine-tuning methods often lack the dynamism to adapt to the diverse and complex nature of various language tasks, highlighting the need for more flexible and adaptable fine-tuning strategies.

Amidst the quest for efficiency in fine-tuning LLMs, the concept of parameter efficiency has gathered attention, particularly due to the vast size and intricate architecture of modern models. Parameter-efficient fine-tuning (PEFT) approaches [\(Liu et al.,](#page-11-0) [2022\)](#page-11-0) aim to adapt LLMs to specialized tasks by fine-tuning a small subset of model parameters, significantly reducing computational and storage costs while mitigating the risk of catastrophic forgetting.

<sup>∗</sup>Corresponding author

Among these approaches, Low-Rank Adaptation (LoRA) [\(Hu et al.,](#page-10-0) [2021\)](#page-10-0) is notable for its ability to introduce adaptability without altering the original model's weights. LoRA applies low-rank decomposition to represent weight updates through smaller matrices, allowing the model to adapt to new data while the core weight matrix remains unchanged, thus embodying a targeted and efficient method for model refinement.

Building on this foundation, recent advancements have combined LoRA with the Mixture of Experts (MoE) [\(Shazeer et al.,](#page-11-1) [2017\)](#page-11-1) framework to further enhance the model's adaptability and performance. LoRA's integration allows for precise modification of weights through low-rank matrices, while MoE leverages a set of expert networks, each specializing in different tasks or aspects of the data. The synergy between LoRA's targeted weight adaptation and MoE's expert-driven approach offers a dynamic avenue for model enhancement. However, the prevalent static top-k expert selection in MoE does not fully leverage the potential for task-specific adaptability, prompting the need for more dynamic selection mechanisms that can respond to the varying complexities and subtleties of different tasks and contents.

Addressing this gap, we present AdaMoLE[1](#page-1-0) , a novel method that synergizes LoRA with an adaptive MoE, incorporating a dynamic threshold network for expert activation, which is illustrated in Figure [1.](#page-1-1) This innovation allows AdaMoLE to fine-tune its activation of experts based on the context of the input, providing a more refined and context-aware approach to model adaptation.

<span id="page-1-1"></span>![](_page_1_Figure_4.jpeg)

Figure 1: Illustration of Adaptive Mixture of Low-Rank Adaptation Experts (AdaMoLE). AdaMoLE employs a gating function alongside a threshold function to determine the activation of experts. In the training phase, pre-trained weights are frozen while the LoRA experts and two functions are updated.

Our main contributions are as follows:

- 1. AdaMoLE represents an advanced integration of LoRA and an adaptive MoE framework, featuring a dynamic threshold network that facilitates context-sensitive expert activation, transcending the limitations of static top-k strategies.
- 2. Through comprehensive evaluations across various tasks, AdaMoLE showcases superior adaptability and performance, highlighting the effectiveness of dynamic expert selection and setting a new baseline in the fine-tuning of LLMs.

<span id="page-1-0"></span><sup>1</sup>GitHub: <https://github.com/zefang-liu/AdaMoLE>

3. Threshold sensitivity and expert activation analyses of AdaMoLE provide crucial insights into the model's operational dynamics, confirming that its adaptive threshold mechanism plays a pivotal role in balancing computational efficiency with expert engagement across diverse tasks.

By introducing AdaMoLE, we aim to not only refine the fine-tuning process for LLMs but also encourage further research in developing models that are inherently flexible and attuned to the specificities of diverse application domains. This work reflects our commitment to enhancing the capabilities of LLMs, suggesting a promising direction for increased personalization and efficiency in NLP.

