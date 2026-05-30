# Model Sharing with Multi-LoRA

Sharing of model parameters is commonly used for reducing GPU memory footprint. S-Lora [\[29\]](#page-8-11), vLLM [\[5,](#page-7-11) [9\]](#page-7-12), CaraServe [\[22\]](#page-8-13) share the common base model across multiple model variants each served with its respective LoRA adapter. However, such techniques are specific to LoRA adapters and do not apply directly to MoE models. Additionally, MoEsaic does not require a common base model. Multiple model variants can share different experts across them.

### Sharing in Mixture of Experts

Recently, Mixture of Experts models have become a focus of multi-task learning, where an MoE model is shared across several tasks either through task-specific gates [\[6,](#page-7-13) [13,](#page-7-14) [25\]](#page-8-14) or a combined gate [\[24\]](#page-8-15). These approaches specifically train the gates to share experts across related tasks. In contrast to these approaches, MoEsaic is agnostic to the tasks or use-cases served by the constituent models. Therefore, MoEsaic does not require any additional training of the gate to incorporate new MoE models.

DeepSeekMoE [\[12\]](#page-7-10) segments the experts into smaller experts and isolates the subset of these experts as shared ones, aiming to capture the common knowledge. The shared experts are invoked for each request. Li et. al. [\[21\]](#page-8-10) consolidate experts into fewer and more knowledgeable experts. The authors note that the consolidated experts are better compressible than the original experts. While the above techniques modify the MoE architecture for performance benefits, MoEsaic simply incorporates the new experts and gates from multiple clients, where each model retains its exact structure as from the original MoEs.

#### Reducing Memory Footprint of MoEs

MoQE[\[17\]](#page-7-15) applies 2-bit quantization to experts. It observed that the expert layers in MoE models are much more robust to the quantization than conventional feed-forward networks (FFN) layers. Chen et. al. [\[10\]](#page-7-16) convert larger model into task specific smaller model through fine-tuning by eliminating less relevant experts. Li et. al. [\[20\]](#page-8-12) apply post-training quantization to MoE models. The work explores structure-aware

<span id="page-2-0"></span>![](_page_2_Figure_2.jpeg)

Figure 1: The figure shows an MoE layer within an MoE model. The experts A, B and C are identical across clients two clients. Whereas, the expert D and E are unique experts belonging to client 1 and 2 respectively. Each request leads to the selection of 2 experts for execution.

quantization at various granularities, e.g., MoE expert to linear block. While these techniques reduce the memory footprint of MoEs and improve their computational efficiency, they are orthogonal our work. MoEsaic can also detect and deduplicate any quantized experts across model instances.

