# <span id="page-3-1"></span>2.3 INSTRUCTION FINE-TUNING

After pretraining, the memory slots produced by the pretrained ICAE are expected to represent the original context. However, for LLMs, the purpose of providing a context extends beyond rote memorization or continuation; instead, the more common use scenario is using the provided context as a basis for accurately and appropriately responding to various prompts, ultimately accomplishing the tasks we want it to perform [\(Wei et al.,](#page-11-4) [2021;](#page-11-4) [Ouyang et al.,](#page-10-6) [2022\)](#page-10-6).

To enhance the interaction of memory slots produced by the ICAE with diverse prompts, we further fine-tune the ICAE with the PWC dataset (Prompt-with-Context), a dataset[1](#page-3-0) introduced in this paper consisting of thousands of (context, prompt, response) samples (as shown in Figure [1\)](#page-0-0).

Formally, the ICAE is fine-tuned for learning to encode the context into the memory slots based on which the decoder (i.e., the target LLM) can produce a desirable response r<sup>1</sup> . . . r<sup>n</sup> according to a given prompt p<sup>1</sup> . . . pm, as shown in Figure [8](#page-12-1) in Appendix [A:](#page-11-3)

$$\mathcal{L}_{\text{FT}} = \max_{\widetilde{m_1} \dots \widetilde{m_k}} P(r_1 \dots r_n | \widetilde{m_1} \dots \widetilde{m_k}, p_1 \dots p_m; \Theta_{LLM})$$

$$= \max_{\Theta_{LoRA}, e_m} P(r_1 \dots r_n | m_1 \dots m_k, p_1 \dots p_m; \Theta_{LLM}, \Theta_{LoRA}, e_m)$$

