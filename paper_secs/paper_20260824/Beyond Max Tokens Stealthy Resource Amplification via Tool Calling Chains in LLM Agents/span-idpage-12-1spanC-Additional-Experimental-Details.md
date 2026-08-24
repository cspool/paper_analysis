# <span id="page-12-1"></span>C Additional Experimental Details

## C.1 Target LLMs configuration

All target models are served under a uniform runtime on a single node with eight H200 GPUs, using bfloat16 precision and a maximum context length of 131,072 tokens. We deploy vLLM [\(Kwon et al.,](#page-9-15) [2023\)](#page-9-15). Decoding follows the same setting across all conditions: nucleus sampling with p = 0.95 and temperature 0.5, and a per-generation completion cap of 16,384 tokens. These settings are held constant for every model and benchmark so that any change in cost, length, or throughput arises from the agent–tool interaction rather than heterogeneous serving choices.

## C.2 Attacker LLMs configuration

We employ a two-stage approach using two distinct LLMs for attack generation. Within the iterative MCTS optimization loop, the Editor LLM is Llama-3.3-70B-Instruct [\(Grattafiori](#page-8-12)

[et al.,](#page-8-12) [2024\)](#page-8-12). Its serving and decoding configuration deliberately mirrors the target LLM setup described above to eliminate experimental confounds and ensure the generated edits are effective. For converting a benign tool description into a protocol-compatible malicious template, we leverage gpt-4o [\(OpenAI,](#page-9-14) [2024\)](#page-9-14). We fix its temperature at 0 to guarantee deterministic and high-fidelity output for our seed templates, leaving all other parameters at the provider's defaults.

