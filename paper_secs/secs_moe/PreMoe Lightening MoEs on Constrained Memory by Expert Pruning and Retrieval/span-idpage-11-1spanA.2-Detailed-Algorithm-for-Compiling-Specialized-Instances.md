# <span id="page-11-1"></span>A.2 Detailed Algorithm for Compiling Specialized Instances

As described in the main text, the identified computational patterns serve as blueprints for compiling specialized MoE instances. The key to this process is its memory efficiency: the full, dense model (which can be hundreds of gigabytes) is never loaded into memory. Instead, a lightweight "skeleton" of the model architecture is first instantiated. Then, only the weights for the selected experts (identified by the PEU-based pattern) are loaded from storage and placed into the appropriate slots in the model. This creates a sparse, powerful,

and ready-to-use model instance without ever incurring the memory cost of the full model. This subsection provides the detailed algorithm for the two primary compilation strategies based on this principle.

- **1. Compiling a Domain-Specific Specialist.** This is the most straightforward application of a computational pattern.
  - 1. **Pattern Identification:** For a target domain T, calculate the PEU scores  $\{PEU_i^T\}_{i=1}^{N_T}$  for all experts in each MoE layer across a calibration dataset  $\mathfrak{X}_T$ . This forms the computational pattern for the domain.
  - 2. **Expert Selection:** For a given expert budget M, select the set of M experts with the highest PEU scores for each layer. This pruned set of experts becomes the new set of routed experts  $\{E_i^r(\mathbf{x})\}_{i=1}^M$  for the compiled instance.
  - 3. **Instance Compilation:** A new, sparse model instance is created containing only the selected routed experts. The router weights are also pruned to remove parameters corresponding to the discarded experts.
- **2. Compiling a High-Efficiency Generalist.** This strategy creates a single, sparse model that retains capability across multiple domains by creating a synthesized, multi-domain computational pattern.
  - 1. **Synthesize Token-Level Scores:** For a set of D target domains  $\{T_1, \ldots, T_D\}$ , first collect the token-level utility scores,  $\{\tilde{s}_i(\mathbf{x})\}$ , by running the model over all tokens in all of their respective calibration datasets,  $\{X_{T_1}, \ldots, X_{T_D}\}$ . All of these individual token-level scores are then aggregated into a single, large collection.
  - 2. Calculate Multi-Domain PEU: A unified PEU score for the generalist model,  $PEU_i^{\text{multi}}$ , is calculated by averaging all of the aggregated token-level scores.

$$PEU_i^{\text{multi}} = \frac{1}{\sum_{d=1}^{D} |\mathcal{X}_{T_d}|} \sum_{d=1}^{D} \sum_{\mathbf{x} \in \mathcal{X}_{T_d}} \tilde{s}_i(\mathbf{x}).$$
 (11)

This creates a single, blended PEU ranking that captures an expert's importance across the full spectrum of targeted domains.

- 3. **Expert Selection:** For a given total expert budget M, select the set of M experts with the highest multi-domain PEU scores. This becomes the new set of routed experts  $\{E_i^r(\mathbf{x})\}_{i=1}^M$ .
- 4. **Instance Compilation:** A new model instance is created containing only the final selected set of routed experts.

This compilation process is performed once at deployment time, creating a static, efficient model instance that is proactively specialized for its intended application, whether that be single-domain or multi-domain.

## A.3 Generation Configuration

For reproducibility, we provide the generation configurations used for each model during both calibration (pattern collection) and evaluation:

For all models, we use a maximum context length (input + output) of 32,768 tokens during evaluation. The exception is Qwen3-30B-A3B on AIME 2024, AIME 2025, and CNMO 2024, where we use 38,912 tokens following the official evaluation setting from the Qwen3 technical report.

