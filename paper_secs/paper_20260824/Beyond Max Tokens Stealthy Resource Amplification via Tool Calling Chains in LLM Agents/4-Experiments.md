# 4 Experiments

## 4.1 Experimental Setup

Agent framework & serving environment. We evaluate all conditions under the same agent policy A and prompts. For safety and isolation, we do not evaluate against production agent stacks; instead, all experiments run on a controlled simulator built by modifying qwen-agent to faithfully emulate a tool calling loop while preventing unintended external actions [\(QwenLM,](#page-10-14) [2025\)](#page-10-14). Runs are executed on a single node with 8× H200 GPUs using a uniform serving stack with a fixed concurrency of 25 queries; no changes to A or the target LLM M are made across conditions. Full configuration details, including target and attacker LLMs configuration, the agent framework setup, and our datasets filtering and wrapping rules, can be found in Appendix [C.](#page-12-1)

Target LLMs. We target six LLMs with strong tool calling support: Qwen-3-32B [\(Yang et al.,](#page-10-15) [2025a\)](#page-10-15), Llama-3.3-70B-Instruct [\(Grattafiori et al.,](#page-8-12) [2024\)](#page-8-12), Llama-DeepSeek-70B [\(DeepSeek-AI et al.,](#page-8-13) [2025\)](#page-8-13), Mistral Large [\(Mistral,](#page-9-13) [2024\)](#page-9-13), Seed-32B [\(ByteDance,](#page-8-14) [2025\)](#page-8-14), and GLM-4.5-Air [\(Zhipu](#page-11-5) [AI et al.,](#page-11-5) [2025\)](#page-11-5).

Datasets. We use two tool-use corpora: Tool-Bench [\(Fan et al.,](#page-8-8) [2025\)](#page-8-8) and BFCL [\(Patil et al.,](#page-9-6)

**Algorithm 1:** MCTS Optimizer for Malicious Template Generation

```
: Seed bank (candidate T_{\theta}); action families
   Input
                 \mathcal{A}_{\mathrm{MT}}, \mathcal{A}_{\mathrm{LEN}}, \mathcal{A}_{\mathrm{REP}}; targets m^* (minimum
                 multi-turn count), L^* (per-turn length
                 target); Stage sizes and gates; UCT constant
                 C; search budget.
   Output: Optimized template T_{\theta^*} and an updated
                 seed bank.
 1 Seed screening: evaluate candidates on a fixed query
     set; pick the most accepted starters.
   while budget not exhausted do
         select node v by UCT using \bar{Q} and N_{\text{uct}};
3
         if v not fully expanded then
              set A \leftarrow A_{\text{MT}} if \phi(v) = \text{pre\_MT} else A_{\text{LEN}};
5
              if omission observed at v then
                A \leftarrow A \cup A_{REP}
 7
              for
each untried a \in \mathcal{A} do
 8
                    create a child by applying the Editor
                      once to obtain T_{\theta'}.
10
              foreach new child u in parallel do
                    Stage-1: run small rollouts; update \bar{Q}(u)
11
                      and increment N_{\text{uct}} along the path;
12
                    if segment sequencing stabilized then
                     | set \phi(u):=post_MT.
13
                    if Stage-1 gate satisfied then
14
                         Stage-2: run additional rollouts;
15
                           refine \bar{Q}(u).
                    if acceptance predicate holds then
16
17
                         record T_{\theta^*} and write back to the
                           seed bank.
18
                    backpropagate Stage-1 statistics to
                      ancestors (value means and N_{\text{uct}}).
19 return T_{\theta^*} and the updated bank.
```

2025). From each, we select all prompts that are single-turn and single-tool in their original specification. For comparability, each original tool is wrapped as an MCP server that preserves its functionality and descriptions. We drop a small number of low-quality prompts that never trigger a tool call under the benign configuration. The final evaluation sets contain: ToolBench: 105 MCP servers and 261 queries; BFCL: 80 MCP servers and 203 queries.

**Baselines.** We compare five conditions under identical agent policy A, target LLM M, prompts, and decoding: (i) Benign MCP server (no attack): the unmodified server for each tool; (ii) Overthink (ICL–Genetic, Agnostic): we reproduce the strongest variant from (Kumar et al., 2025). Because our benchmarks are non-RAG, we place the decoy trigger in the user query (in-context prefix/suffix) rather than in retrieved context; (iii) Overthink-MT (multi-turn Overthink): an aligned multi-turn extension of Overthink for tool-calling agents. We match our multi-turn setting (tool-call

budget and delayed exposure of the true tool output) while keeping the trigger at the context layer and leaving the tool server benign; (iv) Hand-crafted template (no MCTS): a fixed malicious template that follows the same constraints as ours (text-only edits, protocol-compatible, payload-preserving) but without MCTS optimization; (v) Ours: the MCTS-optimized malicious MCP template that preserves functionality and task completion yet induces verbose, multi-turn tool calling trajectories.

**Attack LLMs.** Within the MCTS optimizer, we use Llama-3.3-70B-Instruct as the Editor LLM. For the one-shot rewriting that instantiates the Universal Malicious Template on an MCP server, we employ gpt-40 (OpenAI, 2024).

Attack setup. For each tool/LLM pair we (1) select from the seed bank the starter template with the highest acceptance under a fixed per-turn length target; (2) instantiate a protocol-compatible malicious variant by editing text-only fields of the benign MCP server (argument descriptions and inprogress/corrective messages; function signatures and identifiers unchanged; termination deferred via text-only notices; benign payload preserved), introducing the segment index and full calibration sequence to encourage multi-turn trajectories with verbose tool calling outputs; and (3) run UCT-MCTS refinement with phase gating (multi-turn induction before length induction) and a two-stage evaluation, freezing the template once it meets a fixed acceptance threshold and writing it back to the seed bank. This instantiation aligns with the components and procedures detailed in §4.2–§4.4 and Algorithm 1. Unless otherwise noted, we evaluate under the same serving cap (M=16,384 max completion tokens per generation) and our default multi-turn setting used throughout Table 2.

**Metrics.** All metrics are evaluated on both benchmarks (ToolBench, BFCL). We report: (i) Efficacy: (a) token length per query: average output tokens per eligible query; (b) latency per query: average end-to-end latency; (c) attack success rate (ASR): fraction of eligible queries for which (1) the method's targeted behavior occurs (for ours: multiturn tool calling with long outputs; for Overthink: single-turn think inflation), (2)  $\operatorname{Succ}(u,\tau,o)=1$  (i.e., the final answer o solves the user goal u). (d) task success rate (TSR): success probability under the unmodified (benign) MCP servers, used as the correctness baseline. (ii) Resource impact: (a) total energy consumption (Wh): integrate perdevice power over time; (b) maximum GPU KV

cache usage: peak KV-cache occupancy reported by the serving stack. (iii) Throughput efficiency (tokens/s): tokens-per-second of a fixed, benign co-running workload executed concurrently with the evaluated condition. We present results in the order: effectiveness → resources → throughput → defenses.

