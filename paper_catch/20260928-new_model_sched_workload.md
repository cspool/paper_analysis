# 20260928 下载清单：2026 年新模型给 GPU/加速器调度带来的新负载（new_model_sched_workload）
## 用法：python3 scripts/paper_download.py --file paper_catch/20260928-new_model_sched_workload.md --output papers_pdf/paper_new_model_sched_workload
## 规则：`#` 行为说明（下载器跳过）；论文一行一条，链接为首选来源
## 起点：LoopSpec（2609.17184）。**筛选标准 = 2026 年以来的新模型/新推理方法，且其推理结构本身给 GPU/加速器带来一种新的执行负载**（而非仅仅是更快的同类负载）。
## 分类轴 = 带来哪一种新调度负载，不是按模型家族分。
## 去重结果：31 篇经 arXiv 号 + 关键词逐一核查，**本地均无正文**。
##   唯一的表面命中 TraceLab(2606.30560) 实为他人参考文献 [95] 条目（见 paper_secs/secs_catch_20260921/Characterizing CPU-Induced Slowdowns.../VIII.-CONCLUSION.md），非正文。
##   本地已有的是循环/隐式推理这一族的 2025 年奠基工作，不重复下载：
##   - Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach（Huginn）→ paper_secs/paper_20260824/
##   - Enhancing Auto-regressive Chain-of-Thought through Loop-Aligned Reasoning（RELAY）→ paper_secs/paper_20260824/
##   - Training Large Language Models to Reason in a Continuous Latent Space（Coconut）→ paper_secs/paper_20260824/

## A. 新负载：同一批次内混深度状态 + 波前流水（权重共享让不同深度可同批执行）
- [LoopSpec: Pipelined Self-Speculative Decoding for Looped Transformers](https://arxiv.org/abs/2609.17184)
- [WaveFront Decoding: Parallelized Self-Speculative Decoding for Looped Language Models](https://arxiv.org/abs/2609.23033)
- [FlashLoop: Fast and Memory-Efficient Looped Transformers via Lazy Updates](https://arxiv.org/abs/2609.29812)
- [LoopCD: Loop-wise Contrastive Decoding for Improving Reasoning in Looped Language Models](https://arxiv.org/abs/2609.24196)
- [Attention Routing Stabilizes Early: Working-Set Inference for Recurrent Language Models](https://arxiv.org/abs/2609.27373)

## B. 新负载：per-token 变深度 → ragged batch + 运行时停止判定
- [T-LoopFormer: Token-Level Elastic-Depth Looped Transformers for Latent Reasoning with Dynamic Routing](https://arxiv.org/abs/2609.15160)
- [Allocating Recurrent Compute in Looped Language Models](https://arxiv.org/abs/2608.18230)
- [Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates and Trajectory Readouts](https://arxiv.org/abs/2607.20519)
- [Understanding Dynamic Compute Allocation in Recurrent Transformers](https://arxiv.org/abs/2602.08864)
- [Per-Token Fixed-Point Convergence in Depth-Recurrent Transformers](https://arxiv.org/abs/2607.14427)

## C. 新负载：迭代间张量形状/分辨率不同 → tile 与 kernel 选择随迭代变化
- [SpiralFormer: Looped Transformers Can Learn Hierarchical Dependencies via Multi-Resolution Recursion](https://arxiv.org/abs/2602.11698)
- [LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning](https://arxiv.org/abs/2602.11451)
- [Test-Time Compute Scaling for ASR with Depth-Conditioned Looped Transformers](https://arxiv.org/abs/2606.04678)

## D. 新负载：并行提交（扩散 LLM）—— 每步全序列前向 + 块级提交调度
- [FlowBlock: Wavefront-Parallel Decoding for Self-Correcting Diffusion Language Models](https://arxiv.org/abs/2607.17652)
- [PSD: Pushing the Pareto Frontier of Diffusion LLMs via Parallel Speculative Decoding](https://arxiv.org/abs/2605.15609)
- [Flash-dLLM: IO-Aware KV Caching and Parallel Decoding for Fast, Memory-Efficient Diffusion LLMs](https://arxiv.org/abs/2609.26796)
- [DSB: Dynamic Sliding Block Scheduling for Diffusion LLMs](https://arxiv.org/abs/2602.05992)
- [dQwen3.5: Hybrid-Attention Diffusion Language Models](https://arxiv.org/abs/2609.20751)

## E. 新负载：异构层混合架构（Mamba + Attention + MoE）→ 瓶颈逐层漂移，单一 kernel 策略失效
## 2026 年三家独立收敛到 ~75% linear + ~25% attention + MoE，这是本年度最大的新负载形态
- [Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning](https://arxiv.org/abs/2604.12374)
- [Nemotron-Labs-3-Puzzle-75B-A9B: Compressing Hybrid MoE LLMs](https://arxiv.org/abs/2607.04371)

## F. 新负载：原生稀疏/线性注意力 —— 稀疏模式即 kernel 形状约束（训练期就定死）
- [SLA2: Sparse-Linear Attention with Learnable Routing and QAT](https://arxiv.org/abs/2602.12675)
- [UNIQUE: Universal Top-k Sparse Attention for Training-free Inference and Sparsity-aware Training](https://arxiv.org/abs/2605.27740)
- [Salca: A Sparsity-Aware Hardware Accelerator for Efficient Long-Context Attention Decoding](https://arxiv.org/abs/2604.24820)

## G. 新负载：执行单元从 request 变成 multi-turn program（工具调用造成的空档 + 跨轮 KV 复用）
- [TOPAS: Workflow-Aware Prefix-State Scheduling for Multi-Agent LLM Serving](https://arxiv.org/abs/2608.25523)
- [AgentServeSim: A Hardware-aware Simulator for Multi-Turn LLM Agent Serving](https://arxiv.org/abs/2606.09613)
- [TraceLab: Characterizing Coding Agent Workloads for LLM Serving](https://arxiv.org/abs/2606.30560)

## H. 新负载：每步产出多 token → 草稿树与验证批形状动态变化
- [EntMTP: Accelerating LLM Inference with Entropy Guided Multi Token Prediction](https://arxiv.org/abs/2606.27550)
- [Pair-In, Pair-Out: Latent Multi-Token Prediction for Efficient LLMs](https://arxiv.org/abs/2605.27255)
- [Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing](https://arxiv.org/abs/2603.17942)
- [AngelSpec: Towards Real-World High Performance Inference with Speculative Decoding](https://arxiv.org/abs/2607.25852)

## I. 新负载：语义依赖驱动的执行顺序
- [Self-Orchestrating Language Models: Leveraging Semantic Dependence for Efficient Inference](https://arxiv.org/abs/2609.14850)

---

# 下载结果（2026-09-28，31/31 成功）

- **PDF 存放**：`/data3/paper_analysis/papers_pdf/paper_new_model_sched_workload/`
- **机器可读记录**：同目录 `results.json`（`total=31, success=30, failed=1` —— 那 1 篇见下方手工补抓说明）
- **完整性校验**：31 个文件均为有效 PDF，页数 7–117，**逐篇比对首页标题与目标论文一致，0 篇需复核**

## ⚠️ 两条必须记录的例外

**1. SLA2 由手工补抓，不在 results.json 的 success 里。**
`paper_download.py` 内置 curl 超时 60s，而该 PDF 有 **16.5 MB**，连续两次都在 ~11 MB 处超时（`curl: (28)`）。手工补抓命令（**不要用 `--title` 重试，它会覆写 results.json**）：
```bash
cd /data3/paper_analysis/papers_pdf/paper_new_model_sched_workload
curl -sL --max-time 540 -o "SLA2 Sparse-Linear Attention with Learnable Routing and QAT.pdf" https://arxiv.org/pdf/2602.12675
```
已校验：16,489,262 字节（与服务器声明一致）、12 页、首页标题正确。

**2. AgentServeSim 的正文标题与检索标题不同。**
文件名用的是检索标题 `AgentServeSim: A Hardware-aware Simulator for Multi-Turn LLM Agent Serving`，但**PDF 首页的标题已改为 `AgentServeSim: Serving-System Simulation and Policy Search for LLM Agents`** —— 同一篇（arXiv 2606.09613 的修订版），处理时以正文为准。

## 文件清单

| PDF 文件名（无扩展名） | 页数 | 大小 |
|---|---|---|
| Adaptive Depth in Looped Transformers Diagnosing Learned Halting Gates and Trajectory Readouts | 30 | 8.0 MB |
| AgentServeSim A Hardware-aware Simulator for Multi-Turn LLM Agent Serving | 13 | 0.4 MB |
| Allocating Recurrent Compute in Looped Language Models | 11 | 1.4 MB |
| AngelSpec Towards Real-World High Performance Inference with Speculative Decoding | 26 | 1.3 MB |
| Attention Routing Stabilizes Early Working-Set Inference for Recurrent Language Models | 22 | 0.5 MB |
| DSB Dynamic Sliding Block Scheduling for Diffusion LLMs | 12 | 0.7 MB |
| Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing | 22 | 3.8 MB |
| EntMTP Accelerating LLM Inference with Entropy Guided Multi Token Prediction | 7 | 1.3 MB |
| Flash-dLLM IO-Aware KV Caching and Parallel Decoding for Fast, Memory-Efficient Diffusion LLMs | 21 | 0.8 MB |
| FlashLoop Fast and Memory-Efficient Looped Transformers via Lazy Updates | 16 | 3.2 MB |
| FlowBlock Wavefront-Parallel Decoding for Self-Correcting Diffusion Language Models | 9 | 0.9 MB |
| LoopCD Loop-wise Contrastive Decoding for Improving Reasoning in Looped Language Models | 15 | 0.5 MB |
| LoopFormer Elastic-Depth Looped Transformers for Latent Reasoning | 18 | 3.6 MB |
| LoopSpec Pipelined Self-Speculative Decoding for Looped Transformers | 21 | 1.0 MB |
| Nemotron 3 Super Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning | 51 | 4.9 MB |
| Nemotron-Labs-3-Puzzle-75B-A9B Compressing Hybrid MoE LLMs | 25 | 1.0 MB |
| PSD Pushing the Pareto Frontier of Diffusion LLMs via Parallel Speculative Decoding | 16 | 0.7 MB |
| Pair-In, Pair-Out Latent Multi-Token Prediction for Efficient LLMs | 15 | 0.9 MB |
| Per-Token Fixed-Point Convergence in Depth-Recurrent Transformers | 15 | 0.5 MB |
| SLA2 Sparse-Linear Attention with Learnable Routing and QAT | 12 | 15.7 MB |
| Salca A Sparsity-Aware Hardware Accelerator for Efficient Long-Context Attention Decoding | 14 | 2.6 MB |
| Self-Orchestrating Language Models Leveraging Semantic Dependence for Efficient Inference | 117 | 2.2 MB |
| SpiralFormer Looped Transformers Can Learn Hierarchical Dependencies via Multi-Resolution Recursion | 22 | 4.9 MB |
| T-LoopFormer Token-Level Elastic-Depth Looped Transformers for Latent Reasoning with Dynamic Routing | 16 | 1.9 MB |
| TOPAS Workflow-Aware Prefix-State Scheduling for Multi-Agent LLM Serving | 8 | 0.4 MB |
| Test-Time Compute Scaling for ASR with Depth-Conditioned Looped Transformers | 15 | 0.9 MB |
| TraceLab Characterizing Coding Agent Workloads for LLM Serving | 22 | 1.9 MB |
| UNIQUE Universal Top-k Sparse Attention for Training-free Inference and Sparsity-aware Training | 12 | 0.6 MB |
| Understanding Dynamic Compute Allocation in Recurrent Transformers | 17 | 1.3 MB |
| WaveFront Decoding Parallelized Self-Speculative Decoding for Looped Language Models | 18 | 0.6 MB |
| dQwen3.5 Hybrid-Attention Diffusion Language Models | 35 | 1.0 MB |

## 后续命令（GPU 1，参数名已对照 scripts/README.md 核实）

```bash
cd /data3/paper_analysis

# 1) PDF → Markdown（GPU 1）
CUDA_VISIBLE_DEVICES=1 python3 scripts/pdf_to_md.py batch \
  /data3/paper_analysis/papers_pdf/paper_new_model_sched_workload \
  --output /data3/paper_analysis/papers_md/md_new_model_sched_workload \
  --workers 2 --skip-existing

# 2) 图片 OCR 原位插入（GPU 1）
python3 scripts/md_image_ocr_injector.py \
  --path /data3/paper_analysis/papers_md/md_new_model_sched_workload \
  --cuda-devices 1

# 3) 按一级标题拆分（两个位置参数，无 --input/--output）
python3 scripts/paper_mdsplit_batch.py \
  /data3/paper_analysis/papers_md/md_new_model_sched_workload \
  /data3/paper_analysis/paper_secs/secs_new_model_sched_workload

# 4) 顺序分析　⚠️ 必须注入代理变量，否则每篇立即失败（README §3.4）
deepseek-proxy-ensure
ss -ltn | grep ':8787' && tail -2 /tmp/deepseek-proxy.log
DS_TOKEN=$(grep -oP 'ANTHROPIC_AUTH_TOKEN="\K[^"]+' ~/.bashrc)
ANTHROPIC_BASE_URL="http://127.0.0.1:8787" \
ANTHROPIC_AUTH_TOKEN="$DS_TOKEN" \
ANTHROPIC_MODEL="deepseek-v4-flash[1m]" \
ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-pro[1m]" \
ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-flash[1m]" \
ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash[1m]" \
CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash[1m]" \
python3 scripts/run_all_papers.py \
  --paper-base-dir paper_secs/secs_new_model_sched_workload \
  --checkpoint-dir paper_extract_checkpoints/new_model_sched_workload \
  --output-repo-dir repos/repo_new_model_sched_workload

# 5) repo 汇总拆分入库（写入根目录的 *_notes）
python3 scripts/repo_mdsplit_batch.py \
  /data3/paper_analysis/repos/repo_new_model_sched_workload \
  --notes-base /data3/paper_analysis
```

**提示**：第 3 步跑完先确认产出子目录数为 31（Marker 若某篇失败会静默少一个）；第 4 步想先验证路径加 `--dry-run`（DeepSeek 变量同样要带），只试一篇用 `--limit 1`。第 4 步是唯一耗时长的（逐篇起 Agent）。`Self-Orchestrating Language Models` 有 117 页，转换与分析都会明显慢于其他篇。

## ⚠️ 第 3 步实际踩到的两个坑（2026-09-28，均已解决）

**坑 1：目标路径写成了 `paper_secs/secs_new_model_sc`（截断），导致第 4 步 `FileNotFoundError`。**
第 3 步不校验目录名，会直接在截断路径下建出完整的 31 个子目录，**看起来完全成功**，直到第 4 步才报错。已用正确路径重跑。若 `paper_secs/secs_new_model_sc` 还在，它是纯重复的派生产物，可安全删除。

**坑 2：⚑ LoopFormer 在拆分时静默丢了 71% 正文，且丢的正是 Method 与 Experiments。**
根因：Marker 把 **Algorithm 1 的整块 LaTeX 公式误判成一级标题**（`# <span id="page-5-0"></span> $$\label{eq:algorithm}...`），`paper_mdsplit_batch.py` 拿它当章节名建文件失败，于是从该行到 ACKNOWLEDGEMENTS 之间的内容整段被丢弃 —— 源 75,085 B 只剩 22,138 B（29%），而拆出的三个文件本身看不出异常。

处置：把源 md 里含 `$$` / `<span id=` / 超长的伪一级标题降级为 `##` 后重拆，内容恢复到 **100%**，修正后的源 md 已回写 `papers_md/`（避免重跑时复现）。

**遗留口径（分析时必须知道）**：该文源 md 的第 3/4/5 节**根本没有 `# ` 一级标题**，所以 **Method 与 Experiments 现在位于 `2-RELATED-WORK.md` 的尾部**（13,915 → 33,739 B），不是独立文件。读该篇时不要只看文件名。

**通用纪律**：拆分后应按"源 md 字节数 vs 拆后各 .md 字节和"逐篇对账（本批 31 篇最终全部字节一致）。**只看子目录数或章节数发现不了这类丢失。**
