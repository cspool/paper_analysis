# 5 EXPERIMENTS

#### 5.1 SETTINGS

Dataset. For SFT, we sample 12K instances from LLaVA-CoT(4k) [\(Xu et al., 2024\)](#page-10-14) and Vision-SR1(8k) [\(Li et al., 2025\)](#page-9-3), converting each into the structured format <description>, <think>, and <answer>to train the model to verbalize task-relevant visual evidence before reasoning. For RL, we aggregate multimodal reasoning samples from MMK12(5k) [\(Meng et al., 2025b\)](#page-9-13), LLaVA-CoT(5k), Vision-R1-rl(5k) [\(Huang et al., 2025b\)](#page-9-14), and Mulberry(5k) [\(Yao et al., 2024\)](#page-10-15), spanning domains such

<span id="page-7-0"></span>

| Method                         | MathVista | MMMU | EMMA | AI2D | C-MMBench | C-MMBench-TO |
|--------------------------------|-----------|------|------|------|-----------|--------------|
| Vision-SR1-7B                  | 71.1      | 54.9 | 28.3 | 81.0 | 39.7      | 42.4         |
| Vision-R1-7B                   | 71.3      | 44.9 | 27.4 | 63.2 | 52.6      | 48.5         |
| Perception-R1-7B               | 67.2      | 50.9 | 27.6 | 77.4 | 41.7      | 47.9         |
| Visionary-R1                   | 65.5      | 46.2 | 28.2 | 80.5 | 35.1      | 37.6         |
| MM-Eureka-Qwen-7B              | 71.4      | 54.7 | 28.0 | 78.9 | 44.0      | 45.1         |
| VL-Rethinker-7B                | 72.4      | 56.4 | 27.5 | 79.7 | 41.3      | 45.4         |
| Qwen2.5-VL-7B-Instruct         | 66.4      | 48.4 | 28.0 | 77.2 | 43.1      | 45.0         |
| VTPerception-R1-7B (Before RL) | 66.4      | 50.6 | 26.6 | 80.4 | 46.7      | 47.0         |
| VTPerception-R1-7B             | 71.0      | 52.2 | 28.8 | 82.5 | 53.2      | 50.5         |

Table 2: Performance Comparison Across Multimodal Benchmarks

as mathematics, science, and figure comprehension. This diverse dataset supports both perception enhancement and general reasoning improvement.

**Benchmark.** We evaluate multimodal understanding and reasoning using a comprehensive suite: MMMU, MathVista, AI2D, EMMA, and Creation-MMBench.

MMMU. targets college-level, multi-discipline reasoning with 11.5K image-text questions across six core disciplines.

*MathVista*. contains 6,141 problems drawn from 28 existing datasets plus three newly curated ones, assessing mathematical reasoning in visual contexts.

AI2D. evaluates diagram understanding on thousands of annotated grade-school science diagrams paired with multiple-choice questions.

*EMMA*. measures integrated cross-modal reasoning in mathematics, physics, chemistry, and coding, requiring organic image—text reasoning.

*Creation-MMBench.* specifically assesses context-aware creative capabilities of MLLMs with 765 instances over 51 fine-grained tasks and instance-specific criteria (we also report results on its text-only variant, Creation-MMBench-TO).

**Training details.** For supervised fine-tuning (SFT), we initialized from Qwen2.5-VL-7B-Instruct and tuned all parameters on a merged multimodal dataset of  $\sim$ 12K samples for 3 epochs, with a learning rate of  $1\times10^{-5}$ , weight decay of 0.1, batch size 1, and gradient accumulation of 8. bf16 precision, gradient checkpointing, and DeepSpeed ZeRO-3 were enabled to support long-context multimodal inputs. For reinforcement learning (RL), we adopted a DAPO-style framework implemented in EasyR1-perc, distributed with Ray across one main node and an additional ORM node for reward computation. Tensor parallel size was set to 4. The reward function combined answer accuracy, format compliance, key visual/textual information, n-gram penalty, and consistency, with tuned weights. RL training ran on  $\sim$ 22K samples for 2 epochs.

#### 5.2 EVALUATION

We report performance across six representative multimodal reasoning benchmarks in Table 2. Our method (VTPerception-R1-7B), after reinforcement learning (RL), consistently outperforms its supervised fine-tuning (SFT) baseline and demonstrates competitive or superior results compared to existing strong baselines.

Comparison with Prior Methods. VTPerception-R1-7B achieves new state-of-the-art results on four out of six benchmarks—AI2D (82.5), Creation-MMBench (53.2), C-MMBench-TO (50.5), and EMMA (28.8). Notably, it surpasses VL-Rethinker (79.7) and MM-Eureka (78.9) on AI2D, highlighting its advantage on diagram-heavy perception tasks. On EMMA and both variants of C-MMBench, our model significantly improves over Vision-R1, Perception-R1, and MM-Eureka, demonstrating that incorporating both visual and textual perception rewards yields more reliable and grounded outputs.

**Effectiveness of Reinforcement Learning.** The RL stage contributes substantial improvements across all tasks. Compared to the SFT-only version of our model, RL brings gains of +4.6 on MathVista, +1.6 on MMMU, +2.2 on EMMA, +2.0 on AI2D, +6.5 on C-MMBench, and +3.5 on C-

<span id="page-8-0"></span>

| Method             | MathVista | MMMU | EMMA | AI2D | C-MMBench | C-MMBench-TO |
|--------------------|-----------|------|------|------|-----------|--------------|
| Full model (Ours)  | 65.0      | 47.9 | 26.3 | 80.8 | 44.5      | 47.9         |
| - Consistency      | 64.2      | 47.1 | 26.2 | 79.6 | 41.2      | 46.2         |
| - Textual key-info | 64.3      | 49.2 | 25.6 | 80.4 | 41.9      | 44.6         |
| - Visual key-info  | 66.6      | 46.7 | 26.1 | 78.8 | 43.9      | 46.5         |

Table 3: Ablation results on multiple benchmarks.

MMBench-TO. These gains validate not only the utility of reinforcement learning but also highlight the contribution of our method: explicitly enhancing the model's visual and textual perception capabilities. The perception-aware reward design—targeting key visual elements, textual cues, and their consistency—plays a central role, particularly in benchmarks requiring compositional reasoning and fine-grained evidence tracking. The largest improvements on C-MMBench and AI2D support our hypothesis that grounding the <description> → <think> → <answer> pipeline in concrete perceptual signals leads to more faithful, interpretable, and robust reasoning.

Overall, the consistent improvements across all benchmarks—particularly over perception-focused baselines like Perception-R1, Visionary-R1, and Vision-SR1—demonstrate that VTPerception-R1- 7B's integration of explicit perception and structured RL rewards offers a robust and generalizable path forward for multimodal reasoning.

#### 5.3 ABLATION STUDY

To demonstrate the effectiveness of our proposed method VTPerception-R1, we conducted controlled experiments under different reward configurations. For a fair comparison, all models are initialized from the same SFT checkpoint, trained with 12k samples for 3 epochs, and subsequently fine-tuned on the same RL dataset for one epoch. Performance is evaluated across multiple benchmarks, as shown in Table [3.](#page-8-0)

Ablation Analysis. The removal of any single reward results in a drop in overall performance, confirming the complementary roles of the three reward components. The consistency reward has the broadest impact: its absence causes the most severe declines on reasoning-intensive benchmarks (-3.26 on C-MMBench and -1.70 on C-MMBench-TO) and also reduces performance on MathVista and AI2D. This highlights the importance of enforcing a coherent <description> → <think> → <answer> reasoning chain.

The text perception reward is especially critical for benchmarks like C-MMBench that rely heavily on precise textual cues, where its removal leads to drops of -2.64 and -3.31. Although a slight improvement is observed generally, likely due to reduced over-regularization, most datasets show decreased performance—indicating that weakening textual grounding tends to obscure key constraints.

The visual perception reward contributes most to diagram- or image-intensive tasks, as shown by its impact on AI2D (-2.01) and MMMU (-1.21). Interestingly, its removal slightly improves MathVista (+1.60), suggesting that certain samples may not depend strongly on fine-grained visual grounding.

In summary, the full VTPerception-R1 configuration achieves the best trade-off across all tasks, confirming the necessity of integrating all three reward signals for robust and generalizable performance.

