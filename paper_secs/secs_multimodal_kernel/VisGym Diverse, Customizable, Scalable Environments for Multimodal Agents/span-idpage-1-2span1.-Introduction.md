# <span id="page-1-2"></span>1. Introduction

Humans navigate complex tasks in visually rich and interactive settings: manipulating objects, using devices, or exploring unfamiliar environments. Success in these settings hinges on the tight coupling of perception, memory, and action over multiple steps (Gibson, 1979; Henderson, 2001). Foundation Vision–Language Models (VLMs) have made remarkable progress on static vision–language benchmarks (Yue et al., 2024; Lu et al., 2023; Wang et al., 2024) and on text-based multi-step tasks such as web browsing and coding (Sirdeshmukh et al., 2025; Wei et al., 2025; Jimenez et al., 2023). Yet when visual observations must be integrated into multi-step decision-making, their behavior remains far less understood. Recent evaluations across robotic manipulation, computer-use agents, and gaming agents highlight a range of challenges for visually interactive decision-making, including low task success rates, brittle visual grounding, and weak generalization (Zhang et al., 2025; Xie et al., 2024; Zhang et al., 2025; Liu et al., 2023; Hu et al., 2025; Chen et al., 2025; Shi et al., 2025; Liu et al., 2024). Although these insights are valuable, they tend to be domain-specific and observational, offering limited *systematic*, *controlled* diagnosis of how domain-agnostic factors—such as context length, representation modality, feedback design, or goal visibility—affect model performance across tasks.

We introduce VisGym, a highly diverse, scalable, and customizable gymnasium with 17 long-horizon environments designed to isolate what limits interactive decision-making across domains and to expose where current VLMs break down. The suite spans symbolic puzzles, real-image understanding, navigation, and manipulation tasks, each with distinct observability and dynamics and equipped with oracle multi-step solvers for supervised finetuning (framework comparison in Tab. 1). Crucially, VisGym provides fine-grained controls over input representation, difficulty, history length, planning horizon, and feedback, enabling domain-agnostic, systematic analysis of model behavior. Building on prior domain-specific studies, we conduct cross-domain controlled experiments that examine how these factors, together with module finetuning and data curation, affect performance in multi-step visual decision-making.

Across 12 state-of-the-art models, even the strongest achieve only 46.61% and 26.00% success in the easy and hard settings, respectively. Our analyses reveal several concrete, cross-domain failure modes: (1) models struggle to effectively leverage long-term context, showing a reversed-U relationship where performance degrades as the context grows unbounded; (2) VLMs struggle with low-level perceptual grounding, a limitation highlighted by symbolic variants of tasks being substantially easier than their visually rendered counterparts; (3) models struggle to infer task states and outcomes from purely visual transitions, consistently relying

<span id="page-2-1"></span><span id="page-2-0"></span>Table 2. **VisGym environments.** For each environment, we specify (1) **Domain**: whether observations come from **Real** or **Synthetic** images, (2) **Observability (Obs.)**: **Full** or potentially **Partial**, (3) **Dynamics (Dyn.)**: **Known** vs. **Unknown** dynamics, (4) **Parameters (P.)**: number of difficulty parameters, and (5) **Available Actions**.

| Environment                             | Domain    | Obs.    | Dyn.    | P. | Available Actions                                      |
|-----------------------------------------|-----------|---------|---------|----|--------------------------------------------------------|
| Colorization (103)                      | Real      | Full    | Known   | 1  | $rotate(\theta)$ ; $saturate(\delta)$ ; $stop()$       |
| Counting (30)                           | Real      | Full    | Known   | 2  | mark(x, y); undo(); guess(N); stop()                   |
| Jigsaw (27)                             | Real      | Full    | Known   | 2  | $swap((r_1,c_1),(r_2,c_2)); reorder([\ldots]); stop()$ |
| Matchstick Equation (42)                | Synthetic | Full    | Known   | 1  | move([i, s, j, t]); undo(); stop()                     |
| Matchstick Rotation (44)                | Synthetic | Full    | Unknown | 3  | $move([dx, dy, d\theta]); stop()$                      |
| Maze 2D (43)                            | Synthetic | Full    | Known   | 2  | move(d); stop()                                        |
| Maze 3D (43)                            | Synthetic | Partial | Known   | 2  | move(0); turn(d); stop()                               |
| Mental Rotation 2D (18)                 | Real      | Full    | Known   | 1  | $rotate(\theta)$ ; $stop()$                            |
| Mental Rotation 3D (CUBE) (66; 70)      | Synthetic | Partial | Known   | 3  | rotate([dy, dp, dr]); stop()                           |
| Mental Rotation 3D (Objaverse) (70; 20) | Synthetic | Partial | Known   | 1  | rotate([dr, dp, dy]); stop()                           |
| MuJoCo Fetch (Pick-and-Place) (86)      | Synthetic | Partial | Unknown | 0  | move([x, y, z]); gripper(g); stop()                    |
| MuJoCo Fetch (Reach) (86)               | Synthetic | Partial | Unknown | 0  | move([x, y, z]); stop()                                |
| Patch Reassembly (28)                   | Synthetic | Full    | Known   | 2  | place(p, r, c); remove(p); stop()                      |
| Referring Dot-Pointing (39)             | Real      | Full    | Known   | 0  | mark(x, y); $stop()$                                   |
| Sliding Block (75)                      | Synthetic | Full    | Known   | 1  | move(b,d); stop()                                      |
| Video Unshuffle (29; 60)                | Real      | Full    | Known   | 3  | swap(i, j); reorder([]); stop()                        |
| Zoom-In Puzzle (6)                      | Real      | Full    | Known   | 5  | swap(i,j); reorder([]); stop()                         |

on explicit textual feedback to boost performance; (4) the benefit of providing explicit goal observations is brittle and can backfire: while explicit goals can yield large gains, limited visual perception can cause models to misidentify them and, paradoxically, perform worse than with no goal at all; (5) models fail to learn from standard demonstrations under partial observability or unknown dynamics, requiring information-revealing demonstrations that expose hidden states or clarify dynamics to significantly improve supervised finetuning outcomes.

Together, these findings establish VisGym as a unified and extensible framework for diagnosing, understanding, and ultimately improving VLMs in visually interactive decision-making.

