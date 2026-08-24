# 5 A Framework for Evaluating O1 Replication Claims: The Technical Transparency Index

To systematically evaluate and compare various O1 replication attempts, we propose the Technical Transparency Index (TTI), a comprehensive framework that quantifies the transparency and reproducibility of claimed implementations. This framework aims to provide the research community with objective metrics for assessing the openness and verifiability of different approaches.

## <span id="page-9-0"></span>5.1 Evaluation Dimensions of Transparency

The framework evaluates O1 replication efforts with a primary focus on transparency, which is assessed across several interconnected aspects. These include the data transparency, encompassing the accessibility, quality, and documentation of datasets used for downstream search or post-training; methodological transparency, reflected in the clarity and detail of the described techniques, processes, and experimental setups; and evaluation transparency, which considers the reproducibility and comprehensiveness of performance evaluations. Additionally, the framework examines the openness of resources, such as the availability of code, datasets, and models, to ensure that the work can be independently verified and effectively utilized by the research community. This comprehensive perspective captures the multifaceted nature of transparency in replication efforts. Details will be introduced below.

#### Index 1: Data Transparency

This aspect evaluates whether the origin of the data is clearly specified, including detailed descriptions of the datasets used and their respective sources. It considers whether the dataset names, providers, or publications from which the data is derived are explicitly mentioned. This applies to all datasets used in downstream tasks such as supervised fine-tuning (SFT), reinforcement learning (RL), or search algorithms and becomes even more crucial when the datasets serve as seed data for synthesizing long thought data.

- Data Source: This aspect evaluates whether the origin of the data is clearly specified, including detailed descriptions of the datasets used and their respective sources. It considers whether the dataset names, providers, or publications from which the data is derived are explicitly mentioned.
- Data Selection Process: This focuses on the clarity and rigor of the criteria and methodology used for filtering, cleaning, or preprocessing data prior to its application in downstream tasks such supervised fine-tuning (SFT), searching, or reinforcement learning (RL).

#### Index 2: Methodology Transparency

Methodology transparency ensures that the approach, techniques, and processes employed in the work are described in sufficient detail to enable independent reproduction and validation. This section evaluates multiple components, from foundational model descriptions to training and data curation methods. Moreover, in addition to detailing how a method is implemented, it is even more important to validate the effectiveness of the method itself. It highlights the importance of validating the effectiveness of each method employed. A thorough evaluation should quantify the contributions of individual techniques to the overall system performance, rather than simply reporting final results.

- Foundation Model Details: This evaluates the depth and clarity of information provided about the base model used in the work. It includes details such as the architecture (e.g., transformer layers, attention mechanisms), parameter size (number of trainable parameters). The goal is to ensure that the foundational components of the approach are fully understood and reproducible.
- Search Algorithm: This focuses on the explanation of the search algorithm employed for inference-time scaling. It assesses whether the methodology for applying techniques like beam search, monte carlo tree search (MCTS), or other strategies is well-documented, including parameters, step-by-step processes, and any custom modifications.
- RL Algorithm: This examines the details of reinforcement learning (RL) or preference learning approaches (e.g., Direct Preference Optimization). It includes the specification of reward functions, optimization goals, and training dynamics.
- Long Thought (O1-like) Synthetic Algorithm: This aspect assesses the process of creating or synthesizing long-thought (O1-like) datasets. It includes explanations of any specific algorithms, heuristics, or rules applied in data generation or selection.
- Training Details: This examines the documentation of training procedures, including key hyper-parameters (e.g., learning rate, batch size, optimizer types) and the overall training configuration.

• Effectiveness Validation: This evaluates whether the effectiveness of each method is rigorously validated. For instance, ablation studies, comparative experiments, or incremental analyses should be conducted to quantify how individual techniques contribute to the overall system. Such validations ensure that claims about the importance of a method are backed by clear empirical evidence, fostering transparency and reproducibility.

#### Index 3: Evaluation Transparency

- Benchmark Usage: This evaluates the selection of benchmarks used to assess model performance, considering whether the chosen benchmarks are appropriate for the task and domain.
- Evaluation Metrics: This assesses the metrics used to quantify model performance, such as pass@k, maj@k, or rm@k. It examines the clarity of metric definitions, their relevance to the specific task, and any customizations introduced to address unique aspects of the evaluation. Additionally, it evaluates how metrics are standardized and aligned across baselines to ensure fair and unbiased comparisons.

## Index 4: Open-Source Resources

Open-source resources play a vital role in fostering reproducibility and enabling the research community to build upon existing work. This section evaluates the availability and accessibility of datasets, models, code, and documentation, which are essential for independent validation and further experimentation.

- Data: This evaluates whether the post-training raw data and the synthesized O1-like datasets are made publicly available for use. Open availability of these datasets significantly enhances reproducibility and enables researchers to apply them to additional tasks.
- Model Weights: This assesses the public release of the trained model weights. Sharing model weights facilitates replication and further optimization efforts.
- Code: This considers whether the released codebase includes scripts for both training the model and evaluating its performance. A complete and well-documented codebase is crucial for enabling others to reproduce and validate the work.
- Documentation: This examines the availability of supplemental documentation, such as research papers, technical reports, or blog posts. It assesses whether these materials clearly explain the methodology, results, and underlying ideas, and whether they provide actionable insights for researchers and practitioners.

## 5.2 Checklist for O1-style Technique

Scoring Framework (100 Points) We propose a scoring framework that provides a unified approach to assess O1 replication efforts by focusing exclusively on transparency, with a total score of 100 points (see Table [3\)](#page-11-0). This focus underscores the critical importance of reproducibility and openness in evaluating the quality of replication efforts. The framework evaluates key dimensions as detailed in Section [5.1,](#page-9-0) ensuring a comprehensive and fair assessment of each work's commitment to clarity and accessibility. By emphasizing transparency through a systematic checklist approach, this scoring system highlights the foundational aspects necessary for building trust and driving further advancements in the field.

Binary Score Under this framework, every evaluation indicator in the checklist is assessed through a simple Yes/No question, with each "Yes" response contributing its designated points to the total score. The binary nature of this system ensures clarity and consistency in evaluation, as each indicator is either fully satisfied or not. This method prioritizes transparency over implementation scope. For example, if a work explicitly acknowledges that it does not employ a particular technique (e.g., reinforcement learning), it will still receive full points for transparency in that indicator, as openly documenting such details reflects a commitment to reproducibility and openness.

When assigning point values to each indicator, we carefully weigh their relative importance in the technical pipeline. Indicators deemed to have a more significant impact on the success and reproducibility of O1 replication efforts are given higher point values. For instance, transparency in the search algorithm and the long thought data synthesis algorithm is assigned higher scores, reflecting their critical roles in achieving high-quality and reproducible results. This weighted scoring ensures that the framework aligns with the priorities of the technical process, emphasizing the documentation of key components that drive the overall system's performance and reproducibility.

<span id="page-11-0"></span>

| Evaluation Dimensions | Checklist                                                                                                                         | Score |
|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------|-------|
|                       | Are dataset names, sources, and providers explicitly documented and<br>properly cited?                                            |       |
| Data (14)             | Is there sufficient documentation of data distributions, formats, and<br>characteristics to enable proper replication?            |       |
|                       | Are the criteria and methodology for data selection and filtering clearly<br>justified and documented?                            |       |
|                       | For synthetic data generation, is the entire process transparent, including<br>prompting strategies and quality control measures? | 4     |
|                       | Is there a clear and complete description of the base model (including<br>its architecture, size, etc.)?                          |       |
|                       | Is the complete search algorithm implementation (e.g., beam search,<br>MCTS) detailed with all components?                        | 6     |
| Methodology (33)      | Is the RL algorithm fully specified with its objective function and<br>training procedure?                                        |       |
|                       | Is the long thought data curation/generation algorithm thoroughly ex<br>plained with its complete workflow?                       |       |
|                       | Is the complete training pipeline documented, including all stages and<br>their sequence?                                         |       |
|                       | Are the computational requirements and infrastructure details provided?                                                           | 2     |
|                       | Is there clear documentation of all training hyperparameters and opti<br>mization choices?                                        | 2     |
|                       | Are there comprehensive ablation studies showing the contribution of<br>each major component?                                     | 4     |
|                       | Is there a clear justification for the selection of evaluation benchmarks?                                                        |       |
|                       | Is the evaluation dimension clearly specified (e.g., answer-level, step<br>by-step level)?                                        | 4     |
| Evaluation (24)       | Are all evaluation metrics (e.g., pass@k, maj@k) clearly defined?                                                                 | 4     |
|                       | For any custom metrics (if exists), are they well-justified and clearly<br>documented?                                            |       |
|                       | Are the evaluation metrics consistently applied across all baselines?                                                             | 4     |
|                       | Are the evaluation conditions (e.g., temperature, top-p) explained for<br>all compared methods?                                   | 4     |
|                       | Is the post-training data publicly available?                                                                                     | 3     |
|                       | Is the synthetic long thought data publicly available?                                                                            |       |
|                       | Are trained model weights publicly available?                                                                                     | 5     |
| Open-Source (29)      | Is the complete training codebase publicly available?                                                                             | 4     |
|                       | Is the complete evaluation codebase publicly released?                                                                            | 4     |
|                       | Are there step-by-step guidance and instruction for code usage?                                                                   | 4     |
|                       | Is there a comprehensive technical paper detailing all research aspects<br>instead of a brief blog post?                          | 4     |

Table 3: Transparency scoring framework for O1 replication efforts. Each evaluation point of the checklist is assigned a score based on their transparency criteria. The total transparency score sums up to 100 points.

## 5.3 Compared Works

We include a comprehensive evaluation of existing attempts to replicate O1, assessing them across both transparency and performance dimensions. The works we cover include Open O1 [\(Team,](#page-15-2) [2024b\)](#page-15-2), O1-Journey (Part 1) [\(Qin](#page-15-0) [et al.,](#page-15-0) [2024\)](#page-15-0), LLaMA-O1 [\(Team,](#page-15-3) [2024a\)](#page-15-3), k0Math [\(kimi,](#page-14-0) [2024\)](#page-14-0), Skywork O1 [\(kunlun,](#page-14-1) [2024\)](#page-14-1), Deepseek-R1- Lite [\(deepseek,](#page-14-2) [2024\)](#page-14-2) and this work O1-Journey (Part 2). These comparisons provide a holistic view of the current progress in O1 replication efforts, highlighting their strengths and areas for further improvement.

## 5.4 Leaderboard

<span id="page-12-0"></span>

| Work               | Data (14) | Methodology (33) | Evaluation (24) | Open-Source (29) | Total Score |  |
|--------------------|-----------|------------------|-----------------|------------------|-------------|--|
| Open O1            | 0         | 8                | 20              | 5                | 33          |  |
| O1-Journey (Part1) | 10        | 33               | 24              | 9                | 76          |  |
| LLaMA-O1           | 0         | 6                | 0               | 5                | 11          |  |
| K0Math             | 0         | 0                | 16              | 0                | 16          |  |
| Skywork O1         | 0         | 0                | 0               | 0                | 0           |  |
| DeepSeek-R1-Lite   | 0         | 0                | 20              | 0                | 20          |  |
| O1-Journey (Part2) | 10        | 33               | 24              | 12               | 79          |  |

Table 4: Transparency scores of various O1 replication efforts. Each column represents a specific method, with individual scores provided for each evaluation dimension and indicator. The total transparency score is calculated out of 100 points, reflecting the openness and reproducibility of each approach.

The leaderboard (Table [4\)](#page-12-0) showcases the transparency levels of various O1 replication efforts, with our work achieving a perfect transparency score. This result highlights our commitment to openness and reproducibility, building upon the solid foundation established by O1-Journey (Part 1). Together, the O1-Journey series sets a new benchmark for transparency by excelling across all evaluation dimensions, including data accessibility, methodology clarity, and open-source resource availability.

