# <span id="page-7-0"></span>3 Post-Training

Based on the pre-trained model of Hunyuan-Large, we further conduct a post-training stage that aims to enhance task-specific capabilities and align LLM to human preference. This stage contains a supervised fine-tuning (SFT) phase and a Reinforcement Learning from Human Feedback (RLHF) phase on elaborately selected datasets and outputs of current policy models (Bai et al., 2022). The following subsections contain (a) the data selection, preprocessing, and training process of SFT, (b) the techniques and training strategies of Direct Preference Optimization (DPO) in RLHF.

#### 3.1 Supervised Fine-Tuning

The performance of SFT strongly depends on the quality of instruction data related to various types of LLM capabilities. In SFT, we concentrate on the detailed data collection and processing manners that ensure the effectiveness of Hunyuan-Large's post-training, along with the training settings of SFT.

## 3.1.1 Overview of SFT Data

The central goal of SFT is further enhancing its performance across multiple key capabilities based on the corresponding well-selected data. These capabilities primarily encompass mathematics, coding, logical reasoning, knowledge-based question answering, agent behavior, text generation, NLP comprehension, industrial applications, role-playing, long-text capabilities, etc. We recognize that improving these abilities not only enables the model to be more adept in practical applications, but also better satisfies users' diverse needs across multiple scenarios. Simultaneously, we place great emphasis on data security, striving to ensure that the model aligns with human values under most circumstances. The overall SFT data volume exceeds 1 million.

## 3.1.2 Data Collection and Processing

The key techniques of SFT data collection and processing mainly include instruction extraction, instruction generalization, instruction balancing, and data quality controlling.

Instruction Extraction. To enhance the breadth and diversity of the instruction set, we develop an instruction extraction model specifically for domains such as mathematics, logical reasoning, and knowledge-based question answering, whose primary goal is to effectively extract data suitable for instruction tuning from publicly available data sources (e.g., web pages, encyclopedias, etc.). The extracted data includes both instructions and corresponding reference answers. We develop many specialized models as instruction extractors. With the help of these model, we successfully extract a large set of natural instructions from public data. These instructions play a crucial role as the seed to enhance the final model's generalization performance and diversity.

Instruction Generalization. We propose an instruction generalization method to obtain more diverse and complex instructions in large quantities. Specifically, we design and train an instruction generalization system capable of generalizing targeted instructions while gradually increasing their difficulty and complexity levels. The central recipe of this system lies in training the model by synthesizing numerous mappings between simple and complex instructions. In addition, we construct a well-structured instruction taxonomy with its corresponding classification models, which aims to analyze and balance the distribution of various instruction types in SFT data. Armed with this instruction taxonomy, our instruction generalization system can supplement the original data on specific weak instructions of targeted types.

Instruction Balancing. Through the instruction extraction and generalization processes, we accumulate more than 10 million instructions. Instruction balance is essential for enhancing the model's performance across various scenarios. However, many generated instructions have very similar semantic meanings and the instruction type distribution is naturally unbalanced. To enhance the instruction complexity while maintaining balanced instruction distributions, we attach labels for each instruction. These labels encompass multiple dimensions. By meticulously tagging these labels, we can more accurately understand and analyze the characteristics of our instruction sets. By ensuring adequate amounts and balanced distribution of different types of instructions during the SFT process, we can effectively alleviate overfitting or underfitting problems on specific instruction types, thereby improving the model's generalization capabilities and adaptability across diverse application scenarios.

Data Quality Controlling. The quality of SFT data is the foundation of superior performance. We mainly conduct the following three methods to ensure the high quality of our SFT data.

- Rule-based Filtering. We discover some common issues such as data truncation errors, duplication, garbled characters, and format errors in SFT data. Consequently, we develop a set of rule-based data filtering strategies to prevent the above instruction extraction and generation models from producing undesirable outputs.
- Model-based Filtering. To automatically extract high-quality SFT data from a substantial volume of synthesized instruction data, we train a critique model [\(McAleese et al., 2024\)](#page-16-13) based on a 70B dense model of our Hunyuan series. This model assigns a four-tier quality score to each instruction sample, assessing aspects such as the accuracy, relevance, completeness, usefulness, and clarity of the generated responses, and other possible data quality issues.

• Human-based Filtering. Prior to model training, the SFT data filtered via rule-based and modelbased methods further undergo human annotation, ensuring that answers adhere to the desired task-specific response patterns and avoid introducing additional low-quality issues.

## 3.1.3 Training Details

In SFT, we fine-tune the pre-trained model based on the high-quality data (more than 1 million) for a total of 3 epochs. The learning rate decays from 2e-5 to 2e-6. To mitigate overfitting during SFT, we utilize an attention dropout of 0.1 and a hidden dropout of 0.2. We find that, compared to the dense models, the MoE architecture of Hunyuan series could benefit more from incorporating suitable dropout rates.

## 3.2 Reinforcement Learning from Human Feedback

To align Hunyuan-Large with human preferences, we further train our SFT model using DPO [\(Rafailov](#page-16-14) [et al., 2024\)](#page-16-14). We adopt a single-stage training strategy that integrates both offline and online training, which demonstrates superior controllability and overall performance. In this integrated approach, we utilize a pre-compiled preference dataset to enhance controllability, while simultaneously employing the current policy model to generate multiple responses for each prompt and our reward model to select the most and least preferred responses.

To enhance training stability, we incorporate an SFT loss term on the chosen response, similar to the approaches in [\(Dubey et al., 2024;](#page-14-2) [Adler et al., 2024\)](#page-14-10). This addition helps stabilize DPO training by preventing a decrease in the log probability of chosen responses. Furthermore, we implement an exponential moving average strategy to mitigate reward hacking and reduce alignment tax [\(Ouyang](#page-16-15) [et al., 2022\)](#page-16-15), ensuring a more stable training process across a larger dataset.

