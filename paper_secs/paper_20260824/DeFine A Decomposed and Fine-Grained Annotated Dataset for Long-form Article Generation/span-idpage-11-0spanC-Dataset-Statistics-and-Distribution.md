# <span id="page-11-0"></span>C Dataset Statistics and Distribution

We randomly selected 2,286 outline data samples and 3,625 question-answer pairs, with an equal split between Chinese and English, from De-Fine dataset to fine-tune the Qwen2-7b-Instruct model [\(Yang et al.,](#page-9-16) [2024\)](#page-9-16) for long-form article generation tasks in order to validate the effectiveness of our dataset. To further enhance the model's capabilities in information extraction and compression tasks, we trained the BGE-m3 relation extraction model using abstract set data. This model is designed to extract key information from ref-

## Algorithm 1: Hallucination Detection Algorithm for Citation Reliability (HDACR)

<span id="page-12-0"></span>**Input:** Generated content G and Reference content R

Output: Hallucination detection result H

**Initialization**: Reference content entity set  $E_r \leftarrow \emptyset$ , Generated content entity set  $E_g \leftarrow \emptyset$ , Hallucination detection result  $H \leftarrow$  empty;

Extract Reference Content Entities: For each model M in the model set, extract entities  $E_{rM}$  from the reference content  $R_r$ , add  $E_{rM}$  to  $E_r$ ;

Extract Generated Content Entities: For each model M in the model set, extract entitie $E_{gM}$  from the generated content G, add  $E_{gM}$  to  $E_g$ ;

**Calculate Matching Scores**: Initialize the matching score list  $\gamma \leftarrow []$ . For each entity e in  $E_g$ :

- If e matches hard  $(e \in E_r)$ , then  $\gamma \leftarrow 1.0$ ;
- Otherwise, perform soft matching, calculate Sentence-BERT score  $\gamma_{SBERT}$  and BM25 score  $\gamma_{BM25}$ , calculate weighted score  $\gamma \leftarrow \frac{\gamma_{SBERT} + \gamma_{BM25}}{2}$ , add  $\gamma$  to the matching score list.

**Determine the Presence of Hallucinations**: If  $\exists \gamma_i < 0.6$ , mark hallucination detection result:  $H \leftarrow$  Hallucination present; otherwise, mark hallucination detection result  $H \leftarrow$  No hallucination **Output Results**: Output generated content G, hallucination detection result H, reference content R, and unverifiable entities and their positions in the text;

erences and condense full-text content into summaries through precise relation extraction and compression.

From the remaining data, 270 outline samples and 270 QA pairs were reserved to evaluate the outline generation and long-article generation performance of the fine-tuned model. The fine-tuned model is referred to as *Qwen2-7b-Scribe*. To validate the effectiveness of our methods, we selected 230 distinct topics from various domains, including 130 topics in Chinese and 100 in English, to generate full-length articles.

Table 6 provides comprehensive statistics on the various components of the dataset used in this study. Each category is further divided into English (eng.) and Chinese (ch.) data, with columns detailing the number of examples in the test set, training set, and the total number of examples available.

Notably, the English data contains a significantly larger number of samples compared to the Chinese data. This discrepancy arises from the larger volume of English-language Wikipedia entries; it has a far richer and more extensive collection of articles than its Chinese counterpart, with a greater number of contributors and editors actively engaged in expanding and updating content. According to recent studies, English Wikipedia hosts over 6 million articles, while Chinese Wikipedia has just over 1 million articles (contributors, 2024). This has resulted in a more substantial dataset for English,

leading to the selection of a representative subset to avoid redundancy and overfitting in training. By doing so, we aimed to ensure that the training process remains efficient while still providing the model with sufficient variability and representation from the dataset.

Furthermore, the varying depth and detail within entries also contribute to the richness of the English dataset. Articles on complex topics often include extensive references and citations, which enhance the dataset's quality and comprehensiveness. In contrast, the Chinese entries, while valuable, may not always match this level of detail, particularly in less commonly covered subjects. This situation necessitates a careful curation of data to maintain a balanced representation in the model's training set, ensuring that both languages are adequately represented in the resulting outputs.

The distribution of various data types is shown in Figure 5 and Figure 6. Figure 5 illustrates the access distribution of QA data topics in the dataset, while Figure 6 provides similar insights for Abstract data topics. The categories ending with **ch** represent Chinese data, while all other categories correspond to English data. The percentages displayed in the figures indicate the proportion of accesses from each topic, highlighting the dataset's diversity. This rich distribution is essential for training models capable of understanding and generating content across a variety of topics and styles,

<span id="page-13-0"></span>

| Data Type              | Test. | Train. | All   |
|------------------------|-------|--------|-------|
| Outline data eng.      | 120   | 1026   | 1226  |
| Outline data ch.       | 150   | 941    | 50787 |
| Abstract Set Data eng. | \     | 200    | 8525  |
| Abstract Set Data ch.  | \     | 200    | 1122  |
| QA data eng.           | 133   | 1026   | 5244  |
| QA data ch.            | 120   | 1138   | 1258  |

Table 6: This table provides detailed information about our dataset. The dataset is categorized into Outline data, Abstract Set data, and QA data, with separate columns for English (eng.) and Chinese (ch.). The backslash ('\') denotes that no test data is available for the Abstract Set in either English or Chinese.

further enhancing the robustness and applicability of our findings.

## D Case Study

In this appendix, we present examples of the results generated by different models used in our experiments. We provide outlines, QA data, and full generated articles for several topics.

## **D.1** Outline Generation Examples

Here is the section where we present the outlines generated by different models for the same task and compare them with the correct answers. The task is to generate a directory for the article topic "首尔大学" and "John Gribbel." Below, we list the generated results from each model alongside the correct answers.

#### **D.2** Chinese Outline Generation

Instruction: 请根据'首尔大学'这一主题,生成一篇长篇文章的目录结构。目录应包括多个主要部分和其子部分,以确保涵盖主题的各个方面。

## **D.2.1** Original Outline

- 1. 摘要
- 2. 正文
  - 2.1 历史
    - 2.1.1 早期历史
    - 2.1.2 国立首尔大学时期
    - 2.1.3 冠岳校区
    - 2.1.4 莲建校区
    - 2.1.5 平昌校区
  - 2.2 学生与校友
  - 2.3 大学排名
    - 2.3.1 QS世界大学排名
    - 2.3.2 泰晤士高等教育世界大学排名
    - 2.3.3 软科世界大学学术排名

- 2.4 交通
  - 2.4.1 校本部
  - 2.4.2 莲建校区
  - 2.4.3 平昌校区
- 2.5 参看
- 2.6 外部连结

## D.2.2 Llama3 Output

- 基本信息
- 历史沿革
- 学术研究
- 院系设置
- 校园设施
- 国际交流
- 名人校友
- 社会影响

