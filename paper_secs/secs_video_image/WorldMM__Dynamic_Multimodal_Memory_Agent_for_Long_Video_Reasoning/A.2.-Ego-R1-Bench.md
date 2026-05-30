# A.2. Ego-R1 Bench

Ego-R1 Bench [30] is designed as a complementary evaluation to EgoLifeQA, but with a distinct focus on model reasoning. While both benchmarks focus on the same week-long egocentric video, Ego-R1 Bench targets multistep, tool-augmented reasoning over ultra-long video. We reorganize query types of Ego-R1 Bench to the category adopted by EgoLifeQA, as shown in Tab. 6.

<span id="page-11-3"></span>Table 6. Classification of queries under the EgoLifeQA category.

| Category     | Ego-R1 Category                               |
|--------------|-----------------------------------------------|
| EntityLog    | EntityLog, FoodLog, HealthLog, TechLog        |
| EventRecall  | EventRecall, Event Recollection, Event Memory |
| HabitInsight | HabitInsight, Behavior Habit(s)               |
| RelationMap  | RelationMap, Interpersonal Relationships      |
| TaskMaster   | TaskMaster, Future Plan(s)                    |

## A.3. HippoVlog

HippoVlog [19] contains 25 daily vlog videos with 1,000 multiple-choice questions for continuous audiovisual event understanding. The benchmark evaluates a model's ability to handle modality-specific information, with **Auditory** (**Aud.**) questions requiring reasoning over the audio stream (or transcript) and **Visual** (**Vis.**) questions focusing on the visual content. **Auditory+Visual** (**A+V**) queries test the model's ability to integrate information across both modalities, while **Summarization** (**Summ.**) questions assess higher-level reasoning over long temporal spans, requiring synthesis of events and semantic understanding from the continuous video.

#### A.4. LVBench

LVBench [32] consists of 103 long videos, typically longer than an hour, with 1,549 multiple-choice questions for extreme long video understanding. The videos cover a general and diverse set of domains. Questions include both visual perception for recognizing entities or events in short segments and summarization for higher-level reasoning across

<span id="page-12-2"></span>extended sequences, evaluating models' ability to integrate information over both local and long-horizon contexts. In our experiments, we categorize questions into three groups based on their segment length, defined as the duration of video required to answer the question: Short (<30s), Medium (Med.) (30s∼5min), and Long (>5min). We excluded 15 questions without segment tags, leaving 1,534 questions in total for evaluation.

