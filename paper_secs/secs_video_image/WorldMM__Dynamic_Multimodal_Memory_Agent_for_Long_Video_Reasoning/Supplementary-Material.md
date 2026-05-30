# Supplementary Material

In this supplementary material, we provide additional details on the dataset (Sec. A), additional implementation details (Sec. B) and descriptions on experiments (Sec. C). We also present detailed and additional experimental results (Secs. D and E), qualitative analyses (Sec. F), and a discussion of limitations and broader impacts (Sec. G).

### <span id="page-11-0"></span>A. Additional Details on Dataset

In this section, we provide additional details for each dataset used in our experiments. Tab. 5 summarizes the datasets, including the number of queries, domain categories, and the average video duration.

<span id="page-11-2"></span>Table 5. Summary of benchmark datasets used in experiments.

| Dataset           | # Queries | Domain     | Avg. Video Length |
|-------------------|-----------|------------|-------------------|
| EgoLifeQA [41]    | 500       | Egocentric | 44.3h             |
| Ego-R1 Bench [30] | 300       | Egocentric | 44.3h             |
| HippoVlog [19]    | 1,000     | Vlog       | 0.45h             |
| LVBench [32]      | 1,534     | General    | 1.14h             |
| Video-MME (L) [7] | 900       | General    | 0.69h             |

## A.1. EgoLifeQA

EgoLifeQA [41] is a set of questions designed to test the capability of models to understand and remember everyday life from week-long video recordings. It includes questions that require recalling past events, tracking object locations, and reasoning over long-term activities. In our experiments, we use questions from the perspective of a single participant (A1: JAKE), along with his corresponding video stream, which spans 44.3 hours. The benchmark is organized into five distinct categories as follows.

**EntityLog (Ent.)** Questions that require recalling information about objects, such as their locations, states, or interactions. (Example: "Who used the screwdriver first?")

**EventRecall (EvR.)** Questions that ask about specific past events, including what happened, when it occurred, and relevant context. (Example: "Shure mentioned Tiramisu, when was the last time we discussed making Tiramisu?")

**HabitInsight (Hab.)** Questions aimed at identifying a person's recurring behaviors or long-term activity patterns. (Example: "What food does Alice love to eat?")

**RelationMap (Rel.)** Questions involving understanding social relationships and interactions between people. (Example: "Who usually sings when Shure plays the guitar?")

**TaskMaster** (**Task**) Questions focused on ongoing or pending tasks that require reasoning about what actions still need to be completed. (Example: "What are we planning to do in the afternoon?")

