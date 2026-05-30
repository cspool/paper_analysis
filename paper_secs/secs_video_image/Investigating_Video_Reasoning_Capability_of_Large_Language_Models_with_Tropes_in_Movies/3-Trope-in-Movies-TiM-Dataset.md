# 3 Trope in Movies (TiM) Dataset

Overview TiM comprises (1) 684 movies, each annotated with per-shot keyframes, subtitles, and trope labels, and (2) 95 trope identification queries accompanied by their definitions. The TiM dataset is specifically designed to pose more demanding and intricate reasoning tasks in video analysis, particularly focusing on extended content such as movies. The homepage of the TiM dataset[1](#page-3-0) offers a download link for the TiM data along with detailed explanations of the annotations. Additionally, we have provided a pre-processing script for our baseline models in Section [4](#page-4-0) to facilitate reproduction of our experimental results.

Trope Considering the broad diversity of tropes, we utilize a set of 95 tropes categorized into four groups as introduced by TiMoS [\[19\]](#page-9-7), depicted in Figure [2.](#page-3-1) Subsequent research could explore expanding the dataset by incorporating additional tropes. The categories used are Character Traits, Role Interaction, Situation, and Storyline. Character Traits analyze individual strengths and personalities, showing their impact on behavior and interactions within the story. Role Interaction explores the dynamics between characters and their influence on the film's development. Situation covers specific scene-level scenarios that drive the plot with abstract concepts and emotional dynamics. Storyline focuses on the overall narrative structure, guiding the flow and thematic elements through-

![](_page_3_Figure_6.jpeg)

<span id="page-3-1"></span>Figure 2: Word cloud of trope occurrences in Fullset, size of the tropes in proportion to their frequency in Fullset and color of the tropes correspond to the category they belongs

out the film. Together, these categories offer a comprehensive framework for analyzing the complex interplay of tropes in cinematic narratives.

Task Definition We formulate the task considered here as binary classification: y = f(movie, trope), where y ∈ {True, False} indicates whether a given trope is present in the movie. This simplifies the task and enhances the focus on complex reasoning for single tropes in movies. Future research could consider revisiting the more challenging multi-label tasks [\[19\]](#page-9-7).

Evaluation We have selected the micro F1 score as the primary metric for global comparison within the chosen set in TiM.

<span id="page-3-0"></span><sup>1</sup> https://ander1119.github.io/TiM/

Data Collection We sourced trope occurrences in movies from the TiMoS dataset [\[19\]](#page-9-7), originally compiled from the TVTropes database. Movie frames and subtitles were gathered from the MovieNet dataset [\[36\]](#page-10-6). We aligned the movies with their corresponding tropes using their IMDb IDs. Future research could extend this dataset by collecting more movies.

<span id="page-4-1"></span>Table 1: Comparison between different experiment setups.

| Setting          | Movies    | Frames           | Subtitles<br>Line | Char       | Tropes        |  |
|------------------|-----------|------------------|-------------------|------------|---------------|--|
| Fullset          | 684       | 1545.7           | -                 | -          | 11.91         |  |
| VDset<br>Mainset | 246<br>50 | 1585.9<br>1699.6 | 1587.4<br>1822.2  | 56k<br>65k | 13.38<br>6.08 |  |

Data Statistics This benchmark is tailored for LLM-based methods, utilizing the entire dataset as the test set. Supervised learning evaluations are conducted using 5-fold cross-validation. To accommodate the absence of some subtitles in the MovieNet dataset, we offer the *VDset*, which includes subtitles. Additionally, the *Mainset*—a subset of 50 movies—is provided for more detailed analysis as experiments may require additional time or resources. Table [1](#page-4-1) presents a comparative analysis of different experimental setups.

