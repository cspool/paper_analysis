# <span id="page-13-0"></span>B Benchmarks

We conduct the experiments on several widely used visual understanding benchmarks. In the following, we will give a detailed description of these benchmarks.

GQA. [\[13\]](#page-10-12). The GQA benchmark consists of three components: scene graphs, questions, and images. The image component includes raw images, their spatial features, and the features of all objects within the images. The questions in GQA are crafted to evaluate visual scene understanding and reasoning about various aspects of an image. Our method is evaluated on the subset of "testdev\_balanced\_instructions", which includes 12,578 samples.

MMBench. [\[27\]](#page-11-4). MMBench is a comprehensive benchmark designed to evaluate the multi-modal capabilities of large language models, covering a wide range of tasks including visual question answering, image captioning, cross-modal retrieval, and creative generation. It provides a finegrained assessment from perception to cognition, containing approximately 3,000 multiple-choice questions aggregated from diverse sources. The benchmark aims to measure whether a model is a true "all-around player" in multi-modal understanding and reasoning.

MME. [\[12\]](#page-10-0). The MME benchmark is a comprehensive evaluation suite carefully crafted to assess multiple facets of model performance. It comprises 14 distinct subtasks targeting both perceptual and cognitive capabilities of models. By employing manually curated instruction-answer pairs and succinct instruction formats, MME effectively reduces the risks of data leakage and ensures a fairer assessment of model abilities. We evaluate the performance on the dev split including 4,377 samples. The evaluation metric is the accuracy of the model's answer.

POPE. [\[22\]](#page-11-13) The POPE benchmark focuses on assessing object hallucination in models by presenting them with a set of targeted yes/no questions about object existence within images. This approach reframes the evaluation of hallucination, emphasizing the model's ability to correctly identify whether certain objects are present. To quantitatively analyze performance across three distinct sampling methods, the benchmark utilizes metrics such as accuracy, recall, precision, and F1 score, offering a robust measure of the model's susceptibility to hallucination. We evaluate the model's performance on the test split, including 9,000 samples. The evaluation metric is the F1 score.

ScienceQA (SQA). [\[29\]](#page-11-14) Encompassing a wide array of fields such as natural sciences, linguistics, and social sciences, SQA structures its questions through a hierarchical framework consisting of 26 topics, 127 categories, and 379 distinct skills. This benchmark is designed to rigorously test a model's proficiency in multimodal comprehension, complex reasoning across multiple steps, and interpretability. By organizing questions first by subject area, then by specific category, and finally by the required skill, SQA ensures a thorough and nuanced assessment of scientific understanding across diverse domains. This layered organization enables a detailed evaluation of a model's ability to handle a broad spectrum of scientific queries. The evaluation metric is the accuracy.

TextVQA. [\[36\]](#page-12-11) TextVQA is designed to assess a model's capability to interpret and reason over textual content embedded in images. This benchmark challenges models with visual question answering tasks that require both comprehension of image context and accurate reading of the text present within the images. To perform well, models must effectively integrate visual and textual cues, demonstrating robust understanding and reasoning skills related to text in complex visual environments. We evaluate the model's performance on the test split, including 5,000 samples. The evaluation metric is exact match (EM).

SEEDBench [\[18\]](#page-11-15) SEEDBench features a collection of 19,000 multiple-choice questions curated by human annotators. Covering 12 different evaluation dimensions, this benchmark examines models'

<span id="page-14-2"></span>Table 6: Performance comparison under different vision token configurations. The evaluated model is LLaVA 1.5 13B, where the default number of visual tokens is 576. The first row for each method reports the raw accuracy across benchmarks, and the second row indicates the performance relative to the upper bound.

| Method                         | GQA                         |                   | MMB MME | POPE                            | SQA    | TextVQA SEED-I |       | MMVet   | Avg.   |  |  |
|--------------------------------|-----------------------------|-------------------|---------|---------------------------------|--------|----------------|-------|---------|--------|--|--|
| Upper Bound, 576 Tokens (100%) |                             |                   |         |                                 |        |                |       |         |        |  |  |
|                                | 63.2                        | 67.7              | 1818    | 85.9                            | 72.8   | 61.3           | 66.9  | 35.3    |        |  |  |
| Vanilla (CVPR'24)              | 100%                        | 100%              | 100%    | 100%                            | 100%   | 100%           | 100%  | 100%    | 100%   |  |  |
|                                | Retain 192 Tokens (↓ 66.7%) |                   |         |                                 |        |                |       |         |        |  |  |
|                                | 59.1                        | 66.9              | 1754    | 85.1                            | 73.5   | 59.5           | 65.2  | 37.5    |        |  |  |
| VisionZip (CVPR'25)            |                             | 93.5% 98.8% 96.5% |         | 99.1%                           | 101.0% | 97.1%          | 97.5% | 106.20% | 98.7%  |  |  |
|                                | 59.7                        | 67.6              | 1775    | 86.7                            | 73.8   | 60             | 65.5  | 39.4    |        |  |  |
| Ours                           |                             |                   |         | 94.5% 99.9% 97.6% 100.9% 101.4% |        | 97.9%          | 97.9% | 111.6%  | 100.2% |  |  |
|                                |                             |                   |         | Retain 128 Tokens (↓ 77.8%)     |        |                |       |         |        |  |  |
|                                | 57.9                        | 66.7              | 1743    | 85.2                            | 74     | 58.7           | 63.8  | 37.5    |        |  |  |
| VisionZip (CVPR'25)            |                             | 91.6% 98.5% 95.9% |         | 99.2%                           | 101.6% | 95.8%          | 95.4% | 106.2%  | 97.0%  |  |  |
|                                | 59.3                        | 67.2              | 1735    | 85.9                            | 73.9   | 58.7           | 64.8  | 37.7    |        |  |  |
| Ours                           |                             |                   |         | 93.8% 99.3% 95.4% 100.0% 101.5% |        | 95.8%          | 96.9% | 106.8%  | 98.7%  |  |  |
| Retain 64 Tokens (↓ 88.9%)     |                             |                   |         |                                 |        |                |       |         |        |  |  |
|                                | 56.2                        | 64.9              | 1676    | 76.0                            | 74.4   | 57.4           | 60.4  | 33.9    |        |  |  |
| VisionZip (CVPR'25)            |                             | 88.9% 95.9% 92.2% |         | 88.5%                           | 102.2% | 93.3%          | 90.3% | 96.0%   | 93.7%  |  |  |
|                                | 58.7                        | 65.5              | 1762    | 83.0                            | 73.2   | 58.3           | 63.6  | 35.7    |        |  |  |
| Ours                           |                             | 92.9% 96.8% 96.9% |         | 96.6%                           | 100.5% | 95.1%          | 95.1% | 101.1%  | 96.9%  |  |  |

capabilities in identifying patterns within both images and videos, taking into account spatial as well as temporal characteristics. The evaluation metric is the accuracy.

MMVet [\[45\]](#page-12-12) The MMVet benchmark is constructed with the understanding that tackling complex tasks typically requires a generalist model to effectively combine multiple fundamental visionlanguage skills. MMVet identifies six essential vision-language capabilities and systematically evaluates sixteen specific combinations arising from these core abilities, thereby assessing the model's proficiency in integrating diverse vision-language functions. We evaluate the model's performance on the test split, including 218 samples. The score is evaluated by the GPT model.

