# <span id="page-11-1"></span>C Dataset

Multi-hop QA Datasets HotpotQA features a collection of 2-hop questions directly authored by native speakers, based on two interconnected paragraphs. 2WikiMultihopQA is comprised of complex questions up to 5-hops in length, constructed through carefully designed templates to prevent the possibility of shortcut solutions.

In the MuSiQue dataset, questions are intricately crafted starting from straightforward scenarios that require up to 4-hops reasoning. Annotators subsequently rephrase these with a dual purpose: to avoid shortcut answers and to maintain a natural linguistic quality. Each question within the original datasets is complemented by 2-4 supporting paragraphs, delivering evidence for simple one-step reasoning, alongside multiple paragraphs designed to serve as decoys.

HotpotWikiQA-mixup originates from LV-Eval and employs a construction method known as a mixup. This method randomly blends support documents with various distracting documents to generate five different context lengths for a given QA pair, including 16k, 32k, 64k, 128k, and 256k. Due to the excessive length of this dataset, we select the first 50 data entries from each different context length for experimentation to control costs.

Single-hop QA Datasets NarrativeQA is a dataset designed to test comprehension abilities for long documents, primarily sourced from movie scripts. As a single-hop QA dataset, the information required to answer its questions appears at a single location within the text.

Real-World Datasets QuALITY [\(Pang et al.,](#page-9-21) [2022\)](#page-9-21) is a long-text multiple-choice questionanswering dataset, with questions crafted by contributors who are familiar with the complete passages, making it more representative of real-world QA scenarios. We handle it as straightforward QA problems. Natural Questions [\(Kwiatkowski et al.,](#page-9-22) [2019\)](#page-9-22) includes real anonymous aggregated queries from Google along with corresponding Wikipedia pages, providing another excellent resource for authentic long-text QA situations.

