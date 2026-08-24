# 4 Application Beyond Math Reasoning

In this section, we investigate how the model trained on mathematic long thoughts generalizes when applied to other tasks or applications.

Training Details To investigate the model's generalization capability across different domains, we first construct a diverse bilingual dataset through a systematic data extraction and translation process. From our distilled O1 model outputs, we carefully select approximately 5,000 high-quality samples containing retrospective thinking and self-reflection elements. These samples are then translated into Chinese using GPT-4o mini model, resulting in a balanced bilingual dataset. The final training dataset comprises 10,750 mixed Chinese-English sample pairs, where each sample consists of a query-response pair. We then perform Supervised Fine-Tuning (SFT) on the Qwen2.5-72B-Instruct [\(Yang et al.,](#page-15-16) [2024a\)](#page-15-16) model (named as "baseline") using this curated dataset to obtain our final model (named as "Ours".)

