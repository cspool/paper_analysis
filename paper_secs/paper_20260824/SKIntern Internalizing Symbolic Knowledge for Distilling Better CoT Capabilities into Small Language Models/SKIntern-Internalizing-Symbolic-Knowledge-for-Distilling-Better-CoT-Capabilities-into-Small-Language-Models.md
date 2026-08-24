# *SKIntern*: Internalizing Symbolic Knowledge for Distilling Better CoT Capabilities into Small Language Models

Huanxuan Liao<sup>1</sup>,<sup>2</sup> , Shizhu He<sup>1</sup>,2\*, Yupu Hao<sup>1</sup>,<sup>2</sup> , Xiang Li<sup>1</sup>,<sup>2</sup> , Yuanzhe Zhang<sup>4</sup> , Jun Zhao<sup>1</sup>,<sup>2</sup> , Kang Liu<sup>1</sup>,2,<sup>3</sup>

 The Key Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese Academy of Sciences, Beijing, China School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, China Shanghai Artificial Intelligence Laboratory, Shanghai, China National Science Library, Chinese Academy of Sciences, Beijing, China

{liaohuanxuan2023, haoyupu2023, lixiang2022}@ia.ac.cn {shizhu.he, kliu, jzhao}@nlpr.ia.ac.cn

## Abstract

Small Language Models (SLMs) are attracting attention due to the high computational demands and privacy concerns of Large Language Models (LLMs). Some studies fine-tune SLMs using Chains of Thought (CoT) data distilled from LLMs, aiming to enhance their reasoning ability. Furthermore, Some CoT distillation methods introduce external symbolic knowledge into the generation process to improve the limited knowledge memory, reasoning ability and out-of-domain (OOD) generalization of SLMs. However, the introduction of symbolic knowledge increases computational overhead and introduces potential noise. In this paper, we introduce *SKIntern*, an innovative approach that empowers SLMs to internalize symbolic knowledge and few-shot examples gradually through a progressive finetuning process, guided by a predefined linear decay schedule under curriculum learning. By efficiently internalizing knowledge, *SKIntern* reduces computational overhead and speeds up the reasoning process by focusing solely on the question during inference. It outperforms state-of-the-art baselines by over 5%, while reducing inference costs (measured in FLOPs) by up to 4× across a wide range of SLMs in both in-domain (ID) and out-of-domain (OOD) tasks. Our code will be available at [https:](https://github.com/Xnhyacinth/SKIntern) [//github.com/Xnhyacinth/SKIntern](https://github.com/Xnhyacinth/SKIntern).

