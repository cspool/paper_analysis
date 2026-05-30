# Limitations

Despite the encouraging results achieved by VisualRWKV, several limitations must be acknowledged. Firstly, due to the lack of data following such instructions and the limited context length, VisualRWKV is currently unable to process multiple images. Secondly, although VisualRWKV shows good performance on academic datasets, its ability to handle certain tasks, such as TextVQA, may be constrained by the limitations in the recall ability of efficient language models [\(Arora et al.,](#page-8-5) [2023\)](#page-8-5). These constraints can potentially be mitigated by further architectural improvements. Lastly, to maintain consistency with LLaVA-1.5, this study did

not investigate the effects of the choice of vision encoder or the quality of training data on VisualRWKV. In the future, we aim to explore more advanced visual encoders and utilize higher-quality training data to further enhance its performance.

Risks Although VisualRWKV significantly reduces the occurrence of hallucinations, it can still generate hallucinations and occasionally disseminate misinformation. Therefore, its application in critical fields, such as the medical industry, should be approached with great caution.

