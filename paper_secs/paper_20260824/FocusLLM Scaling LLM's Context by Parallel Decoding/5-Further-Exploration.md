# 5 Further Exploration

#### 5.1 Visualization of Candidate Tokens

To further illustrate how candidate tokens function, we provide a more intuitive explanation by visualizing the information carried by these tokens through attention weight heatmaps when decoding the next token. Due to space limitations, we place the visualization results in Appendix [H.](#page-11-0) We have the following observations: i) In Passkey Retrieval task, the model assigns a high attention weight to *one certain candidate token*, indicating that this to-

ken effectively carry the passkey information from its respective chunk. In contrast, candidate tokens from chunks containing noisy text carry no useful information, resulting in near-zero attention weights. ii) In LongBench NarrativeQA task, the model shows a slightly different pattern, where *many candidate tokens receive attention*, as multiple chunks' information may be aggregated for the QA task. The visualization results demonstrate that FocusLLM effectively uses candidate tokens to transmit information from the context while ignoring irrelevant noise.

