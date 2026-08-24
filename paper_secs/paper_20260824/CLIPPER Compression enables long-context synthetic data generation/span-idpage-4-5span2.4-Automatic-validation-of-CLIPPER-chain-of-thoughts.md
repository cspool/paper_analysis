# <span id="page-4-5"></span>2.4 Automatic validation of CLIPPER chain-of-thoughts

We assess the groundedness of chain-of-thought (CoT) reasoning by prompting an LLM to verify whether each event in the CoT is supported by the chapter outline. Accuracy is measured as the percentage of events in true claim CoTs that are grounded in the book. To scale up evaluation, we use an LLM judge, DeepSeek-R1-Distill-Llama-70B (DeepSeek-AI et al., 2025). We find that 98.5% of CoTs are grounded. The remaining ungrounded CoTs typically involve events open to multiple interpretations. Compared to NAÏVE, CLIPPER's CoTs are significantly easier to verify due to their explicit chapter references.

