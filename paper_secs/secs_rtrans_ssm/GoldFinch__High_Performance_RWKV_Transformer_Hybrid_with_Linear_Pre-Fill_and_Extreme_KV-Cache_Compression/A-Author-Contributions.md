# **A Author Contributions**

**Daniel Goldstein** Entire GPTAlpha design, research, and code. GoldFinch code, architecture design, and research. Full manuscript initial draft except [4.3.](#page-8-1) Manuscript edits. Proofreading and revisions of full manuscript. Core experiments featured herein.

**Fares Obeid** Research discussions and experiments during development of the GoldFinch architecture. Significant input on all aspects of final architecture design.

**Eric Alcaide** Research discussions and experiments during development of the GoldFinch architecture. Significant input and experiments leading to Finch-C2 design.

**Guangyu Song** Section [4.3.](#page-8-1) Experiments for [4.3.](#page-8-1)

**Eugene Cheah** GoldFinch code proofreading, development of release code and testing, contributions to pre-fill mechanism details.

## **B Other Related Work**

Ring Attention [\(Liu et al.,](#page-12-3) [2023\)](#page-12-3) allows the attention calculation to be split across many discrete processors that do not share VRAM. Keys and values can be split up among these processors, linearly amortizing the amount of KV-Cache required to remain resident within each processor's VRAM. This enables unbounded scaling of attention given enough hardware, but does not address the cost of O(*N* 2 ) compute, and still imposes total memory costs that scale with the sequence length.