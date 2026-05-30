# <span id="page-6-1"></span>**6 Discussion and Analysis**

To thoroughly evaluate the specific contributions of each individual module, in this section, we present a detailed analysis of **DIG**. Our evaluation is structured around the following key questions:

- *How does the choice of frame selection strategy impact performance on global versus localized queries?* ([§6.1\)](#page-7-0)
- *How effective is the CAFS module at selecting representative frames, and what is its contribution to the overall performance of DIG?* ([§6.2\)](#page-7-1)
- *How does the LMM-based reward model compare against the CLIPScore [\[63\]](#page-14-9) in reward assignment?* ([§6.3\)](#page-9-0)
- *What is the influence of the temporal window length (wlen) on model performance?* ([§6.4\)](#page-9-1)
- *What is the computational efficiency of DIG?* ([§6.5\)](#page-9-2)

<span id="page-7-2"></span>**Figure 5:** Comparison of our proposed frame selection pipeline (Sections 4.2–4.4) versus uniform sampling across different query types. The base LMMs are Owen2.5-VL-7B [16] and Owen2.5-VL-32B [16].

<span id="page-7-3"></span>![](_page_7_Figure_4.jpeg)

**Figure 6:** GlC and LoC scores across varying video durations. We compare three sampling strategies: FPS, UNI, and CAFS. In each sub-figure, the lines represent the score (left y-axis), while the bars indicate the number of sampled frames (right y-axis).

