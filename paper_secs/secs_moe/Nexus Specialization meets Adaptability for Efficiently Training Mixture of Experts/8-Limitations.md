# 8 Limitations

The MoE architecture is often employed for larger models in the multi-billion parameter range, where efficiency is paramount. However, to facilitate a broader set of experiments, we limit our setup to using 2.8B parameter seed models for the main results and 470M parameter seed models for ablations. Furthermore, our dense experts are based on existing data sources in the SlimPajama dataset which is pre-defined. Future work could extend our method by discovering specialized data domains through unsupervised clustering similar to [Gururangan et al.](#page-17-5) [\[2023\]](#page-17-5).

