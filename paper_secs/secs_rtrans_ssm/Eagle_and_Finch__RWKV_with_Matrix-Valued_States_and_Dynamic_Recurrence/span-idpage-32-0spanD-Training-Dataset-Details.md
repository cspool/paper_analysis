# <span id="page-32-0"></span>**D Training Dataset Details**

Most of the component data sources for the RWKV World v2 dataset are used intact, with no upor down-sampling done so all tokens are given equal weighting. Recent works have demonstrated the impact that automated data mixing can have on pretraining [\(Albalak et al.,](#page-17-13) [2023;](#page-17-13) [Xie et al.,](#page-25-9) [2024\)](#page-25-9), but we leave this as an exploration for future work. Some sub-sampling is done for overrepresented languages within a few data sources. All tokens are given equal weighting unless otherwise noted in Table [9.](#page-33-0)

