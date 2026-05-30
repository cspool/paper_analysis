# <span id="page-15-5"></span>A.4 Energy Savings

We measure the energy consumption of various baselines using Zeus [47], a Python-based energy measurement toolkit. The proposed methods, MoDM-SDXL and MoDM-SANA, are compared against two references: a standard SD3.5-Large (Vanilla) model and Nirvana. Fig. 18 presents the energy savings of different systems relative to the vanilla baseline. Nirvana achieves a modest 23.9% energy improvement, primarily due to skipping de-noising steps. However, these benefits are limited as inference still relies on a single, large

model. In contrast, MoDM significantly enhances energy efficiency: (1) MoDM-SDXL, which utilizes the SDXL model for cache-hit requests, achieves a 46.7% energy savings, and (2) MoDM-SANA, which leverages the smaller SANA-1.6B model for cache-hit requests while maintaining comparable image quality, achieves even greater efficiency, reaching 66.3% energy savings. These results underscore two key insights. First, caching image generations from a large base model effectively reduces redundant computational overhead. Second, using lighter, more efficient downstream diffusion models (e.g., SANA-1.6B) further amplifies energy savings.

<span id="page-16-2"></span>![](_page_16_Figure_3.jpeg)

**Figure 18.** Energy Savings of different baselines normalized to Vanilla (SD3.5L) on DiffusionDB.

