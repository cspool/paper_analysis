# <span id="page-16-1"></span>G Improvement on Text-only Capability

In this section, you can find full results on text-only capability, as shown in the Table [11](#page-16-2) and Table [12.](#page-16-3)

<span id="page-16-2"></span>

| Method     | Size | LBD  | Eng   | LAM   | PIQA  | SC16  | HSW   | WG    | ARC-C | ARC-E | HQA   | OBQA  | SCIQ  |
|------------|------|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|            |      | ppl  | avg%  | acc   | acc   | acc   | acc-n | acc   | acc-n | acc   | acc-n | acc-n | acc   |
| RWKV       | 1.6B | 4.63 | 59.82 | 67.39 | 74.37 | 74.50 | 61.06 | 60.93 | 33.70 | 64.18 | 35.22 | 37.4  | 89.40 |
| VisualRWKV | 1.6B | 4.15 | 61.01 | 67.64 | 73.44 | 75.09 | 61.50 | 61.95 | 38.31 | 67.88 | 36.46 | 38.0  | 89.80 |

Table 11: The table lists the English performance metrics for various benchmarks: LBD (LAMBADA), PIQA, SC16 (StoryCloze16), HSW (Hellaswag), WG (WinoGrande), ARC-C (arc\_challenge), ARC-E (arc\_easy), HQA (headQA\_en), OBQA (openbookQA), SCIQ. Metric units are ppl (perplexcity), acc (accuracy) and acc-n (normalized accuracy).

For multilingual evaluations, we assess LAMBADA in English, French, German, Italian, and Spanish. We evaluate StoryCloze as per [\(Lin et al.,](#page-9-22) [2021\)](#page-9-22) in Arabic, English, Spanish, Basque, Hindi, Indonesian, Burmese, Russian, Swahili, Telugu, and Chinese. COPA is evaluated in Estonian, Haitian Creole, Indonesian, Italian, Cusco-Collao Quechua, Kiswahili, Tamil, Thai, Turkish, Vietnamese, and Chinese, following [\(Ponti et al.,](#page-10-15) [2020\)](#page-10-15). We also evaluate multilingual WinoGrande in English, French, Japanese, Portuguese, Russian, and Chinese, as demonstrated in [\(Tikhonov and Ryabinin,](#page-10-16) [2021;](#page-10-16) [Muennighoff et al.,](#page-10-17) [2022\)](#page-10-17).

| Method     | Size | MultiLang | xLBD  | xSC   | xWG   | xCOPA |  |
|------------|------|-----------|-------|-------|-------|-------|--|
|            |      | avg%      | acc   | acc   | acc   | acc   |  |
| RWKV       | 1.6B | 59.97     | 47.17 | 58.24 | 76.46 | 58.03 |  |
| VisualRWKV | 1.6B | 59.83     | 46.73 | 58.90 | 75.07 | 58.65 |  |

<span id="page-16-3"></span>Table 12: The table lists the Multi-Language performance metrics for various benchmarks: xLBD (Multilingual LAMBADA), xSC (Multilingual StoryCloze), xWG (Multilingual WinoGrande), xCOPA (Multilingual COPA).

