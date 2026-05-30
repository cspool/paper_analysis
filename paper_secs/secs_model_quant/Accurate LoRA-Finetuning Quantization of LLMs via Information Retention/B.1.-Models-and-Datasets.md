# B.1. Models and Datasets

For models, we establish IR-QLoRA upon the LLaMA [\(Touvron et al.,](#page-10-0) [2023a\)](#page-10-0) and LLaMA2 [\(Touvron et al.,](#page-10-1) [2023b\)](#page-10-1) families. Specifically, we finetune the 7B, 13B, 30B, and 65B models of LLaMA and the 7B and 13B models of LLaMA2.

For datasets, we choose Alpaca [\(Taori et al.,](#page-10-13) [2023\)](#page-10-13) and FLAN v2 [\(Longpre et al.,](#page-10-14) [2023\)](#page-10-14) as our finetuning datasets. Alpaca contains 52K instruction-following data generated from text-davinci-003 (GPT 3.5) [\(Wang et al.,](#page-10-19) [2022\)](#page-10-19). Flan v2 is a collection of 1,836 tasks combining the mixture with CoT, Muffin, T0-SF, and NIV2.

### <span id="page-14-1"></span>B.2. NormalFloat Quantization

Quantile Quantization [\(Dettmers et al.,](#page-9-8) [2021\)](#page-9-8) is an information-theoretically optimal data type that ensures each quantization bin has an equal number of values from the input tensor. Essentially, it distributes the data evenly across the available quantization levels, leading to efficient and balanced utilization of the quantized representation.

Building on Quantile Quantization, NormalFloat (NF) Quantization [\(Dettmers et al.,](#page-9-7) [2023\)](#page-9-7) introduces the quantization principle that the weights conform to a zero-centered normal distribution. By converting all weights into a fixed distribution, their distribution is fully adapted to the range of the specified data type. Then the quantile constant can be calculated. Following the processing steps in QLoRA [\(Dettmers et al.,](#page-9-7) [2023\)](#page-9-7), we can get different quantized points according to different quantized bit widths. Table [11](#page-14-2)[-13](#page-14-3) shows the NF quantization for 2-4 bits, respectively. Note that we use symmetrical settings in NF2 to prevent excessive deviation of information.

Table 11: The exact values of the NormalFloat 2-bit (NF2) data type

<span id="page-14-2"></span>

| Index | Value                | Index | Value              |
|-------|----------------------|-------|--------------------|
| 0     | -1.0                 | 2     | 0.2525685131549835 |
| 1     | -0.25256848335266113 | 3     | 1.0                |

Table 12: The exact values of the NormalFloat 3-bit (NF3) data type

| Index | Value               | Index | Value               |
|-------|---------------------|-------|---------------------|
| 0     | -1.0                | 4     | 0.16093020141124725 |
| 1     | -0.4786292016506195 | 5     | 0.33791524171829224 |
| 2     | -0.217141792178154  | 6     | 0.5626170039176941  |
| 3     | 0.0                 | 7     | 1.0                 |

Table 13: The exact values of the NormalFloat 4-bit (NF4) data type

<span id="page-14-3"></span>

| Index | Value                | Index | Value               |
|-------|----------------------|-------|---------------------|
| 0     | -1.0                 | 8     | 0.07958029955625534 |
| 1     | -0.6961928009986877  | 9     | 0.16093020141124725 |
| 2     | -0.5250730514526367  | 10    | 0.24611230194568634 |
| 3     | -0.39491748809814453 | 11    | 0.33791524171829224 |
| 4     | -0.28444138169288635 | 12    | 0.44070982933044434 |
| 5     | -0.18477343022823334 | 13    | 0.5626170039176941  |
| 6     | -0.09105003625154495 | 14    | 0.7229568362236023  |
| 7     | 0.0                  | 15    | 1.0                 |

Table 14: Examples for each the evaluation datasets

<span id="page-15-0"></span>

| Dataset       | Question                                                                               |                                          | Answer |  |  |  |  |
|---------------|----------------------------------------------------------------------------------------|------------------------------------------|--------|--|--|--|--|
|               | Which of the following factors is associated with a decreased risk of Alzheimer's?     |                                          |        |  |  |  |  |
| MMLU          | (A) Being African or Hispanic American                                                 | (B) Eating fish                          | B      |  |  |  |  |
|               | (C) A lower level of education                                                         | (D) Being married                        |        |  |  |  |  |
| HellaSwag     | A man is sitting on a roof. he                                                         |                                          |        |  |  |  |  |
|               | (A) is using wrap to wrap a pair of skis.                                              | (B) is ripping level tiles off.          | D      |  |  |  |  |
|               | (C) is holding a rubik's cube.                                                         | (D) starts pulling up roofing on a roof. |        |  |  |  |  |
| PIQA          | How do I ready a guinea pig cage for it's new occupants?                               |                                          |        |  |  |  |  |
|               | (A) Provide the guinea pig with a cage full of a few inches of bedding made of ripped  |                                          |        |  |  |  |  |
|               | paper strips, you will also need to supply it with a water bottle and a food dish.     |                                          |        |  |  |  |  |
|               | (B) Provide the guinea pig with a cage full of a few inches of bedding made of ripped  |                                          |        |  |  |  |  |
|               | jeans material, you will also need to supply it with a water bottle and a food dish.   |                                          |        |  |  |  |  |
| WinoGrande    | John moved the couch from the garage to the backyard to create space. The<br>is small. |                                          |        |  |  |  |  |
|               | (A) garage<br>(B) backyard                                                             |                                          |        |  |  |  |  |
| ARC-Easy      | Which factor will most likely cause a person to develop a fever?                       |                                          |        |  |  |  |  |
|               |                                                                                        | (B) a bacterial population in the        |        |  |  |  |  |
|               | (A) a leg muscle relaxing after exercise                                               | bloodstream                              |        |  |  |  |  |
|               |                                                                                        | (D) carbohydrates being digested in the  |        |  |  |  |  |
|               | (C) several viral particles on the skin                                                | stomach                                  |        |  |  |  |  |
| ARC-Challenge | George wants to warm his hands quickly by rubbing them. Which skin surface will        |                                          |        |  |  |  |  |
|               | produce the most heat?                                                                 |                                          | A      |  |  |  |  |
|               | (A) dry palms                                                                          | (B) wet palms                            |        |  |  |  |  |
|               | (C) palms covered with oil                                                             | (D) palms covered with lotion            |        |  |  |  |  |
| BoolQ         | Phantom pain sensations are described as perceptions that an individual experiences    |                                          |        |  |  |  |  |
|               | relating to a limb or an organ that is not physically part of the body. Limb loss is a |                                          |        |  |  |  |  |
|               | result of either removal by amputation or congenital limb deficiency. However,         |                                          |        |  |  |  |  |
|               | phantom limb sensations can also occur following nerve avulsion or spinal cord injury. |                                          |        |  |  |  |  |
|               | Is pain experienced in a missing body part or paralyzed area?                          |                                          |        |  |  |  |  |
|               | (A) True                                                                               | (B) False                                |        |  |  |  |  |
| OpenBookQA    | A magnet will stick to                                                                 |                                          |        |  |  |  |  |
|               | (A) a belt buckle                                                                      | (B) a wooden table,                      | A      |  |  |  |  |
|               | (C) a plastic cup                                                                      | (D) a paper plate                        |        |  |  |  |  |

