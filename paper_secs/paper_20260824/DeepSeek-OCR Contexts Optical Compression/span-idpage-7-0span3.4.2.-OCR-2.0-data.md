# <span id="page-7-0"></span>*3.4.2. OCR 2.0 data*

Following GOT-OCR2.0 [\[38\]](#page-21-4), we refer to chart, chemical formula, and plane geometry parsing data as OCR 2.0 data. For chart data, following OneChart [\[7\]](#page-19-7), we use pyecharts and matplotlib

<span id="page-8-4"></span>> **[图片提取文字 (无描述)]:**
> "Line": ( "1100": "(-8.8, -1.4) -- (-6.6, 2.87) -- (-3.99, 5.54) -- (-3.19, 6.89)", <td "(-8.0, -1.4) -- (-6.62, 0.30) -- (-0.00, 2.10) -- (-0.6, 2.3)", "(-8.6, -1.4) -- (-4.36, -3.86) -- (-6.72, -4.72) -- (2.92, -6.38) -- (3.18, -6.51)", td>至厘子(右轴)\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\* "(-8.6, -1.4) -- (-4.97, -8.65) -- (-0.14, 0.1) -- (2.79, 0.85) -- (7.72, 1.6) -- (8.0, 1.66)", td>58.6%td>58.6%td>58.6%</tr "(-3.10, 6.89) -- (-2.4, 2.07) -- (-1.75, -8.2)", "(-3.15, 6.89) — (-1.5, 3.26) — (8.19, -4.37) — (1.88, -3.99) — (3.15, -6.51)", 6.71 td>6.7174.1%42.7%47.7%41.5% "(-3.18, 6.89) - (0.45, 5.23) - (6.89, 3.57) - (7.73, 1.91) - (8.9, 1.66)"前1,2799,46%88,84% "(-1.75, -0.2) - (-0.6, 2.3)", 33.52% "(-1.35, -0.21 -- 10.9, -2.33", tr>tr>\text{\text{d}}<\td>-0.4477.52%4046. "(-0.6, 2.3) - (1.9, 0.2)", \*(0.5, -2.3) - (1.9, 4.2)\*, 50.0% 43% "(8.6, -2.3) - (8.48, -2.51) - (8.8, 1.66)", "(1.9, 8.2) - (2.69, -1.72) - (5.19, -0.51)". "(3.19, -6.51) - (5.19, -3.84) - (7.2, 8.43) - (8.8, 1.66)" 酸辣豆花 修水双井茶(右轴) 车厘子(右轴) 20.2% "time type": | 奇峰船 -3.9158.6% 38.3% and the transfer and the transfer and the transfer and the transfer and the "Line\_endpoint": 打油茶 42.7% 6.71 74.1% 75: (-8.8, -1.4)", "f: (-3-19, 6.89)", 1.27 99,46% 80.04% "X: (-1.75, -0.21", "Rt. (8.9, -2.3)" 4185 77.52% 46.43%
![](_page_8_Figure_0.jpeg)

<span id="page-8-5"></span>(a) Image-text ground truth of chart (b) Image-text ground truth of geometry

Figure 6 | For charts, we do not use OneChart's [\[7\]](#page-19-7) dictionary format, but instead use HTML table format as labels, which can save a certain amount of tokens. For plane geometry, we convert the ground truth to dictionary format, where the dictionary contains keys such as line segments, endpoint coordinates, line segment types, etc., for better readability. Each line segment is encoded using the Slow Perception [\[39\]](#page-21-8) manner.

to render 10M images, mainly including commonly used line, bar, pie, and composite charts. We define chart parsing as image-to-HTML-table conversion task, as shown in Figure [6\(a\).](#page-8-4) For chemical formulas, we utilize SMILES format from PubChem as the data source and render them into images using RDKit, constructing 5M image-text pairs. For plane geometry images, we follow Slow Perception [\[39\]](#page-21-8) for generation. Specifically, we use perception-ruler size as 4 to model each line segment. To increase the diversity of rendered data, we introduce geometric translation-invariant data augmentation, where the same geometric image is translated in the original image, corresponding to the same ground truth drawn at the centered position in the coordinate system. Based on this, we construct a total of 1M plane geometry parsing data, as illustrated in Figure [6\(b\).](#page-8-5)

## <span id="page-8-0"></span>*3.4.3. General vision data*

DeepEncoder can benefit from CLIP's pretraining gains and has sufficient parameters to incorporate general visual knowledge. Therefore, we also prepare some corresponding data for DeepSeek-OCR. Following DeepSeek-VL2 [\[40\]](#page-21-9), we generate relevant data for tasks such as caption, detection, and grounding. Note that DeepSeek-OCR is not a general VLM model, and this portion of data accounts for only 20% of the total data. We introduce such type of data mainly to preserve the general vision interface, so that researchers interested in our model and general vision task can conveniently advance their work in the future.

## <span id="page-8-1"></span>*3.4.4. Text-only data*

To ensure the model's language capabilities, we introduced 10% of in-house text-only pretrain data, with all data processed to a length of 8192 tokens, which is also the sequence length for DeepSeek-OCR. In summary, when training DeepSeek-OCR, OCR data accounts for 70%, general vision data accounts for 20%, and text-only data accounts for 10%.

## <span id="page-8-2"></span>**3.5. Training Pipelines**

<span id="page-8-3"></span>Our training pipeline is very simple and consists mainly of two stages: a).Training DeepEncoder independently; b).Training the DeepSeek-OCR. Note that the Gundam-master mode is obtained by continuing training on a pre-trained DeepSeek-OCR model with 6M sampled data. Since the training protocol is identical to other modes, we omit the detailed description hereafter.

## *3.5.1. Training DeepEncoder*

Following Vary [\[36\]](#page-21-2), we utilize a compact language model [\[15\]](#page-19-8) and use the next token prediction framework to train DeepEncoder. In this stage, we use all OCR 1.0 and 2.0 data aforementioned, as well as 100M general data sampled from the LAION [\[31\]](#page-20-8) dataset. All data is trained for 2 epochs with a batch size of 1280, using the AdamW [\[23\]](#page-20-9) optimizer with cosine annealing scheduler [\[22\]](#page-20-10) and a learning rate of 5e-5. The training sequence length is 4096.

## <span id="page-9-0"></span>*3.5.2. Training DeepSeek-OCR*

After DeepEncoder is ready, we use data mentioned in Section [3.4](#page-6-1) to train the DeepSeek-OCR. with the entire training process conducted on the HAI-LLM [\[14\]](#page-19-9) platform. The entire model uses pipeline parallelism (PP) and is divided into 4 parts, with DeepEncoder taking two parts and the decoder taking two parts. For DeepEncoder, we treat SAM and the compressor as the vision tokenizer, place them in PP0 and freeze their parameters, while treating the CLIP part as input embedding layer and place it in PP1 with unfrozen weights for training. For the language model part, since DeepSeek3B-MoE has 12 layers, we place 6 layers each on PP2 and PP3. We use 20 nodes (each with 8 A100-40G GPUs) for training, with a data parallelism (DP) of 40 and a global batch size of 640. We use the AdamW optimizer with a step-based scheduler and an initial learning rate of 3e-5. For text-only data, the training speed is 90B tokens/day, while for multimodal data, the training speed is 70B tokens/day.

<span id="page-9-3"></span>Table 2 | We test DeepSeek-OCR's vision-text compression ratio using all English documents with 600-1300 tokens from the Fox [\[21\]](#page-20-0) benchmarks. Text tokens represent the number of tokens after tokenizing the ground truth text using DeepSeek-OCR's tokenizer. Vision Tokens=64 or 100 respectively represent the number of vision tokens output by DeepEncoder after resizing input images to 512×512 and 640×640.

|             |       | Vision Tokens =64     | Vision Tokens=100 |                             |    |
|-------------|-------|-----------------------|-------------------|-----------------------------|----|
| Text Tokens |       | Precision Compression |                   | Precision Compression Pages |    |
| 600-700     | 96.5% | 10.5×                 | 98.5%             | 6.7×                        | 7  |
| 700-800     | 93.8% | 11.8×                 | 97.3%             | 7.5×                        | 28 |
| 800-900     | 83.8% | 13.2×                 | 96.8%             | 8.5×                        | 28 |
| 900-1000    | 85.9% | 15.1×                 | 96.8%             | 9.7×                        | 14 |
| 1000-1100   | 79.3% | 16.5×                 | 91.5%             | 10.6×                       | 11 |
| 1100-1200   | 76.4% | 17.7×                 | 89.8%             | 11.3×                       | 8  |
| 1200-1300   | 59.1% | 19.7×                 | 87.1%             | 12.6×                       | 4  |

