**==> picture [960 x 198] intentionally omitted <==**

**----- Start of picture text -----**<br>
Oaken<br>: Fast and Efficient LLM Serving with<br>Online-Offline Hybrid KV Cache Quantization<br>**----- End of picture text -----**<br>


**Minsu Kim*** Seongmin Hong*† RyeoWook Ko Soongyu Choi Hunjong Lee† Junsoo Kim† Joo-Young Kim† Jongse Park 

**==> picture [233 x 76] intentionally omitted <==**

## KAIST 

**==> picture [318 x 49] intentionally omitted <==**

† HyperAccel 

**==> picture [174 x 109] intentionally omitted <==**

- Co-first authors who contributed 

- equally to this work 

**ISCA 2025** 

## **LLM Serving at Scale** 

**==> picture [869 x 5] intentionally omitted <==**

**==> picture [371 x 315] intentionally omitted <==**

**----- Start of picture text -----**<br>
...<br>Long Context Length<br>...<br>Documents Query<br>...<br>Query<br>Thoughts<br>**----- End of picture text -----**<br>


# ▪ LLM serving system should simultaneously handle **a large number of, long-context requests** 

**Large Batch Size** 

**==> picture [270 x 145] intentionally omitted <==**

LLM serving system batches multiple requests (+10,000) from users 

Recent LLM tasks (e.g., RAG, reasoning) involve over tens of thousands of tokens 

2 / 19 

## **KV Cache Matters for “Capacity”** 

**==> picture [869 x 5] intentionally omitted <==**

▪ **Larger batch and longer context result in larger KV cache** Oaken means made **Large Batch Size Long Context Length** of oak trees . 450 **Key-Value** 450 KV Size Model Size **Cache** KV Size Model Size **Larger Batch** ~~**&**~~ **Longer Context put** Decoder 1 Decoder 1 Decoder 1 Decoder 1 300 300 **pressure on M** Decoder 2 ~~**em**~~ **ory Capacity &** Decoder 2 Decoder 2 **Bandwi** Decoder 2 ~~**dt**~~ **h** 150 150 **80 80** 0 Decoder N Decoder N 0Decoder N Decoder N 1 8 16 32 64 128 256 1K 2K 4K 8K 16K 32K 64K Batch Size Context Length * Llama2-13B, context length: 2K oak trees * Llama2-13B, batch size: 8 . <EOS> 

**Generation** 3 / 19 **Phase** 

**Prefill Phase** 

## **KV Cache Matters for “Bandwidth”** 

**==> picture [869 x 5] intentionally omitted <==**

**==> picture [835 x 266] intentionally omitted <==**

**----- Start of picture text -----**<br>
Single Request Batched Requests<br>Shared<br>B<br>A 1<br>1<br>A2 ⨉ = B2<br>Weight<br>A3 B3<br>Not Shared<br>22.0<br>20.0<br>or<br>Key<br>C1 ⨉ = D1<br>⨉ =<br>⨉ Value =<br>QKV QKV<br>Attn FFN Attn FFN<br>Gen Gen<br>GPU Utilization (%) GPU Utilization (%)<br>**----- End of picture text -----**<br>


   - NVIDIA A100, Llama2-13B, context length: 1K 

- Increasing batch size improves utilization **except for attention operation** 

- ▪ Attention operation is **bandwidth-bound** due to un-sharable KV cache 

4 / 19 

## **KV Cache Matters for “Bandwidth”** 

**==> picture [869 x 5] intentionally omitted <==**

**Single Request Batched Requests Shared** B A 1 1 A2 ⨉ **=** B2 Weight A3 B3 **KV cache uantization** We leverage **q Not Shared 22.0** to overcome bandwidth and capacity limitations **20.0** or Key C1 ⨉ **=** D1 ⨉ **=** ⨉ Value **=** QKV QKV Attn FFN Attn FFN Gen Gen 

## * NVIDIA A100, Llama2-13B, context length: 1K 

▪ Increasing batch size improves utilization **except for attention operation** 

- Attention operation is **bandwidth-bound** due to un-sharable KV cache 

5 / 19 

## **Prior Quantization Techniques** 

**==> picture [869 x 5] intentionally omitted <==**

**==> picture [86 x 35] intentionally omitted <==**

× **Oaken a** ~~**chiev**~~ **e** ~~**s both orman**~~ × 𝜆 **high perf ce & accuracy through co-designing Per-Vector Mixed-Precision Channel Reordering / Matrix Transformation Fine-grained grouping** minimizes **quantization algorithm & hardware modules** accuracy loss **Hardware-friendly algorithm** with 

**Mixed-precision** increases complexity 

**==> picture [57 x 49] intentionally omitted <==**

**Online KV profiling** incurs overhead 

**KIVI [ICML’24], KVQuant [NeurIPS’24]** 

minimal overhead 

**Coarse-grained grouping** leads to large accuracy loss 

**==> picture [56 x 48] intentionally omitted <==**

**Atom [MLSys’24], Tender [ISCA’24], QServe [MLSys’25]** 

6 / 19 

## **Overview of Oaken** 

**==> picture [869 x 5] intentionally omitted <==**

**==> picture [217 x 24] intentionally omitted <==**

**----- Start of picture text -----**<br>
Design Objectives<br>**----- End of picture text -----**<br>


**==> picture [836 x 139] intentionally omitted <==**

**----- Start of picture text -----**<br>
① Address memory bottleneck ② Find sweet spot between  ③ Maximize hardware<br>in LLM serving accuracy & performance utilization & performance<br>**----- End of picture text -----**<br>


## **Algorithm Design** 

**==> picture [411 x 114] intentionally omitted <==**

**----- Start of picture text -----**<br>
Threshold-based Group shift Dense-and-sparse<br>hybrid grouping quantization encoding<br>**----- End of picture text -----**<br>


## **Hardware Design** 

**==> picture [394 x 114] intentionally omitted <==**

**----- Start of picture text -----**<br>
Streamlined Page-based memory<br>module architecture management<br>**----- End of picture text -----**<br>


7 / 19 

## **Key Observations on KV Distribution** 

**==> picture [869 x 5] intentionally omitted <==**

## **Observation 1** 

KV distribution **varies** across models and decoder layers 

## **Observation 2** 

KV distribution is **consistent** across input datasets 

**Insight 1** Oaken should determine quantization scale **for each model and decoder layer** Layer * Llama-7B, OPT-6.7B **Insight 2** Oaken can use shared quantization scale **regardless of model inputs** * WikiText2, PIQA, Layer HellaSwag 

## **Observation 3** 

KV distribution has **exceptions** to channel-wise pattern 

## **Insight 3** 

Oaken should use **multiple quantization groups** segmented by magnitude Channel * Plotted top 4% values 

8 / 19 

## **Threshold-based Online-Offline Quantization** 

**==> picture [869 x 5] intentionally omitted <==**

**==> picture [858 x 314] intentionally omitted <==**

**----- Start of picture text -----**<br>
Offline Profiling Online Quantization<br>... past<br>tokens<br>μ<br>Layer 1 topK<br>new token<br>Layer 2<br>future<br>tokens<br>Profiling 𝑜 𝑖 𝑖 𝑜<br>Result 𝑇𝑙𝑜 𝑇𝑙𝑜 0 𝑇ℎ𝑖 𝑇ℎ𝑖<br>Layer N Per-token<br>quantization<br>Frequency<br>... ...<br>**----- End of picture text -----**<br>


▪ **one-time cost** for each model Offline profiling requires (~100 inferences, ~10 min) 

9 / 19 

## **Threshold-based Online-Offline Quantization** 

**==> picture [869 x 5] intentionally omitted <==**

**==> picture [723 x 253] intentionally omitted <==**

**----- Start of picture text -----**<br>
*<br>*<br>* *<br>Inner<br>4-bit<br>Group<br>Outer Middle Middle Outer<br>idx g val<br>Group Group<br>Group Group<br>idx g val<br>idx g val<br>idx g val<br>6-bit 1-bit 16-bit<br>bit/entry<br>Inliers<br>4<br>* Filled with zeros<br>Frequency<br>bit/entry<br>Outliers<br>23<br>**----- End of picture text -----**<br>


## **Challenges:** 

▪ Outliers add **storage and hardware costs** 

▪ Outliers are **hard to quantize** due to large magnitude 

10 / 19 

## **Group Shift Quantization** 

**==> picture [869 x 5] intentionally omitted <==**

**==> picture [879 x 312] intentionally omitted <==**

**----- Start of picture text -----**<br>
*<br>INT4 *<br>* *<br>4-bit<br>𝑖 𝑖<br>𝑇 0 𝑇<br>𝑙𝑜 ℎ𝑖<br>idx g s val<br>INT5 idx g s val<br>idx g s val<br>𝑇𝑙𝑜𝑜 𝑇𝑙𝑜𝑖 0 𝑇ℎ𝑖𝑖 𝑇ℎ𝑖𝑜 idx g s val<br>6-bit 1 / 5-bit<br>𝑇𝑜 0 𝑇𝑜<br>𝑙𝑜 ℎ𝑖<br>bit/entry<br>Inliers<br>4<br>* Filled with zeros<br>bit/entry<br>Outliers<br>12<br>**----- End of picture text -----**<br>


## ▪ **Group shift algorithm** reduces average bitwidth **from 5.9 to 4.8** 

* 10% Sparsity 

11 / 19 

## **Fused Dense-and-Sparse Encoding** 

**==> picture [869 x 5] intentionally omitted <==**

**==> picture [826 x 362] intentionally omitted <==**

**----- Start of picture text -----**<br>
*<br>*<br>* *<br>4-bit 4-bit<br>idx g s valval idx g s<br>idx g s valval idx g s<br>idx g s valval idx g s<br>idx g s valval idx g s<br>6-bit 1 / 5-bit 6-bit 1/1-bit<br>▪<br>8-bit sparse matrices are  hardware-efficient  and  memory-aligned<br>▪<br>Fused encoding  reduces average bitwidth  from 4.8 to 4.4 * 10% Sparsity<br>bit/entry<br>Inliers bit/entry Dense 4<br>4<br>* Filled with zeros<br>bit/entry<br>bit/entry<br>Sparse 8<br>Outliers 12<br>**----- End of picture text -----**<br>


12 / 19 

## **Oaken Accelerator Integration** 

**==> picture [869 x 5] intentionally omitted <==**

**==> picture [865 x 207] intentionally omitted <==**

**----- Start of picture text -----**<br>
Device Memory<br>Vector Module Area<br>DMA<br>Control<br>Processing<br>Mem. Host Unit Quant. VPU 22.86%<br>MC ... MC Unit<br>Ctrl. Interface Engine<br>Direct<br>MPU 6.03%<br>Memory<br>Dequant.<br>Matrix<br>Access<br>Interconnect Register Engine Quant. Engine 1.86%<br>Processing<br>File<br>Unit<br>MMU Dequant. Engine 6.35%<br>Comp. Core Core Core ... Core<br>Core Total 100%<br>Oaken Compute Core<br>* Synthesized on TSMC 28nm<br>**----- End of picture text -----**<br>


## Oaken Accelerator 

- Oaken modules do **not modify** the existing compute logic in the accelerator 

- Oaken modules are integrated into existing accelerator **with low overhead** 

13 / 19 

## **Oaken Hardware Modules** 

**==> picture [869 x 5] intentionally omitted <==**

**==> picture [837 x 364] intentionally omitted <==**

**----- Start of picture text -----**<br>
Control Register Matrix Processing Unit Management Table Physical Memory<br>Dense<br>Thresholds KV Cache<br>Decomposer<br>Inlier Outlier<br>Min&Max Min&Max<br>Sparse<br>Scale calc Scale calc<br>Quantizer Quantizer<br>Splitter<br>Memory Management Unit<br>OR<br>Shifter<br>Dense Sparse ▪ Oaken modules are designed to maximize<br>Device Memory hardware  and  memory utilization<br>**----- End of picture text -----**<br>


▪ Oaken modules are designed to maximize **hardware** and **memory utilization** 

## **Quantization Engine** 

14 / 19 

## **Evaluation Methodology** 

**==> picture [869 x 5] intentionally omitted <==**

## ▪ **Models** 

## ▪ **Datasets** 

- Llama2 

- 7B, 13B, 70B* 

◦ WikiText2, PIQA, WinoGrande, and HellaSwag 

- 

- ◦ OPT 6.7B, 13B, 30B* 

– ◦ Mistral 7B 

- 

- ◦ Mixtral 8x7B* 

## ▪ **Group Configuration** 

◦ **4%, 90%, 6%** for outer, middle and inner group 

- Used **2GPUs** with pipeline parallelism 

## ▪ **Baselines** 

◦ Tender (ASIC) ◦ Atom (GPU) ◦ QServe (GPU) ◦ KIVI (GPU) ◦ KVQuant (GPU) 

## ▪ **Hardware Specification** 

|||**NVIDIA A100 **|**Oaken-HBM **|**Oaken-LPDDR**|
|---|---|---|---|---|
||**FP16 TFLOPS**|312|270|270|
||**Memory type**<br>**Memory capacity**<br>**Memory bandwidth**|HBM<br>80 / 160***** GB<br>2.0 TB/s|HBM<br>80 GB<br>2.0 TB/s|LPDDR<br>256 GB<br>1.1 TB/s|



- Used **2GPUs** with pipeline parallelism 

15 / 19 

## **Evaluation Results** 

## **Throughput** 

**==> picture [869 x 5] intentionally omitted <==**

Oaken-HBM achieves performance improvement of **1.79** ⨉ over vLLM (FP16) Oaken-LPDDR is also a competitive option for **larger models** and **larger batches** 

**==> picture [859 x 310] intentionally omitted <==**

**----- Start of picture text -----**<br>
* Context length : 2K<br>GPU (vLLM) GPU (KIVI) GPU (QServe) Oaken-LPDDR Oaken-HBM<br>6K 3K 1000<br>750<br>4K 2K<br>500<br>2K 1K<br>250<br>0 0 0<br>16 32 64 128 256 16 32 64 128 256 16 32 64 128 256<br>Batch Size Batch Size Batch Size<br>(1) Llama2-7B (2) Llama2-13B (3) Llama2-70B<br>Throughput (token/sec)<br>**----- End of picture text -----**<br>


16 / 19 

## **Evaluation Results** 

## **Accuracy** 

**==> picture [869 x 5] intentionally omitted <==**

|**Model**|Llama2|Llama2|Llama2|Llama2|
|---|---|---|---|---|
||13B<br>70B|13B<br>70B|13B<br>70B|13B<br>70B|
|**Dataset**|WikiText2|PIQA|WinoGrande|HellaSwag|
|**Metric**|Perplexity (↓)|Accuracy (%)|Accuracy (%)|Accuracy (%)|
|**Original**|**4.88**<br>**3.32**|**80.52**<br>**82.70**|**72.80**<br>**80.20**|**79.38**<br>**83.82**|
|**KIVI**<br>**QServe***|**4.90**<br>**3.33**<br>5.12<br>3.36|79.05<br>78.07<br>77.48<br>81.77|**70.96**<br>**76.81**<br>66.80<br>76.09|**78.97**<br>83.47<br>76.69<br>83.24|
|**Oaken**|4.93<br>3.34|**79.71**<br>**82.59**|70.56<br>76.64|78.24<br>**83.50**|



## * Activated KV quantization feature only 

Oaken incurs **0.87%** and **0.32% accuracy loss** compared to FP16 and KIVI Oaken achieves **1.38% higher** accuracy compared to QServe 

17 / 19 

## **Additional Results in Our Paper** 

**==> picture [869 x 5] intentionally omitted <==**

▪ Performance evaluation using other LLMs and baselines ▪ Accuracy and effective bits with varying group configurations ▪ End-to-end latency breakdown ▪ Sensitivity study to total sequence length ▪ Performance evaluation using real-world benchmark ▪ Synthesized area and power 

18 / 19 

**==> picture [960 x 138] intentionally omitted <==**

**----- Start of picture text -----**<br>
Conclusion<br>**----- End of picture text -----**<br>


## • **Oaken** 

# ◦Acceleration solution for LLM inference serving including algorithm-hardware co-designed KV cache quantization technique 

- **Contributions** 

   - ◦Addresses memory bandwidth and capacity bottlenecks in modern LLM serving ◦Finds sweet spot in accuracy-performance trade-off of KV cache quantization 

- **Future works** 

   - ◦Extending Oaken to handle recent attention architectures (e.g., latent attention, linear attention) ◦HyperAccel's high efficiency LLM accelerator with broad quantization support 

19 / 19 

