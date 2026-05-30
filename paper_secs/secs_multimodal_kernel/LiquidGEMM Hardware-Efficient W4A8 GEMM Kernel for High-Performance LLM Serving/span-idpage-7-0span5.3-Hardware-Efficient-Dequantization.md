# <span id="page-7-0"></span>5.3 Hardware-Efficient Dequantization

After loading weights from SMEM to RF, each thread holds 32 UINT4 elements packed into four 32-bit registers, as illustrated in Figure 8. Elements w8-w15 correspond to the first MMA operation, and w16-w31 to the second. We dequantize these weights from UINT4 to INT8 on CUDA Cores using LQQ (Section 4).

Figure 8 illustrates the dequantization process. We first apply the unpacking method from QServe [15] to expand eight 4-bit elements from one register into two registers holding 8-bit values. We then perform dequantization using Equation 12: multiplying by the scale factor  $s_{u8}$ , adding the offset a, and applying a final XOR. Because LQQ ensures no overflow, all operations can be executed using native 32-bit hardware instructions, specifically, IMAD for multiply-add and XOR for offset correction. Note that both  $s_{u8}$  and a can be precomputed offline. After dequantization, the resulting UINT8 elements share the same binary representation as the target INT8 values, making them directly usable for subsequent MMA operations on Tensor Cores.

In summary, our method dequantizes four elements using just two hardware arithmetic instructions. Including the unpacking step, eight elements are dequantized with only seven instructions, significantly reducing computational overhead on CUDA Cores, well below the threshold required for effective overlap with weight loading and MMA (Section 3.3). The first-level dequantization is fused into the GEMM epilogue and incurs negligible cost.

## <span id="page-7-1"></span>5.4 Other GEMM Optimizations

As discussed above, GPU MMA instructions are constrained to fixed matrix shapes defined by hardware. For INT8, H800 fixes the m

<span id="page-7-3"></span>![](_page_7_Figure_10.jpeg)

Figure 9: Overview of dataflows in our LLM serving system for LLaMA models.

dimension to 64, while n can vary from 8 to 256 across several configurations. To better utilize Tensor Cores under small batch sizes, we apply a hardware-specific optimization by rewriting  $Y = XW^T$  as  $Y = (WX^T)^T$ , allowing us to choose WGMMA instructions based on the batch size and maximize compute efficiency. Additionally, we adopt standard GEMM optimizations such as persistent kernels. As these techniques are widely used, we omit the details for brevity.

Leveraging the programming primitives of CUTLASS and Cute, we integrate and adapt components such as the tile scheduler, mainloop, and epilogue into a warp-specialized ping-pong kernel. Specifically, our dequantization algorithm is fused into the MMA mainloop, and a Dual-MMA packed layout is used during data loading. We implement WGMMA instructions, barrier synchronization, and general components like TMA in PTX, wrapped by CUTLASS. In contrast, the dequantization logic is implemented directly in CUDA.

### <span id="page-7-4"></span>6 LLM Serving System and Offline Quantization

To support end-to-end performance evaluation, we implement an LLM serving system by integrating open-source techniques for key system-level components, including attention computation, KV cache management, and quantization schemes. This section briefly outlines their implementation, along with the offline quantization. Serving System. Figure 9 illustrates the dataflow of our LLM serv-

ing system. Figure 9 illustrates the datanow of our LLM serving system for LLaMA models. Query, Key, Value, Output, and FFN layers are executed using our proposed LiquidGEMM with W4A8 quantization on weights and activations, producing FP16 outputs. Following TensorRT-LLM [20], KV caches are quantized to INT8 using per-channel static quantization, with scale factors computed offline. To improve memory efficiency, we adopt PagedAttention [12] for KV cache management and use FlashAttention-2 [6] for runtime attention computation. We do not adopt FlashAttention-3 [22], as it is tailored for FP8. For activation quantization, we follow SmoothQuant [29], dynamically mapping FP16 activations to INT8 on-the-fly via per-token quantization after dividing by the smooth scale. As activation tensors have small memory footprints and low computational overhead, quantization is lightweight and typically fused into other kernels.

Offline Quantization. We adopt the SmoothQuant [29] post-training quantization method to quantize weights offline. Specifically, weights are first scaled by a smooth factor and then quantized using the two-level approach described in Section 4: per-channel quantization from FP16 to INT8, followed by per-group quantization to UINT4. Following OutlierSuppression+ [28], we apply a grid search to determine the optimal smooth scale. Note that our focus

<span id="page-8-3"></span>Table 1: Peak token generation throughput (tokens/s) of LiquidServe, QServe, and TRT on H800 with 80 GB memory constraint. The number in parentheses indicates the batch size at which peak throughput is achieved. Speedup is reported relative to the best-performing baseline between QServe and TRT. LiquidServe/wo uses the W4A8 GEMM kernel from QServe.

| System         | LLaMA1     |             | LLaMA2      |             | LLaMA3       | Mistral      | Yi          | Mixtral      |
|----------------|------------|-------------|-------------|-------------|--------------|--------------|-------------|--------------|
|                | 30B        | 7B          | 13B         | 70B         | 8B           | 7B           | 34B         | 8×7B         |
| TRT-FP16       | 410 (13)   | 5,521 (128) | 2,701 (64)  | OOM         | 13,920 (256) | 14,573 (256) | 1,931 (64)  | OOM          |
| TRT-W4A16      | 1,170 (48) | 4,953 (128) | 2,906 (109) | 2,266 (128) | 12,997 (256) | 13,513 (256) | 4,645 (256) | 5,712 (256)  |
| TRT-W8A8       | 1,006 (36) | 5,083 (128) | 2,922 (100) | 1,166 (46)  | 13,012 (256) | 13,636 (256) | 3,860 (128) | NA           |
| TRT-FP8        | 986 (36)   | 5,913 (144) | 3,402 (96)  | 948 (45)    | 16,820 (256) | 17,433 (256) | 4,206 (225) | 8,296 (256)  |
| QServe         | 1,478 (64) | 5,402 (128) | 3,311 (124) | 871 (64)    | 5,240 (128)  | 5,361 (124)  | 1,415 (64)  | NA           |
| LiquidServe/wo | 1,309      | 5,926       | 3,299       | 1,869       | 10,956       | 11,091       | 3,699       | 6,135        |
| LiquidServe    | 1,607 (53) | 6,721 (194) | 4,105 (119) | 3,695 (184) | 16,694 (256) | 17,011 (256) | 6,999 (256) | 10,745 (256) |
| Speedup        | 1.09x      | 1.14x       | 1.21x       | 1.63x       | 0.99x        | 0.98x        | 1.51x       | 1.30x        |

is on optimizing the efficiency of W4A8 GEMM; our method is orthogonal to techniques that improve quantization accuracy and can be seamlessly integrated with such approaches.

