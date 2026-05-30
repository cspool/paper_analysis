# <span id="page-14-0"></span>C. Additional Main Results

## C.1. Results on Stable Diffusion

To demonstrate that our method generalizes to larger-scale datasets and higher resolutions, we conduct experiments on MS-COCO 2014 [\(Lin et al.,](#page-9-14) [2014\)](#page-9-14) using Stable Diffusion v1.4 with DPM solvers[\(Lu et al.,](#page-9-18) [2022\)](#page-9-18). We apply tensor-wise dynamic quantization and evaluate the quantized models within the Q-Diffusion framework. A total of 30,000 images are generated using 50 sampling steps. As shown in Table [7,](#page-14-2) the resulting FID scores confirm that MoDiff consistently performs well on large-scale diffusion models.

<span id="page-14-2"></span>Table 7. The FID and sFID on MS-COCO with Stable Diffusion using PLMS solver under different precisions. The best performance is bolded.

| Methods           | Bits (W/A) | FID ↓  | sFID ↓ |
|-------------------|------------|--------|--------|
| LTQ               | 8/8        | 12.15  | 19.05  |
| LTQ+MoDiff (Ours) |            | 12.14  | 19.05  |
| LTQ               | 8/6        | 71.38  | 59.74  |
| LTQ+MoDiff (Ours) |            | 13.21  | 20.07  |
| LTQ               | 8/4        | 408.42 | 199.59 |
| LTQ+MoDiff (Ours) |            | 225.22 | 104.12 |

