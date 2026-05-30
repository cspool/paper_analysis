# 6 Related Work

Quantization-Aware Training (QAT) is often used to obtain quantized models that are adapted in downstream tasks [\(Peri et al.,](#page-15-8) [2020;](#page-15-8) [Liu et al.,](#page-15-9) [2023\)](#page-15-9). It involves quantization and full model fine-tuning at the same time. However, QAT requires massive training cost, such as the gradient and optimization state. Moreover, it is difficult to compute the gradient of quantized weights. Our method, with the help of LoRA, sidesteps the aforementioned issues, providing a light approach for downstream task adaptation.

Post-Training Quantization (PTQ) is a category of popular quantization frameworks [\(Frantar et al.,](#page-14-9) [2022;](#page-14-9) [Xiao et al.,](#page-16-6) [2023\)](#page-16-6), which can also be used for task adaptation. It calibrates the high-precision

model with a small subset of the training dataset. Therefore, the subsequent quantization is guided by the training dataset, providing task-specific quantized models. Besides, it does not involve any gradient backpropagation, so it is cost-efficient. However, it usually results in lower accuracy compared to QAT.

