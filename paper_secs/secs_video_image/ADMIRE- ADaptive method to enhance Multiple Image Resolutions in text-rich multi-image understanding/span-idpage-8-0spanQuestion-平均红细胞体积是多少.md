# <span id="page-8-0"></span>**Question:** 平均红细胞体积是多少?

Figure 7: Case study of ADMIRE in PRQA.

6.4.4 Case Study. In this section, we visualize some cases of Qwen2VL-7B in PRQA as shown in Figure [7.](#page-8-0) More cases are depicted in Appendix [B.](#page-10-0)

## 6.5 Deployment

Our framework has been widely adopted in real-world scenarios at Alipay since October 2024. It enhances performance on complex text-rich multi-image understanding tasks based on mainstream Large Vision-Language Models (LVLMs) without requiring additional training, particularly excelling in multi-page medical report QA scenarios. ADMIRE is deployed in Alipay's medical report interpretation service, which accepts user-uploaded, anonymized multi-page medical document images to enable VQA functionality. The system is accessible to everyone via Alipay App (Medical Health → Health Manager → Upload Medical Examination Report File). Targeting end users, it is planned for future expansion to medical professionals. Our approach significantly improves the accuracy of medical VQA, enabling more reliable and precise answers to clinical questions while ensuring rapid inference speeds essential for real-time applications. Specifically, we assess our framework on the multi-page medical report QA scenarios. As shown in Table [6,](#page-8-1) demonstrate that our approach, ADMIRE, is more convincing compared to the baseline in PRQA. As shown in Figure [9,](#page-11-1) we illustrate the differences in the multi-page medical report QA scenarios before and after applying ADMIRE. We observe that ADMIRE produces more accurate and reliable responses compared to the baseline Qwen2VL-7B model.

<span id="page-8-1"></span>Table 6: Performance of ADMIRE based on physical report task. The bold font indicates the best performance.

| Model              | PRQA  |
|--------------------|-------|
| Qwen2VL-7B [24]    | 29.05 |
| /w. ADMIRE-Top5-X4 | 33.04 |

## 7 Conclusion

In this work, we presented ADMIRE, an innovative approach for enhancing large visual language models (LVLMs) in high-resolution, text-rich multi-image comprehension tasks. By leveraging a textguided image scorer based on attention weights produced by the LVLM itself, ADMIRE dynamically adjusts the resolution of each image according to its relevance to the given question . We conduct extensive experiments to evaluate effectiveness of method and validated it in industrial multi-page medical document QA scenarios. Since our proposed approach primarily leverages the inherent capabilities of LVLMs, ADMIRE can achieve state-of-the-art results for OCR-free methods on mainstream multi-page image QA datasets without additional training.

