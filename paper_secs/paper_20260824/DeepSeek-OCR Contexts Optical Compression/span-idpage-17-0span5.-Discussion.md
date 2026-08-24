# <span id="page-17-0"></span>**5. Discussion**

Our work represents an initial exploration into the boundaries of vision-text compression, investigating how many vision tokens are required to decode text tokens. The preliminary results are encouraging: DeepSeek-OCR achieves near-lossless OCR compression at approximately 10× ratios, while 20× compression still retains 60% accuracy. These findings suggest promising directions for future applications, such as implementing optical processing for dialogue histories beyond rounds in multi-turn conversations to achieve 10× compression efficiency.

<span id="page-18-1"></span>> **[图片提取文字 (无描述)]:**
> Very Clear Crystal Clear Clear Blurry Very Blurry Almost Gone Time → Just happened 1 day 1 hour 1 week 1 month 1 year Memory Crystal Clear Very Clear Clear Blurry Very Blurry Almost Gone Distance 1 Vision 10cm 50cm 1m 3m 10m 20m Crystal Clear Very Clear Blurry Very Blurry Almost Gone Clear Resolution 1 Text Text token Gundam Large Base Small Tiny
![](_page_18_Figure_0.jpeg)

Figure 13 | Forgetting mechanisms constitute one of the most fundamental characteristics of human memory. The contexts optical compression approach can simulate this mechanism by rendering previous rounds of historical text onto images for initial compression, then progressively resizing older images to achieve multi-level compression, where token counts gradually decrease and text becomes increasingly blurred, thereby accomplishing textual forgetting.

For older contexts, we could progressively downsizing the rendered images to further reduce token consumption. This assumption draws inspiration from the natural parallel between human memory decay over time and visual perception degradation over spatial distance—both exhibit similar patterns of progressive information loss, as shown in Figure [13.](#page-18-1) By combining these mechanisms, contexts optical compression method enables a form of memory decay that mirrors biological forgetting curves, where recent information maintains high fidelity while distant memories naturally fade through increased compression ratios.

While our initial exploration shows potential for scalable ultra-long context processing, where recent contexts preserve high resolution and older contexts consume fewer resources, we acknowledge this is early-stage work that requires further investigation. The approach suggests a path toward theoretically unlimited context architectures that balance information retention with computational constraints, though the practical implications and limitations of such vision-text compression systems warrant deeper study in future research.

