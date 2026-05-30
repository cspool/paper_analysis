# Q.5. How is our approach similar or different from speculative decoding in LLMs?

**Response:** Speculative decoding differs from caching-based image generation in MoDM. In speculative decoding, a small draft model predicts tokens for text generation, while a large verification model checks and refines them. In contrast, MoDM does not involve verification by a large model. Instead, each prompt is processed by either the small or large model, depending on cache availability: ensuring efficiency without additional verification overhead.

Q.6. How well do cross-model queries work in MoDM? Response: We demonstrate cross-model compatibility using two model families: Stable Diffusion and SANA. §7 presents overall throughput and how MoDM utilizes different models to handle high request loads. §7.3 evaluates the image quality produced by different model families. §A.7 provides visual examples of images generated using a mix of models from different families.

### Q.7. How are the thresholds on k decided?

**Response:** §5.2 explains how thresholds on k are determined using text-to-image similarity scores, ensuring a high image quality factor of  $\geq 0.95$ .

