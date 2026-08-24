# Limitations

Despite the significant contributions of the DeFine dataset to the field of long-form article generation, the current implementation presents certain limitations. One of the primary issues is the imbalance

between the English and Chinese data, which stems from the limited number of Chinese Wikipedia entries compared to the far more extensive English Wikipedia. This may impact the model's ability to generalize across both languages, leading to potential performance gaps. Additionally, another key challenge lies in the reliance on automated evaluation metrics such as ROUGE. While useful for measuring surface-level similarities between generated and reference texts, these metrics may fail to account for deeper aspects of text quality, such as coherence, logical flow, and factual correctness.

Future work should focus on balancing the language data, expanding coverage of specialized topics, and developing evaluation frameworks that better align with human judgment while improving model robustness to reduce factual inconsistencies.

