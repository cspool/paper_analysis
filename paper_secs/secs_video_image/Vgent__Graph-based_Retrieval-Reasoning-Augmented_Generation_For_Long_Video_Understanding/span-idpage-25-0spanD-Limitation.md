# <span id="page-25-0"></span>D Limitation

In this work, we represent video content using textual descriptions—such as entities and their associated details—as a lightweight and efficient alternative to raw visual features. However, we do not incorporate visual embeddings or frame-level features into our graph. While computing similarity across frames can be computationally intensive, it remains a promising direction for future improvement.

Additionally, our framework is model-agnostic and compatible with any LVLM, meaning its performance is inherently bounded by the capabilities of the base LVLM. As more powerful LVLMs emerge, our pipeline can be readily adapted to take advantage of their enhanced video understanding and reasoning abilities.