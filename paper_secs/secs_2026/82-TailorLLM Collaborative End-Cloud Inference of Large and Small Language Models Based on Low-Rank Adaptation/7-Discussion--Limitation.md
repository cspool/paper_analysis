# 7 Discussion & Limitation

## Q1: How does TailorLLM handle long-tail distribution tasks, such as those in private domains with specific rules or highly customized, domain-specific tasks?

For scarce-data, domain-specific tasks, TailorLLM decomposes them into subtasks and combines lightweight LoRA adapters (e.g., for key info extraction, terminology, style) to address requirements compositionally. For private domains, it fine-tunes lightweight adapters with domain data and rules [\[50\]](#page-14-28), ensuring adherence to domain constraints without degrading the base model.

## Q2: What is the specific role of AdapterMgr — is it used to manage memory or storage?

AdapterMgr manages LoRA modules in RAM, not storage. It balances memory limits and fast inference via caching, eviction, and prefetching. This avoids excessive RAM usage while keeping LoRA loading latency low ( 0.26 ms vs. much slower storage).

## Q3: How does TailorLLM handle new task types that do not fit any known category beyond offloading to the cloud?

TailorLLM groups uncategorized queries in the feature space based on similarity. When a dense cluster forms with sufficient instances, the cloud fine-tunes a new adapter for that task. This requires accumulating enough data before establishing a new category.

#### Q4: What are the limitations of TailorLLM?

The LoRA modules need to be transmitted between the edge and the cloud, which can lead to high latency or transmission failures under limited bandwidth. Moreover, when the SLM model is updated, the existing LoRA adapters need to be retrained.

