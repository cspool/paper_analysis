# ABSTRACT

Mixture-of-Experts (MoE) in Large Language Models (LLMs) routes each token through a subset of specialized Feed-Forward Networks (FFN), known as experts. We present SteerMoE, a framework to steer MoE models by detecting and controlling behavior-associated experts. We detect key experts by comparing how often they activate between paired inputs that demonstrate opposite behaviors (e.g., safe vs. unsafe). By selectively activating or deactivating such experts during inference, we control behaviors like faithfulness and safety without fine-tuning. Across 11 benchmarks and 6 LLMs, our steering raises safety by up to +20% and faithfulness by +27%. Alternatively, unsafe steering drops safety by -41% alone, and -100% when combined with existing jailbreak methods, bypassing *all* safety guardrails. Overall, SteerMoE offers a lightweight, effective, and widely applicable test-time control, while revealing unique vulnerabilities in MoE LLMs.

![](_page_0_Figure_6.jpeg)

Figure 1: Steering MoE models by routing through behavior-linked experts at inference enables lightweight, interpretable control. Red and green FFNs are controlled by our method; others follow the router's choice. Generations are from Qwen3-30B-A3B. (See more examples in Table [1\)](#page-1-0)

