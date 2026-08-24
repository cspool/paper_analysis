# 5 Conclusion

We propose a novel automated DoS attack on the LLM agent tool-calling layer, using an MCTS optimizer to convert benign servers into malicious variants that induce costly multi-turn dialogues. The attack maintains task correctness while often evading detection by standard monitors. Experiments show it inflates per-query costs by up to 658× and generates over 60,000 tokens. Our work highlights the agent-tool interface as a critical attack surface, stressing the need for defenses that monitor the entire workflow. Future systems should develop defenses based on behavioral baselines to differentiate between legitimate and maliciously inefficient tool-calling patterns.

