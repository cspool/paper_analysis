# 7 Conclusion

In this work, we investigate the challenge of complex instruction following for generating long-form text. We introduce Suri, a dataset of long humanwritten responses accompanied by backtranslated and corrupted instructions. We demonstrate the effectiveness of Suri in improving the constraintfollowing capabilities of LLMs for long-form gen-

eration through supervised fine-tuning and I-ORPO. Human and automated evaluations show that our models generate high-quality, long-form responses while effectively satisfying constraints.

### Limitations

Fine-tuning additional LLMs on **Suri** While we demonstrate the effectiveness of Suri and I-ORPO on Mistral-7b-Instruct-v0.2, we have yet to experiment with fine-tuning other models on our dataset using I-ORPO.

Impact of surface features on I-ORPO Even though I-ORPO works well on our dataset, we would like to explore how surface features, such as instruction length and the degree of information overlap between the instruction and response, affect its performance.

Impact of truncating gold responses In our experiments, we truncate gold responses to lengths between 2,048 and 5,024 words to make fine-tuning more cost-effective and computationally efficient. However, our released code includes an option that allows users to recover the full response text, and thus bypass the truncated version if needed.

Ranking accuracy on out-of-domain datasets We report the ranking accuracy on the Suritest set, where Suri-SFT and Suri-I-ORPO may have an advantage over the baseline models due to their fine-tuning on Suri.

