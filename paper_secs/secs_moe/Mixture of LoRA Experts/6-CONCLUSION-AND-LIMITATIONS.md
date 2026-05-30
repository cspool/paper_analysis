# 6 CONCLUSION AND LIMITATIONS

In this study, we introduce the Mixture of LoRA Experts (MOLE) as a versatile and dynamic approach for composing multiple trained LoRAs. The key innovation of MOLE lies in its learnable gating functions, which utilize the outputs of multiple LoRAs at each layer to determine composition weights. Our comprehensive evaluation in both the both NLP and V&L domains establishes that MOLE outperforms existing LoRA composition methods.

Limitations. As described in Section [5,](#page-7-0) when the number of LoRAs increases to a very large value (e.g., 128), despite our MOLE exhibiting superior performance, the performance of all LoRA composition methods, including our MOLE, tends to decrease. This suggests that our MOLE still faces challenges when performing large-scale LoRA composition. It also highlights the significance of researching better approaches for handling large-scale LoRA composition effectively.

