# *B. Implications of Expert Skews on Expert Co-processing*

A prior work [\[19\]](#page-13-20) argue that the number of tokens each expert processes can substantially differ. In cases where there are hot experts processing a large number of tokens and cold experts processing a smaller number of tokens, Duplex can efficiently process the MoE layer by flexibly handling experts with both xPU and Logic-PIM, exploiting expert co-processing. However, in ideal cases where each expert processes the same number of tokens, expert co-processing may not be as effective.

