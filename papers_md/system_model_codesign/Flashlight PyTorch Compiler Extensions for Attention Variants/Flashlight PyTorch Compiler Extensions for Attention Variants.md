## FLASHLIGHT:

0.5

(Batch Size, Sequence Length)

(Batch Size, Sequence Length)

(Batch Size, Sequence Length)

![](_page_0_Picture_1.jpeg)

![](_page_0_Picture_2.jpeg)

![](_page_0_Picture_3.jpeg)

## © PyTorch Compiler Extensions to Accelerate Attention Variants

Bozhi You, Irene Wang, Zelal Su Mustafaoglu, Abhinav Jangda, Angélica Moreira, Roshan Dathathri, Divya Mahajan, Keshav Pingali

![](_page_0_Figure_6.jpeg)

Moves the optimization burden from the user to the compiler

## Superior performance

- Always faster than default PyTorch Compiler
- For FlexAttention-supported variants: similar performance
- Beyond FlexAttention: **5x** faster for Evoformer
- Improves end-to-end inference latency for AlphaFold by **6-9**%