# <span id="page-11-2"></span>C GPT-4o Evaluator

For GRB and GRF, we prompt GPT-4o (2024-08- 06 version) as the MT evaluator in the reference-

based and reference-free manners, respectively. The corresponding prompts borrow from [Kocmi](#page-8-4) [and Federmann](#page-8-4) [\(2023\)](#page-8-4), and make some adaptions to literature translation.

## GRB Prompt:

Score the following translation from English to Chinese with respect to the human reference on a continuous scale from 0 to 100, where score of zero means "no meaning preserved" and score of one hundred means "perfect preservation of meaning, with faithfulness, expressiveness, and elegance".

English source: {src}

Chinese human reference: {ref} Chinese translation: {hyp}

Score:

## GRF Prompt:

Score the following translation from English to Chinese on a continuous scale from 0 to 100, where score of zero means "no meaning preserved" and score of one hundred means "perfect preservation of meaning, with faithfulness, expressiveness, and elegance".

English source: {src} Chinese translation: {hyp}

Score:

