# J GPT-4o-mini's performance on **Suri**

We note that although GPT-4o-mini produces less repetitive text, it can generate only an average of 1,134 tokens, which is still lower than Mixtral-8x7B-Instruct (Table [20\)](#page-31-3). The higher repetition rate in the fine-tuned models may simply be due to these models generating longer text. Upon analyzing 30 generation samples from GPT-4o-mini, we observe that while the model can satisfy the constraints, it still suffers from formulaic generation and unnatural incorporation of those constraints.

#### <span id="page-28-0"></span>Prompt: Assign constraint scope (broad, specific) to each constraint

You are a helpful assistant. You are given a constraint that you need to determine if it is a specific or broad constraint. Specific constraints focus on an element that can be found in a specific part of the text. Broad constraints focus on an element that can be found throughout the text.

### Examples:

Constraint: Throughout the narrative, use a first-person perspective that centers on the protagonist's perspective.

Your response: Broad

Constraint: Include cliffhangers at the end of the first chapter to encourage readers to continue reading.

Your response: Specific

Constraint: Introduce a new character in the middle of the story to add depth to the narrative.

Your response: Specific

Constraint: Include cliffhangers at the end of each chapter to encourage readers to continue reading.

Your response: Broad

### Constraints:

Constraint: { x<sup>w</sup> constraint}

### Your response:

<span id="page-28-1"></span>Table 12: Prompt to assign constraint scope (broad/specific) to each constraint. The placeholder {x<sup>w</sup> constraint} is replaced with a single constraint from each backtranslated instruction.

| Configurations                                | Values           |
|-----------------------------------------------|------------------|
| Hardware (Training and Inference)<br>Tracking | 4xA100s<br>wandb |
| lora_r                                        | 16               |
| lora_alpha                                    | 16               |
| lora_dropout                                  | 0.05             |
| beta (for ORPO only)                          | 0.4              |
| gradient_accumulation_steps                   | 1                |
| gradient_checkpointing                        | True             |
| learning_rate                                 | 5.0e-5           |
| lr_scheduler_type                             | cosine           |
| max_length                                    | 15024            |
| max_completion_length                         | 15000            |
| max_prompt_length                             | 5000             |
| num_train_epochs                              | 2                |
| optim                                         | adamw_torch      |
| per_device_train_batch_size                   | 1                |

Table 13: Training details for SFT and ORPO

#### <span id="page-29-0"></span>Prompt: **p(preference|prompt)** evaluation

You are an expert instruction rater. You will be given a text and two instructions, one of which is used to generate the text. Read through the text carefully, then determine which of the two instructions was used to generate the text. Answer only with "1" if the first instruction is correct, or "2" if the second instruction is correct. DO NOT give any reasoning.

### Text: {text}

### First Instruction:

{ins1}

### Second Instruction:

{ins2}

Which instruction is correct? Answer only with "1" if the first instruction is correct, or "2" if the second instruction is correct. DO NOT give any reasoning.

Your response:

Table 14: Prompt used in the p(preference|prompt) evaluation. The {text} placeholder is replaced with gold responses, while the placeholders {ins1} and {ins2} are replaced with the correct and corrupted instructions, respectively. To mitigate any potential ordering bias, the order of the correct and corrupted instructions is shuffled. We will consider a response correct only if the model chooses the correct instruction, regardless of the ordering.

<span id="page-29-1"></span>

| Types                                             | Krippendorff's<br>alpha | Satisfied<br>vs<br>Partially<br>Satisfied | Partially Satis<br>fied vs Not Sat<br>isfied | Satisfied vs<br>Not<br>Satis<br>fied |
|---------------------------------------------------|-------------------------|-------------------------------------------|----------------------------------------------|--------------------------------------|
| Instruction Validation (Section 2.3)              | 0.1                     | 0.30                                      | 0.03                                         | 0.0                                  |
| Constraint Satisfaction - Suri-SFT (Section 5)    | 0.0                     | 0.52                                      | 0.24                                         | 0.24                                 |
| Constraint Satisfaction - Suri-I-ORPO (Section 5) | 0.2                     | 0.60                                      | 0.34                                         | 0.06                                 |
| I-ORPO vs SFT - Coherence (Section 5)             | 0.0                     | -                                         | -                                            | -                                    |
| I-ORPO vs SFT - Informativeness (Section 5)       | 0.0                     | -                                         | -                                            | -                                    |
| I-ORPO vs SFT - Enjoyability (Section 5)          | 0.1                     | -                                         | -                                            | -                                    |

Table 15: Types of disagreement among annotators in the instruction validation and constraint satisfaction tasks. Most disagreements arise over whether the text fully or partially satisfies the constraints.

#### <span id="page-30-0"></span>Prompt: LLM evaluation

You will be given a text and its corresponding instruction, which contains the text's main goal and a constraint. Determine whether the text satisfies the constraint (not the main goal). You should return your answer (Yes/No/Partially) along with your reasoning and a quote in the text that supports your reasoning (the quote should not contain any double quotation marks). Your answer should contain 3 fields: "answer", "reasoning", and "quote". DO NOT output anything else other than the response, which starts with "«" and ending with "»".

