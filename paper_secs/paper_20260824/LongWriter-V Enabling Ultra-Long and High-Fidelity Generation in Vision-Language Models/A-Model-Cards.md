# A Model Cards

Table [3](#page-11-1) demonstrates the detailed information of the LLMs and VLMs evaluated in our experiments.

<span id="page-11-1"></span>

| Model name                                  | Model version               | Context window   | Max output tokens |
|---------------------------------------------|-----------------------------|------------------|-------------------|
| Large Language Models                       |                             |                  |                   |
| GLM-4-9B-chat (GLM et al., 2024)            | -                           | 128,000 tokens   | -                 |
| Mistral-Large-Instruct (Jiang et al., 2023) | Mistral-Large-Instruct-2407 | 128,000 tokens   | -                 |
| Deepseek-r1 (Guo et al., 2025)              | deepseek-reasoner           | 64,000 tokens    | 8,000 tokens      |
| Vison Language Models                       |                             |                  |                   |
| MiniCPM-V2.6 (Yao et al., 2024)             | MiniCPM-V-2-6               | 32,000 tokens    | -                 |
| Qwen2.5-VL-7B (Team, 2025)                  | Qwen2.5-VL-7B-Instruct      | 32,000 tokens    | -                 |
| Qwen2.5-VL-72B (Team, 2025)                 | Qwen2.5-VL-72B-Instruct     | 32,000 tokens    | -                 |
| Claude 3 Opus (Anthropic, 2024)             | claude-3-opus-20240229      | 200,000 tokens   | 4,096 tokens      |
| Gemini-1.5-pro (Team et al., 2024)          | gemini-1.5-pro              | 2,000,000 tokens | 8,192 tokens      |
| GPT-4o (OpenAI, 2024)                       | gpt-4o-2024-08-06           | 128,000 tokens   | 8,192 tokens      |

Table 3: Model cards.

## B Model Prompts

## B.1 Prompts for Collecting Visual Instructions

## Prompt for selecting user requests that require 1,000+ word response.

You will receive an image and an instruction from a user to an AI assistant, please determine whether the instruction requires the AI assistant to write an article for the given image, and the length of the article is more than 1,000 words in English (or 1,000 characters in Chinese). If the instruction does not mention the word requirement, please determine whether the user's intention of the response length is more than 1,000 words. If the instruction is irrelated with the image, please reply "no". Instruction: {*User Instruction*}

## Prompt for constructing multi-image instruction.

You will receive {*Image Number*} images and an instruction from a user to an AI assistant, this original instruction is targeted for the first image solely. Now please rewrite this instruction to a challenging long-output one that need using visual information from all the input images, and the length of the expected output should be more than 2,000 words in English (or 2,000 characters in Chinese). Here are three examples of challenging long-output instructions:

Example instruction 1: {*Example Instruction 1*}

Example instruction 2: {*Example Instruction 2*}

Example instruction 3: {*Example Instruction 3*}

Now, you should rewrite the following instruction:

Instruction: {*User Instruction* }

Please rewrite this user instruction to a challenging long-output instruction that requires the use of all the input images. Please output only the rewritten instruction, do not output other content.

### <span id="page-11-0"></span>B.2 Prompts for the LongWrite Agent-V Pipeline

## Prompt for planning the writing outline.

You are an expert planner. Your task is to break down a writing task into clear subtasks based on the provided images and writing instruction.

Please analyze the images and writing instruction carefully, then create a detailed outline in this format:

Section 1 - Main Point: [Key points to cover based on images and instruction] - Word Count: [200-1000 words]

Section 2 - Main Point: [Key points to cover based on images and instruction] - Word Count: [200-1000 words]

...

Make each section focused and specific while ensuring the full outline:

- 1. Covers all key content from both images and writing instruction
- 2. Flows logically from section to section
- 3. Has reasonable word count targets (200-1000 words per section)
- 4. Forms a cohesive whole that fulfills the writing instruction

Writing instruction: {*User Instruction*} Output only the outline with no other text.

### Prompt for generating each paragraph according to the writing outline.

You are an expert writer. Your task is to write the next section of a longer piece based on:

- 1. The provided images and writing instruction
- 2. The outline plan
- 3. Previously written sections

Writing instruction: {*User Instruction*}

Outline plan: {*PLAN*} Previous sections: {*TEXT*}

Please write section {*STEP*} following these guidelines:

- 1. Focus on the main points specified in the outline
- 2. Stay within the target word count
- 3. Flow naturally from previous sections
- 4. Integrate relevant details from the images
- 5. Maintain a consistent tone and style
- 6. Write only this section, not a full conclusion

Output only the new section with no other text.

