# <span id="page-18-0"></span>A.2 Stage-2 Write Prompt

This appendix provides an overview of the prompt modules used in *SuperWriter*-Agent Stage-2 Write. There are a total of 2 modules, each serving a specific role in the writing pipeline:

- Write-thinker: This module is designed to guide the planning phase for each paragraph. It takes the structured outline, the previous paragraphs, and the key point for the current paragraph as input. The module then prompts the model to develop a detailed thought process covering the paragraph's purpose, structure, transitions, details and examples, language style, and other relevant aspects. It ensures that each paragraph is planned thoroughly before the actual writing begins.
- Write: This module transforms the thought process from the Write-thinker stage into the actual written paragraph. It uses the same structured outline, previous paragraphs, key

point, and the generated thought process as guidance to produce a coherent and logically sound paragraph. The output is formatted with delimiters (e.g., \$\$content\$\$) to clearly separate the paragraph from other text.

The following sections provide the detailed prompt templates and usage notes for each module.

## Write-thinker

You are a writing expert skilled in thoughtful planning before generating each paragraph.

Outline: outline

Previous Paragraphs: previous\_paragraphs Key Point for the Current Paragraph: key\_point

Please carefully develop a writing plan for the new paragraph. You may consider the following aspects:

- 1. Purpose: What is the main objective of this paragraph? What message or emotion should it convey?
- 2. Structure: How should the content of this paragraph be organized? What logical sequence would best ensure clarity and coherence, and how will it connect tightly with the previous content?
- 3. Transitions: How will this paragraph naturally link to the one before it? Are there specific transition sentences or bridging techniques that can be used?
- 4. Details and Examples: What details, facts, or examples are needed to support the main idea? How should these be arranged for maximum impact?
- 5. Language Style and Techniques: What kind of language style should be used to achieve the goal? Are there rhetorical devices (such as metaphors or analogies) that could enhance the paragraph's impact—while still being clear, readable, and easy to understand for the audience?
- 6. Markdown Format: Use Markdown to structure the output neatly, including headings, bullet points, or bold text to improve readability.

Based on the outline and the key point for this paragraph, construct a detailed writing plan. Add any other relevant considerations as needed, and keep the word count requirements in mind. Only the thought process behind the paragraph is needed, not the paragraph itself.

