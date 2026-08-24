# <span id="page-17-0"></span>A.1 Stage-1 Plan Prompt

This appendix provides a brief overview of the prompt modules used in *SuperWriter*-Agent Stage-1 Plan. There are a total of 6 modules, each serving a specific role in the writing and evaluation pipeline:

- BrainStorm: Generates an initial in-depth thinking process to analyze and develop a preliminary writing plan for a given task, ensuring a comprehensive and thorough design.
- BrainStorm Review: Critically evaluates the task design, raising questions about potential flaws, ambiguities, or unclear requirements to refine the task's overall logic and readability.
- BrainStorm Refine: Integrates reviewer feedback into the task design by applying editorial judgment to revise or completely rewrite the task, ensuring it is rigorous and well-structured.
- Outline: Constructs a structured article outline based on the task design, including the estimated word count per paragraph and a logical framework to guide the writing process.
- Check outline: Acts as a reviewer evaluating the logical structure and completeness of the outline, pointing out logical gaps or missing elements to ensure it aligns with the intended objectives.
- Refine outline: Edits and improves the previously generated outline based on reviewer feedback, ensuring clarity, completeness, and alignment with the writing objectives.

The following sections provide the detailed prompt templates and usage notes for each module.

