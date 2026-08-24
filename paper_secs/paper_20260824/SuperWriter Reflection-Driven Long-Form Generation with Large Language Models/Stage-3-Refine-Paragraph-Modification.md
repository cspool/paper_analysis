# Stage-3 Refine: Paragraph Modification

As a text editor, your task is to revise the paragraph based on the following feedback:

review\_feedback

Original Paragraph:

updated\_document[idx]

Ensure the revision strictly follows the specific suggestions in the feedback. Only provide the revised paragraph. Enclose the paragraph content with \$\$, like: \$\$content\$\$ Revised Paragraph:

## <span id="page-20-0"></span>A.4 Evaluation Prompt for Hierarchical DPO

To support structured preference data construction for Direct Preference Optimization (DPO), we design and apply a sequence of modular prompts. Each serves a specific role within the evaluation pipeline. Below is an overview of their usage:

The evaluation pipeline comprises the following steps:

- 1. Rubric Definition (evaluation\_criteria): Defines the complete set of General and Special evaluation dimensions. This rubric is reused across all queries.
- 2. Criterion Selection Schema (format\_query): Specifies the JSON format for selecting six criteria (three General, three query-relevant Special) and rewriting their Definitions and Standards to match the specific query context.
- 3. Criterion Selection Prompt: Combines the rubric and schema to instruct the model to select and customize criteria. The output is a JSON object referred to as evaluate\_standard.
- 4. Scoring Format Schema (format\_eval): Specifies the expected evaluation output format: for each selected criterion, the model must return an *Analysis* string and a numeric *Score*.
- 5. Final Scoring Prompt: Provides the model with a query, its generated result, the customized evaluate\_standard, and the format\_eval schema. The model performs criterion-wise evaluation and outputs a structured JSON.

Outcome: This pipeline yields structured, query-specific evaluations that are interpretable, machineparsable, and suitable for training with DPO loss.

## **evaluation\_criteria Prompt**

Evaluation Criteria

#### 1. General Criteria (Applicable to All Genres)

#### 1.1 Relevance

Definition: How well the content matches the user's request, and whether it addresses the intended purpose or topic.

#### Standards:

- 10: Fully aligned with the user's needs, highly relevant to the request.
- 7–9: Mostly relevant, with some minor deviations or less-than-perfect alignment.
- 4–6: Partially relevant, with the majority of the content not matching the user's request.
- 1–3: Completely irrelevant, fails to meet the user's needs.

