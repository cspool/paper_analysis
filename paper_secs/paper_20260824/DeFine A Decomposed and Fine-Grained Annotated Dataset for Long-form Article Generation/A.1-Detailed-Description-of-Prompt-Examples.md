# A.1 Detailed Description of Prompt Examples

The table [5](#page-11-1) presents three examples of prompt templates designed for long-form question-answer generation based on provided abstracts. These templates help structure the interaction between the question and abstract, ensuring diverse and highquality responses.

Structure and Functionality: Each prompt is structured to include one or more abstract references (denoted as *{Abstract[1]...Abstract[2]}*) and a dynamically generated question (denoted as *{Question}*). These prompts serve to guide the model in providing detailed, relevant answers while ensuring that the content aligns with the source material.

Prompt Diversity: To ensure that the generated content covers a wide range of information points and avoids repetition, multiple prompt templates are provided. For instance:

- Prompt 1 requests an answer based on the references, emphasizing completeness.
- Prompt 2 adds a directive that the model must answer, reinforcing the requirement for response.
- Prompt 3 simplifies the directive but maintains the expectation of a detailed answer.

Customization for Content Length: The prompts are adaptable for abstracts of different lengths. By adjusting the amount of information in {Abstract[1]...Abstract[2]}, the system ensures that both short and long references are effectively processed.

## Advantages of Diverse Prompting:

- Increased Diversity: Multiple prompts ensure a broader coverage of information, reducing redundancy and improving the model's capability to generate varied responses.
- Improved Robustness: These prompts expose the model to different question structures and contexts, improving its ability to respond appropriately across diverse scenarios.
- Enhanced Alignment: By consistently referring to abstracts and generating questions

dynamically, the prompts ensure that the answers remain aligned with the source information, enhancing both factual accuracy and relevance.

