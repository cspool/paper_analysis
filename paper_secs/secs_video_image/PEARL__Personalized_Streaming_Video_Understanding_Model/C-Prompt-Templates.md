# C Prompt Templates

## C.1 Concept Description Generation Prompts

As mentioned in the main text, we provide standardized prompting templates used to generate a compact description that summarizes the concept's salient characteristics. Since PEARL-Bench evaluates both Frame-level and Video-level concepts, we design two distinct prompts for their respective characteristics:

Frame-level Prompt. This prompt is designed to guide the model to focus on permanent and stable visual features (such as gender, facial features, hair, and body type) while instructing it to ignore temporary elements like clothing, accessories, or poses, which are likely to change across a long video stream.

Video-level Prompt. Conversely, this prompt directs the model to focus on the core kinematics and stable movement patterns of a customized action, while explicitly ignoring the specific identity or appearance of the person performing it, as well as the background and surrounding environment, ensuring the extracted action features are generalizable across different characters and scenes.

Specifically, both templates include two key placeholders:

- {concept\_name}: The user-defined name assigned to the concept.
- {original\_description}: The initial user instruction from the Concept-Definition QA. For frame-level concepts, it helps the model locate the target subject (e.g., "The character wearing white clothes is named {Adaliz}."). For video-level concepts, it helps the model identify the specific action sequence (e.g., "The sequence of movements shown in this clip is {Action A}.").

The complete prompts for both levels are provided below:

