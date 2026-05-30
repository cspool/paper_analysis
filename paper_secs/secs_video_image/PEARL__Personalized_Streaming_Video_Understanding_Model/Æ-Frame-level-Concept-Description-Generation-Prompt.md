# Æ Frame-level Concept Description Generation Prompt

Based on the image and the original description provided, generate a concise visual description of this character/object that focuses on PERMANENT/STABLE features for video clip retrieval.

Original description: "{original\_description}" Concept name: {concept\_name}

Your task:

1. Use the original description to understand WHICH

character/object to focus on in the image

- 2. Generate a description focusing on STABLE features that DON'T change throughout the video:
  - Gender (male/female/other)
  - Face features (eye shape, facial structure, distinctive marks)
  - Hair (color, length, style if distinctive)
  - Body type (build)
  - Age appearance (young/middle-aged/elderly)

AVOID or minimize:

- Clothing details (they change in long videos)

- Accessories (they may be removed)
- Temporary expressions or poses
- Background, location, surroundings, or nearby objects in the scene
- Relative position or size compared to objects/environment in the scene

#### Requirements:

- Keep it concise and simple (1 sentence, around 10-15 words)
- Focus on features that remain consistent across different scenes
- Write in English using simple descriptive terms
- Use third person (e.g., "a young male with...", "the girl with...")
- Make it natural enough to replace the concept name in a question

Please provide the distinctive visual description focusing on PERMANENT features:

