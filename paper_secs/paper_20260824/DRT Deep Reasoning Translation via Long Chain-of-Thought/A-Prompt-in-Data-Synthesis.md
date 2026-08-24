# A Prompt in Data Synthesis

### <span id="page-10-0"></span>A.1 Prompts in Literature Book Mining

SYSTEM PROMPT:

You are assigned to translate an English literary work into Chinese. The text may include descriptions or expressions that embody English cultural nuances, which may not resonate with Chinese language habits. In such instances, a literal translation may not be appropriate; instead, these sentences should be paraphrased to convey their intended meaning effectively.

USER PROMPT:

The English sentence is provided as follows: <english sentence> {sentence}

</english sentence>

Please begin by assessing whether the English sentence contains any metaphors or similes. If there are none, respond with "no metaphors and no similes."

If the English sentence does contain metaphors or similes, provide a literal translation of them, and then evaluate whether the literal translation is appropriate and easy for Chinese natives to understand.

If it is suitable, format your response as follows (two lines):

"your literal translation for metaphors/similes here (in Chinese)"

"suitable"

If it is unsuitable, please provide the reason for the unsuitability. Format your response as follows (three lines):

"your literal translation for metaphors/similes here (in Chinese)"

"unsuitable"

"reason for unsuitability here (in Chinese)"

### <span id="page-10-1"></span>A.2 Prompts in Multi-Agent Framework

## Translator Agent (Word-level translation)

Given an English sentence, identify the important words (usually nouns, verbs, technical terms, and named entities that require special attention in translation) and translate them into Chinese. Output the translations in JSON format, for example:

{"EnglishWord1": "ChineseTranslation", "English-Word2": "ChineseTranslation"}

The Chinese translations can be a single translation or multiple options as deemed appropriate.

## Translator Agent (Preliminary translation)

SYSTEM PROMPT:

Given an English sentence and a JSON object containing potential translations of important keywords, produce a Chinese literal translation of the entire sentence. Please directly output the Chinese translation without any descriptions.

USER PROMPT:

<English Sentence> {sentence} </English Sentence> <Potential Keyword Translation> {keyword translation}

</Potential Keyword Translation>

## Translator Agent (Refinement translation)

In the refine loop, the translator agent receives the feedback of the previous translation, and then provides a new translation. The prompt is a multiturn dialogue between the translator and advisor, where the system prompt is the same as the preliminary translation.

### Advisor Agent

Please rate the Chinese translation of the following English text and provide your comments and suggestions.

## Evaluator Agent

SYSTEM PROMPT:

Please evaluate the following Chinese translation of an English text. Rate the translation on a scale of 0 to 100, where:

- 10 points: Poor translation; the text is somewhat understandable but contains significant errors and awkward phrasing that greatly hinder comprehension for a Chinese reader.
- 30 points: Fair translation; the text conveys the basic meaning but lacks fluency and contains several awkward phrases or inaccuracies, making it challenging for a Chinese reader to fully grasp the intended message.
- 50 points: Good translation; the text is mostly fluent and conveys the original meaning well, but may have minor awkwardness or slight inaccuracies that could confuse a Chinese reader.
- 70 points: Very good translation; the text is smooth and natural, effectively conveying the intended meaning, but may still have minor issues that could slightly affect understanding for a Chinese reader.
- 90 points: Excellent translation; the text is fluent and natural, conveying the original meaning clearly and effectively, with no significant issues that would hinder understanding for a Chinese reader.

Please provide the reason first, followed by a score. Format your evaluation in the JSON structure below: {"reason": "reason for the score", "score": int}

## <span id="page-11-0"></span>A.3 Prompts in Thought Reformulation

A student is engaged in the task of translating an English sentence into Chinese.

The English sentence is as follows: <English Sentence> {sentence} </English Sentence>

This student constantly thinks about and optimizes his translation. The whole process is shown as follows:

<Translation Process> {translation process} </Translation Process>

Please polish the whole translation process into a long first-person self-reflection description (use the present tense).

The self-reflection should begin with selecting the keywords from the English sentence, translating the keywords, and then attempt to translate the whole sentence, and then think about whether the translation is good or not, and iteratively make translation attempts. Finally, make a final translation decision.

Output the self-reflection description directly without any additional descriptions or explanations. Each line in the self-reflection description can be regarded as a reasoning step to the translation.

