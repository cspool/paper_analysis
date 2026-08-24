# <span id="page-12-2"></span>C PROMPT-WITH-CONTEXT DATASET

We introduce the PROMPT-WITH-CONTEXT (PWC) dataset where each sample entry is a triple (text, prompt, answer), as depicted in Figure [9.](#page-13-1) To construct this dataset, we first sample 20k texts from the Pile dataset. Then, for each text, we employ the GPT-4 to provide 15 prompts (10 specific prompts and 5 general prompts) about the text and give the corresponding answers. The prompt instructing the GPT-4 is outlined in Listing [1.](#page-12-4)

The dataset is composed of 240k examples for training purposes, with an additional 18k examples for testing. The context length distribution of test samples is presented in Table [10.](#page-14-0)

### Listing 1: Prompt used by GPT4 API to generate the PWC dataset.

<span id="page-12-4"></span>Design 10 prompts specified to the above text to test understanding of the above text. These prompts should be diverse and cover as many

#### <span id="page-13-1"></span>*As ar&ficial intelligence becomes an increasingly powerful force, some of the world's biggest companies are worrying about how the technology will be used ethically, and how the public will perceive its spread. To combat these problems (among others), five tech companies — Google, Amazon, MicrosoD, Facebook, and IBM — set up a research group called the Partnership on AI. … New trustees joining include Dario Amodei of OpenAI — another industry AI research group founded by Elon Musk and Y Combinator president Sam Altman — and Carol Rose of the ACLU. AI taking white collar jobs, eroding trust in public media, becoming embedded in public ins&tu&ons like the courts and hospitals: these are the sorts of problems facing the industry in the future.* **Context Prompt:** List the five tech companies that ini6ally set up the Partnership on AI. **Answer:** *Google, Amazon, MicrosoD, Facebook and IBM.* **Prompt:** What are poten6al challenges the AI industry might face in the future? **Answer:** *Challenges include AI taking white collar jobs, eroding trust in public media, and becoming embedded in public ins&tu&ons like courts and hospitals..* **Prompt:** Which organiza6on did Dario Amodei previously work for? **Answer:** *OpenAI* **… …**

Figure 9: Construction of the PWC dataset: we use the GPT-4 to generate a variety of prompt-answer pairs according to contexts. The resulting dataset is used for instruction fine-tuning (240k for training) and evaluation (18k for testing) in this work.

aspects (e.g., topic, genre, structure, style, polarity, key information and details) of the text as possible. The first half of these prompts should be like an instruction, the other should be like a question. In addition to the prompts specified to the above text, please also design 5 general prompts like "rephrase the above text", "summarize the above text", "write a title for the above text", "extract a few keywords for the above text" and "write a paragraph (i.e., continuation) that follows the above text". Each prompt should be outputted in the following format: [{"prompt": your generated prompt, "answer": the answer to the prompt}]

