# <span id="page-10-1"></span>B Hallucination Detection Algorithm for Citation Reliability

The Hallucination Detection Algorithm for Citation Reliability (HDACR) is a crucial component designed to ensure the factual consistency between generated content and reference material in longform article generation tasks. This algorithm specifically targets the identification of hallucinations, which are instances where the generated content introduces false or unverifiable information that is not present in the reference material. Below is a detailed explanation of the steps involved in the HDACR algorithm, as presented in Algorithm [1.](#page-12-0)

<span id="page-11-1"></span>

| Prompt   | Prompt Text                                                                                                                                                                                                                     |  |  |  |
|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|--|--|
| Prompt 1 | Abstract[1]Abstract[2]\n Answer the following questions based<br>on the provided references. Please provide detailed answers with a minimum of<br>300 words: {Question}                                                         |  |  |  |
| Prompt 2 | You cannot refuse to answer the question. Please refer to the following<br>information:\nAbstract[1]Abstract[2]\n Answer the following<br>questions. Please provide detailed answers with a minimum of 300 words:<br>{Question} |  |  |  |
| Prompt 3 | Based on the provided references, answer the following questions. Please<br>provide detailed answers with a minimum of 300<br>words:\nAbstract[1]Abstract[2]\n Question: {Question}                                             |  |  |  |

Table 5: Prompt example (Using a paragraph length of 200-400 characters as an example.)

