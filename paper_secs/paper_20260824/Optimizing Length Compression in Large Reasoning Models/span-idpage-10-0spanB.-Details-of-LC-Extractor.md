# <span id="page-10-0"></span>B. Details of LC-Extractor

We train Qwen-2.5-3B-Instruct [\(Team,](#page-9-15) [2024\)](#page-9-15) as the LC-Extractor model. We construct a dataset consisting of 5,000 <*Question, Thinking Process, Answer*> triplets from MATH dataset and identify the position of the first correct token using Gemini-2.5-Flash [\(Google,](#page-9-16) [2025b\)](#page-9-16), followed by rigorous rule-based filtering. We then distill this knowledge into a smaller model through training for 2 epochs with these curated samples. LC-Extractor's effectiveness is validated on a 100-sample test set, achieving 98% accuracy as confirmed by human evaluation as shown in in Figure [6.](#page-12-0) LC-Extractor model is activated by the prompt in Figure [5.](#page-11-0)

