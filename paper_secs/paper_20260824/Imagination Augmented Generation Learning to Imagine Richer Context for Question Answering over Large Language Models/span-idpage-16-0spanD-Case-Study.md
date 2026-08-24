# <span id="page-16-0"></span>D Case Study

This study illustrates the differences in how three paradigms—RAG, GAG, and AAG—utilize documents during reasoning as shown in Table [11.](#page-17-2) RAG

<span id="page-16-1"></span>

|     | 1                      | 5     | 10    | 20    | 50    |
|-----|------------------------|-------|-------|-------|-------|
| FiD | 34.69                  | 41.27 | -     | -     | 46.59 |
|     | # Document Compression |       |       |       |       |
| AAG | 32.57                  | 38.19 | 35.17 | 32.12 | 36.83 |

Table 10: Performance Metrics for Different Configurations

retrieves ten documents from an external knowledge base, while GAG employs ChatGPT to generate ten documents with higher similarity. For illustration, we present only the content of the first document. Conversely, AAG uses its proprietary context generator to create virtual compressed documents containing more information. The token counts in parentheses show that AAG requires significantly fewer tokens for processing compared to the other two methods, thus enhancing inference speed and reducing computational overhead.

<span id="page-17-2"></span>

| Question: w | hat style o | of art did henri matisse use?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|-------------|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Method      | # Docs      | Documents                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| RAG         | 10          | Henri Matisse Henri Émile Benoît Matisse (; 31 December 1869 – 3 November 1954) was a French artist, known for both his use of colour and his fluid and original draughtsmanship. He was a draughtsman, printmaker, and sculptor, but is known primarily as a painter. Matisse is commonly regarded, along with Pablo Picasso, as one of the artists who best helped to define the revolutionary developments in the visual arts throughout the opening decades of the twentieth century, responsible for significant developments in painting and sculpture. The intense colorism of the works he painted between 1900 and 1905 brought him (1860 tokens)                                                                                                                                                                                                                                                                                                                                                                                                         |
| GAG         | 10          | Henri Matisse is considered one of the most important artists of the 20th century. He is known for his use of color and his distinctive style of painting and sculpture. Matisse was a member of the French avant-garde movement and his work was influenced by other artists such as Paul Cŏ0e9zanne and Pablo Picasso (1540 tokens)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| AAG(ours)   | 1           | Henri Matisse context: a period of convalescence attack of appendic. He discovered ä kind of paradise as he later described it, to become an deeply disappointing his. In 19 he returned to study art the AcadÕ0e9mie and became a student of William-Adolphe Bouguereau Gustave Moreau Initially he painted still lif and landscapes a traditional style at which reasonable proficiency Mat was influenced the works earlier masters such as Jean-Bapt-SimÕ0e9on Ch Nicolas Pous Watteau, as well artists, such as Õ0c9douard Manet a body of work spanning over a half-century, won him recognition as a leading figure in modern art. Matisse was born in Le Cateau-CambrÕ0e9sis, in the Nord department in northern France, the oldest son of a prosperous grain merchant. He grew up in Bohain-en-Vermandois, Picardie, France. In 1887 he went to Paris to study law, working as a court administrator in Le Cateau-CambrÕ0e9sis after gaining his qualification. He first started to paint in 1889, after his mother brought him art supplies (280 tokens) |

Table 11: A inference case with used documents from WQ.

<span id="page-17-0"></span>

| Models      | # Docu- | Trainable |       | NQ    |       |      | TQA   |      |      | WQ    |       |
|-------------|---------|-----------|-------|-------|-------|------|-------|------|------|-------|-------|
| Models      | ments   | Params    | NQ    | TQA   | WQ    | NQ   | TQA   | WQ   | NQ   | TQA   | WQ    |
| T5          | 0       | 220M      | 22.16 | 3.18  | 4.12  | 2.65 | 21.8  | 3.15 | 0.88 | 2.95  | 28.3  |
| LoRA-Base   | 0       | 28.3M     | 5.43  | 3.15  | 4.02  | 0.00 | 9.60  | 0.00 | 0.22 | 1.77  | 20.47 |
| w FFN       | 0       | 141.5M    | 16.17 | 4.71  | 6.89  | 3.15 | 21.16 | 0.00 | 1.33 | 3.04  | 26.38 |
| w FFN & LCD | 0       | 141.5M    | 21.37 | 2.82  | 6.89  | 1.99 | 17.94 | 3.74 | 0.00 | 2.82  | 32.50 |
| AAG         | 0       | 26.1M     | 5.31  | 3.82  | 5.71  | 0.22 | 10.34 | 2.12 | 0.55 | 2.30  | 16.58 |
| w FFN       | 0       | 139.3M    | 21.05 | 4.52  | 6.50  | 3.51 | 19.08 | 3.15 | 2.11 | 3.84  | 28.17 |
| w FFN & LCD | 0       | 141.5M    | 23.89 | 6.21  | 10.94 | 5.31 | 22.69 | 6.30 | 3.23 | 5.10  | 30.31 |
| T5-1        | 0       | 770M      | 28.5* | 3.18  | 4.12  | 2.65 | 28.7* | 3.15 | 0.88 | 2.95  | 30.6* |
| LoRA-l      | 0       | 42.5M     | 4.42  | 6.50  | 7.87  | 3.98 | 10.03 | 3.94 | 1.99 | 6.71  | 18.11 |
| w FFN       | 0       | 445.1M    | 17.70 | 7.49  | 8.66  | 3.54 | 23.87 | 4.72 | 0.00 | 5.65  | 29.13 |
| w FFN & LCD | 0       | 445.1M    | 28.32 | 4.52  | 10.94 | 5.31 | 25.71 | 6.12 | 1.75 | 4.52  | 29.92 |
| AAG-1       | 0       | 34.8M     | 7.08  | 8.90  | 9.45  | 4.42 | 13.14 | 8.66 | 2.43 | 10.17 | 17.72 |
| w FFN       | 0       | 437.5M    | 23.01 | 8.33  | 11.02 | 3.51 | 20.08 | 3.15 | 3.51 | 5.65  | 31.50 |
| w FFN & LCD | 0       | 437.5M    | 29.32 | 10.17 | 14.06 | 7.02 | 30.11 | 7.81 | 2.65 | 7.06  | 32.68 |

Table 12: OOD and ablation experiment results in closed-book setting. \* denotes the results are from the existing papers and LCD denotes Long Context Distillation.

<span id="page-17-1"></span>

| Hallucinations                                                                                                                                                                  | Meaningless                                         |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| 4%                                                                                                                                                                              | 6%                                                  |
| Question: When is the next Deadpool movie being released?                                                                                                                       | Question: Who got the first Nobel Prize in Physics? |
| Document: "Deadpool (film) Deadpool is a 2016 American superhero film based on the Marvel Comics character of the same name, produced by Marvel Studios and distributed by Walt | Document: The Nobel Prize is not a prize in itself. |
| Disney Studios Motion Pictures.  Correct answer: May 18, 2018                                                                                                                   | Correct answer: Wilhelm Conrad Röntgen              |

Table 13: Hallucinations and Meaningless Analysis.

<span id="page-18-0"></span>

| Methods   | Prompt                                                                                                                                                                                                                                                                                                                                                                               |  |  |  |  |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|--|--|--|
| CBQA      | Please write a high-quality answer for the given question using your knowledge.<br>Only give me the answer and do not output any other words.<br>Question: {question}<br>Answer:                                                                                                                                                                                                     |  |  |  |  |
| Retrieval | Please write a high-quality answer for the given question using only the provided<br>search results (some of which might be irrelevant). Only give me the answer<br>and do not output any other words.<br>Context: {context}<br>Answer the question based on the given passages.<br>Question: {question}<br>Answer:                                                                  |  |  |  |  |
| Awakening | Please write a high-quality answer for the given question using your knowledge<br>and the provided imagined compressed results (some of which might be irrelevant).<br>Only give me the answer and do not output any other words.<br>Generated Context: {context}<br>Answer the question based on your knowledge and the given generated context.<br>Question: {question}<br>Answer: |  |  |  |  |

Table 14: Prompts for different methods on Zero-Shot setting. CBQA denotes closed-book QA that just prompts the model with the question.

<span id="page-19-0"></span>

| Models                                        | Reader<br>Params | # Docu<br>ments | NQ    | TriviaQA | WebQ  |
|-----------------------------------------------|------------------|-----------------|-------|----------|-------|
| # Closed-book Setting                         |                  |                 |       |          |       |
| T5∗<br>(Roberts et al., 2020a)                | 220M             | 0               | 25.9  | 23.8     | 27.9  |
| T5-l∗<br>(Roberts et al., 2020a)              | 770M             | 0               | 28.5  | 28.7     | 30.6  |
| T5-xl (Roberts et al., 2020a)                 | 3b               | 0               | 28.30 | 33.92    | 34.43 |
| LoRA-Base                                     | 220M             | 0               | 5.43  | 9.60     | 20.47 |
| LoRA-l                                        | 770M             | 0               | 17.70 | 23.87    | 29.13 |
| LoRA-xl                                       | 3b               | 0               | 23.15 | 32.16    | 35.24 |
| AAG (Ours)                                    | 220M             | 0               | 23.89 | 22.69    | 30.31 |
| AAG-l (Ours)                                  | 770M             | 0               | 29.32 | 30.11    | 32.68 |
| AAG-xl (Ours)                                 | 3b               | 0               | 29.59 | 35.71    | 37.40 |
| # Retrieval Augmented Generation              |                  |                 |       |          |       |
| DPR∗<br>(Karpukhin et al., 2020)              | 110M             | 100             | 41.5  | 56.8     | 41.1  |
| RAG∗<br>(Lewis et al., 2020)                  | 400M             | 10              | 44.5  | 56.1     | 45.2  |
| FiD∗<br>(Izacard and Grave, 2021)             | 220M             | 100             | 48.2  | 65.0     | 46.71 |
| FiD-l∗<br>(Izacard and Grave, 2021)           | 770M             | 100             | 51.4  | 67.6     | 50.52 |
| FiD-xl (Izacard and Grave, 2021)              | 3b               | 20              | 55.18 | 72.92    | 52.85 |
| FiD-l∗<br>(Izacard and Grave, 2021)           | 770M             | 10              | 46.7  | 61.9     | 48.1  |
| FiD-xl∗<br>(Izacard and Grave, 2021)          | 3b               | 10              | 50.1  | 66.3     | 50.8  |
| EAR-l (Chuang et al., 2023)                   | 770M             | 10              | 39.6  | 60.0     | -     |
| EAR-xl∗<br>(Chuang et al., 2023)              | 3b               | 10              | 42.3  | 64.6     | -     |
| RFiD-l (Wang et al., 2023a)                   | 770M             | 10              | 48.3  | 63.4     | -     |
| RFiD-xl (Wang et al., 2023a)                  | 3b               | 10              | 50.5  | 67.8     | -     |
| FILCO-xl∗<br>(Wang et al., 2023d)             | 3b               | 1               | 44.7  | 59.0     | -     |
| AAG (Ours)                                    | 220M             | 10              | 47.01 | 64.95    | 46.36 |
| AAG-l (Ours)                                  | 770M             | 10              | 49.92 | 69.67    | 51.52 |
| AAG-xl (Ours)                                 | 3b               | 5‡              | 50.87 | 70.34    | 52.78 |
| AAG-l (Ours)                                  | 770M             | 30              | 53.1  | 70.5     | 52.0  |
| # Generation Augmented Generation             |                  |                 |       |          |       |
| GENREAD-l (sampling)∗<br>(Yu et al., 2023)    | 770M             | 10†             | 40.3  | 67.8     | 51.5  |
| GENREAD-l (clustering)∗<br>(Yu et al., 2023)  | 770M             | 10†             | 43.5  | 70.2     | 53.5  |
| GENREAD-xl (sampling)∗<br>(Yu et al., 2023)   | 3b               | 10†             | 42.6  | 69.6     | 52.6  |
| GENREAD-xl (clustering)∗<br>(Yu et al., 2023) | 3b               | 10†             | 45.6  | 71.6     | 54.4  |
| AAG (Ours)                                    | 220M             | 10†             | 46.22 | 66.70    | 51.43 |
| AAG-l (Ours)                                  | 770M             | 10†             | 48.83 | 70.85    | 54.52 |
| AAG-xl (Ours)                                 | 3b               | 5†‡             | 49.23 | 72.18    | 55.39 |
| # Awakening Augmented Generation (Ours)       |                  |                 |       |          |       |
| LoRA-Base                                     | 220M             | 1†              | 34.51 | 54.05    | 32.28 |
| LoRA-l                                        | 770M             | 1†              | 40.05 | 62.81    | 43.70 |
| LoRA-xl                                       | 3b               | 1†              | 44.15 | 66.92    | 48.23 |
| AAG                                           | 220M             | 1†              | 40.14 | 60.75    | 41.73 |
| AAG-l                                         | 770M             | 1†              | 42.32 | 65.48    | 45.28 |
| AAG-xl                                        | 3b               | 1†              | 46.51 | 68.38    | 50.45 |

Table 15: Full QA performances (%) of different methods on three datasets. The first part (closed-book setting) indicates that explicit documentation was not utilized; The latter three parts utilize explicit augmented documents. The best results are in bold. \* means that those results are from existing papers, † denotes that the number of documents is generated (‡ indicates that the number of documents is reduced due to insufficient memory for distillation).