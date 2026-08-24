# <span id="page-17-0"></span>E Effect of Training Data Bias on Format Collapse

Since training data may implicitly favor certain reasoning formats, it is important to examine whether such bias can induce format collapse. To this end, we analyze an ARM trained solely on the AIME dataset (1983-2024), which primarily favors *Long CoT* solutions due to its competition-level complexity. We present the result in Table [4.](#page-17-3) We evaluate this model on two simpler tasks—CSQA and OBQA—and observe that the model overwhelmingly selects *Long CoT* (∼80%) even when simpler formats would suffice. This confirms that training on a biased dataset can indeed lead to over-reliance on a single reasoning format.

To mitigate this, ARM is trained on a diverse mixture of datasets across a wide range of difficulties. As shown in Table [5,](#page-17-4) the full ARM recipe achieves both higher accuracy and significantly reduced token usage on the tasks, demonstrating that our approach effectively prevents format collapse and encourages adaptive reasoning behavior across domains.

