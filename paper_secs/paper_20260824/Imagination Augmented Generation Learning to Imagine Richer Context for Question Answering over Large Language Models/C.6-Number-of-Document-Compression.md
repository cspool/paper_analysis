# C.6 Number of Document Compression

In the first stage, we sampled 30,000 instances from the training sets of NQ and TQA, respectively, and used all 3,417 instances from the WebQ training set. To determine the number of retrieved documents to use for each dataset in stage 1, we conducted tests using the FiD (T5-Base) experiment. As shown in Table [10,](#page-16-1) we can find that compressing five documents yielded relatively good performance. Consequently, we decided to compress five documents for each instance.

