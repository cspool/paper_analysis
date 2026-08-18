# *B. Application to Other Predictors*

Although targeting MDPs, SSBench can be easily extended to automatically characterize other predictors, such as branch predictors [\[72\]](#page-14-7), address predictors [\[29\]](#page-13-23), and value predictors [\[28\]](#page-13-22). Benefiting from its model-based methods, SSBench loosely couples its testing algorithms with the test target. Specifically, components in the Test Suites, including characterization of the state machine, hash function, eviction sets, organization, and replacement policy, can be directly reused to characterize other predictors.

To adapt SSBench to other predictors, three modifications are required. First, a new taxonomy should be developed to identify and categorize different predictor designs. Second, training and probing strategies need to be redesigned according to the behavior of the predictors, so that the automated testcase generator can be implemented, enabling the predictor identification. Third, the instruction set used for input needs to be extended to ensure cross-platform compatibility.

