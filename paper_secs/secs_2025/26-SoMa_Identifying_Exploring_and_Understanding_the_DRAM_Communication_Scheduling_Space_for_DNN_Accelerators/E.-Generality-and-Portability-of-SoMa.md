# *E. Generality and Portability of SoMa*

The proposed encoding and SoMa framework exhibit excellent generality for two main reasons: 1) the template depicted in Fig. 1 is highly general, encompassing many accelerators from both the industry [9], [25], [26], [34], [53] and academia [12], [17], [30]; 2) The hardware behavior information encoded by our notation is very general (i.e., computing, data loading, and storing, instruction dependency, etc.).

Our SoMa also possesses excellent portability, due not only to the aforementioned reasons but also because our framework is designed with a robust modular architecture. This design allows for easy adaptation to different accelerators, which may have distinct core micro-architectures, by simply replacing the relevant Core Array Scheduler & Evaluator modules and Instruction Generation module. We have developed a comprehensive compilation flow for our accelerator [1], which can serve as a concrete example for porting to other accelerators.

