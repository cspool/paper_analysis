# A.4.4 ExperimentE3:PredictionMethod [0.5 compute-

hours]. This experiment evaluates the prediction method using di"erent pool sizemapping approaches(i.e., constant, Poisson, and concentrated) using di"erent prediction intervals. The results are used to generate Figure 8 and support claim C3. [Preparation] None.

[Execution] Run the following commands:

cd /<repo\_root>/experiments/

./scripts/fig7.sh

[Results] Two !gures similar to Figure 8a and Figure 8b are generated under/<repo\_root>/experiments/directory. Raw results for failure rate and cost breakdown are generated in csv format under /<repo\_root>/experiments/fig7 directory.

#### A.4.5 Experiment E4: Reactive Method [1 compute-

hour]. This experiment evaluates the reactive method with di"erent scale-up and scale-down factors. The results are used to generate Figure 11 and support claim C4.

[Preparation] None.

[Execution] Run the following commands:

cd /<repo\_root>/experiments/

./scripts/fig10a.sh

./scripts/fig10b.sh

[Results] Two !gures similar to the ones shown in Figure 11 are generated under /<repo\_root>/experiments/ directory. Raw results for failure rate and cost breakdown are generatedincsvformat under/<repo\_root>/experiments/ fig10a and /<repo\_root>/experiments/fig10b directories.

#### A.4.6 Experiment E5: Aggressive Container Creation

[0.5 compute-hours]. This experiment evaluates the aggressive container creation optimization. The results are used to generate Figure 14 and support claim C5.

[Preparation] None.

[Execution] Run the following commands:

cd /<repo\_root>/experiments/

./scripts/fig12.sh

[Results] A !gures similar to Figure 14a is generated under /<repo\_root>/experiments/ directory. Raw results for failure rate and cost breakdown are generated in csv format under /<repo\_root>/experiments/fig12 directory.

#### A.5 Notes on Reusability

Our artifact includes a simulator of a real serverless platform. This simulator can be used to compare the performance and cost of di"erent resource sizing methods. Also, it enables estimating the bene!ts of di"erent optimizations or improvements in the platform (e.g., the estimated cost reduction of reducing the container creation latency by 50%). Our simulator requires input traces in a speci!c format. We include sample traces for the allocation trace and the life cycle trace. Other traces must be transformed to the required format before running the simulator. Also, the simulator accepts a JSON!le specifying a list of experiments to simulate. Each experiment speci!es an optimization method along with a set of parameters.