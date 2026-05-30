# VIII. EVALUATION

#### A. Evaluation Settings

We evaluate PIMphony using a range of LLMs (Table I) with context lengths up to 128K, across four tasks from LongBench [7] and LV-Eval [73] (Table II). Non-GQA models (LLM-7B/72B-32K) are tested with LongBench, while GQA-enabled models (LLM-7B/72B-128K) are evaluated with LV-Eval. For comparison, we use two PIM-based baselines: CENT [16], a PIM-only system with 16GB per module, and NeuPIMs [21], a hybrid xPU+PIM system with 32GB per module. We modify both simulators [16], [21] to integrate our techniques, configuring 128GB for 7B and 512GB for 72B models, following prior PIM studies. PIMphony is modeled using a validated Ramulator-based simulator incorporating AiMX [30], [35], [37] architecture, and is evaluated using the parameters detailed in Table IV.

