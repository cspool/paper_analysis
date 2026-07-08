# 1. paper_isca26 (2 papers)
python3 scripts/run_all_papers.py \
  --paper-base-dir papers_md/paper_isca26 \
  --checkpoint-dir paper_extract_checkpoints/paper_isca26 \
  --output-repo-dir repos/repo_paper_isca26

# 2. system_model_codesign (16 papers)
python3 scripts/run_all_papers.py \
  --paper-base-dir papers_md/system_model_codesign \
  --checkpoint-dir paper_extract_checkpoints/system_model_codesign \
  --output-repo-dir repos/repo_system_model_codesign

# 3. Vendor-Specific_SDKs (1 entry)
python3 scripts/run_all_papers.py \
  --paper-base-dir papers_md/Vendor-Specific_SDKs \
  --checkpoint-dir paper_extract_checkpoints/Vendor-Specific_SDKs \
  --output-repo-dir repos/repo_Vendor-Specific_SDKs

# 4. On-device_AI_Accelerators (2 papers)
python3 scripts/run_all_papers.py \
  --paper-base-dir "papers_md/On-device_AI_Accelerators-_Performance_Characterization_&_Optimization" \
  --checkpoint-dir "paper_extract_checkpoints/On-device_AI_Accelerators-_Performance_Characterization_&_Optimization" \
  --output-repo-dir "repos/repo_On-device_AI_Accelerators-_Performance_Characterization_&_Optimization"

# 5. edge_serving (9 papers)
python3 scripts/run_all_papers.py \
  --paper-base-dir papers_md/edge_serving \
  --checkpoint-dir paper_extract_checkpoints/edge_serving \
  --output-repo-dir repos/repo_edge_serving

# 6. LLM_&_GenAI_Specialized (1 entry)
python3 scripts/run_all_papers.py \
  --paper-base-dir "papers_md/LLM_&_GenAI_Specialized" \
  --checkpoint-dir "paper_extract_checkpoints/LLM_&_GenAI_Specialized" \
  --output-repo-dir "repos/repo_LLM_&_GenAI_Specialized"

# 7. LLM_Inference_on_Mobile_SoCs (7 papers)
python3 scripts/run_all_papers.py \
  --paper-base-dir papers_md/LLM_Inference_on_Mobile_SoCs \
  --checkpoint-dir paper_extract_checkpoints/LLM_Inference_on_Mobile_SoCs \
  --output-repo-dir repos/repo_LLM_Inference_on_Mobile_SoCs
