# PDF → Markdown 转换指令

生成时间: 2026-07-03 11:19
脚本: `scripts/pdf_to_md.py batch`

---

## 1. agent_system (1 PDF)

```bash
python3 scripts/pdf_to_md.py batch \
  "papers_pdf/2026-07-02_downloads/agent_system/" \
  -o "paper_markdown/2026-07-02_downloads_agent_system"
```

## 2. Application-centric_On-device_AI_Systems (2 PDFs)

```bash
python3 scripts/pdf_to_md.py batch \
  "papers_pdf/2026-07-02_downloads/Application-centric_On-device_AI_Systems/" \
  -o "paper_markdown/2026-07-02_downloads_Application-centric_On-device_AI_Systems"
```

## 3. Attention_Acceleration (4 PDFs)

```bash
python3 scripts/pdf_to_md.py batch \
  "papers_pdf/2026-07-02_downloads/Attention_Acceleration/" \
  -o "paper_markdown/2026-07-02_downloads_Attention_Acceleration"
```

## 4. Compiler-based_ML_Optimization (1 PDF)

```bash
python3 scripts/pdf_to_md.py batch \
  "papers_pdf/2026-07-02_downloads/Compiler-based_ML_Optimization/" \
  -o "paper_markdown/2026-07-02_downloads_Compiler-based_ML_Optimization"
```

## 5. edge_serving (8 PDFs)

```bash
python3 scripts/pdf_to_md.py batch \
  "papers_pdf/2026-07-02_downloads/edge_serving/" \
  -o "paper_markdown/2026-07-02_downloads_edge_serving"
```

## 6. Industry_Report (10 PDFs)

```bash
python3 scripts/pdf_to_md.py batch \
  "papers_pdf/2026-07-02_downloads/Industry_Report/" \
  -o "paper_markdown/2026-07-02_downloads_Industry_Report"
```

## 7. LLM_Inference_on_Mobile_SoCs (6 PDFs)

```bash
python3 scripts/pdf_to_md.py batch \
  "papers_pdf/2026-07-02_downloads/LLM_Inference_on_Mobile_SoCs/" \
  -o "paper_markdown/2026-07-02_downloads_LLM_Inference_on_Mobile_SoCs"
```

## 8. Multi-DNN_-_Heterogeneous_Runtime_Scheduling (1 PDF)

```bash
python3 scripts/pdf_to_md.py batch \
  "papers_pdf/2026-07-02_downloads/Multi-DNN_-_Heterogeneous_Runtime_Scheduling/" \
  -o "paper_markdown/2026-07-02_downloads_Multi-DNN_-_Heterogeneous_Runtime_Scheduling"
```

## 9. MultiModal_Serving (32 PDFs)

```bash
python3 scripts/pdf_to_md.py batch \
  "papers_pdf/2026-07-02_downloads/MultiModal_Serving/" \
  -o "paper_markdown/2026-07-02_downloads_MultiModal_Serving"
```

## 10. News (117 PDFs)

```bash
python3 scripts/pdf_to_md.py batch \
  "papers_pdf/2026-07-02_downloads/News/" \
  -o "paper_markdown/2026-07-02_downloads_News"
```

## 11. On-device_AI_Accelerators (1 PDF)

```bash
python3 scripts/pdf_to_md.py batch \
  "papers_pdf/2026-07-02_downloads/On-device_AI_Accelerators-_Performance_Characterization_&_Optimization/" \
  -o "paper_markdown/2026-07-02_downloads_On-device_AI_Accelerators-_Performance_Characterization_&_Optimization"
```

## 12. On-device_Training,_Model_Adaptation (6 PDFs)

```bash
python3 scripts/pdf_to_md.py batch \
  "papers_pdf/2026-07-02_downloads/On-device_Training,_Model_Adaptation/" \
  -o "paper_markdown/2026-07-02_downloads_On-device_Training,_Model_Adaptation"
```

## 13. Profilers (1 PDF)

```bash
python3 scripts/pdf_to_md.py batch \
  "papers_pdf/2026-07-02_downloads/Profilers/" \
  -o "paper_markdown/2026-07-02_downloads_Profilers"
```

## 14. Quantization-Sparsity (1 PDF)

```bash
python3 scripts/pdf_to_md.py batch \
  "papers_pdf/2026-07-02_downloads/Quantization-Sparsity/" \
  -o "paper_markdown/2026-07-02_downloads_Quantization-Sparsity"
```

## 15. Serving (58 PDFs)

```bash
python3 scripts/pdf_to_md.py batch \
  "papers_pdf/2026-07-02_downloads/Serving/" \
  -o "paper_markdown/2026-07-02_downloads_Serving"
```

## 16. system_model_codesign (15 PDFs)

```bash
python3 scripts/pdf_to_md.py batch \
  "papers_pdf/2026-07-02_downloads/system_model_codesign/" \
  -o "paper_markdown/2026-07-02_downloads_system_model_codesign"
```

---

共 16 个子目录，264 篇 PDF。

## 一键批量

```bash
src="papers_pdf/2026-07-02_downloads"
for d in "$src"/*/; do
    name=$(basename "$d")
    [ "$(find "$d" -name '*.pdf' 2>/dev/null | wc -l)" -eq 0 ] && continue
    echo "=== $name ==="
    python3 scripts/pdf_to_md.py batch "$d" -o "paper_markdown/2026-07-02_downloads_$name"
done
```
