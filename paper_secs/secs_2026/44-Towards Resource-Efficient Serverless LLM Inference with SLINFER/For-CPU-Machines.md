# For CPU Machines:

cd \$PROJECT\_BASE/vLLM\_modify pip install -r requirements-build.txt --extra-index-url https://download.pytorch. org/whl/cpu

PIP\_PRE=1 PIP\_EXTRA\_INDEX\_URL="https:// download.pytorch.org/whl/cpu https://storage.openvinotoolkit.org/simple /wheels/nightly/" VLLM\_TARGET\_DEVICE= openvino python -m pip install -v -e .

- *3) Model Preparation:* Download the three models mentioned above from Hugging Face into \$PROJECT\_BASE/huggingface\_models/.
  - *On GPU machine:*

\$PROJECT\_BASE/huggingface\_models/export \_gpu\_models.sh

• *On CPU machine:*

\$PROJECT\_BASE/huggingface\_models/export \_cpu\_models.sh

