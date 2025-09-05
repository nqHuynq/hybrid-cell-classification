# hybrid-cell-classification
Official Implementation of "A Novel Expert-Annotated Single-Cell Dataset for Thyroid Cancer Diagnosis with Deep Learning Benchmarks"

## 🚀 Usage

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/hybrid-cell-classification.git
cd hybrid-cell-classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train and evaluate (example: ConvNeXt + SPA Loss)
python train.py --model convnext_base --loss spa --epochs 10 --batch_size 4
python evaluate.py --model convnext_base --loss spa --threshold tuned

```
## Evaluate with Pretrained Weights

If you don’t have time to train the model from scratch, you can directly evaluate using our pretrained weights hosted on - [ConvNext_SCTC (Hugging Face)](https://huggingface.co/SoftmaxSamurai/ConvNext_SCTC)
