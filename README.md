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

👁️ Visualization Tool (New Feature)

This tool allows you to visualize key pathological cells directly on the original microscopic images, highlighting high-risk areas based on AI predictions.

1. Prepare Dataset Structure

Ensure your data is organized as follows within the repository or adjacent to it:

hybrid-cell-classification/
├── Dataset/
│   ├── images/       # Contains original microscopic images (.jpg/.png)
│   └── contour/      # Contains contour JSON files (e.g., B5_141_contours.json)
├── visualization.py       # Visualization entry point
└── ...


2. Run Visualization

Run the main script to process images end-to-end (Crop -> Predict -> Visualize):

# Run for a specific image (Recommended for testing)
# Edit TARGET_IMG_ID in src/main.py before running
python src/visualization.py