# GoT-SLM: A Small Language Model Trained on "A Song of Ice and Fire" 📚⚔️

This repository contains the code and a pre-trained tokenizer for building a small, GPT-2 style language model from scratch. The model is designed to be trained on the text of the first five books of George R. R. Martin's "A Song of Ice and Fire" series. It learns the writing style, character names, locations, and lore of Westeros and can generate new text in a similar vein.

This project is intended as an educational tool to demonstrate the key steps of building a modern language model: data preparation, tokenization, model architecture, training, and inference.

## Model Architecture 🤖

The model is a decoder-only Transformer, similar in architecture to the original GPT-2 model. It is built using PyTorch.

* **Parameters:** ~4.5 Million
* **Architecture:** GPT-2 Style Decoder-Only Transformer
* **Activation Function:** SiLU (Swish)
* **Attention:** Causal Self-Attention (with Flash Attention support)
* **Normalization:** Pre-Layer Normalization

### Key Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `vocab_size` | 8000 | The number of unique tokens in the vocabulary. |
| `block_size` | 256 | The context window size for the model. |
| `n_layer` | 3 | The number of Transformer blocks. |
| `n_head` | 4 | The number of attention heads. |
| `n_embd` | 256 | The embedding dimension for tokens. |
| `dropout` | 0.1 | Dropout rate for regularization. |

## Project Pipeline ⚙️

1.  **Data Preparation:** The dataset (the text of five GoT books) is downloaded from an external source (Kaggle). The notebook then combines them into a single corpus.
2.  **Tokenization:** A custom **Byte-Pair Encoding (BPE)** tokenizer is used. A pre-trained tokenizer (`got_tokenizer.json`) is included in this repository to save time.
3.  **Training:** The SLM is trained on the corpus using the AdamW optimizer, gradient accumulation, and mixed-precision training for efficiency.
4.  **Inference:** After training, the model can generate new text given a starting prompt.

## Files in this Repository

```
.
├── slm.ipynb
├── README.md
├── got_tokenizer.json
└── requirements.txt
```

## How to Use 🚀

Follow these steps to set up the project and train your own model.

### Step 1: Clone the Repository
Clone this repository to your local machine:
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

### Step 2: Set Up the Environment
This project includes a `requirements.txt` file to make setup easy. Simply run the following command to install all necessary libraries:
```bash
pip install -r requirements.txt
```

### Step 3: Download the Dataset
The raw text data for the books is not included in this repository. You must download it from Kaggle.

1.  Go to this Kaggle dataset link: **[(https://www.kaggle.com/datasets/khulasasndh/game-of-thrones-books)]**
2.  Download the files.
3.  Create a folder named `got_data` in the root of this project directory.
4.  Place the `.txt` files for the five books into the `got_data` folder. The notebook expects the files to be found there.

Your project structure should now look like this:
```
.
├── got_data/
│   ├── book1.txt
│   ├── book2.txt
│   └── ... (etc.)
├── slm.ipynb
├── README.md
├── got_tokenizer.json
└── requirements.txt
```

### Step 4: Run the Notebook
Now you are ready to start!
1.  Open the `slm.ipynb` notebook in an environment like Jupyter Lab or Google Colab.
2.  Run the cells in order.
3.  **Note:** You can **skip the tokenizer training cell** since the `got_tokenizer.json` file is already provided and the notebook will load it.

