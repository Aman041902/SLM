# GoT-SLM: A Small Language Model Trained on "A Song of Ice and Fire" 📚⚔️

This repository contains the code for training a small, GPT-2 style language model from scratch on the text of the first five books of George R. R. Martin's "A Song of Ice and Fire" series. The model learns the writing style, character names, locations, and lore of Westeros and can generate new text in a similar vein.

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

The entire process is contained within the `slm.ipynb` notebook and follows these steps:

1.  **Data Preparation:** All five `.txt` files of the books are combined into a single corpus.
2.  **Tokenization:** A custom **Byte-Pair Encoding (BPE)** tokenizer is trained on the combined text, creating a vocabulary of 8000 tokens. The corpus is then encoded into token IDs and saved as binary files (`.bin`).
3.  **Data Splitting:** The tokenized data is split into an 80/20 train/validation set.
4.  **Training:** The SLM is trained on the training set using the AdamW optimizer, gradient accumulation, and mixed-precision training for efficiency. The learning rate is managed by a linear warmup followed by a cosine decay schedule.
5.  **Inference:** After training, the model can generate new text given a starting prompt.

## How to Use 🚀

### 1. Prerequisites
* Python 3.8+
* PyTorch
* A GPU is highly recommended for training.

### 2. Setup

Clone the repository and install the required packages:
```bash
git clone [https://github.com/your-username/GoT-SLM.git](https://github.com/your-username/GoT-SLM.git)
cd GoT-SLM
pip install -r requirements.txt
```

Create a `requirements.txt` file with the following contents:
```
torch
numpy
tokenizers
matplotlib
tqdm
```

### 3. Prepare the Data
1.  Create a folder named `got_data` in the root of the project directory.
2.  Place the `.txt` files for the first five books of "A Song of Ice and Fire" inside this folder.

### 4. Run the Notebook
Open and run the `slm.ipynb` notebook in an environment like Jupyter Lab or Google Colab. The notebook is divided into clear steps that will:
1.  Process the dataset.
2.  Train the tokenizer.
3.  Define the model.
4.  Train the model.
5.  Run inference to generate text.

## Example Generation

Here's an example of how to generate text once the model is trained:

```python
# Load the trained model and tokenizer
# ... (code from the notebook) ...

# Generate text
prompt = "The king sat upon the Iron Throne, his voice echoing through the silent hall."
context = torch.tensor(tokenizer.encode(prompt).ids).unsqueeze(0).to(device)

# Generate 200 new tokens
generated_tokens = model.generate(context, max_new_tokens=200)
generated_text = tokenizer.decode(generated_tokens.squeeze().tolist())

print(generated_text)
```
> **Sample Output:** *The king sat upon the Iron Throne, his voice echoing through the silent hall. He had seen the way of it, and the old gods had been made to stand beside him. "The boy is a Lannister," he said, his voice a whisper. "He will not be the first to die." He was not a man to be trifled with. He had been a boy of sixteen when he had last seen his father. He had been a boy of ten when he had last seen his sister. He had been a boy of ten when he had last seen his brother. He had been a boy of ten when he had last seen his mother. He had been a boy of ten when he had last seen his father. He had been a boy of ten when he had last seen his sister. He had been a boy of ten when he had last seen his brother. He had been a boy of ten when he had last seen his mother. He had been a boy of ten when he had last seen his father. He had been a boy of ten when he had last seen his sister. He had been a boy of ten when he had last seen his brother. He had been a boy of ten when he had last seen his mother...*
