# ORPO Llama Finetuner

A streamlined framework for fine-tuning Llama-like models using the **Monolithic Preference Optimization without Reference Model (ORPO)** algorithm. This project provides a clean, modular, and easy-to-use codebase to replicate and extend ORPO fine-tuning experiments.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🚀 Features

-   **Modular Design**: Code is split into logical components (`config`, `model`, `trainer`, `utils`) for better readability and maintenance.
-   **ORPO Implementation**: A self-contained implementation of the ORPO loss function.
-   **Llama Architecture**: Includes the complete Llama model definition, making it independent of external custom model files.
-   **Automated Dataset Handling**: Downloads and processes the `mlabonne/orpo-dpo-mix-40k` dataset, with local caching for faster subsequent runs.
-   **Pre-trained Model Loading**: Automatically downloads a 138M parameter Llama model and tokenizer from Google Drive if not found locally.
-   **Experiment Tracking**: Integrated with Weights & Biases (W&B) for logging metrics, losses, and hyperparameters.
-   **Efficient Training**: Supports `bfloat16` for faster training on compatible hardware and `torch.compile()` for significant speedups.

---

## 📂 Project Structure

The repository is organized to separate concerns, making it easy to navigate and modify.

```
ORPO-Llama-Finetuner/
│
├── .gitignore
├── README.md
├── requirements.txt
├── config.py           # All hyperparameters and configuration settings
├── main.py             # Main script to run the training
│
├── model/
│   ├── __init__.py
│   └── llama.py        # Llama model architecture
│
└── utils/
    ├── __init__.py
    └── gdrive.py       # Google Drive download utilities
```

---

## 🛠️ Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/frezazadeh/ORPO-Llama-Finetuner.git
cd ORPO-Llama-Finetuner
```

### 2. Install Dependencies

It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. Google Drive API Credentials

This project requires credentials to download the base model and tokenizer from Google Drive.

1.  Follow the [Google Drive API Python Quickstart](https://developers.google.com/drive/api/quickstart/python) to enable the API and download your `credentials.json` file.
2.  Place the `credentials.json` file in the root directory of the project.
3.  The first time you run the script, you will be prompted to authorize the application. Follow the URL, grant permission, and paste the authorization code back into the terminal. A `token.json` file will be created to store your credentials for future runs.


### 4. Hugging Face API Token (Recommended)

To download datasets and models from the Hugging Face Hub, it's best practice to use an API token.

1.  Get your token from your Hugging Face account settings: [**hf.co/settings/tokens**](https://hf.co/settings/tokens).
2.  Open the `main.py` file and add your token to the following lines at the top of the script:

    ```python
    # main.py

    os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN_HERE"
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    ```

---

## ⚙️ Configuration

All training hyperparameters, paths, and settings are located in `config.py`. You can easily modify this file to experiment with different setups.

Key parameters include:
- `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`
- `CONTEXT_LENGTH`
- `ALPHA_ORPO`: The $\lambda$ coefficient for the ORPO loss term.
- `ENABLE_WANDB_LOGGING`: Set to `False` to disable Weights & Biases logging.

---

## ▶️ How to Run

To start the fine-tuning process, simply run the `main.py` script:

```bash
python main.py
```

The script will:
1.  Check for the model and tokenizer, downloading them from Google Drive if needed.
2.  Check for the dataset, downloading and processing it from the Hugging Face Hub if needed.
3.  Initialize the model, optimizer, and data loaders.
4.  Begin the training loop, logging progress and metrics to the console and W&B.
5.  Save a model checkpoint at the end of each epoch in the `llm_models/` directory.

---

## 📜 License
This llm.py file includes code from the open source LLaMA2.c repository.

This project is licensed under the MIT License.
