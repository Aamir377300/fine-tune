# Fine-Tuning Mistral-7B-Instruct with QLoRA for YouTube Comment Responses

Fine-tuning a 7-billion parameter LLM to act as a personalized YouTube comment responder — using QLoRA so it runs on consumer-grade hardware without needing massive GPU memory.

The fine-tuned persona is called **AamirGPT**: a virtual data science consultant that replies to viewer comments in a consistent, accessible tone and always signs off with `–AamirGPT`.

---

## What is QLoRA?

Training a 7B model from scratch requires updating billions of parameters — that demands enormous GPU memory (often 40–80GB+). QLoRA solves this with two ideas stacked together:

**1. Quantization (the "Q")**
The base model weights are loaded in **4-bit precision** (using GPTQ format) instead of the usual 16-bit or 32-bit. This alone cuts memory usage by 4–8x, making the model fit on a single consumer GPU.

**2. Low-Rank Adaptation (the "LoRA")**
Instead of updating all model weights during fine-tuning, LoRA injects small trainable matrices into specific layers. Here's the core idea:

A weight matrix `W` in the model is large (e.g., 4096×4096). Rather than updating all of it, LoRA approximates the update as a product of two small matrices:

```
ΔW = A × B
where A is (4096 × r) and B is (r × 4096), with r = 8
```

This reduces trainable parameters from `O(n²)` to `O(n × r)`. The original weights stay **frozen** — only `A` and `B` are trained.

**The result in this project:**
- Total model parameters: **264,507,392**
- Trainable LoRA parameters: **2,097,152**
- That's only **0.79%** of the model being trained

During inference, the LoRA updates are merged back into the base weights — zero extra overhead.

---

## LoRA Configuration

```python
LoraConfig(
    r=8,                        # Rank — controls the size of the adapter matrices
                                # Lower rank = fewer params, less expressive
    lora_alpha=32,              # Scaling factor — higher value = stronger adapter influence
                                # Effective learning rate scales as lora_alpha / r = 4x
    target_modules=["q_proj"], # Only the query projection in attention layers is adapted
    lora_dropout=0.05,          # Dropout on adapter layers to reduce overfitting
    bias="none",
    task_type="CAUSAL_LM"
)
```

Only `q_proj` (query projection) is targeted here — a conservative choice that keeps training fast and stable on a small dataset.

---

## Project Structure

```
Fine-Tuning_With_QLoRA/
├── create-dataset.ipynb        # Step 1: Process raw CSV → HuggingFace dataset
├── QLoRA_Fine-Tuning.ipynb     # Step 2: Fine-tune, evaluate, and run inference
├── data/
│   ├── YT-comments.csv         # Raw YouTube comments (input)
│   ├── train/                  # 50 training examples (Arrow format)
│   └── test/                   # 9 test examples (Arrow format)
└── model/
    ├── checkpoint-3/ ... checkpoint-30/   # LoRA adapter checkpoints per epoch
    └── runs/                              # TensorBoard training logs
```

> The `model/` checkpoints only store the **LoRA adapter weights** (`adapter_model.safetensors`), not the full 7B base model. The base model is loaded separately from HuggingFace Hub at runtime.

---

## Setup

### Prerequisites

- Python **3.9 or higher**
- A **CUDA-capable GPU** with at least **8GB VRAM** (the 4-bit quantized model + small batch size fits on most modern consumer GPUs like RTX 3080/4080)
- CUDA toolkit installed and accessible (`nvidia-smi` should work in your terminal)

### 1. Clone the Repository

```bash
git clone https://github.com/Shreyash-Gaur/Fine-Tuning_With_QLoRA.git
cd Fine-Tuning_With_QLoRA
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install transformers peft datasets accelerate bitsandbytes optimum auto-gptq
```

| Package | Purpose |
|---|---|
| `transformers` | Load and run Mistral model + tokenizer |
| `peft` | LoRA adapter creation and training |
| `datasets` | Dataset loading and processing |
| `accelerate` | Multi-GPU / device mapping support |
| `bitsandbytes` | 8-bit optimizer (`paged_adamw_8bit`) |
| `auto-gptq` | Load GPTQ 4-bit quantized model |
| `optimum` | Backend support for GPTQ models |

### 4. Launch Jupyter

```bash
pip install jupyter
jupyter notebook
```

### Hardware Notes

- `fp16=True` and `paged_adamw_8bit` are used to keep memory usage low during training
- If you run out of VRAM, reduce `per_device_train_batch_size` from `4` to `2` in the training config
- CPU-only machines are not supported — the model requires CUDA for both training and inference

---

## How to Run

The project runs in two sequential notebooks. Open them in Jupyter in this order:

### Step 1 — `create-dataset.ipynb`

This notebook prepares the dataset from the raw CSV file and saves it locally.

1. Open `create-dataset.ipynb` in Jupyter
2. Update the `output_dir` variable to point to your local `data/` folder path
3. Run all cells (`Kernel → Restart & Run All`)
4. After it finishes, you'll see `data/train/` and `data/test/` folders created with Arrow files

> Skip this step if the `data/` folder already exists in the repo — the processed dataset is already included.

### Step 2 — `QLoRA_Fine-Tuning.ipynb`

This is the main notebook — it loads the model, fine-tunes it, and runs inference.

1. Open `QLoRA_Fine-Tuning.ipynb` in Jupyter
2. Update the `output_dir` variable in the dataset loading cell to match your local `data/` path
3. Run all cells in order (`Kernel → Restart & Run All`)

The notebook runs through these stages automatically:

| Stage | What happens |
|---|---|
| **Load model** | Downloads `TheBloke/Mistral-7B-Instruct-v0.2-GPTQ` from HuggingFace (~4GB) |
| **Base model test** | Runs a sample comment through the untuned model so you can compare output later |
| **Prepare for training** | Enables gradient checkpointing and wraps model with LoRA adapters |
| **Tokenize dataset** | Converts text examples to token IDs with left-side truncation at 512 tokens |
| **Train** | Runs 10 epochs, saves a checkpoint after each epoch to `model/` |
| **Inference** | Runs the fine-tuned model on sample comments and prints responses |

Training takes approximately **60–90 minutes** on a mid-range GPU (e.g., RTX 3080).

### Optional — Push to HuggingFace Hub

If you want to share your fine-tuned adapter weights publicly:

```python
from huggingface_hub import login
login("your_hf_token_here")   # get token from huggingface.co/settings/tokens

model.push_to_hub("your-username/AamirGPT")
trainer.push_to_hub("your-username/AamirGPT")
```

---

## What Each Notebook Does (Code Walkthrough)

### `create-dataset.ipynb` — Build the Dataset

- Reads `YT-comments.csv` containing real YouTube comments and their ideal responses
- Formats each pair into Mistral's instruction format:
  ```
  [INST] {system_prompt} \n{comment} \n[/INST] {response}
  ```
- Saves the result as a HuggingFace `DatasetDict` to `data/` with an 85/15 train-test split (50 train, 9 test)

### `QLoRA_Fine-Tuning.ipynb` — Fine-Tune the Model

**Load the base model**
```python
model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", revision="main")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
```
The GPTQ variant is already 4-bit quantized — no extra quantization step needed.

**Test the base model (before fine-tuning)**

A simple prompt like `[INST] Great content, thank you! [/INST]` gets a generic, verbose response. The model doesn't know about the AamirGPT persona yet.

**Prepare for training**
```python
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)  # enables training on quantized weights
model = get_peft_model(model, lora_config)       # wraps model with LoRA adapters
```

**Tokenize the dataset**
- Truncation from the left side (preserves the most recent context)
- Max sequence length: 512 tokens
- Pad token set to EOS token
- `DataCollatorForLanguageModeling` handles batching (causal LM, no masked LM)

**Training configuration**
```python
TrainingArguments(
    output_dir="model/",
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    gradient_accumulation_steps=4,   # effective batch size = 4 × 4 = 16
    warmup_steps=2,
    fp16=True,
    optim="paged_adamw_8bit",        # memory-efficient optimizer
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
```

**Run training**
```python
trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    args=training_args,
    data_collator=data_collator
)
trainer.train()
```

---

## Training Results

The model was trained for ~9.2 epochs (30 steps total). Loss dropped consistently across both train and eval sets:

| Epoch | Train Loss | Eval Loss |
|-------|-----------|-----------|
| 0.92  | 4.42      | 4.04      |
| 1.85  | 3.92      | 3.44      |
| 2.77  | 3.34      | 2.93      |
| 4.00  | 2.16      | 2.41      |
| 4.92  | 2.52      | 2.06      |
| 5.85  | 2.23      | 1.78      |
| 6.77  | 1.98      | 1.56      |
| 8.00  | 1.36      | 1.40      |
| 8.92  | 1.73      | 1.35      |
| 9.23  | 1.17      | **1.35** ← best checkpoint |

- Best checkpoint: `model/checkpoint-30`
- Best eval loss: `1.347`
- Total training time: ~70 minutes
- Training throughput: ~0.119 samples/second

---

## Inference with the Fine-Tuned Model

```python
instructions = """AamirGPT, functioning as a virtual data science consultant on YouTube,
communicates in clear, accessible language, escalating to technical depth upon request.
It reacts to feedback aptly and ends responses with its signature '–AamirGPT'.
AamirGPT will tailor the length of its responses to match the viewer's comment."""

prompt_template = lambda comment: f"[INST] {instructions} \n{comment} \n[/INST]"

comment = "What is fat-tailedness?"
prompt = prompt_template(comment)

model.eval()
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)
print(tokenizer.batch_decode(outputs)[0])
```

**Sample output:**
```
Fat-tailedness is a statistical property where the distribution of data has heavier tails
than a normal distribution. In simpler terms, extreme values occur more frequently than
what a normal distribution would predict.

For example, in finance, stock prices follow a fat-tailed distribution...

–AamirGPT
```

### Loading from a Saved Checkpoint

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
checkpoint_path = "model/checkpoint-30"

model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", revision="main")
model = PeftModel.from_pretrained(model, checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
```

---

## Key Observations

- **Small dataset effect** — with only 59 examples, the model learned the persona well for technical questions but showed repetitive output patterns on very short comments (a sign of overfitting on response structure)
- **Adapter-only checkpoints** — each checkpoint is only a few MB (just the LoRA weights), not the full 7B model
- **Memory efficiency** — 4-bit base model + 8-bit optimizer + fp16 training makes this runnable on a single consumer GPU
- **Only `q_proj` targeted** — expanding `target_modules` to include `v_proj`, `k_proj`, or `o_proj` would increase expressiveness at the cost of more trainable parameters

---

## Future Work

- Expand the dataset — more examples would reduce overfitting and improve generalization across comment types
- Target more attention modules (`v_proj`, `k_proj`) and experiment with higher rank values (`r=16`, `r=32`)
- Add ROUGE or BERTScore metrics for more meaningful evaluation beyond loss
- Merge the LoRA adapter into the base model weights for single-file deployment
