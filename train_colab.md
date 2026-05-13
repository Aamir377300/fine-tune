# AamirGPT — Colab Training & HuggingFace Deployment Guide

This guide walks you through the full workflow:
1. Train the model on Google Colab (free T4 GPU)
2. Deploy the fine-tuned adapter to HuggingFace Hub
3. Use the model via HuggingFace Inference API locally

---

## Before You Start

You need two accounts — both free:
- **Google Account** — for Colab + Google Drive
- **HuggingFace Account** — go to [huggingface.co](https://huggingface.co) and sign up

### Get your HuggingFace Token
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token**
3. Name it anything (e.g. `colab-training`)
4. Set permission to **Write**
5. Copy the token — it looks like `hf_xxxxxxxxxxxxxxxxxxxxxxxx`
6. Keep it safe, you'll paste it into the notebooks

---

## Part 1 — Upload Files to Google Colab

### Step 1 — Open Google Colab
Go to [colab.research.google.com](https://colab.research.google.com)

### Step 2 — Upload the notebooks
You need to upload **both notebooks** from this repo:
- `create-dataset.ipynb`
- `QLoRA_Fine-Tuning.ipynb`

In Colab: click **File → Upload notebook** and upload them one at a time.

### Step 3 — Upload the data file
The CSV file needs to be accessible inside Colab. Two options:

**Option A — Upload directly to Colab session (simpler)**
In the left sidebar click the folder icon → click the upload button → upload `data/YT-comments.csv`
Then create the folder structure: right-click → New folder → name it `data`, then move the CSV inside it.

**Option B — Upload to Google Drive (recommended)**
1. Go to [drive.google.com](https://drive.google.com)
2. Create a folder called `AamirGPT`
3. Inside it, create a folder called `data`
4. Upload `YT-comments.csv` into that `data` folder
5. The path will be `/content/drive/MyDrive/AamirGPT/data/YT-comments.csv`
6. Update the CSV path in `create-dataset.ipynb` accordingly

### Step 4 — Set the Runtime to GPU
In Colab: **Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save**

> Without this, training will be extremely slow or fail entirely.

---

## Part 2 — Prevent Colab from Disconnecting

Colab disconnects after ~90 minutes of inactivity. Training takes ~70 minutes, so you need to keep it alive.

**Method — Browser Console Trick**
1. Open your browser's developer console: `F12` (Windows) or `Cmd+Option+J` (Mac)
2. Paste this and press Enter:

```javascript
function ClickConnect(){
  console.log('Keeping Colab alive...');
  document.querySelector('#top-toolbar > colab-connect-button')
    .shadowRoot.querySelector('#connect').click();
}
setInterval(ClickConnect, 60000);
```

This clicks the connect button every 60 seconds, preventing idle timeout.

> Even if it does disconnect — checkpoints are saved to Google Drive after every epoch, so you won't lose your training progress.

---

## Part 3 — Run `create-dataset.ipynb`

This notebook processes the raw CSV into a HuggingFace dataset and saves it to Drive.

### What to fill in before running:
In the **"(Optional) Push Dataset to HuggingFace Hub"** section:
```python
HF_TOKEN    = "hf_xxxxxxxxxxxxxxxxxxxx"   # your HF write token
HF_USERNAME = "your-hf-username"          # your HuggingFace username
```

### How to run:
Click **Runtime → Run all** (or press `Ctrl+F9`)

When it asks to mount Google Drive, click **Connect to Google Drive** and allow access.

After it finishes, you'll see the dataset saved at `/content/drive/MyDrive/AamirGPT/data/`

---

## Part 4 — Run `QLoRA_Fine-Tuning.ipynb`

This is the main notebook — loads the model, trains it, and pushes to HuggingFace.

### What to fill in before running:

**Step 10 — Push to Hub cell:**
```python
HF_TOKEN    = "hf_xxxxxxxxxxxxxxxxxxxx"   # your HF write token
HF_USERNAME = "your-hf-username"          # your HuggingFace username
```

### How to run:
Click **Runtime → Run all** (or press `Ctrl+F9`)

When it asks to mount Google Drive, click **Connect to Google Drive** and allow access.

### What happens during training:
| Time | What's happening |
|------|-----------------|
| 0–5 min | Installing packages, mounting Drive |
| 5–10 min | Downloading base model (~4GB from HuggingFace) |
| 10–80 min | Training — 10 epochs, loss drops from ~4.4 → ~1.35 |
| 80–85 min | Pushing adapter weights to HuggingFace Hub |
| 85–90 min | Running inference test on the fine-tuned model |

You'll see training logs like this as it runs:
```
{'loss': 4.4215, 'epoch': 0.92}
{'eval_loss': 4.04, 'epoch': 0.92}
{'loss': 3.34, 'epoch': 2.77}
...
{'loss': 1.17, 'epoch': 9.23}
```

When it finishes you'll see:
```
Model pushed to: https://huggingface.co/your-username/AamirGPT
```

---

## Part 5 — Verify Your Model on HuggingFace

1. Go to [huggingface.co/your-username/AamirGPT](https://huggingface.co)
2. You should see your model repository with:
   - `adapter_config.json` — LoRA configuration
   - `adapter_model.safetensors` — the trained adapter weights
   - `README.md` — auto-generated model card

---

## Part 6 — Use the Model Locally via API

Once your model is on HuggingFace Hub, you can call it from your local machine using the Inference API.

### Get your HuggingFace API key
Same token you used for training — the `hf_xxxx` token with **read** permission is enough for inference.

### Python example — call the model locally

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/your-username/AamirGPT"
HEADERS = {"Authorization": "Bearer hf_xxxxxxxxxxxxxxxxxxxx"}  # your HF token

def ask_aamir_gpt(comment):
    instructions = """AamirGPT, functioning as a virtual data science consultant on YouTube,
communicates in clear, accessible language, escalating to technical depth upon request.
It reacts to feedback aptly and ends responses with its signature '–AamirGPT'.
AamirGPT will tailor the length of its responses to match the viewer's comment."""

    prompt = f"[INST] {instructions}\n\nPlease respond to the following comment.\n\n{comment}\n[/INST]"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 280,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()[0]["generated_text"]

# Test it
print(ask_aamir_gpt("What is fat-tailedness?"))
print(ask_aamir_gpt("Great video, learned a lot!"))
```

### Install requests if needed:
```bash
pip install requests
```

### Important notes about the free Inference API:
- **Cold start** — if the model hasn't been used recently, the first request takes 20–60 seconds to load. Subsequent requests are fast.
- **Rate limits** — the free tier allows a reasonable number of requests per day, enough for testing and personal use
- **Model size** — Mistral-7B may not always be available on the free serverless tier. If you get a 503 error, the model is loading — just retry after 30 seconds

### Handle the cold start in code:
```python
import time

def ask_with_retry(comment, retries=3):
    for attempt in range(retries):
        response = requests.post(API_URL, headers=HEADERS, json={
            "inputs": comment,
            "parameters": {"max_new_tokens": 280, "return_full_text": False}
        })
        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        elif response.status_code == 503:
            print(f"Model loading... waiting 30s (attempt {attempt+1}/{retries})")
            time.sleep(30)
        else:
            print(f"Error {response.status_code}: {response.text}")
            break
    return None
```

---

## Quick Reference

| What | Where |
|------|-------|
| HuggingFace tokens | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| Your model after training | `https://huggingface.co/your-username/AamirGPT` |
| Inference API endpoint | `https://api-inference.huggingface.co/models/your-username/AamirGPT` |
| Checkpoints during training | `/content/drive/MyDrive/AamirGPT/model/` |
| Dataset on Drive | `/content/drive/MyDrive/AamirGPT/data/` |

---

## Troubleshooting

**"CUDA out of memory" during training**
Reduce batch size in `QLoRA_Fine-Tuning.ipynb`:
```python
batch_size = 2  # change from 4 to 2
```

**"Model too large" on Inference API**
The free serverless API sometimes rejects large models. Use the retry logic above, or test inference directly in Colab after training.

**Colab disconnected mid-training**
Don't panic — checkpoints are saved to Drive every epoch. Reopen the notebook, re-run from Step 9 (Fine-tune), and set `resume_from_checkpoint=True`:
```python
trainer.train(resume_from_checkpoint=True)
```

**"Repository not found" when pushing to Hub**
Make sure you're logged in with a **write** token, not a read token.
