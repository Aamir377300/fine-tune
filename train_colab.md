# Answer Reviewer — Colab Training & HuggingFace Deployment Guide

This guide walks you through the full workflow:
1. Train the model on Google Colab (free T4 GPU)
2. Deploy the fine-tuned adapter to HuggingFace Hub
3. Use the model via HuggingFace Inference API locally

---

## What This Model Does

Given a question, a reference answer, and a student's answer — the model outputs:
- **Grade:** `correct`, `partially_correct`, or `incorrect`
- **Feedback:** a brief explanation of why

**Dataset used:** [Meyerger/ASAG2024](https://huggingface.co/datasets/Meyerger/ASAG2024) — loaded directly from HuggingFace.

---

## Before You Start

You need three accounts — all free:
- **Google Account** — for Colab + Google Drive
- **HuggingFace Account** — sign up at [huggingface.co](https://huggingface.co)
- **GitHub Account** — for repo sync (you already have this)

### Get your HuggingFace Write Token
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token**
3. Name it anything (e.g. `colab-training`)
4. Set permission to **Write**
5. Copy the token — it looks like `hf_xxxxxxxxxxxxxxxxxxxxxxxx`

### Get your GitHub Personal Access Token (PAT)
1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. Click **Generate new token (classic)**
3. Give it a name (e.g. `colab-sync`)
4. Check the **repo** scope
5. Click **Generate token** and copy it — looks like `ghp_xxxxxxxxxxxxxxxx`

### Add GitHub Token to Colab Secrets
This is how Colab securely reads your token without it being visible in the notebook:
1. Open your notebook in Colab
2. Click the **🔑 key icon** in the left sidebar (or go to `Tools → Secrets`)
3. Click **+ Add new secret**
4. Name: `GITHUB_TOKEN`
5. Value: paste your GitHub PAT
6. Toggle **Notebook access** to ON

---

## Part 1 — Upload the Notebook to Google Colab

Only **one notebook** is needed: `QLoRA_Fine-Tuning.ipynb`

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload `QLoRA_Fine-Tuning.ipynb` from this repo

> `create-dataset.ipynb` is no longer needed — the dataset loads directly from HuggingFace and the repo syncs automatically via GitHub.

---

## Part 2 — Set Runtime to GPU

In Colab: **Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save**

Without this, training will fail or take days.

---

## Part 3 — Prevent Colab from Disconnecting

Training takes **2–3 hours** on a T4. Colab disconnects after ~90 minutes of inactivity.

**Fix — Browser Console Trick:**
1. Open browser developer console: `F12` (Windows) or `Cmd+Option+J` (Mac)
2. Go to the **Console** tab
3. Paste this and press Enter:

```javascript
function ClickConnect(){
  console.log('Keeping Colab alive...');
  document.querySelector('#top-toolbar > colab-connect-button')
    .shadowRoot.querySelector('#connect').click();
}
setInterval(ClickConnect, 60000);
```

> Even if it disconnects — checkpoints are saved to Google Drive after every epoch. You can resume with one line (see Part 6).

---

## Part 4 — Fill in Your Credentials

Before running, find **Step 13** in the notebook and fill in:

```python
HF_TOKEN    = "hf_xxxxxxxxxxxxxxxxxxxx"   # your HF write token
HF_USERNAME = "your-hf-username"          # your HuggingFace username
```

The GitHub token is read automatically from Colab Secrets — no need to paste it anywhere in the notebook.

---

## Part 5 — Run the Notebook

Click **Runtime → Run all** (`Ctrl+F9`)

When prompted to mount Google Drive, click **Connect to Google Drive** and allow access.

### What happens at each step:

| Step | What happens | Time |
|------|-------------|------|
| Step 0 | Mounts Google Drive | ~1 min |
| Step 1 | Clones repo (first run) or pulls latest (subsequent runs) | ~30s |
| Step 2 | Installs all packages | ~2 min |
| Step 5 | Downloads ASAG2024 dataset from HuggingFace | ~1 min |
| Step 6 | Formats 5000 training examples into prompts | ~1 min |
| Step 7 | Downloads base model (~4GB from HuggingFace) | ~5 min |
| Step 8 | Tests base model before training | ~1 min |
| Step 9–10 | Applies LoRA, tokenizes dataset | ~2 min |
| Step 11 | **Training — 3 epochs** | ~2–3 hours |
| Step 12 | Tests fine-tuned model on 3 examples | ~2 min |
| Step 13 | Pushes adapter to HuggingFace Hub | ~2 min |

### Training logs you'll see:
```
{'loss': 1.85, 'epoch': 1.0}
{'eval_loss': 1.62, 'epoch': 1.0}
{'loss': 1.43, 'epoch': 2.0}
{'eval_loss': 1.31, 'epoch': 2.0}
{'loss': 1.21, 'epoch': 3.0}
{'eval_loss': 1.18, 'epoch': 3.0}
Training complete!
```

When it finishes:
```
Model pushed to: https://huggingface.co/your-username/answer-reviewer
```

---

## Part 6 — If Colab Disconnects Mid-Training

Don't panic. Checkpoints are saved to `/content/drive/MyDrive/AnswerReviewer/model/` after every epoch.

To resume:
1. Reopen the notebook in Colab
2. Run Steps 0–9 again (fast, no training)
3. In Step 10, uncomment the resume cell at the bottom:

```python
trainer.train(resume_from_checkpoint=True)
```

---

## Part 7 — Verify Your Model on HuggingFace

After training completes, go to:
```
https://huggingface.co/your-username/answer-reviewer
```

You should see:
- `adapter_config.json` — LoRA configuration
- `adapter_model.safetensors` — trained adapter weights (a few MB)
- `README.md` — auto-generated model card

---

## Part 8 — Use the Model Locally via API

Once deployed, call it from your local machine or React frontend.

### Python example

```python
import requests

HF_TOKEN    = "hf_xxxxxxxxxxxxxxxxxxxx"   # read token is enough for inference
HF_USERNAME = "your-hf-username"

API_URL = f"https://api-inference.huggingface.co/models/{HF_USERNAME}/answer-reviewer"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

SYSTEM_PROMPT = """You are an expert answer reviewer. You are given a question, a reference answer, and a student's answer.
Your task is to evaluate the student's answer by comparing it to the reference answer.
Respond with:
- Grade: one of [correct, partially_correct, incorrect]
- Feedback: a brief explanation of why the student's answer is correct, partially correct, or incorrect."""

def review_answer(question, reference_answer, student_answer):
    prompt = f"""[INST] {SYSTEM_PROMPT}

Question: {question}
Reference Answer: {reference_answer}
Student Answer: {student_answer}
[/INST]"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.1,
            "return_full_text": False
        }
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)

    if response.status_code == 200:
        return response.json()[0]["generated_text"].strip()
    elif response.status_code == 503:
        return "Model is loading, please wait 30 seconds and try again."
    else:
        return f"Error {response.status_code}: {response.text}"

# Test it
result = review_answer(
    question="What is photosynthesis?",
    reference_answer="Photosynthesis is the process by which plants use sunlight, water, and CO2 to produce glucose and oxygen.",
    student_answer="Photosynthesis is when plants make food from sunlight."
)
print(result)
```

### Handle cold start (model takes time to load on first request)

```python
import time

def review_with_retry(question, reference_answer, student_answer, retries=3):
    for attempt in range(retries):
        result = review_answer(question, reference_answer, student_answer)
        if "loading" in result.lower():
            print(f"Model loading... retrying in 30s (attempt {attempt+1}/{retries})")
            time.sleep(30)
        else:
            return result
    return "Model unavailable, please try again later."
```

---

## Quick Reference

| What | Where |
|------|-------|
| HuggingFace tokens | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| Your model after training | `https://huggingface.co/your-username/answer-reviewer` |
| Inference API endpoint | `https://api-inference.huggingface.co/models/your-username/answer-reviewer` |
| Checkpoints during training | `/content/drive/MyDrive/AnswerReviewer/model/` |
| Dataset source | [huggingface.co/datasets/Meyerger/ASAG2024](https://huggingface.co/datasets/Meyerger/ASAG2024) |

---

## Troubleshooting

**"CUDA out of memory"**
Reduce batch size in Step 10:
```python
batch_size = 1  # change from 2 to 1
```

**"Model too large" on Inference API**
The free serverless API sometimes rejects large models. Retry after 30 seconds — the model needs to load into memory first.

**Colab disconnected mid-training**
See Part 6 above — resume from checkpoint.

**"Repository not found" when pushing**
Make sure your token has **Write** permission, not just Read.

**Want to train on more data?**
In Step 5, increase the sample size:
```python
TRAIN_SAMPLES = 10000  # or 20000 for better accuracy (longer training)
EVAL_SAMPLES  = 1000
```
