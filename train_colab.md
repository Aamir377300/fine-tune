# Answer Reviewer — Colab Training & HuggingFace Deployment Guide

This guide walks you through the full workflow:
1. Train the model on Google's T4 GPU via Colab
2. Deploy the fine-tuned adapter to HuggingFace Hub
3. Use the model via HuggingFace Inference API locally

---

## What This Model Does

Given a question, a reference answer, and a student's answer — the model outputs:
- **Grade:** `correct`, `partially_correct`, or `incorrect`
- **Feedback:** a brief explanation of why

**Dataset:** [Meyerger/ASAG2024](https://huggingface.co/datasets/Meyerger/ASAG2024) — loaded directly from HuggingFace, no CSV needed.

---

## Before You Start

You need two accounts — both free:
- **Google Account** — for Colab GPU access + Google Drive
- **HuggingFace Account** — sign up at [huggingface.co](https://huggingface.co)

### Get your HuggingFace Write Token
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token** → name it anything → set permission to **Write**
3. Copy the token — looks like `hf_xxxxxxxxxxxxxxxxxxxxxxxx`
4. Paste it into **Step 10** of the notebook before running

---

## How to Run (VS Code — Recommended)

Running from VS Code is the best experience — you can switch tabs, close VS Code, or even shut your Mac screen and training keeps going on Google's servers.

### One-time setup
1. Open VS Code
2. Go to **Extensions** → search **Google Colab** → install it
3. Open `QLoRA_Fine-Tuning.ipynb` in VS Code

### Every run
1. Click **Select Kernel** (top right of the notebook)
2. Click **Colab** → sign in with your Google account
3. Select **T4 GPU** runtime
4. Fill in your HF token and username in **Step 10**
5. Click **Run All** (`Ctrl+Shift+P` → "Run All Cells" or the ▶▶ button)
6. Switch tabs, do other work — training runs on Google's GPU, not your machine

---

## How to Run (Browser Colab)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook** → upload `QLoRA_Fine-Tuning.ipynb`
3. Set runtime: **Runtime → Change runtime type → T4 GPU → Save**
4. Fill in your HF token and username in **Step 10**
5. Click **Runtime → Run all**

### Prevent disconnection when switching tabs
Training takes 2–3 hours. If you want to switch browser tabs, paste this in your browser console (`F12` → Console tab) **before** running:

```javascript
function ClickConnect(){
  console.log('Keeping Colab alive...');
  document.querySelector('#top-toolbar > colab-connect-button')
    .shadowRoot.querySelector('#connect').click();
}
setInterval(ClickConnect, 60000);
```

> The notebook also prints this snippet at the end of Step 0 as a reminder.

---

## What Happens at Each Step

| Step | What happens | Time |
|------|-------------|------|
| Step 0 | Mounts Google Drive, installs packages, prints keep-alive snippet | ~3 min |
| Step 1 | Imports | instant |
| Step 2 | Downloads ASAG2024 dataset from HuggingFace | ~1 min |
| Step 3 | Formats 5000 training examples into prompts | ~1 min |
| Step 4 | Downloads base model (~4GB from HuggingFace) | ~5 min |
| Step 5 | Tests base model before training | ~1 min |
| Step 6–7 | Applies LoRA, tokenizes dataset | ~2 min |
| Step 8 | **Training — 3 epochs** | ~2–3 hours |
| Step 9 | Tests fine-tuned model on 3 examples | ~2 min |
| Step 10 | Pushes adapter to HuggingFace Hub | ~2 min |

### Training logs you'll see:
```
{'loss': 1.85, 'epoch': 1.0}
{'eval_loss': 1.62, 'epoch': 1.0}
{'loss': 1.43, 'epoch': 2.0}
{'eval_loss': 1.31, 'epoch': 2.0}
{'loss': 1.21, 'epoch': 3.0}
{'eval_loss': 1.18, 'epoch': 3.0}
✓ Training complete!
```

When it finishes:
```
✓ Model pushed to: https://huggingface.co/your-username/answer-reviewer
```

---

## If the Session Disconnects Mid-Training

Checkpoints are saved to `/content/drive/MyDrive/AnswerReviewer/model/` after every epoch.

To resume:
1. Reopen the notebook and reconnect to Colab T4
2. Run Steps 0–7 again (fast, no training)
3. Uncomment and run the **Resume Training** cell at the bottom:

```python
trainer.train(resume_from_checkpoint=True)
```

---

## Verify Your Model on HuggingFace

After training completes, go to:
```
https://huggingface.co/your-username/answer-reviewer
```

You should see:
- `adapter_config.json` — LoRA configuration
- `adapter_model.safetensors` — trained adapter weights (a few MB)
- `README.md` — auto-generated model card

---

## Use the Model Locally via API

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
```

### Handle cold start

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
Reduce batch size in Step 8:
```python
batch_size = 1  # change from 2 to 1
```

**"Model too large" on Inference API**
Retry after 30 seconds — the model needs to cold-start into memory.

**Session disconnected mid-training**
See "If the Session Disconnects" section above — resume from checkpoint.

**"Repository not found" when pushing to HuggingFace**
Make sure your HF token has **Write** permission, not just Read.

**Want to train on more data?**
In Step 3, increase the sample size:
```python
TRAIN_SAMPLES = 10000  # or 20000 for better accuracy (longer training)
EVAL_SAMPLES  = 1000
```
