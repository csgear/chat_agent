# Step 1 — LoRA Fine-tuning on Qwen3-1.7B

Fine-tune [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) with LoRA on an e-commerce customer-service dialogue dataset, then run inference with the fine-tuned adapter.

## Files

| File                  | Purpose                                                 |
| --------------------- | ------------------------------------------------------- |
| `train_lora_small.py` | **Recommended** — LoRA training optimised for 16 GB GPU |
| `train_lora.py`       | Original training script (24 GB GPU, larger batch)      |
| `demo.py`             | Sanity-check the base model before fine-tuning          |
| `customer_reply.py`   | Single-turn inference with the fine-tuned LoRA adapter  |
| `customers_reply.py`  | Multi-turn interactive chat with merged LoRA weights    |
| `get_dataset.py`      | Inspect tokenized dataset statistics                    |

## Requirements

```bash
pip install torch transformers datasets peft huggingface_hub safetensors tqdm
```

## Workflow

### 1. (Optional) Test the base model

```bash
cd step1
python demo.py
```

Runs a single inference with the unmodified `Qwen3-1.7B` to confirm the environment works.

### 2. Train with LoRA

```bash
cd step1
python train_lora_small.py
```

**Switching datasets** — edit the two lines at the top of `train_lora_small.py`:

```python
# Quick experiment: 10k samples from dev split
TRAIN_FILE = "../E-commerce dataset/dev.txt"

# Full training: 1 million samples
TRAIN_FILE = "../E-commerce dataset/train.txt"
```

`EVAL_FILE` always points to `dev.txt` so eval loss is comparable across runs.

**Key training settings (16 GB GPU):**

| Parameter                     | Value      | Reason                    |
| ----------------------------- | ---------- | ------------------------- |
| `per_device_train_batch_size` | 2          | Fits in VRAM              |
| `gradient_accumulation_steps` | 8          | Effective batch = 16      |
| `gradient_checkpointing`      | True       | Trades compute for memory |
| `bf16`                        | True       | Half-precision training   |
| `max_length`                  | 256        | Sequence length cap       |
| LoRA `r`                      | 8          | Adapter rank              |
| LoRA `target_modules`         | q/k/v proj | Attention layers only     |

The adapter is saved to `qwen_lora_finetuned/` on completion.

### 3. Compare dev.txt vs train.txt results

Eval loss is printed after every epoch. Run twice and compare:

```
# Run 1 — TRAIN_FILE = dev.txt
Epoch 1: eval_loss = X.XX
Epoch 2: eval_loss = X.XX
Epoch 3: eval_loss = X.XX

# Run 2 — TRAIN_FILE = train.txt
Epoch 1: eval_loss = X.XX   ← should be lower
```

Lower `eval_loss` on `dev.txt` indicates better generalisation from the larger training set.

### 4. Test the fine-tuned model

**Single-turn** (one question, one answer):

```bash
python customer_reply.py
```

Edit the `messages` list at the bottom to change the test prompt.

**Multi-turn interactive chat:**

```bash
python customers_reply.py
```

Type your message and press Enter. Type `退出` / `exit` / `quit` to stop.

## Data format

The E-commerce dataset uses tab-separated format:

```
label \t utt1 \t utt2 \t ... \t response
```

- `label=1` — positive (correct) response pair, used for training
- `label=0` — negative sample, skipped
- Utterances are space-tokenized; the loader strips spaces automatically

| Split | File                           | Samples    |
| ----- | ------------------------------ | ---------- |
| Train | `E-commerce dataset/train.txt` | ~1 000 000 |
| Dev   | `E-commerce dataset/dev.txt`   | 10 000     |
| Test  | `E-commerce dataset/test.txt`  | 10 000     |

## Output

After training, `qwen_lora_finetuned/` contains the LoRA adapter weights only (not the full base model). Both inference scripts load the base model separately and apply the adapter at runtime via `PeftModel.from_pretrained`.
