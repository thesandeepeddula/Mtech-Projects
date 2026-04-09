import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from utils.io import read_jsonl

BASE = "distilgpt2"
lr = 2e-4; bs = 8; epochs = 5

train = read_jsonl("data/qa/train.jsonl")
valid = read_jsonl("data/qa/valid.jsonl")

def to_instruct(example):
    return {
        "text": (
            "Instruction: Answer the financial question.\n"
            f"Question: {example['question']}\n"
            f"Answer: {example['answer']}\n"
        )
    }

train_txt = list(map(to_instruct, train))
valid_txt = list(map(to_instruct, valid))

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

train_ds = Dataset.from_list(train_txt)
valid_ds = Dataset.from_list(valid_txt)

def tok_map(batch):
    out = tok(batch["text"], truncation=True, padding="max_length", max_length=256)
    out["labels"] = out["input_ids"].copy()
    return out

train_tok = train_ds.map(tok_map, batched=True, remove_columns=["text"])
valid_tok = valid_ds.map(tok_map, batched=True, remove_columns=["text"])

base = AutoModelForCausalLM.from_pretrained(BASE)

peft_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["c_attn", "c_proj"]
)
model = get_peft_model(base, peft_cfg)

collator = DataCollatorForLanguageModeling(tok, mlm=False)

args = TrainingArguments(
    output_dir="models/ft_gen",
    learning_rate=lr,
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs,
    num_train_epochs=epochs,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=valid_tok,
    data_collator=collator,
)

trainer.train()
model.save_pretrained("models/ft_gen")
tok.save_pretrained("models/ft_gen")

with open("models/ft_gen/hparams.json", "w", encoding="utf-8") as f:
    json.dump({
        "lr": lr,
        "batch_size": bs,
        "epochs": epochs,
        "base": BASE,
        "technique": "Adapter‑based (LoRA)"
    }, f, indent=2)