# train_llama8b_lora_multigpu.py
# ------------------------------------------------------
# https://www.acorn.io/resources/learning-center/fine-tuning-llama-2/
# https://www.philschmid.de/fine-tune-google-gemma#2-create-and-prepare-the-dataset
# https://www.philschmid.de/fine-tune-google-gemma#2-create-and-prepare-the-dataset
import os, torch, time, logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType


from accelerate import init_empty_weights

from accelerate.big_modeling import infer_auto_device_map, dispatch_model, init_empty_weights

# enable logging
logging.basicConfig(
    level=logging.INFO,                    
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

# Calculate the time it takes for each epoch
class EpochTimer(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.t0 = time.perf_counter()

    def on_epoch_end(self, args, state, control, **kwargs):
        dt = time.perf_counter() - self.t0
        logging.info(f"🕒 Epoch {int(state.epoch)} took {dt/60:.2f} min "
                     f"({dt:.1f} s)")


# ❷ Paths & LoRA hyper‑params
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR = "/scratch/gjf3sa/llama8b_lora_accerate_entireModel"

lora_cfg = LoraConfig(
    r             = 8,           # tiny adapters → small VRAM
    lora_alpha    = 16,          # ≈ 2 × r
    lora_dropout  = 0.05,
    task_type     = TaskType.CAUSAL_LM,
    target_modules= ["q_proj","k_proj","v_proj","o_proj",
                     "gate_proj","up_proj","down_proj"],
)


base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",          
)    

#Add LoRA (real FP16 adapters on CPU)
model = get_peft_model(base_model, lora_cfg)

model.gradient_checkpointing_enable()
model.enable_input_require_grads() 
model.config.use_cache = False

# Tokeniser & data
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
tok.pad_token = tok.eos_token

def to_prompt(ex):
    ctx, q  = ex["context"], ex["question"]
    ans_txt = ex["answers"]["text"][0] if ex["answers"]["text"] else "Unanswerable"
    prompt  = (
        "<|begin_of_text|><|system|>\n"
        "You are a precise question-answering assistant. Extract exact spans from the provided context to answer questions. Do not generate information beyond what is explicitly stated in the context. If the answer cannot be found in the context, respond with only Unanswerable."
        "<|user|>\n"
        f"Context: {ctx}\n\nQuestion: {q}\n"
        "<|assistant|>\n"
        f"{ans_txt}<|end_of_text|>"
    )
    return {"text": prompt}

ds = load_dataset("squad_v2")
train_ds = ds["train"].map(to_prompt)     
val_ds   = ds["validation"].select(range(50)).map(to_prompt)

def tok_fn(batch):
    return tok(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
val_tok   = val_ds.map(tok_fn,   batched=True, remove_columns=["text"])

collator = DataCollatorForLanguageModeling(tok, mlm=False)

# TrainingArguments & Trainer
args = TrainingArguments(
    output_dir             = OUTPUT_DIR,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 32,
    learning_rate          = 1e-4,
    num_train_epochs       = 1,
    logging_steps          = 10,
    save_strategy = "steps",    
    save_steps    = 4070,         
    save_total_limit = 3, 
    fp16                   = True,        # fp16 on V100s
    ddp_find_unused_parameters=False,
    report_to              = "none",
)

trainer = Trainer(          
    model = model,
    args  = args,
    train_dataset = train_tok,
    eval_dataset  = val_tok,
    data_collator = collator,
    callbacks=[EpochTimer()],
)


#Train!
trainer.train()

# Save 
trainer.model.save_pretrained(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
