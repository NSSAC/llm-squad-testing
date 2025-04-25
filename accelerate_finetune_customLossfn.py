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

            
class LoRATrainer(Trainer):
    # Look at lines 1064- 1069
    # https://github.com/huggingface/transformers/blob/7293fdc5b9cc809c2aa2ceb84f903ad47e5c06f0/src/transformers/models/llama/modeling_llama.py#L1061
    
    # Also used chatgpt prompting to obtain a solution to override the RuntimeError: Expected all tensors to be on the same device, but found at least two devices,


    # Detailed explaination of below code

    
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        **kwargs              # <‑‑ catches num_items_in_batch or anything new
    ):
        import torch.nn as nn

        labels = inputs.pop("labels") #Labels are removed  the model doesn’t compute its built-in loss.
        outputs = model(**inputs)      # Run a forward pass to get logits         # logits on the layer’s GPU
        logits  = outputs.logits

        # move labels to same GPU as logits
        labels = labels.to(logits.device) # Move labels to the same device as logits

        # causal‑LM loss 

        # Shifting for next-token prediction 
        shift_logits = logits[..., :-1, :].contiguous() # drop the last time step’s scores since we cant predict the next token for the last word
        shift_labels = labels[..., 1:].contiguous() #   drop the first token’s real label since we dont have any token to predict the first token

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss


# (optional) bitsandbytes 4‑bit to save VRAM
#   pip install bitsandbytes
USE_4BIT = False                 # flip to True if you want 4‑bit base weights

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

# ❸ Build an **empty** model just to compute the device‑map
# https://medium.com/%40syedhamzatahir1001/how-hugging-faces-accelerate-helps-us-handle-huge-models-97ae9fe32fa6
# https://preemo.medium.com/squeeze-more-out-of-your-gpu-for-llm-inference-a-tutorial-on-accelerate-deepspeed-610fce3025fd

with init_empty_weights():
    empty_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,      
    )
    empty_full = get_peft_model(empty_base, lora_cfg)  # adapters (still empty!) 

# 4 × 32 GB V100 – leave a couple GB per card for gradients/activations
max_mem = {i: "14GiB" for i in range(4)}   # 14 GiB is safe on 32 GiB cards
max_mem["cpu"] = "32GiB"                   # optional off‑load overflow


device_map = infer_auto_device_map(
    empty_full,
    max_memory=max_mem,
    no_split_module_classes=["LlamaDecoderLayer"],  # keep full decoder blocks together
)
print("Device map:")
for k,v in device_map.items():
    print(f"  {k:10s} → {v}")

# ❹ Reload **real** weights
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map={"": "cpu"},          # keep on CPU for now
    load_in_4bit=USE_4BIT,           # bitsandbytes, optional
)    

#Add LoRA (real FP16 adapters on CPU)
model = get_peft_model(base_model, lora_cfg)

# Dispatch … every param now lands on its GPU from the map
model = dispatch_model(model, device_map)

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

trainer = LoRATrainer(          
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
