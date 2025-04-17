import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType


# Model configuration 
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
new_model = "squad2/llama-3-3b-check-10K"

# Load tokenizer from pretrained model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load model without quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)
# Enable gradient checkpointing
model.config.gradient_checkpointing = True

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    # bias="none",
    task_type=TaskType.CAUSAL_LM,
     target_modules=[
        "q_proj",
        "k_proj", 
        "v_proj", 
        "o_proj", 
        "gate_proj", 
        "up_proj", 
        "down_proj"
    ],
)

# Get PEFT model
model = get_peft_model(model, config)

# Helper function to print trainable parameter stats
def print_trainable_parameters(m):
    trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in m.parameters())
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}")

# Print the percentage of trainable parameters
print_trainable_parameters(model)

# Function to extract answer from model's response
def extract_answer(prompt):
    answer_marker = "ANSWER:\n"
    
    start_pos = prompt.find(answer_marker) + len(answer_marker)
    
    answer = prompt[start_pos:len(prompt)].strip()
    
    return answer

# Load the SQuAD v2 dataset
qa_dataset = load_dataset("squad_v2")

# Function to create prompts - adapted for Llama instruction format
def create_prompt(example):
    context = example['context']
    question = example['question']
    answer = example['answers']
    
    if len(answer["text"]) < 1:
        answer_text = "Unanswerable"
    else:
        answer_text = answer["text"][0]
    
    # Format adjusted for Llama-3.2-Instruct
    prompt = f"""<|begin_of_text|><|system|>
You are a precise question-answering assistant. Extract exact spans from the provided context to answer questions. 
Do not generate information beyond what is explicitly stated in the context.
If the answer cannot be found in the context, respond with only "Unanswerable".
<|user|>
Context: {context}

Question: {question}
<|assistant|>
{answer_text}<|end_of_text|>"""
    return {"text": prompt}

# Process the dataset - use a smaller subset to fit in memory
processed_train = qa_dataset["train"].select(range(10000)).map(create_prompt)  # Using only 500 samples
processed_val = qa_dataset["validation"].select(range(50)).map(create_prompt)  # Small validation set

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_train = processed_train.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_val = processed_val.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Enable gradient checkpointing to save memory
# model.gradient_checkpointing_enable()

# # Disable caching for more efficient training
# model.config.use_cache = False

# Training arguments
training_args = TrainingArguments(
    output_dir='/scratch/gjf3sa/llama3b_squad_peft_10K',
    per_device_train_batch_size=1,  # Very small batch size to fit in memory
    gradient_accumulation_steps=16,  # Accumulate gradients to compensate for small batch size
    learning_rate=1e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,  # Use bfloat16 mixed precision
    optim="adamw_torch",
    # Make sure we don't use DataParallel since we're only using one GPU
    ddp_find_unused_parameters=False,
    report_to="none"
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# # — With this —
# trainer.train(resume_from_checkpoint="/scratch/gjf3sa/squad_peft/checkpoint-6250")


# Save the fine-tuned model and tokenizer
trainer.model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)
