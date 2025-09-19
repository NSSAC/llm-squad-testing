from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import pandas as pd
import string
import re
import collections
import torch
from evaluate import load
import json
import Levenshtein as lev

squad_metric = load("squad_v2")

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# Meta-Llama-3-8B-Instruct
# Llama-3.2-3B-Instruct
# Llama-3.2-1B-Instruct


df = pd.read_csv("/home/gjf3sa/sneha/midas/qwen/html_dataset_final_with_answerStart.csv")

df["answers"] = df["answers"].apply(json.loads)

bnb_config4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4"  # "nf4" usually gives better results than "fp4" for LLMs
)

# Configure 8-bit quantization
bnb_config8 = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=False
)

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    quantization_config=bnb_config4,
    device_map="auto"
)

def clean_extracted(extracted: str) -> str:
    """
    Given the bit you pulled from your logs, this will try to grab whatever
    lives between the ChatML SYS-tags (<<SYS>> … </SYS>) and return it stripped.
    Falls back to stripping any remaining brackets or tags.
    """
    # 1) collapse all whitespace so tags are easier to match
    t = re.sub(r"\s+", " ", extracted)
    
    # 2) look for <<SYS>> ... </SYS>>
    m = re.search(r"<<SYS>>(.*?)</SYS>>", t)
    if m:
        return m.group(1).strip()
    
    # 3) fallback: look for [SYS] ... [/SYS]
    m = re.search(r"\[SYS\](.*?)\[/SYS\]", t)
    if m:
        return m.group(1).strip()
    
    # 4) nothing matched? strip out any <...> or [...] tags and return the rest
    return re.sub(r"<[^>]+>|\[[^\]]+\]", "", t).strip()

# prepare the model input
# system_message = "Give me a short introduction to large language model."
system_message=f"""You are a precise and reliable assistant trained to answer questions in a single word or single phrase strictly based on 
the given background. Use the provided background to answer questions. When answering, you must **locate** the answer span in the provided context
and **copy it exactly**, preserving every character, space, and punctuation mark."""


# def build_prompt(system_message, context, question):
#     return f"""
# {system_message}
# Background: {context}
# Question: {question}
# Answer:"""


# If the answer is explicitly mentioned in the context, provide the **exact phrasing from the context**. 
def format_prompt(system_message, context, question):
    prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_message}
<|end_header_id|><|start_header_id|>user<|end_header_id|>

Background: '''{context}'''
The following question is about what is stated in the given context. Do not rely on external knowledge.
According to the passage, {question}
Answer (only a single phrase, no explanations, no hashtags, no extra text):
<|end_header_id|>assistant<|end_header_id|>
"""

    return prompt + tokenizer.eos_token
def clean_answer(response_text):
    # Strip leading/trailing whitespace and split into lines
    lines = response_text.strip().split("\n")

    # Remove empty lines and lines that just say "assistant"
    cleaned_lines = [line.strip() for line in lines if line.strip().lower() != "assistant" and line.strip() != ""]

    # Return the first meaningful line as the answer
    return cleaned_lines[0] if cleaned_lines else "Unanswerable"




predictions=[]
references=[]
f1_scores = []
exact_match=[]
lev_ratios=[]
num_examples=len(df)
total_time=0
for i in range(0,num_examples):
    context=df.iloc[i]["context"]
    question=df.iloc[i]["question"]
    qid=str(df.iloc[i]["id"])
    gold_answers = df.iloc[i]["answers"]
    prompt=format_prompt(system_message,context,question)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
   
    
    start_time = time.perf_counter()
    # conduct text completion
    generated_text = model.generate(
        input_ids,
        # stopping_criteria=stopping_criteria,  # Use this instead of `stop`
        max_new_tokens=100,  # Adjust based on desired response length
        # do_sample=True,  # Enable sampling for variability
        # early_stopping=True,
        # top_p=0.80,  # Nucleus sampling for more controlled diversity
        # temperature=0.7,  # Lower temperature for less randomness
        # num_return_sequences=1,  # Ensure only one response is generated
        # pad_token_id=tokenizer.eos_token_id,  # Ensures consistent behavior
        # eos_token_id=tokenizer.eos_token_id  # Stop generation at the end of an answer
    )


    end_time = time.perf_counter()
    response = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    # print("response",response)
    ans= response.split("Answer (only a single phrase, no explanations, no hashtags, no extra text):")[-1].strip()
    print("predicted",clean_answer(ans))
    content=clean_answer(ans)

    # print("extracted",clean_extracted_answer(response.split("Answer:")[-1].strip()))
    # ans= response.split("Answer (only a single phrase, no explanations, no hashtags, no extra text):")[-1].strip()

    # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    
    # print("thinking content:", thinking_content)

    print("Gold Answer", df.iloc[i]["answers"]["text"])
    inference_time=end_time-start_time
    print("Time per inference",end_time-start_time)
    total_time += inference_time

   

    pred={
        'id': qid,
        'prediction_text': content,
        'no_answer_probability': 1.0 if content=="Unanswerable" else 0.0
    }
    ref={
      'id':qid,
      'answers':gold_answers
    }
    gold_answers_texts = gold_answers["text"]
    row_result =  squad_metric.compute(predictions=[pred], references=[ref], no_answer_threshold=0.5)
    f1_scores.append(row_result["f1"])
    exact_match.append(row_result["exact"])
    predictions.append(pred)
    references.append(ref)
    ratios = [lev.ratio(content, gold) for gold in gold_answers_texts]
    max_ratio = max(ratios) if ratios else 0.0
    lev_ratios.append(max_ratio)




results = squad_metric.compute(predictions=predictions, references=references, no_answer_threshold=0.5)
print(results)


print("total time",total_time)
average_inference_time = total_time / num_examples
print(f"Average inference time per prediction: {average_inference_time:.4f} seconds")


# 1) build a simple mapping from id → prediction_text
pred_map = { p['id']: p['prediction_text'] for p in predictions }

# 2) make sure your DataFrame’s id column is the same type (string)
df['id'] = df['id'].astype(str)

# 3) map each id to its prediction
df['predicted_answer'] = df['id'].map(pred_map)
df["exact_match"]=exact_match
df["f1"] = f1_scores
df["lev_ratios"]=lev_ratios

# 4) write out a new CSV with your old columns + the new one
df.to_csv("/home/gjf3sa/sneha/midas/qwen/epi_html/llama_logs/html_epi_final_llama8B_quant4.csv", index=False)

