from transformers import  Gemma3ForCausalLM, AutoTokenizer, BitsAndBytesConfig
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

model_id = "google/gemma-3-1b-it"

df = pd.read_csv("/home/gjf3sa/sneha/midas/qwen/pdf_dataset_final_with_answerStart.csv")

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
model = Gemma3ForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16,device_map="auto",
    quantization_config=bnb_config8
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)



# prepare the model input
# system_message = "Give me a short introduction to large language model."
system_message=f"""You are a precise and reliable assistant trained to answer questions in a single word or single phrase strictly based on 
the given background. Use the provided background to answer questions. When answering, you must **locate** the answer span in the provided context
and **copy it exactly**, preserving every character, space, and punctuation mark."""

def format_prompt(context, question):
  prompt = """Background: {} Question: {} Answer:
  """.format(context, question)
  return prompt

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
    prompt=(format_prompt(context,question))
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
        },
        {
            "role": "user",
            "content": [
                # {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
                {"type": "text", "text": prompt}
            ]
        }
    ]
   
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]


    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True,
    #     enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    # )
    # model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    start_time = time.perf_counter()

    with torch.inference_mode():
      outputs = model.generate(**inputs, max_new_tokens=2000)


    end_time = time.perf_counter()
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    raw = decoded[0]

# simple split-based cleanup:
    if "Answer:" in raw:
      answer_section = raw.rsplit("Answer:", 1)[-1]
    else:
      answer_section = raw
    
    lines = [l for l in answer_section.splitlines() if l.strip().lower() not in {"model", "assistant", "user"} and l.strip()]

    # 3) Take the last real line
    answer = lines[-1].strip()
    content=answer
   
    # print("thinking content:", thinking_content)
    print("content:", content)
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

    # if (i + 1) % 1000 == 0 or (i + 1) == len(df):
    #     print(f"\nComputing scores after {i+1} samples...\n")
    #     results = squad_metric.compute(predictions=predictions, references=references, no_answer_threshold=0.5)
    #     print(results)


results = squad_metric.compute(predictions=predictions, references=references, no_answer_threshold=0.5)
print(results)


print("total time",total_time)
average_inference_time = total_time / num_examples
print(f"Average inference time per prediction: {average_inference_time:.4f} seconds")
# df['predicted_answer'] = df['id'].map(predictions)

# df.to_csv("/home/gjf3sa/sneha/midas/qwen/epi_with_answers_predictions.csv", index=False)
# print(predictions)
# print(len(predictions))

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
df.to_csv("/home/gjf3sa/sneha/midas/qwen/epi_pdf/gemma_logs/pdf_epi_final_gemma1B_quant8.csv", index=False)

print("Wrote new file")

