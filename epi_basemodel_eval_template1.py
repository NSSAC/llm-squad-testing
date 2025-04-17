import time
import torch
import pandas as pd
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
import string
import re
import collections
import json

splits = {'train': 'squad_v2/train-00000-of-00001.parquet', 'validation': 'squad_v2/validation-00000-of-00001.parquet'}
# df = pd.read_parquet("hf://datasets/rajpurkar/squad_v2/" + splits["validation"])
# print(df.head())
df = pd.read_csv("/home/gjf3sa/sneha/midas/smol-course-llm-finetuning/1_instruction_tuning/notebooks/epi_data/epi_dataset.csv")

df["answer"] = df["answer"].apply(json.loads)

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# meta-llama/Llama-3.1-8B-Instruct
# meta-llama/Meta-Llama-3-8B-Instruct

# Configure 8-bit quantization
bnb_config8 = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config8, device_map="auto")
# If the answer is explicitly mentioned in the context, provide the **exact phrasing from the context**. 
def format_prompt(system_message, context, question, example):
    prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_message}
<|end_header_id|><|start_header_id|>user<|end_header_id|>
Context: '''{example['context']}'''
The following question is about what is stated in the given context. Do not rely on external knowledge.
According to the passage, {example['question']}
If the answer is **not present in the context**, respond with exactly: "Unanswerable".
Answer (only a single phrase, no explanations, no hashtags, no extra text): {example['answer']}

Context: '''Albert Einstein developed the theory of relativity.'''
The following question is about what is stated in the given context. Do not rely on external knowledge.
According to the passage, "Who developed the theory of relativity?"

If the answer is **not present in the context**, respond with exactly: "Unanswerable".
Answer (only a single phrase, no explanations, no hashtags, no extra text): "Albert Einstein"

Context: '''No information is given about the president.'''
The following question is about what is stated in the given context. Do not rely on external knowledge.
According to the passage, "Who is the president?"
If the answer is **not present in the context**, respond with exactly: "Unanswerable".
Answer (only a single phrase, no explanations, no hashtags, no extra text): "Unanswerable"

Now, answer the following:
Context: '''{context}'''
The following question is about what is stated in the given context. Do not rely on external knowledge.
According to the passage, {question}
If the answer is **not present in the context**, respond with exactly: "Unanswerable".
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


def get_answer(context, question, example):
    # system_message = "You are a precise and reliable assistant trained to answer questions strictly based on the given context. If the answer is explicitly stated in the context, provide the exact phrasing without alterations. If the answer is not found in the context, respond only with 'Unanswerable'. Do not add explanations, hashtags, or extra text. Follow these guidelines strictly."
    system_message = "You are a precise and reliable assistant trained to answer questions strictly based on the given context. Use the provided articles delimited by triple quotes to answer questions in a single word or single phrase. If the answer cannot be found in the articles, write \"Unanswerable\""
    prompt = format_prompt(system_message, context, question, example)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    # Measure inference time
    start_time = time.perf_counter()
   

    
    generated_text = model.generate(
        input_ids,
        # stopping_criteria=stopping_criteria,  # Use this instead of `stop`
        max_length=2000,  # Adjust based on desired response length
        do_sample=True,  # Enable sampling for variability
        top_p=0.80,  # Nucleus sampling for more controlled diversity
        temperature=0.7,  # Lower temperature for less randomness
        num_return_sequences=1,  # Ensure only one response is generated
        pad_token_id=tokenizer.eos_token_id,  # Ensures consistent behavior
        eos_token_id=tokenizer.eos_token_id  # Stop generation at the end of an answer
    )
    end_time = time.perf_counter()
    
    response = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    print("response",response)
    ans= response.split("Answer (only a single phrase, no explanations, no hashtags, no extra text):")[-1].strip()

    inference_time = end_time - start_time
    print(f"Inference Time: {inference_time:.4f} seconds")
    return clean_answer(ans), inference_time
example = {
    "context": "The Eiffel Tower was constructed in 1889 and is located in Paris, France.",
    "question": "When was the Eiffel Tower built?",
    "answer": "1889"
}

squad_metric = evaluate.load("squad")

em_scores = []
f1_scores = []
def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()
def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1
def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))
def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_raw_scores(df, preds):
    exact_scores = {}
    f1_scores = {}
    
    for i in range(50):
        qid = df.iloc[i]["id"]
        gold_answers = df.iloc[i]["answer"]["text"]
        
        if len(gold_answers) == 0:
            # For unanswerable questions, the only correct answer is an empty string
            gold_answers = ['Unanswerable']
        
        a_pred = preds.get(qid, '')  # Use `.get()` to avoid KeyError

        # if a_pred == "Unanswerable":
        #     a_pred = ''  # Convert 'Unanswerable' to ''
        print("qid",qid)
        print("cleaned answer:",a_pred)
        print("gold_answers",gold_answers)
        
        if qid not in preds:
            print(f'Missing prediction for {qid}')
            continue
        
        a_pred = preds[qid]
        
        # Take max over all gold answers
        exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
    
    return exact_scores, f1_scores
predictions = {}
total_time=0
for i in range(50):
    passage = df.iloc[i]["context"]
    question = df.iloc[i]["question"]
    qid = df.iloc[i]["id"]
    
    print(f"\nProcessing Example {i+1}")
    print("Question:", question)
    print("id:", qid)

    # Get model's predicted answer
    predicted_answer, inference_time = get_answer(passage, question, example)
    total_time += inference_time
    print("Predicted Answer:", predicted_answer)

    # Store prediction
    predictions[qid] = predicted_answer

    # Get gold standard answers
    gold_answers = df.iloc[i]["answer"]["text"]
    print("Gold Answers:", gold_answers)

        # Compute scores every 1000 samples or at the last iteration
    if (i + 1) % 1000 == 0 or (i + 1) == len(df):
        print(f"\nComputing scores after {i+1} samples...\n")
        exact_scores, f1_scores = get_raw_scores(df.iloc[:i+1], predictions)

        exact_match_score = sum(exact_scores.values()) / len(exact_scores)
        f1_score = sum(f1_scores.values()) / len(f1_scores)

        print(f"Exact Match Score after {i+1} samples: {exact_match_score:.4f}")
        print(f"F1 Score after {i+1} samples: {f1_score:.4f}\n")

# Compute scores using get_raw_scores
exact_scores, f1_scores = get_raw_scores(df, predictions)

print(exact_scores)
print(f1_scores)
exact_match_score = sum(exact_scores.values()) / len(exact_scores)
f1_score = sum(f1_scores.values()) / len(f1_scores)

print(f"Average Exact Match Score: {exact_match_score:.4f}")
print(f"Average F1 Score: {f1_score:.4f}")
average_inference_time = total_time / 50
print(f"Average inference time per prediction: {average_inference_time:.4f} seconds")