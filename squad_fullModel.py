import time
import torch
import pandas as pd
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
import string
import re
import collections
from evaluate import load
splits = {'train': 'squad_v2/train-00000-of-00001.parquet', 'validation': 'squad_v2/validation-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/rajpurkar/squad_v2/" + splits["validation"])

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float32,  # Use FP32 (high memory usage!)
    device_map="auto"
)

# model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config8, device_map="auto")
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
    
    response = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    print("response",response)
    ans= response.split("Answer (only a single phrase, no explanations, no hashtags, no extra text):")[-1].strip()
    return clean_answer(ans)
example = {
    "context": "The Eiffel Tower was constructed in 1889 and is located in Paris, France.",
    "question": "When was the Eiffel Tower built?",
    "answer": "1889"
}



squad_metric = load("squad_v2")


predictions = []
references=[]
for i in range(10):
    passage = df.iloc[i]["context"]
    question = df.iloc[i]["question"]
    qid = df.iloc[i]["id"]
    
    print(f"\nProcessing Example {i+1}")
    print("Question:", question)
    print("id:", qid)

    # Get model's predicted answer
    predicted_answer = get_answer(passage, question, example)
    print("Predicted Answer:", predicted_answer)

    # Store prediction
    
    # predictions[qid] = predicted_answer
    # predictions = [{'prediction_text': '1976', 'id': '56e10a3be3433e1400422b22', 'no_answer_probability': 0.}]
    prediction= {'prediction_text':predicted_answer,'id':qid, 'no_answer_probability': 0. }
    predictions.append(prediction)
    # Get gold standard answers
    gold_answers = df.iloc[i]["answers"]
    print("Gold Answers:", gold_answers)
    reference={'answers':gold_answers,'id':df.iloc[i]["id"]}
    references.append(reference)
    

        # Compute scores every 1000 samples or at the last iteration
    if (i + 1) % 1000 == 0 or (i + 1) == len(df):
        print(f"\nComputing scores after {i+1} samples...\n")
        results = squad_metric.compute(predictions=predictions, references=references)

        print(f"Results after {i+1} samples: {results:.4f}")
        


# print(references)
# print(predictions)
results = squad_metric.compute(predictions=predictions, references=references)


print(results)
