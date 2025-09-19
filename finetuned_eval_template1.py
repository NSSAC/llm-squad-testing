import os
import re
import time
import string
import collections
import torch
import pandas as pd
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set environment variable to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load dataset
splits = {'train': 'squad_v2/train-00000-of-00001.parquet', 'validation': 'squad_v2/validation-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/rajpurkar/squad_v2/" + splits["validation"])

# Model path
model_path = "/home/gjf3sa/sneha/midas/smol-course-llm-finetuning/1_instruction_tuning/notebooks/squad2/llama-3-8b-squad_50K"

# Load the model and tokenizer directly with Transformers
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float16,
    device_map="auto"
)

# Define a system message that emphasizes precision
system_message = """You are a precise and reliable assistant trained to answer questions strictly based on the given context. 
Use the provided context to answer questions in a single word or single phrase.
Never include information that cannot be directly found in the context.
If the answer cannot be found in the context, write "Unanswerable".
Keep your answers extremely brief (typically 1-5 words) and always extracted directly from the context.
Do not add any explanation or additional text."""

# Format prompt using a simplified template
def format_prompt(system_message, context, question, example):
    prompt = f"""<|system|>
{system_message}

<|user|>
Context: '''{example['context']}'''
According to the passage, {example['question']}

<|assistant|>
{example['answer']}

<|user|>
Context: '''Albert Einstein developed the theory of relativity.'''
According to the passage, Who developed the theory of relativity?

<|assistant|>
Albert Einstein

<|user|>
Context: '''No information is given about the president.'''
According to the passage, Who is the president?

<|assistant|>
Unanswerable

<|user|>
Context: '''The Eiffel Tower was built in 1889 in Paris.'''
According to the passage, When was the Eiffel Tower constructed?

<|assistant|>
1889

<|user|>
Context: '''{context}'''
According to the passage, {question}

<|assistant|>"""
    return prompt

# Improved clean_answer function with simpler regex
def clean_answer(response_text):
    # Check for the unanswerable keyword with more flexibility
    if "unanswerable" in response_text.lower():
        return "Unanswerable"
    
    # Try to find answers in the raw text using pattern matching
    # First, look for any text before common interruptions
    answer = response_text.strip()
    
    # Remove the word "assistant" if present
    answer = answer.replace("assistant", "").replace("Assistant", "")
    
    # Split by common markers that may appear in the response
    for splitter in ["\n", "What", "I'm happy", "Am I"]:
        if splitter in answer:
            answer = answer.split(splitter)[0].strip()
    
    # Maximum length for answers
    max_length = 30
    if len(answer) > max_length:
        # Try to find a reasonable breaking point
        shorter_answer = answer[:max_length].rsplit(' ', 1)[0]
        answer = shorter_answer
    
    # Remove quotes if present
    answer = answer.replace('"', '').replace("'", '').strip()
    
    # If answer is suspiciously long or empty, default to Unanswerable
    if len(answer) > 50 or not answer:
        return "Unanswerable"
    
    return answer

# Completely revised get_answer function 
def get_answer(context, question, example):
    prompt = format_prompt(system_message, context, question, example)
    
    # Add padding token to inputs to avoid the attention mask warning
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    generated_text = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=input_ids.shape[1] + 50,
        do_sample=False,  # More deterministic results
        top_p=0.95,
        temperature=0.1,  # Very low temperature for predictable answers
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Get just the newly generated tokens
    new_tokens = generated_text[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Print full response for debugging
    print("Raw generated response:", response)
    
    # Simple cleanup - take only the first sentence or up to the first newline
    answer = response.strip()
    
    # If the answer seems complicated, try to extract just the first part
    if len(answer.split()) > 10 or "\n" in answer:
        # Take everything before the first newline or punctuation mark
        for splitter in ["\n", ".", "!", "?"]:
            if splitter in answer:
                answer = answer.split(splitter)[0].strip() + (splitter if splitter != "\n" else "")
                break
    
    return clean_answer(answer)

# Evaluation functions
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

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

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

# Improved get_raw_scores function with better handling of unanswerable questions
def get_raw_scores(df, preds, num_examples):
    exact_scores = {}
    f1_scores = {}

    for i in range(min(num_examples, len(df))):
        qid = df.iloc[i]["id"]
        gold_answers = df.iloc[i]["answers"]["text"]
        
        is_impossible = len(gold_answers) == 0
        
        if is_impossible:
            gold_answers = ['Unanswerable']
        
        if qid not in preds:
            print(f'Missing prediction for {qid}')
            continue
            
        a_pred = preds[qid]
        
        # Special handling for impossible questions
        if is_impossible:
            exact_scores[qid] = 1 if a_pred.lower() == "unanswerable" else 0
            f1_scores[qid] = 1 if a_pred.lower() == "unanswerable" else 0
        else:
            # For answerable questions, use the original logic
            exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
            f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)

    return exact_scores, f1_scores

# Example for in-context learning
example = {
    "context": "The Eiffel Tower was constructed in 1889 and is located in Paris, France.",
    "question": "When was the Eiffel Tower built?",
    "answer": "1889"
}

# Run predictions
predictions = {}
num_examples = len(df)  # Number of examples to process

try:
    for i in range(num_examples):
        passage = df.iloc[i]["context"]
        question = df.iloc[i]["question"]
        qid = df.iloc[i]["id"]

        print(f"\nProcessing Example {i+1}")
        print("Question:", question)
        print("id:", qid)

        # Generate the answer using HuggingFace
        start_time = time.time()
        predicted_answer = get_answer(passage, question, example)
        generation_time = time.time() - start_time
        print(f"Generation time: {generation_time:.2f} seconds")

        print("Predicted Answer:", predicted_answer)
        predictions[qid] = predicted_answer

        # Get gold standard answers
        gold_answers = df.iloc[i]["answers"]["text"]
        print("Gold Answers:", gold_answers)

        # Compute intermediate scores every 1000 samples
        if (i + 1) % 1000 == 0 or (i + 1) == num_examples:
            print(f"\nComputing intermediate scores after {i+1} samples...\n")
            exact_scores, f1_scores = get_raw_scores(df, predictions, i+1)

            exact_match_score = sum(exact_scores.values()) / len(exact_scores)
            f1_score = sum(f1_scores.values()) / len(f1_scores)

            print(f"Exact Match Score after {i+1} samples: {exact_match_score:.4f}")
            print(f"F1 Score after {i+1} samples: {f1_score:.4f}\n")

except Exception as e:
    print(f"Error occurred: {e}")
    print("Saving predictions so far...")

# Compute final scores for whatever we processed
if predictions:
    exact_scores, f1_scores = get_raw_scores(df, predictions, min(num_examples, len(predictions)))

    exact_match_score = sum(exact_scores.values()) / len(exact_scores)
    f1_score = sum(f1_scores.values()) / len(f1_scores)

    print(f"\nFinal Results:")
    print(f"Predictions collected: {len(predictions)}")
    print(f"Average Exact Match Score: {exact_match_score:.4f}")
    print(f"Average F1 Score: {f1_score:.4f}")
else:
    print("No predictions were collected.")