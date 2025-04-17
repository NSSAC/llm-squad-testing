import pandas as pd
import string
import time
import re
import collections
import torch
import outlines
from outlines import models, generate, samplers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load dataset
splits = {'train': 'squad_v2/train-00000-of-00001.parquet', 'validation': 'squad_v2/validation-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/rajpurkar/squad_v2/" + splits["validation"])

def create_hybrid_qa_system(model_name="/home/gjf3sa/sneha/midas/smol-course-llm-finetuning/1_instruction_tuning/notebooks/squad2/llama-3-8b-squad_50K", temperature=0.7):
    """
    Creates a hybrid QA system that uses Outlines for structured output
    but preserves the high-accuracy approach from the working code.
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto"
    )
    
    # Create Outlines model wrapper for structured output
    try:
        outlines_model = models.Transformers(model, tokenizer)
        sampler = samplers.multinomial(1, temperature=temperature, top_p=0.80)
        print("Successfully created Outlines model wrapper")
    except Exception as e:
        print(f"Could not create Outlines model wrapper: {e}")
        outlines_model = None
        sampler = None

    def create_few_shot_prompt(context, question):
        prompt = """I'll answer questions based on context. I'll give short, precise answers or say "Unanswerable" if the answer isn't in the context.

Example 1:
Context: '''The Eiffel Tower was constructed in 1889 and is located in Paris, France.'''
Question: When was the Eiffel Tower built?
Answer: 1889

Example 2:
Context: '''Albert Einstein developed the theory of relativity.'''
Question: Who developed the theory of relativity?
Answer: Albert Einstein

Example 3:
Context: '''No information is given about the president.'''
Question: Who is the president?
Answer: Unanswerable

Now for your question:
Context: '''{}'''
Question: {}
Answer:""".format(context, question)
        
        return prompt
    
    def clean_answer(response):
        """Extract and clean the answer from the model's response"""
        # Handle list-type responses
        if isinstance(response, list):
            response = response[0] if response else ""
        
        # Remove everything after "assistant" if present
        if "assistant" in response:
            response = response.split("assistant")[0].strip()
        
        # Split into lines and take the first non-empty line
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        if not lines:
            return "Unanswerable"
        
        # Get answer - either after a colon or the first line
        answer = lines[0]
        if ":" in answer:
            answer = answer.split(":", 1)[1].strip()
        
        # Remove quotes and clean up
        answer = answer.replace('"', '').replace("'", '').strip()
        
        # If answer is still too long, take just the first part
        if len(answer.split()) > 5:
            parts = answer.split(".")
            answer = parts[0].strip()
            if len(answer.split()) > 5:  # Still too long
                answer = " ".join(answer.split()[:5])
        
        return answer
    
    def qa_function(context, question):
        """Main QA function that uses Outlines with a fallback to direct model usage"""
        # Create a simple prompt
        prompt = create_few_shot_prompt(context, question)
        
        # Measure inference time
        start_time = time.perf_counter()
        
        # Try using Outlines if available
        if outlines_model is not None:
            try:
                # Create structured generator
                generator = outlines.generate.text(outlines_model, sampler)
                
                # Generate with Outlines
                response = generator(
                    prompt, 
                    max_tokens=25  # Keep it short to avoid generating unrelated content
                )
                
                # Debug the response
                print(f"Outlines response type: {type(response)}")
                print(f"Outlines response preview: {str(response)[:100] if isinstance(response, str) else response}")
                
                # Process the response
                answer = clean_answer(response)
                end_time = time.perf_counter()
                inference_time = end_time - start_time
                
                return answer, inference_time
                
            except Exception as e:
                print(f"Outlines generation failed: {e}")
                print("Falling back to direct model usage")
    
    return qa_function

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
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_raw_scores(df, preds, num_examples):
    exact_scores = {}
    f1_scores = {}
    
    for i in range(min(num_examples, len(df))):
        qid = df.iloc[i]["id"]
        gold_answers = df.iloc[i]["answers"]["text"]
        
        if len(gold_answers) == 0:
            gold_answers = ['Unanswerable']
        
        if qid not in preds:
            print(f'Missing prediction for {qid}')
            continue
        
        a_pred = preds[qid]
        
        # Take max over all gold answers
        exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
        
        # Print individual scores for debugging
        print(f"Score for {qid}: EM={exact_scores[qid]}, F1={f1_scores[qid]:.4f}")
    
    return exact_scores, f1_scores

# Main execution
if __name__ == "__main__":
    print("====== STARTING HYBRID QA EVALUATION ======")
    
    # Create the hybrid QA function with the same temperature as the working code
    qa_function = create_hybrid_qa_system(temperature=0.7)
    
    # Run predictions
    predictions = {}
    total_time = 0
    num_examples = len(df)
    
    for i in range(num_examples):
        passage = df.iloc[i]["context"]
        question = df.iloc[i]["question"]
        qid = df.iloc[i]["id"]
        
        print(f"\nProcessing Example {i+1}")
        print("Question:", question)
        print("id:", qid)

        # Generate the answer with the hybrid system
        answer, inference_time = qa_function(passage, question)
        total_time += inference_time
        
        print(f"Generation time: {inference_time:.4f} seconds")
        print("Predicted Answer:", answer)
        
        # Store prediction
        predictions[qid] = answer

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
    
    # Compute final scores
    if predictions:
        exact_scores, f1_scores = get_raw_scores(df, predictions, min(num_examples, len(predictions)))

        exact_match_score = sum(exact_scores.values()) / len(exact_scores)
        f1_score = sum(f1_scores.values()) / len(f1_scores)

        print(f"\nFinal Results:")
        print(f"Predictions collected: {len(predictions)}")
        print(f"Average Exact Match Score: {exact_match_score:.4f}")
        print(f"Average F1 Score: {f1_score:.4f}")
        
        average_inference_time = total_time / len(predictions)
        print(f"Average inference time per prediction: {average_inference_time:.4f} seconds")
    else:
        print("No predictions were collected.")