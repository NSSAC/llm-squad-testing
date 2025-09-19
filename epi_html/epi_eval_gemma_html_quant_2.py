import time
import json
import pandas as pd
import torch
from evaluate import load
import Levenshtein as lev
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,  # Use this instead of Gemma3ForConditionalGeneration
    BitsAndBytesConfig
)

# 1. Load metric and data
squad_metric = load("squad_v2")
df = pd.read_csv("/home/gjf3sa/sneha/midas/qwen/html_dataset_final_with_answerStart.csv")
df["answers"] = df["answers"].apply(json.loads)

# 2. Model & quantization config (keeping your original settings)
model_id = "google/gemma-3-12b-it"

# Your exact 8-bit quantization config
bnb_config8 = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=False
)

# Load model with AutoModelForCausalLM (more stable than Gemma3ForConditionalGeneration)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config8,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

# 3. Load tokenizer properly
tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    use_fast=True,
    trust_remote_code=True
)

# Set pad token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

system_message=f"""You are a precise and reliable assistant trained to answer questions in a single word or single phrase strictly based on 
the given background. Use the provided background to answer questions. When answering, you must **locate** the answer span in the provided context
and **copy it exactly**, preserving every character, space, and punctuation mark."""

# 4. Fixed prompt formatting for Gemma-3
def format_prompt_fixed(context: str, question: str) -> str:
    
    # Simplified, cleaner prompt
    prompt = f"""
    {system_message}
    Background: {context} Question: {question} Answer:"""
    return prompt

# 5. Improved generation function
def generate_answer_fixed(context: str, question: str) -> str:
    """Generate answer with better parameters to avoid garbled output"""
    
    # Create proper prompt
    prompt = format_prompt_fixed(context, question)
    
    # Tokenize with proper settings
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024  # Shorter context to avoid issues
    ).to(model.device)
    
    # Key fix: Better generation parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,  # Shorter answers to avoid rambling
            do_sample=False,    # Pure greedy decoding
            num_beams=1,        # No beam search
            temperature=None,   # Disable temperature
            top_k=None,         # Disable top-k
            top_p=None,         # Disable top-p
            repetition_penalty=1.0,  # No repetition penalty
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            output_scores=False,
            return_dict_in_generate=False
        )
    
    # Decode only new tokens
    input_length = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Clean up output
    generated_text = generated_text.strip()
    
    # Remove common artifacts
    generated_text = generated_text.replace("<end_of_turn>", "")
    generated_text = generated_text.replace("<start_of_turn>", "")
    
    # Stop at first newline or punctuation for cleaner answers
    for stop_char in ['\n', '.', '!', '?']:
        if stop_char in generated_text:
            generated_text = generated_text.split(stop_char)[0].strip()
            break
    
    # If still too long or garbled, take first few words
    words = generated_text.split()
    if len(words) > 10 or any(len(word) > 20 for word in words):
        generated_text = ' '.join(words[:5])  # Take first 5 words only
    
    return generated_text

# 6. Main evaluation loop with debugging
num_examples = len(df) # Start small for testing
predictions = []
references = []
f1_scores = []
exact_match = []
lev_ratios = []
total_time = 0.0

print(f"Starting evaluation with Gemma-3-4B (8-bit) on {num_examples} examples...")
print("Model loaded successfully!")

for i in range(num_examples):
    try:
        context = str(df.iloc[i]["context"])[:500]  # Limit context length
        question = str(df.iloc[i]["question"])
        qid = str(df.iloc[i]["id"])
        gold_answers = df.iloc[i]["answers"]
        
        print(f"\n[{i+1}/{num_examples}]")
        print(f"Question: {question}")
        print(f"Expected: {gold_answers['text']}")
        
        # Generate answer
        start_time = time.perf_counter()
        generated = generate_answer_fixed(context, question)
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        total_time += elapsed
        
        print(f"Generated: '{generated}' (took {elapsed:.2f}s)")
        
        # Check if output looks reasonable
        if len(generated) > 100 or not generated.replace(' ', '').isascii():
            print("Warning: Generated text may be garbled")
        else:
            print("Generated text looks clean")
        
        # Build prediction/reference
        pred = {
            "id": qid,
            "prediction_text": generated,
            "no_answer_probability": 1.0 if generated.lower() in ["no answer", "unanswerable", ""] else 0.0
        }
        ref = {"id": qid, "answers": gold_answers}
        
        # Calculate metrics
        try:
            result = squad_metric.compute(
                predictions=[pred],
                references=[ref],
                no_answer_threshold=0.5
            )
            f1_scores.append(result["f1"])
            exact_match.append(result["exact"])
            print(f"F1: {result['f1']:.3f}, EM: {result['exact']:.3f}")
        except Exception as e:
            print(f"Metrics error: {e}")
            f1_scores.append(0.0)
            exact_match.append(0.0)
        
        # Levenshtein ratio
        if gold_answers["text"]:
            max_ratio = max(lev.ratio(generated, ga) for ga in gold_answers["text"])
            lev_ratios.append(max_ratio)
        else:
            lev_ratios.append(0.0)
        
        predictions.append(pred)
        references.append(ref)
        
        print("-" * 60)
        
    except Exception as e:
        print(f"Error processing example {i}: {e}")
        import traceback
        traceback.print_exc()
        continue

# 7. Final results
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

if predictions:
    try:
        final_scores = squad_metric.compute(
            predictions=predictions,
            references=references,
            no_answer_threshold=0.5
        )
        print(final_scores)
        
        # for key, value in final_scores.items():
        #     print(f"{key}: {value:.4f}")
        
        # print(f"\nDetailed Metrics:")
        # print(f"Average F1: {sum(f1_scores)/len(f1_scores):.4f}")
        # print(f"Average EM: {sum(exact_match)/len(exact_match):.4f}")
        # print(f"Average Levenshtein: {sum(lev_ratios)/len(lev_ratios):.4f}")
        # print(f"Average inference time: {total_time/len(predictions):.4f}s")
        
    except Exception as e:
        print(f"Error computing final metrics: {e}")
else:
    print("No successful predictions generated!")


if predictions:
    results_df = df.iloc[:len(predictions)].copy()
    results_df["predicted_answer"] = [p["prediction_text"] for p in predictions]
    results_df["exact_match"] = exact_match[:len(predictions)]
    results_df["f1"] = f1_scores[:len(predictions)]
    results_df["lev_ratio"] = lev_ratios[:len(predictions)]
    
    print(f"\nResults ready for saving ({len(predictions)} examples)")
    # Uncomment to save:
    # results_df.to_csv("gemma3_4b_8bit_fixed_results.csv", index=False)