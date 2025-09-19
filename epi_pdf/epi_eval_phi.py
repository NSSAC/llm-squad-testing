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
import transformers

squad_metric = load("squad_v2")



df = pd.read_csv("/home/gjf3sa/sneha/midas/qwen/pdf_dataset_final_with_answerStart.csv")

df["answers"] = df["answers"].apply(json.loads)

pipeline = transformers.pipeline(
    "text-generation",
    model="microsoft/Phi-4-mini-instruct",
    model_kwargs={"torch_dtype": torch.float16},
    device_map="auto",
)




# prepare the model input
# system_message = "Give me a short introduction to large language model."
system_message=f"""You are a precise and reliable assistant trained to answer questions in a single word or single phrase strictly based on 
the given background. Use the provided background to answer questions. When answering, you must **locate** the answer span in the provided context
and **copy it exactly**, preserving every character, space, and punctuation mark."""

def format_prompt(context, question):
  prompt = """
  Example 1:
  Background: Special Focus: Update on SARS-CoV-2 variants of interest and variants of concern WHO, in collaboration with national authorities, institutions and researchers, routinely assesses if variants of SARS-CoV-2 alter transmission or disease characteristics, or impact the effectiveness of vaccines, therapeutics, diagnostics or public health and social measures (PHSM) applied to control disease spread. Potential variants of concern (VOCs), variants of interest (VOIs) or variants under monitoring (VUMs) are regularly assessed based on the risk posed to global public health. The classifications of variants will be revised as needed to reflect the continuous evolution of circulating variants and their changing epidemiology. Criteria for variant classification, and the lists of currently circulating and previously circulating VOCs, VOIs and VUMs, are available on the WHO Tracking SARS-CoV-2 variants website. National authorities may choose to designate other variants and are strongly encouraged to investigate and report newly emerging variants and their impact. Geographic spread and prevalence of VOCs  The Omicron variant of concern is the dominant variant circulating globally, accounting for nearly all sequences reported to GISAID. Since its designation as a VOC by WHO on 26 November 2021, Omicron has continued to evolve, leading to variants with slightly different genetic constellations of mutations. Each constellation may differ in the public health risk it poses, including the change in epidemiology and or the severity profile. The main features of Omicron sublineages are the high growth advantage over other variants, which is mainly driven by immune evasion.  Omicron sublineages have led and are still leading to a high number of cases and, as a result, to a high number of hospitalizations and deaths. Three Omicron sublineages BA.4, BA.5 and BA.2.12.1 have acquired a few additional mutations that may impact their characteristics (BA.4 and BA.5 have the del69/70, L452R and F486V mutations; BA.2.12.1 has the L452Q and S704L mutations). Based on GISAID data and reports from WHO regional offices and countries, the number of cases and the number of countries reporting the detection of these three variants are rising. Limited evidence to date, does not indicate a rise in hospital admissions or other signs of increased severity. Preliminary data from South Africa using S gene target failure data (absent in BA.2, present in BA.4 and BA.5) indicate no difference in the risk of hospitalization for BA.4 and BA.5, as compared to BA.1; however, the short follow-up of BA.4 and BA.5 cases does not allow for conclusions on disease severity of these sublineages to be drawn at this stage. WHO continues to closely monitor the BA.4, BA.5, and BA.2.12.1 variants as part of Omicron VOC and provide further updates as more evidence on severity becomes available. WHO requests countries to continue to be vigilant, to monitor and report sequences, as well as to conduct independent and comparative analyses of the different emerging variants. Question: Which variant has continued to evolve leading to variants with slightly different genetic constellations of mutations? Answer: Omicron
  Example 2: 
  Background: The COVID-19 pandemic continues to evolve, as does our understanding of the SARS-CoV-2 virus, and the response needed to control the spread and impact of the virus. In WHO‚Äôs most recent global rapid risk assessment, conducted on 07 January 2022, the global public health risk was assessed as remaining very high. Under the Emergency Response Framework, WHO undertakes risk assessments and situation analyses on a regular basis to inform our response to emerging situations. In addition, WHO periodically reviews the current risk status of ongoing public health events through an in-depth hazard, exposure and context assessment. This also includes a review of the vulnerabilities and capacities available to respond to the public health event and to investigate the current risk to human health, risks of ongoing global spread, and the risk of insufficient control capacities. Such assessments are used as an internal WHO decision-making tool, and to support independent deliberations, including but not limited to, meetings of the International Health Regulations (IHR 2005) Emergency Committee for the COVID-19 pandemic. To date, 13 global rapid risk assessments have been undertaken for COVID-19, and additional assessments have been conducted for specific events such as the emergence of SARS-CoV-2 variants. Here, we provide a synopsis of the most recent   global rapid risk assessment for COVID-19 which covers the period from mid-October to the end of December 2021 and was finalised on 7 January 2022.  Since the last risk assessment, the incidence of COVID-19 has increased globally, initially driven predominantly by transmission of the Delta variant, particularly in the European Region. Subsequently, following the identification of the Omicron variant in South Africa, reported to WHO on 24 November 2021, there has been a sharp increase in the incidence of cases in the African Region. This has been followed by increases in the other WHO Regions in weeks 51 (20-26 December 2021) and 52 (27 December 2021 to 2 January 2022). In contrast, a global decline has been seen for COVID-19 mortality since a peak in August 2021. Nonetheless, a combination of the increased and rapid spread of the Omicron variant, increased population movements and social mixing during and after the end of year holiday period and challenges with ongoing adherence to public health and social measures (PHSM) are expected to lead to increased number of cases, hospitalizations and deaths in the coming weeks.  On 26 November, WHO designated Omicron as the fifth SARS-CoV-2 Variant of Concern (VOC) due to its large number of mutations, evidence indicative of a detrimental change in COVID-19 epidemiology and potential for increased risk of reinfection. The variant has since spread to countries in all six WHO regions, replacing the Delta variant in most countries for which data are available. Question: On which date did WHO designate Omicron as the fifth SARS-CoV-2 Variant of Concern? Answer: 26 November
  Background: {} Question: {} Answer:
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
    {"role": "system", "content": system_message},
    # {"role": "system", "content": {prompt} },
    {"role": "user", "content": prompt}
    ]

    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True,
    #     enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    # )
    # model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    start_time = time.perf_counter()
    # conduct text completion
    # generated_ids = model.generate(
    #     **model_inputs,
    #     max_new_tokens=100
    # )
    outputs = pipeline(messages, max_new_tokens=100)
    
    end_time = time.perf_counter()
    

    # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = outputs[0]["generated_text"][-1]['content']
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
df.to_csv("/home/gjf3sa/sneha/midas/qwen/epi_final/phi_logs/pdf_epi_final_phi_2shot.csv", index=False)

# print("Wrote new file with predictions in column `predicted_answer`")

