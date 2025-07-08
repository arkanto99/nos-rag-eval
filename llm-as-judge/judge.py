import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
from dotenv import load_dotenv
import json
from dataloader_evaluation import load_qa_pairs
from prompts import (
    HALLUCINATION_PROMPT,
    COMPLETENESS_PROMPT,
    CONTEXT_RECALL_PROMPT,
    CONTEXT_PRECISION_PROMPT
)
import re

env_path = os.path.join("/home/pablo.fernandez.rodriguez/configs/ragas.env")
load_dotenv(env_path)

def get_env_variable(var_name):
    """Retrieve an environment variable or raise an error if not set."""
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"{var_name} not set in .env file")
    return value

cache_dir = get_env_variable('CACHE_DIR')

with open('/home/compartido/pabloF/llm-as-judge/evaluation_dataset_fixed.json', 'r', encoding='utf-8') as f:
    evaluation_data = json.load(f)

def split_sentences(text): #Naive approach to split sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s) > 3]

def build_context_recall_prompt(sentence, context):
    return CONTEXT_RECALL_PROMPT.format(sentence=sentence, context=context)

def build_context_precision_prompt(context_sentence, question, ground_truth):
    return CONTEXT_PRECISION_PROMPT.format(
        context_sentence=context_sentence,
        question=question,
        ground_truth=ground_truth
    )

def load_selene():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    selene_model = AutoModelForCausalLM.from_pretrained(
        "AtlaAI/Selene-1-Mini-Llama-3.1-8B",
        device_map="auto",
        cache_dir=cache_dir,
        #quantization_config=quantization_config, # remove to load FP16 model
    )
    selene_tokenizer = AutoTokenizer.from_pretrained(
        "AtlaAI/Selene-1-Mini-Llama-3.1-8B",
        cache_dir=cache_dir,
    )
    return selene_model, selene_tokenizer

def evaluate(prompt, model, tokenizer, device='cpu', temperature=0.01, max_new_tokens=512):
    try:
        # Format the prompt into messages
        messages = [{"role": "user", "content": prompt}]

        # Apply chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Prepare model inputs
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        # Apply attention mask
        attention_mask = model_inputs.attention_mask

        # Generate response
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=attention_mask,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Extract the newly generated tokens
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # Decode the response
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    except Exception as e:
        print(f"Error in evaluate function: {e}")
        return None

def parse_atla_response(response):
    """
    Parse ATLA model response to extract reasoning and score.

    Args:
        response (str): Raw response from ATLA model

    Returns:
        tuple: (critique, score) where critique is a string and score is an integer
    """
    try:
        # Split into lines and clean up
        lines = [line.strip() for line in response.split('\n') if line.strip()]

        # Extract critique (everything between **Reasoning:** and **Result:**)
        critique = None
        score = None

        for i, line in enumerate(lines):
            if line.startswith("**Reasoning:**"):
                critique = lines[i].replace("**Reasoning:**", "").strip()
            elif line.startswith("**Result:**"):
                score = lines[i].replace("**Result:**", "").strip()

        # Remove style tag if present
        if critique and "<userStyle>" in critique:
            critique = critique.split("<userStyle>")[0].strip()

        return critique, score

    except Exception as e:
        print(f"Error parsing ATLA response: {e}")
        return None, None

def compute_context_recall(contexts, ground_truth, model, tokenizer, device):
    gt_sentences = split_sentences(ground_truth)
    if not gt_sentences:
        return 0.0
    relevant_count = 0
    for sent in gt_sentences:
        is_supported = False
        for ctx in contexts:
            prompt = build_context_recall_prompt(sent, ctx)
            result = evaluate(prompt, model, tokenizer, device=device)
            if "yes" in result.lower():
                is_supported = True
                break  # One supporting context is enough
        if is_supported:
            relevant_count += 1

    return relevant_count / len(gt_sentences)

def compute_context_recall_per_context(contexts, ground_truth, model, tokenizer, device):
    gt_sentences = split_sentences(ground_truth)
    if not gt_sentences:
        return {ctx: 0.0 for ctx in contexts}

    context_scores = {}
    for ctx in contexts:
        supported_count = 0
        for sent in gt_sentences:
            prompt = build_context_recall_prompt(sent, ctx)
            result = evaluate(prompt, model, tokenizer, device=device)
            if "yes" in result.lower():
                supported_count += 1
        context_scores[ctx] = supported_count / len(gt_sentences)
    return context_scores

def compute_context_precision(contexts, question, ground_truth, model, tokenizer, device):
    all_sentences = []
    for ctx in contexts:
        all_sentences.extend(split_sentences(ctx))
    if not all_sentences:
        return 0.0

    relevant_count = 0
    for sent in all_sentences:
        prompt = build_context_precision_prompt(sent, question, ground_truth)
        result = evaluate(prompt, model, tokenizer, device=device)
        if "yes" in result.lower():
            relevant_count += 1

    return relevant_count / len(all_sentences)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    selene_model, selene_tokenizer = load_selene()

    for i, example in enumerate(evaluation_data[205:208]):
        print(f"--------------Evaluating example {i}-----------------\n")
        user_input = example['user_input']
        assistant_response = example['response']
        reference_response = example['reference']
        retrieved_contexts = example['retrieved_contexts']
        print(f"User Input: {user_input}\n")
        print(f"Assistant Response: {assistant_response}\n")
        print(f"Reference Response: {reference_response}\n")

        # # Build prompts
        # hallucination_prompt = build_hallucination_prompt(user_input, assistant_response)
        # completeness_prompt = build_completeness_prompt(user_input, assistant_response, reference_response)

        # # Evaluate hallucination
        # hallucination_result = evaluate(hallucination_prompt, selene_model, selene_tokenizer, device=device)
        # critique, score = parse_atla_response(hallucination_result)
        # print(f"Hallucination Score: {score}\nCritique: {critique}\n")

        # # Evaluate completeness
        # completeness_result = evaluate(completeness_prompt, selene_model, selene_tokenizer, device=device)
        # critique, score = parse_atla_response(completeness_result)
        # print(f"Completeness Score: {score}\nCritique: {critique}\n")

        # Evaluate context recall and precision
        recall = compute_context_recall(retrieved_contexts, reference_response, selene_model, selene_tokenizer, device)
        separated_recall = compute_context_recall_per_context(retrieved_contexts, reference_response, selene_model, selene_tokenizer, device)
        precision = compute_context_precision(retrieved_contexts, user_input, reference_response, selene_model, selene_tokenizer, device)
        i=0
        for _, score in separated_recall.items():
            print(f"Context {i}, Recall Score: {score:.2f}\n")
            i+=1
        print(f"Total Context Recall: {recall:.2f}")

        print(f"Context Precision: {precision:.2f}\n")

