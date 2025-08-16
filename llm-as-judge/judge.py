from Selene import Selene
from GPT import GPT
import torch
import random
import os
import sys
from dotenv import load_dotenv
import json
import argparse
import re

# Add parent directory to Python path to find utils module
current_dir = os.path.dirname(os.path.abspath(__file__))  # llm-as-judge
parent_dir = os.path.dirname(current_dir)  # nos-rag-eval
sys.path.append(parent_dir)

from utils.dataloader_evaluation import load_questions_with_metadata
from prompts import (
    CONTEXT_RECALL_PROMPT,
    CONTEXT_PRECISION_PROMPT
)


env_path = os.path.join("/home/pablo.fernandez.rodriguez/configs/ragas.env")
load_dotenv(env_path)

def get_env_variable(var_name):
    """Retrieve an environment variable or raise an error if not set."""
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"{var_name} not set in .env file")
    return value

cache_dir = get_env_variable('CACHE_DIR')

def split_sentences(text): #Naive approach to split sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s) > 3]

def build_context_recall_prompt(sentence, context):
    return CONTEXT_RECALL_PROMPT.format(
        sentence=sentence,
        context=context)

def build_context_precision_prompt(context, question, ground_truth):
    return CONTEXT_PRECISION_PROMPT.format(
        context=context,
        question=question,
        ground_truth=ground_truth
    )

def compute_context_recall(judge, contexts, ground_truth):
    gt_sentences = split_sentences(ground_truth)
    if not gt_sentences:
        return 0.0
    relevant_count = 0
    for sent in gt_sentences:
        for ctx in contexts:
            prompt = build_context_recall_prompt(sent, ctx)
            #print(prompt)
            result = judge.evaluate(prompt)
            if "yes" in result.lower():
                relevant_count += 1
                break  # One supporting context is enough
    return relevant_count / len(gt_sentences)

# def compute_context_recall_per_context(judge, contexts, ground_truth):
#     gt_sentences = split_sentences(ground_truth)
#     if not gt_sentences:
#         return {ctx: 0.0 for ctx in contexts}

#     context_scores = {}
#     for ctx in contexts:
#         supported_count = 0
#         for sent in gt_sentences:
#             prompt = build_context_recall_prompt(sent, ctx)
#             result = judge.evaluate(prompt)
#             if "yes" in result.lower():
#                 supported_count += 1
#         context_scores[ctx] = supported_count / len(gt_sentences)
#     return context_scores

# def compute_context_precision(judge, contexts, question, ground_truth):
#     all_sentences = []
#     for ctx in contexts:
#         all_sentences.extend(split_sentences(ctx))
#     if not all_sentences:
#         return 0.0
#     relevant_count = 0
#     for sent in all_sentences:
#         prompt = build_context_precision_prompt(sent, question, ground_truth)
#         result = judge.evaluate(prompt)
#         #print(f"Evaluating context sentence: {sent}\nResult: {result}\n")
#         if "yes" in result.lower():
#             relevant_count += 1

#     return relevant_count / len(all_sentences)

def compute_context_precision(judge, contexts, question, ground_truth):
    """
    Compute Context Precision as the mean of precision@k for each chunk in contexts.
    Precision@k = (number of relevant chunks in top k) / k

    - For each context chunk, ask the judge if it is relevant (true positive).
    - For each k (from 1 to N), compute Precision@k.
    - Return the mean of all Precision@k values.

    Args:
        judge: The LLM judge object.
        contexts: List of retrieved context chunks.
        question: The original question.
        ground_truth: The reference answer.

    Returns:
        float: Context Precision score.
    """
    if not contexts:
        return 0.0
    relevance = []
    for ctx in contexts:
        prompt = build_context_precision_prompt(ctx, question, ground_truth)
        result = judge.evaluate(prompt)
        # Consider "yes" as relevant
        is_relevant = "yes" in result.lower()
        relevance.append(is_relevant)
    precisions = []
    relevant_so_far = 0
    for k, rel in enumerate(relevance, start=1):
        if rel:
            relevant_so_far += 1
        # Precision@k: relevant_so_far / k
        precisions.append(relevant_so_far / k)
    # Context Precision: mean of all Precision@k
    return sum(precisions) / len(precisions) if precisions else 0.0

def evaluate_file(results_path, questions, judge_llm, metric="recall"):
    """Evaluate a single results file for the specified metric."""
    with open(results_path) as f:
        eval_dataset = json.load(f)
    metric_scores = []
    for i, example in enumerate(eval_dataset):
        user_input = example['user_input']
        reference_response = questions[i]['answer'][0]
        retrieved_contexts = [context_json['context'] for context_json in example['retrieved_contexts']]
        print(f"--------------Evaluating question {i}: {user_input}-----------------\n")
        if metric == "recall":
            score = compute_context_recall(judge_llm, retrieved_contexts, reference_response)
            print(f"Context Recall: {score:.2f}\n")
        else:
            score = compute_context_precision(judge_llm, retrieved_contexts, user_input, reference_response)
            print(f"Context Precision: {score:.2f}\n")
        metric_scores.append(score)
    avg_score = sum(metric_scores) / len(metric_scores) if metric_scores else 0.0
    print(f"Average Context {metric.capitalize()} for {os.path.basename(results_path)}: {avg_score:.3f}")
    return avg_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate with LLM-as-Judge Retrieval Results")
    parser.add_argument('--results', type=str, default=None, help='Path to a single results file')
    parser.add_argument('--folder', type=str, default=None, help='Path to a folder with multiple results files')
    parser.add_argument('--output', type=str, default="context_metric_results.jsonl", help='Output file for folder mode')
    parser.add_argument('--metric', type=str, choices=['recall', 'precision'], default='recall', help='Metric to evaluate: recall or precision')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    judge_llm = Selene(cache_dir=cache_dir, device=device)

    with open("/home/compartido/pabloF/nos-rag-eval/datasets/Revised_Dataset/preguntas_117_Revisado.json") as f:
        questions = json.load(f)

    if args.folder:
        output_path = args.output
        for filename in sorted(os.listdir(args.folder)):
            if filename.endswith('.json'):
                results_path = os.path.join(args.folder, filename)
                print(f"\nEvaluating file: {results_path}")
                avg_score = evaluate_file(results_path, questions, judge_llm, metric=args.metric)
                # Save result after each file
                with open(output_path, "a") as out_f:
                    out_f.write(json.dumps({
                        "file": filename,
                        f"average_context_{args.metric}": avg_score
                    }) + "\n")
    elif args.results:
        avg_score = evaluate_file(args.results, questions, judge_llm, metric=args.metric)
        print(f"Average Context {args.metric.capitalize()}: {avg_score:.3f}")
    else:
        print("Please provide either --results <file> or --folder <folder> argument.")
