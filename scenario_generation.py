import pandas as pd
import random
from itertools import combinations
from typing import Optional, Union
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import inflect
import random
from itertools import combinations
from typing import Optional, Union
import pandas as pd
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['usr_tkn_consose_read'] = "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"
os.environ['ie_model_id'] = 'roncmic/t5-base-disasters'
inflect_engine = inflect.engine()

model_id = os.environ['ie_model_id']
tokenizer = T5Tokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(model_id)

def singularize(text):
    """Convert a word to its singular form."""
    if inflect_engine.singular_noun(text):
        return inflect_engine.singular_noun(text)
    return text

def is_singular_plural_pair(word1, word2):
    """Check if two words are singular/plural forms of each other."""
    return singularize(word1) == singularize(word2)


def extract_edge_and_clean(row, relations):
    for relation in relations:
        if relation in row['source']:
            row['source'] = row['source'].replace(relation, '').strip()
            row['edge'] = relation
        elif relation in row['target']:
            row['target'] = row['target'].replace(relation, '').strip()
            row['edge'] = relation
    return row

def generate_with_temperature(prompt, temperature=1.0, top_k=50, top_p=0.95, max_length=50):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )

    decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded_texts

def generate_new_relations(
    graph_df: pd.DataFrame,
    new_node: str,
    max_combinations_fraction: float = 0.3,
    num_beams: int = 6,  # Note: Beams are ignored in sampling
    max_length: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    seed: Optional[int] = None,
    verbose: bool = False
) -> pd.DataFrame:
    if seed is not None:
        random.seed(seed)

    records = graph_df.to_dict('records')
    all_combos = list(combinations(records, 3))
    max_iters = max(1, int(max_combinations_fraction * len(all_combos)))
    num_iters = random.randint(1, max_iters)

    if verbose:
        print(f"Total possible combinations: {len(all_combos)}")
        print(f"Sampling {num_iters} combos")

    all_predictions = []

    for _ in range(num_iters):
        combo = random.choice(all_combos)
        for choice in ('source', 'target'):
            last = combo[-1].copy()
            if choice == 'target':
                prompt_template = f"<extra_id_0> {new_node}."
            else:
                prompt_template = f"{new_node} <extra_id_0>."

            for _ in range(2):
                perm = list(combo[:-1])
                random.shuffle(perm)

                parts = ["If"]
                for row in perm:
                    parts.append(f"{row['source']} {row['edge']} {row['target']},")
                    parts.append("and")
                parts.append(f"{last['source']} {last['edge']} {last['target']},")
                parts.append("then")
                parts.append(prompt_template)
                prompt = " ".join(parts)

                if verbose:
                    print("Generated Prompt:", prompt)

                preds = generate_with_temperature(
                    prompt, temperature, top_k, top_p, max_length
                )

                if verbose:
                    print("Predictions:", preds)

                for text in preds:
                    #print("Generated Prompt:", prompt)
                    #print("text = ", text)
                    #print("last = ", last)
                    all_predictions.append((choice, text, last))
                    
    

    if not all_predictions:
        return graph_df.copy()

    grouped = {}
    for choice, text, last in all_predictions:
        key = (choice, last['source'], last['edge'], last['target'])
        grouped.setdefault(key, []).append(text)

    new_edges = []
    for (choice, src, edge, tgt), texts in grouped.items():
        common = set(texts)
        if not common:
            continue
        for pred in common:
            if choice == 'target':
                new_edges.append({'source': pred, 'edge': None, 'target': new_node})
            else:
                new_edges.append({'source': new_node, 'edge': None, 'target': pred})

    new_df = pd.DataFrame(new_edges).drop_duplicates()

    relations = ['causes', 'prevents']
    new_df = new_df.apply(lambda row: extract_edge_and_clean(row, relations), axis=1)

    result = pd.concat([graph_df, new_df], ignore_index=True)

    result['pair'] = result.apply(lambda x: tuple(sorted([x['source'], x['target']])), axis=1)
    result = result.drop_duplicates(subset=['pair'])
    result = result[result['source'] != result['target']]
    result = result.drop(columns=['pair'])
    # Remove duplicates based on plural/singular forms
    result['source_singular'] = result['source'].apply(singularize)
    result['target_singular'] = result['target'].apply(singularize)
    result = result.drop_duplicates(subset=['source_singular', 'edge', 'target_singular'])
    result = result[result['source'] != result['target']]
    result = result.drop(columns=['source_singular', 'target_singular'])

    return result