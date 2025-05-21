from openai import OpenAI
from utils import extract_unique_nodes
import time
import random
import json

with open('./data/gpt_token.json', 'r') as file:
    config = json.load(file)
    TOKEN = config['EMM_RETRIEVERS_OPENAI_API_KEY']


client1 = OpenAI(
    api_key=TOKEN,
    base_url="https://api-gpt.jrc.ec.europa.eu/v1",
)



def ensemble_graph(row, max_retries=10):
    
    text = (
        f"key information: {row['key information']}\n"
        f"severity: {row['severity']}\n"
        f"key drivers: {row['key drivers']}\n"
        f"main impacts, exposure, and vulnerability: {row['main impacts, exposure, and vulnerability']}\n"
        f"likelihood of multi-hazard risks: {row['likelihood of multi-hazard risks']}\n"
        f"best practices for managing this risk: {row['best practices for managing this risk']}\n"
        f"recommendations and supportive measures for recovery: {row['recommendations and supportive measures for recovery']}"
    )
    nodes = extract_unique_nodes(row['causal graph'])
    
    prompt = f"""You are an expert in disaster management and causal inference. Your goal is to identify relationships between a set of given nodes based on the provided text. You must use only the relationship types "causes" or "prevents." Follow the instructions below carefully:
    Nodes: You will be given a list of nodes. Do not change the names or add any new nodes.
    Text: Analyze the text to determine how the nodes are related through causal or preventive relationships.
    Output Format: Present your findings in the following format: [["Node1", "relationship", "Node2"]]. List all relationships in a single array.
    Rules: Use only "causes" or "prevents" to describe relationships.
    Example:
        Nodes: ["heavy rain", "flooding", "damages", "casualties", "early warning"]
        Text: "In the past few days, severe weather caused by heavy rain and hailstorms led to flooding and related incidents, resulting in casualties and damage. Local authorities issued early warnings."
        Output: [["heavy rain", "causes", "flooding"], ["flooding", "causes", "damages"], ["flooding", "causes", "casualties"], ["early warning", "prevents", "casualties"]]
    New Task:
    Nodes: {nodes}
    Text:{text}
    Output Graph: """
    
    retries = 0
    while retries < max_retries:
        try:
            completion = client1.chat.completions.create(
                model="nous-hermes-2-mixtral-8x7b-dpo",
                messages=[
                    {"role": "system", "content": "You are a disaster manager expert in risk dynamics."},
                    {"role": "user", "content": prompt},
                ]
            )
            
            message_content = completion.choices[0].message.content
            return message_content
        
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            retries += 1
            backoff_time = random.uniform(10, 30)
            time.sleep(backoff_time)
    
    print("Max retries exceeded. Returning None.")
    return None