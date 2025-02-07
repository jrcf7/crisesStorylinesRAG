import pandas as pd 
from utils import process_graph, process_storyline, remove_duplicated_triplets
from llm_utils import ensemble_graph
import os 

folder_path = './data'
files = os.listdir(folder_path)
csv_files = [f for f in files if f.startswith('emdat2_') and f.endswith('.csv')]
dataframes = []
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    dataframes.append(df)

emdats = pd.concat(dataframes, ignore_index=True)
print(len(emdats))

columns_to_check = [
    'key information',
    'severity',
    'key drivers',
    'main impacts, exposure, and vulnerability',
    'likelihood of multi-hazard risks',
    'best practices for managing this risk',
    'recommendations and supportive measures for recovery', 
    'causal graph'
]

emdats = emdats.apply(process_storyline, axis=1)
emdats = emdats.dropna(subset=columns_to_check).reset_index(drop=True)
print(len(emdats))
emdats = emdats[emdats["causal graph"].notna()]
print(len(emdats))
emdats["causal graph"] = emdats["causal graph"].apply(process_graph)
emdats = emdats[emdats['causal graph'].apply(lambda x: isinstance(x, list))]
print(len(emdats))
emdats["mixtral graph"] = emdats.apply(ensemble_graph, axis=1)
emdats["mixtral graph"] = emdats["mixtral graph"].apply(process_graph)
emdats["ensemble graph"] = emdats["causal graph"]+emdats["mixtral graph"]
emdats["ensemble graph"] = emdats["ensemble graph"].apply(remove_duplicated_triplets)

emdats.to_csv("./data/CG_emdat_proc.csv", index=False)