import pandas as pd 
from utils import safe_extract_list_from_string, process_storyline

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
emdats["causal graph"] = emdats["causal graph"].apply(safe_extract_list_from_string)
filtered_emdats = emdats[emdats['causal graph'].apply(lambda x: isinstance(x, list))]
print(len(filtered_emdats))

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

filtered_emdats = filtered_emdats.apply(process_storyline, axis=1)
filtered_emdats = filtered_emdats.dropna(subset=columns_to_check).reset_index(drop=True)
print(len(filtered_emdats))
