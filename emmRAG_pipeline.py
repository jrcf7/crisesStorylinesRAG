import httpx
import os
import json
import pandas as pd
from client_v1.settings import EmmRetrieversSettings
from client_v1.jrc_openai import JRCChatOpenAI
from client_v1.formatting_utils import fixed_width_wrap, format_docs, format_doc_minimal
from langchain_core.prompts import ChatPromptTemplate
from utils import iso3_to_iso2, generate_date_ranges, process_documents, add_sections_as_columns
from openai import OpenAI
from httpx import ReadTimeout
import re 

model = "llama-3.3-70b-instruct"

def gpt_graph(prompt):
    completion = client1.chat.completions.create(
        model=model,  # Replace with the appropriate model for your use case
        messages=[
        {"role": "system", "content": "You are a disaster manager expert in risk dynamics."},
        {
            "role": "user",
            "content": prompt,
        }
    ]
    )

    # Extract the content from the response
    message_content = completion.choices[0].message.content
    return message_content


apindex = {"2014":'mine_e_emb16-e1f7_prod4_2014', "2015":'mine_e_emb16-e1f7_prod4_2015', "2016":'mine_e_emb16-e1f7_prod4_2016', "2017":'mine_e_emb16-e1f7_prod4_2017',
           "2018":'mine_e_emb16-e1f7_prod4_2018', "2019":'mine_e_emb16-e1f7_prod4_2019', "2020":'mine_e_emb16-e1f7_prod4_2020'}

EMM_RETRIEVERS_API_BASE="https://api.emm4u.eu/retrievers/v1"
with open('./data/emm_token.json', 'r') as file:
    config = json.load(file)
    EMM_RETRIEVERS_API_KEY = config['EMM_RETRIEVERS_API_KEY']
    
EMM_RETRIEVERS_OPENAI_API_BASE_URL="https://api-gpt.jrc.ec.europa.eu/v1"
with open('./data/gpt_token.json', 'r') as file:
    config = json.load(file)
    EMM_RETRIEVERS_OPENAI_API_KEY = config['EMM_RETRIEVERS_OPENAI_API_KEY']


settings = EmmRetrieversSettings(
        API_BASE=EMM_RETRIEVERS_API_BASE,
        API_KEY=EMM_RETRIEVERS_API_KEY,
        OPENAI_API_BASE_URL=EMM_RETRIEVERS_OPENAI_API_BASE_URL,
        OPENAI_API_KEY=EMM_RETRIEVERS_OPENAI_API_KEY,
        LANGCHAIN_API_KEY="your_langchain_api_key"
    )


client = httpx.Client(
    base_url=settings.API_BASE,
    headers={"Authorization": f"Bearer {settings.API_KEY.get_secret_value()}"},
)

client1 = OpenAI(
    api_key=EMM_RETRIEVERS_OPENAI_API_KEY,
    base_url="https://api-gpt.jrc.ec.europa.eu/v1",
)


# Read EM-DAT dataset containing disaster events 
emdat = pd.read_excel("./data/public_emdat_1419.xlsx")

#f = open('./data/skipped_rows.txt')
#disno = f.read().splitlines()
#f.close()
#emdat = emdat[emdat["DisNo."].isin(disno)]

print(len(emdat))



def extract_disaster_info(disaster, month, year, country, formatted_docs):
    """
    Extracts specific disaster information from formatted documents using a language model
    and outputs it in a structured text format.

    :param disaster: Name of the disaster event.
    :param month: Month of the disaster event.
    :param year: Year of the disaster event.
    :param country: The country where the disaster occurred.
    :param formatted_docs: The formatted content of the documents.
    :return: A dictionary with extracted information or None if not available.
    """
    # Construct the prompt for the LLM
    prompt = (
        f"You are an expert in disaster event analysis. Based on the content related to the {disaster} disaster "
        f"that occurred in {country} during {month} {year}, please fill in the factsheet below. "
        f"For 'Locations', list all mentioned cities or provinces only within {country}, ignoring any outside of {country}. "
        f"For 'People affected' and 'Fatalities' return only the total amount according to the Document Content; do not include any additional words or text."
        f"For 'Economic losses' return the total amount plus the currency as reported in the Document Content; do not include any additional words or text."
        f"Use 'None' for any field where the information is not available.\n\n"
        f"Document Content: {formatted_docs}\n\n"
        f"Factsheet:\n"
        f"People affected: \n"
        f"Fatalities: \n"
        f"Economic losses: \n"
        f"Locations: \n"
    )

    # Call the language model with the prompt
    completion = client1.chat.completions.create(
        model=model,  # Replace with your model
        messages=[
            {"role": "system", "content": "You are an expert in disaster event analysis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    # Extract the response content
    response_content = completion.choices[0].message.content.strip()

    return response_content

def parse_factsheet(response_content):
    """
    Parses the structured text response to extract disaster information.

    :param response_content: The structured text response from the language model.
    :return: A dictionary with extracted information.
    """

    def extract_value(label, text):
        # Improved pattern: Look for the label followed by any content until the next label or end of text
        pattern = rf"{label}:\s*([^\n]*?)(?=\n[A-Za-z\s]+:|$)"
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    factsheet = {
        "People affected": extract_value("People affected", response_content),
        "Fatalities": extract_value("Fatalities", response_content),
        "Economic losses": extract_value("Economic losses", response_content),
        "Locations": extract_value("Locations", response_content),
    }

    return factsheet




emdat = pd.read_excel("./data/public_emdat_1419.xlsx")

#f = open('./data/skipped_rows.txt')
#disno = f.read().splitlines()
#f.close()
#emdat = emdat[emdat["DisNo."].isin(disno)]

print(len(emdat))

emdat['Start Month'] = emdat['Start Month'].fillna(1).astype(int)
emdat['Start Day'] = emdat['Start Day'].fillna(1).astype(int)
emdat['start_dt'] = pd.to_datetime(emdat['Start Year'].astype(str) + '-' +
                                   emdat['Start Month'].astype(str).str.zfill(2) + '-' +
                                   emdat['Start Day'].astype(str).str.zfill(2))
emdat['start_dt'] = emdat['start_dt'].dt.strftime('%Y-%m-%d')

mnts = {
    "01": "January",
    "02": "February",
    "03": "March",
    "04": "April",
    "05": "May",
    "06": "June",
    "07": "July",
    "08": "August",
    "09": "September",
    "10": "October",
    "11": "November",
    "12": "December"
}


cutoff_date = pd.to_datetime('2014-04-15', format='%Y-%m-%d')
emdat['start_dt'] = pd.to_datetime(emdat['start_dt'])
filtered_emdat = emdat[emdat["start_dt"] >= cutoff_date]
filtered_emdat["start_dt"] = filtered_emdat["start_dt"].dt.strftime('%Y-%m-%d')

folder_path = './data'
files = os.listdir(folder_path)
csv_files = [f for f in files if f.startswith('emdat2_') and f.endswith('.csv')]
dataframes = []
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    dataframes.append(df)

emdats = pd.concat(dataframes, ignore_index=True)
filtered_emdat = filtered_emdat[~filtered_emdat['DisNo.'].isin(emdats['DisNo.'])]
print(len(filtered_emdat))

output_csv_dir = "./data/"
skipped_rows_file = os.path.join(output_csv_dir, "skipped_rows1.txt")
os.makedirs(output_csv_dir, exist_ok=True)

events = []

def save_and_log_skipped(events, row):
    """Save processed events and log skipped rows."""
    if events:
        emdat2 = pd.concat(events)
        emdat2.to_csv(os.path.join(output_csv_dir, f"emdat2_{emdat2.iloc[-1]['DisNo.'].replace('-', '')}.csv"), index=False)
        print("Data saved up to disaster num. =", emdat2.iloc[-1]["DisNo."], "File saved at:", os.path.join(output_csv_dir, f"emdat2_{emdat2.iloc[-1]['DisNo.'].replace('-', '')}.csv"))
    with open(skipped_rows_file, "a") as f:
        f.write(f"{row['DisNo.']}\n")

for index, row in filtered_emdat.iterrows():
    #i+=1
    print("Processing Disaster Num = ", row["DisNo."])  
    disaster = row["Disaster Type"]
    country = row["Country"]
    start_dt = row['start_dt']
    TEST_INDEX = apindex[start_dt.split("-")[0]]
    #end_dt = (pd.to_datetime(start_dt) + pd.Timedelta(weeks=1)).strftime('%Y-%m-%d')
    location = row["Location"]
    iso2 = iso3_to_iso2(row['ISO'])
    start_end_dates = generate_date_ranges(start_dt, num_weeks=4)
    month = mnts[start_dt.split("-")[1]]
    year = int(start_dt.split("-")[0])


    EXAMPLE_QUESTION = f"What are the latest developments on the {disaster} disaster occurred in {country} on {month} {year} that affected {location}?"
    #print(EXAMPLE_QUESTION)
    
    all_documents = []  # List to store all retrieved documents
    for start_dt, end_dt in start_end_dates:
        #print("start : ", start_dt, ", end : ", end_dt)
        st = pd.to_datetime(start_dt)
        en = pd.to_datetime(end_dt)
        if en > pd.Timestamp(year=st.year, month=12, day=27):
            print("End of Year!")
            TEST_INDEX = apindex[str(st.year+1)]
        try:
            response = client.post(
                "/r/rag-minimal/query",
                params={"cluster_name": settings.DEFAULT_CLUSTER, "index": TEST_INDEX},
                json={
                    "query": EXAMPLE_QUESTION,
                    "lambda_mult": 0.9,
                    "spec": {"search_k": 20, "fetch_k": 100},
                    "filter": {
                        "max_chunk_no": 1,
                        "min_chars": 100,
                        "start_dt": start_dt,
                        "end_dt": end_dt,
                    # "language": ["en", "fr", "es"],
                    },
                },
                timeout=30.0
            )

            response.raise_for_status()  # Ensure the request was successful
            search_resp = response.json()
            documents = search_resp["documents"]
            all_documents.extend(documents)  # Append retrieved documents
            
        except (ReadTimeout, httpx.HTTPStatusError, Exception) as e:
            print(f"An error occurred: {e}")
            #save_and_log_skipped(events, row)
            #events = []
            break  
    
    #print(f"Total documents retrieved: {len(all_documents)}")
        
    #docs = retriever.invoke(EXAMPLE_QUESTION)
    llm_model = JRCChatOpenAI(model=model, 
                          api_key=settings.OPENAI_API_KEY,
                          base_url=settings.OPENAI_API_BASE_URL)

    system_prompt = (
    "You are an assistant tasked with providing concise and factual updates on disaster events. "
    "Based on the context provided, answer the queries with clear and actionable information. "
    "If the information is uncertain or not found, recommend verifying with official channels. "
    "For incomplete or unknown answers, respond with 'unknown'."
    "\n\n"
    "{context}"
    "\n\n"
    "The original question was {question}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
        ("system", system_prompt),
        ("human", "{instructions}"),
        ]
    )
    
    def extract_question(d):
        return d["question"]
    
    def extract_instructions(d):
        return d["instructions"]
    
    
    
    #attempt = 0
    #while attempt < max_retries:
    if all_documents:
        try:
            formatted_docs, num_relevant_docs = process_documents(all_documents, iso2, country, disaster, month, year, location)
            if num_relevant_docs>0:
                #time.sleep(1)
                rag_chain = (
                {
                    "context": lambda _: formatted_docs,
                    "question": extract_question,
                    "instructions": extract_instructions
                }
                | prompt
                | llm_model
                )
                r = rag_chain.invoke({"question": EXAMPLE_QUESTION, "instructions": f"Complete the following factsheet on the {disaster} event in {country}: \n - Key information: [Quick summary with location and date.] \n - Severity: [Low, Medium, High] \n - Key drivers: [Main causes of the disaster.] \n - Main impacts, exposure, and vulnerability: [Economic damage, people affected, fatalities, effects on communities and infrastructure.] \n- Likelihood of multi-hazard risks: [Chances of subsequent related hazards.] \n- Best practices for managing this risk: [Effective prevention and mitigation strategies.] \n- Recommendations and supportive measures for recovery: [Guidelines for aid and rebuilding.]\n Important: Use only the information provided about the event. Do not add any assumptions or external data. If specific details are missing or uncertain, indicate them as 'unknown'."})
                story = fixed_width_wrap(r.content)
                graph_prompt = f"""Create a knowledge graph that captures the main causal relationships presented in the text. Follow these guidelines: \n - Only use two relationship types: 'causes' or 'prevents' (so e.g. either 'A causes B' or 'A prevents B'), no other type of relations are allowed. \n - Minimize the number of nodes, use short node names with no more than two words if possible. \n - Focus on drivers and impacts, specifying the type of impact or damage if mentioned explicitly (e.g. 'blackout' or 'infrastructure damage' or 'crop devastation'). \n - Do not use both a factor and its opposite (e.g., "early warning" and "lack of early warning") in the same graph. Represent the relationship with either "causes" or "prevents," not both. Avoid duplicating similar nodes.  \n Example: 
                prompt: In the past few days, several districts of Limpopo province, north-eastern South Africa experienced heavy rain and hailstorms, causing floods and severe weather-related incidents that resulted in casualties and damage. Local authorities warned population, and many were evacuated. 
                \n graph: [["heavy rain", "causes", "flooding"], ["hailstorms", "causes", "flooding"], ["flooding", "causes", "damages"],  ["flooding", "causes", "casualties"], ["early warning", "prevents", "casualties"]]
                \n
                prompt: {story}
                \n graph:"""
        
                updated_row = add_sections_as_columns(row, story, gpt_graph(graph_prompt))
                #all_keys = ["People affected", "Fatalities", "Economic losses", "Locations"]
                info = parse_factsheet(extract_disaster_info(disaster, month, year, country, formatted_docs))
                #print(info)
                for key, value in info.items():
                    updated_row[key] = value if value is not None else np.nan
                    
                updated_row["nNews"] = num_relevant_docs      
                events.append(updated_row)
            else:
                print("No relevant documents found for disaster: ", row["DisNo."])
                #save_and_log_skipped(events, row)
                #events = []
        
        except Exception as e:
            print(f"An error occurred: {e}")
            save_and_log_skipped(events, row)
            events = []
    else:
        print("No news retrieved about disaster: ", row["DisNo."])
        save_and_log_skipped(events, row)
        events = []

if events:
    emdat2 = pd.concat(events)
    emdat2.to_csv(os.path.join(output_csv_dir, f"emdat2_{emdat2.iloc[-1]['DisNo.'].replace('-', '')}.csv"), index=False)
    print("Final data saved up to disaster num. =", emdat2.iloc[-1]["DisNo."], "File saved at:", os.path.join(output_csv_dir, f"emdat2_{emdat2.iloc[-1]['DisNo.'].replace('-', '')}.csv"))
