import httpx
import pandas as pd
from client_v1.settings import EmmRetrieversSettings
from client_v1.jrc_openai import JRCChatOpenAI
from client_v1.formatting_utils import fixed_width_wrap, format_docs, format_doc_minimal
from langchain_core.prompts import ChatPromptTemplate
from utils import iso3_to_iso2, generate_date_ranges, process_documents, add_sections_as_columns
from openai import OpenAI


def gpt_graph(prompt):
    completion = client1.chat.completions.create(
        model="llama-3.1-70b-instruct",  # Replace with the appropriate model for your use case
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


TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjQyNjA3M2JiLTllYWQtNGRmYy04MmU5LTY3NWYyNGFlZjQzOSIsImlzcyI6ImdwdGpyYyIsImlhdCI6MTczNDUxNDI0NiwiZXhwIjoxNzYxOTU1MjAwLCJpc19yZXZva2VkIjpmYWxzZSwiYWNjb3VudF9pZCI6ImIyNGJiZGEwLWY5YjEtNGFkNS1hNGU2LWYyYjE2MzA5ZGI5ZiIsInVzZXJuYW1lIjoiTWljaGVsZS5ST05DT0BlYy5ldXJvcGEuZXUiLCJwcm9qZWN0X2lkIjoiSU5GT1JNIiwiZGVwYXJ0bWVudCI6IkpSQy5FLjEiLCJxdW90YXMiOlt7Im1vZGVsX25hbWUiOiJncHQtNG8iLCJleHBpcmF0aW9uX2ZyZXF1ZW5jeSI6ImRhaWx5IiwidmFsdWUiOjQwMDAwMH0seyJtb2RlbF9uYW1lIjoiZ3B0LTM1LXR1cmJvLTExMDYiLCJleHBpcmF0aW9uX2ZyZXF1ZW5jeSI6ImRhaWx5IiwidmFsdWUiOjQwMDAwMH0seyJtb2RlbF9uYW1lIjoiZ3B0LTQtMTEwNiIsImV4cGlyYXRpb25fZnJlcXVlbmN5IjoiZGFpbHkiLCJ2YWx1ZSI6NDAwMDAwfV0sImFjY2Vzc19ncm91cHMiOlt7ImFjY2Vzc19ncm91cCI6ImdlbmVyYWwifV19.rMVr1vKb_HUvgowbeH8LhC9g7ZICcvWzDGgaY2yr-8o"


client1 = OpenAI(
    api_key=TOKEN,
    base_url="https://api-gpt.jrc.ec.europa.eu/v1",
)


apindex = {"2014":'mine_e_emb16-e1f7_prod4_2014', "2015":'mine_e_emb16-e1f7_prod4_2015', "2016":'mine_e_emb16-e1f7_prod4_2016', "2017":'mine_e_emb16-e1f7_prod4_2017',
           "2018":'mine_e_emb16-e1f7_prod4_2018', "2019":'mine_e_emb16-e1f7_prod4_2019', "2020":'mine_e_emb16-e1f7_prod4_2020'}

EMM_RETRIEVERS_API_BASE="https://api.emm4u.eu/retrievers/v1"
#EMM_RETRIEVERS_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJlMS11c2VyIiwiZXhwIjoxNzM3MDMyMzMxLCJyb2xlcyI6bnVsbCwiZW1tX2FkbWluIjpmYWxzZSwiZW1tX2FsbG93X2luZGljZXMiOlsibWluZV9lX2VtYi1yYWdfbGl2ZSIsIm1pbmVfZV9lbWItcmFnX2xpdmVfdGVzdF8wMDEiLCJtaW5lX2VfZW1iMTYtZTFmN19wcm9kNF8qIiwibWluZV9lX2VtYjE2LWUxZjdfcHJvZDRfMjAxNCIsIm1pbmVfZV9lbWIxNi1lMWY3X3Byb2Q0XzIwMTUiLCJtaW5lX2VfZW1iMTYtZTFmN19wcm9kNF8yMDE2IiwibWluZV9lX2VtYjE2LWUxZjdfcHJvZDRfMjAxNyIsIm1pbmVfZV9lbWIxNi1lMWY3X3Byb2Q0XzIwMTgiLCJtaW5lX2VfZW1iMTYtZTFmN19wcm9kNF8yMDE5Il0sImVtbV9hbGxvd19jbHVzdGVycyI6WyJyYWctb3MiXX0.My2po1YcbAWlGMG6fvgqMdj3qvXw-vNETFVVrOxCEBQ"
EMM_RETRIEVERS_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJlMS11c2VyIiwiZXhwIjoxNzUzMTgwMTA2LCJyb2xlcyI6bnVsbCwiZW1tX2FkbWluIjpmYWxzZSwiZW1tX2FsbG93X2luZGljZXMiOlsibWluZV9lX2VtYi1yYWdfbGl2ZSIsIm1pbmVfZV9lbWItcmFnX2xpdmVfdGVzdF8wMDEiLCJtaW5lX2VfZW1iMTYtZTFmN19wcm9kNF8qIiwibWluZV9lX2VtYjE2LWUxZjdfcHJvZDRfMjAxNCIsIm1pbmVfZV9lbWIxNi1lMWY3X3Byb2Q0XzIwMTUiLCJtaW5lX2VfZW1iMTYtZTFmN19wcm9kNF8yMDE2IiwibWluZV9lX2VtYjE2LWUxZjdfcHJvZDRfMjAxNyIsIm1pbmVfZV9lbWIxNi1lMWY3X3Byb2Q0XzIwMTgiLCJtaW5lX2VfZW1iMTYtZTFmN19wcm9kNF8yMDE5Il0sImVtbV9hbGxvd19jbHVzdGVycyI6WyJyYWctb3MiXX0.aTDc9ZR0gCw0tJ6W__8uXxHmLEE2iHBYD8bZAPXaGi8"
EMM_RETRIEVERS_OPENAI_API_BASE_URL="https://api-gpt.jrc.ec.europa.eu/v1"
EMM_RETRIEVERS_OPENAI_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjQyNjA3M2JiLTllYWQtNGRmYy04MmU5LTY3NWYyNGFlZjQzOSIsImlzcyI6ImdwdGpyYyIsImlhdCI6MTczNDUxNDI0NiwiZXhwIjoxNzYxOTU1MjAwLCJpc19yZXZva2VkIjpmYWxzZSwiYWNjb3VudF9pZCI6ImIyNGJiZGEwLWY5YjEtNGFkNS1hNGU2LWYyYjE2MzA5ZGI5ZiIsInVzZXJuYW1lIjoiTWljaGVsZS5ST05DT0BlYy5ldXJvcGEuZXUiLCJwcm9qZWN0X2lkIjoiSU5GT1JNIiwiZGVwYXJ0bWVudCI6IkpSQy5FLjEiLCJxdW90YXMiOlt7Im1vZGVsX25hbWUiOiJncHQtNG8iLCJleHBpcmF0aW9uX2ZyZXF1ZW5jeSI6ImRhaWx5IiwidmFsdWUiOjQwMDAwMH0seyJtb2RlbF9uYW1lIjoiZ3B0LTM1LXR1cmJvLTExMDYiLCJleHBpcmF0aW9uX2ZyZXF1ZW5jeSI6ImRhaWx5IiwidmFsdWUiOjQwMDAwMH0seyJtb2RlbF9uYW1lIjoiZ3B0LTQtMTEwNiIsImV4cGlyYXRpb25fZnJlcXVlbmN5IjoiZGFpbHkiLCJ2YWx1ZSI6NDAwMDAwfV0sImFjY2Vzc19ncm91cHMiOlt7ImFjY2Vzc19ncm91cCI6ImdlbmVyYWwifV19.rMVr1vKb_HUvgowbeH8LhC9g7ZICcvWzDGgaY2yr-8o"


settings = EmmRetrieversSettings(
        API_BASE=EMM_RETRIEVERS_API_BASE,
        API_KEY=EMM_RETRIEVERS_API_KEY,
        OPENAI_API_BASE_URL=EMM_RETRIEVERS_OPENAI_API_BASE_URL,
        OPENAI_API_KEY=EMM_RETRIEVERS_OPENAI_API_KEY,
        LANGCHAIN_API_KEY="your_langchain_api_key"
    )

# instantiate an httpx client once with base url and auth
client = httpx.Client(
    base_url=settings.API_BASE,
    headers={"Authorization": f"Bearer {settings.API_KEY.get_secret_value()}"},
)


# Read EM-DAT dataset containing disaster events 
emdat = pd.read_excel("./data/public_emdat_1419.xlsx")
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



events = []

for index, row in filtered_emdat.iloc[:50].iterrows():
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


    EXAMPLE_QUESTION = f"What are the latest developments on the {disaster} disaster occurred in {country} on {mnts[start_dt.split("-")[1]]} {int(start_dt.split("-")[0])} that affected {location}?"
    print(EXAMPLE_QUESTION)
    
    all_documents = []  # List to store all retrieved documents
    for start_dt, end_dt in start_end_dates:
        print("start : ", start_dt, ", end : ", end_dt)
        st = pd.to_datetime(start_dt)
        en = pd.to_datetime(end_dt)
        if en > pd.Timestamp(year=st.year, month=12, day=27):
            print("End of Year!")
            TEST_INDEX = apindex[str(st.year+1)]            
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
        )

        response.raise_for_status()  # Ensure the request was successful

        search_resp = response.json()
        documents = search_resp["documents"]
        all_documents.extend(documents)  # Append retrieved documents

    print(f"Total documents retrieved: {len(all_documents)}")
    
    #docs = retriever.invoke(EXAMPLE_QUESTION)
    llm_model = JRCChatOpenAI(model="llama-3.1-70b-instruct", 
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
    
    rag_chain = (
        {
            "context": lambda _: process_documents(all_documents, iso2=iso2, country=country),
            "question": extract_question,
            "instructions": extract_instructions
        }
        | prompt
        | llm_model
    )
    
    #attempt = 0
    #while attempt < max_retries:
    try:
        r =  rag_chain.invoke({"question":EXAMPLE_QUESTION, "instructions": f"Complete this factsheet on the {disaster}             event in {country}: \n - Key information: [Quick summary with location and date.] \n - Severity: [Very low,             Low, Medium, High, Very high.] \n - Key drivers: [Main causes of the disaster.] \n - Main impacts, exposure,             and vulnerability: [Economic damage, people affected, fatalities, effects on communities and                             infrastructure.] \n- Likelihood of multi-hazard risks: [Chances of subsequent related hazards.] \n- Best                 practices for managing this risk: [Effective prevention and mitigation strategies.] \n- Recommendations and             supportive measures for recovery: [Guidelines for aid and rebuilding.]\n If specific details are unavailable             or uncertain, indicate as 'unknown'."})
        story = fixed_width_wrap(r.content)
        graph_prompt = f"""Create a knowledge graph that captures the main causal relationships presented in the text.               You have to use only these two relationship types: 'causes' or 'prevents' (so e.g. either 'A causes B' or 'A             prevents B'), no other type of relations are allowed. When constructing the graph, minimize the number of               nodes, exclude specific instances or dates to maintain the graph's relevance across various scenarios and               try to use short node names.\n Example: 
            prompt: In the past few days, several districts of Limpopo province, north-eastern South Africa experienced             heavy rain and hailstorms, causing floods and severe weather-related incidents that resulted in casualties               and damage. Local authorities warned population, and many were evacuated. 
            \n graph: [["heavy rain", "causes", "flooding"], ["hailstorms", "causes", "flooding"], ["flooding",                     "causes", "damages"],  ["flooding", "causes", "casualties"], ["early warning", "prevents", "casualties"]]
            \n
            prompt: {story}
            \n graph:"""
    
        updated_row = add_sections_as_columns(row, story, gpt_graph(graph_prompt))
        events.append(updated_row)
        
    except Exception as e:
        emdat2 = pd.concat(events)
        emdat2.to_csv("./data/emdat2_"+emdat2.iloc[-1]["DisNo."].replace('-', '')+".csv", index=False)
        print("Processing ended at disaster num. = ", emdat2.iloc[-1]["DisNo."], " File saved at: ", "./data/emdat2_"+emdat2.iloc[-1]["DisNo."].replace('-', '')+".csv")
        
