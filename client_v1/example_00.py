# %%
from pprint import pprint
import os
import httpx

# from pydantic_settings import BaseSettings, SettingsConfigDict
# from pydantic import SecretStr
#
# model_config = SettingsConfigDict(env_prefix="EMM_RETRIEVERS_", env_file="/eos/jeodpp/home/users/consose/PycharmProjects/disasterStories-prj/.env")
#
# class RetrieverSettings(BaseSettings):
#     api_base: str
#     api_key: SecretStr
#
#     class Config:
#         config_dict = model_config
#
# settings = RetrieverSettings()
# print(settings.api_base)
#print(settings.api_key.get_secret_value())


from client_v1.formatting_utils import fixed_width_wrap, format_docs
from client_v1.settings import EmmRetrieversSettings



# %%
settings = EmmRetrieversSettings()

settings.API_BASE

# the test index configuration
# TEST_INDEX = "mine_e_emb-rag_live_test_001"
# INDEX_MIN = "2024-09-14"
# INDEX_MAX = "2024-09-28"
#
TEST_INDEX = "mine_e_emb16-e1f7_prod4_2014"
INDEX_MIN = "2015-09-14"
INDEX_MAX = "2015-09-20"
# TEST_INDEX = "mine_e_emb16-e1f7_prod4_2015"
# INDEX_MIN = "2015-09-14"
# INDEX_MAX = "2015-09-20"
# TEST_INDEX = "mine_e_emb16-e1f7_prod4_2016"
# INDEX_MIN = "2016-09-14"
# INDEX_MAX = "2016-09-20"
# TEST_INDEX = "mine_e_emb16-e1f7_prod4_2017"
# INDEX_MIN = "2017-09-14"
# INDEX_MAX = "2017-09-20"
# TEST_INDEX = "mine_e_emb16-e1f7_prod4_2018"
# INDEX_MIN = "2018-09-14"
# INDEX_MAX = "2018-09-20"
# TEST_INDEX = "mine_e_emb16-e1f7_prod4_2019"
# INDEX_MIN = "2019-09-14"
# INDEX_MAX = "2019-09-20"





# instantiate an httpx client once with base url and auth
client = httpx.Client(
    base_url=settings.API_BASE,
    headers={"Authorization": f"Bearer {settings.API_KEY.get_secret_value()}"},
)


# %%
# get your auth info
client.get("/_cat/token").json()

EXAMPLE_QUESTION = "What natural disasters are currently occuring?"

# %%
r = client.post(
    "/r/rag-minimal/query",
    params={"cluster_name": settings.DEFAULT_CLUSTER, "index": TEST_INDEX},
    json={
        "query": EXAMPLE_QUESTION,
        "spec": {"search_k": 20},
        "filter": {
            "max_chunk_no": 1,
            "min_chars": 200,
            "start_dt": INDEX_MIN, #"2024-09-19",
            "end_dt": INDEX_MAX, #"2024-09-20",
        },
    },
)

r.raise_for_status()

search_resp = r.json()

documents = search_resp["documents"]
print(len(documents))


titles = [d["metadata"]["title"] for d in documents]

print("\n".join([f"- {title}" for title in titles]))

# %%
# full chunk formatting:

print(format_docs(documents, fixed_width=True))

# %%
# Using the gpt@jrc language models


from client_v1.jrc_openai import JRCChatOpenAI

llm_model = JRCChatOpenAI(model="llama-3.3-70b-instruct", openai_api_key=settings.OPENAI_API_KEY.get_secret_value(), openai_api_base=settings.OPENAI_API_BASE_URL)

resp = llm_model.invoke("What is the JRC?")
print(resp.content)
pprint(resp.response_metadata)

# %%

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

rag_chain = prompt | llm_model

# Add the API key to the LLM model
#llm_model.api_key = settings.OPENAI_API_KEY.get_secret_value()

r = rag_chain.invoke({"input": EXAMPLE_QUESTION, "context": format_docs(documents)})

print(fixed_width_wrap(r.content))
print("-" * 42)
pprint(r.response_metadata)

# %% [markdown]

# notes:
# - custom retriever class
# - multiquery retrieval https://python.langchain.com/docs/how_to/MultiQueryRetriever/
# - self query https://python.langchain.com/docs/how_to/self_query/


# %%
# using prompt hubs

import langchain.hub

if hasattr(settings, 'LANGCHAIN_API_KEY'):
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY.get_secret_value()

    rag_prompt = langchain.hub.pull("rlm/rag-prompt")
    print(
        fixed_width_wrap(
            rag_prompt.format(**{k: "{" + k + "}" for k in rag_prompt.input_variables})
        )
    )


# %%
