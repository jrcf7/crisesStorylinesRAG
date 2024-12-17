# %%
from pprint import pprint

import httpx

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
# TEST_INDEX = "mine_e_emb16-e1f7_prod4_2014"
# INDEX_MIN = "2014-09-14"
# INDEX_MAX = "2014-09-20"
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
TEST_INDEX = "mine_e_emb16-e1f7_prod4_2019"
INDEX_MIN = "2019-09-14"
INDEX_MAX = "2019-09-20"

# %%

from client_v1.client import EmmRetrieverV1

# we can build a concrete retriver by specifying all but the actual `query`
# here for example we build a retriver for just a specific date
retriever = EmmRetrieverV1(
    settings=settings,
    params={"index": TEST_INDEX},
    route="/r/rag-minimal/query",
    spec={"search_k": 20},
    filter={
        "max_chunk_no": 1,
        "min_chars": 200,
        "start_dt": INDEX_MIN,
        "end_dt": INDEX_MAX,
    },
)

# %%

EXAMPLE_QUESTION = "What natural disasters are currently occuring?"

docs = retriever.invoke(EXAMPLE_QUESTION)

docs
# %%
# very similar except `metadata` is an attribute
titles = [d.metadata["title"] for d in docs]

print("\n".join([f"- {title}" for title in titles]))

# %%

print(format_docs(docs))

# %%
# Using the gpt@jrc language models


from client_v1.jrc_openai import JRCChatOpenAI

llm_model = JRCChatOpenAI(model="llama-3.1-70b-instruct", openai_api_key=settings.OPENAI_API_KEY.get_secret_value(), openai_api_base=settings.OPENAI_API_BASE_URL)

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

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm_model
)

# %%
r = rag_chain.invoke(EXAMPLE_QUESTION)

print(fixed_width_wrap(r.content))
print("-" * 42)
pprint(r.response_metadata)

# %%
r = rag_chain.invoke("Outline the ongoing Health emergencies in Europe")

print(fixed_width_wrap(r.content))
print("-" * 42)
pprint(r.response_metadata)

# %%
