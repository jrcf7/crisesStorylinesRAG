from typing import Any, Coroutine

import httpx
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr, model_validator

from .settings import EmmRetrieversSettings


def as_lc_docs(dicts: list[dict]) -> list[Document]:
    return [
        Document(page_content=d["page_content"], metadata=d["metadata"]) for d in dicts
    ]


# the simple retriver is built with fixed spec/filter/params/route config
# and the can be used many times with different queries.
# Note these are cheap to construct.


class EmmRetrieverV1(BaseRetriever):
    settings: EmmRetrieversSettings
    spec: dict
    filter: dict | None = None
    params: dict = Field(default_factory=dict)
    route: str = "/r/rag-minimal/query"
    add_ref_key: bool = True

    _client: httpx.Client = PrivateAttr()
    _aclient: httpx.AsyncClient = PrivateAttr()

    # ------- interface impl:
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        r = self._client.post(**self.search_post_kwargs(query))
        if r.status_code == 422:
            print("ERROR:\n", r.json())
        r.raise_for_status()
        resp = r.json()
        return self._as_lc_docs(resp["documents"])

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> Coroutine[Any, Any, list[Document]]:
        r = await self._aclient.post(**self.search_post_kwargs(query))
        if r.status_code == 422:
            print("ERROR:\n", r.json())
        r.raise_for_status()
        resp = r.json()
        return self._as_lc_docs(resp["documents"])

    # ---------
    @model_validator(mode="after")
    def create_clients(self):
        _auth_headers = {
            "Authorization": f"Bearer {self.settings.API_KEY.get_secret_value()}"
        }

        kwargs = dict(
            base_url=self.settings.API_BASE,
            headers=_auth_headers,
            timeout=self.settings.DEFAULT_TIMEOUT,
        )

        self._client = httpx.Client(**kwargs)
        self._aclient = httpx.AsyncClient(**kwargs)
        return self

    @model_validator(mode="after")
    def apply_default_params(self):
        self.params = {
            **{
                "cluster_name": self.settings.DEFAULT_CLUSTER,
                "index": self.settings.DEFAULT_INDEX,
            },
            **(self.params or {}),
        }
        return self

    def _as_lc_docs(self, dicts: list[dict]) -> list[Document]:
        docs = as_lc_docs(dicts)
        if self.add_ref_key:
            for i, d in enumerate(docs):
                d.metadata["ref_key"] = i

        return docs

    def search_post_kwargs(self, query: str):
        return dict(
            url=self.route,
            params=self.params,
            json={"query": query, "spec": self.spec, "filter": self.filter},
        )
