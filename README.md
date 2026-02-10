# crisesStorylinesRAG

![Example Storyline and Knowledge Graph](AI4DRM_KG.png)

## Overview

`crisesStorylinesRAG` is a pipeline designed to **augment disaster event information with factual storylines and knowledge graphs** at scale. By combining **Large Language Models (LLMs)** with **Retrieval-Augmented Generation (RAG)** on news data from the **European Media Monitor (EMM)**, we generate structured representations of hazards, impacts, drivers, and responses for each event.  

The pipeline produces:

- **Storylines**: coherent textual summaries extracted from multiple news sources.  
- **Knowledge Graphs (KGs)**: condensed, structured representations of the key relationships (causal, preventive, or descriptive) between entities in each disaster.  

This approach allows fast, interpretable insights into global disasters and can support operational decision-making, research, or further modeling efforts.

---

## Data Sources

### European Media Monitor (EMM)

News are retrieved from [EMM](https://emm.newsbrief.eu/NewsBrief/alertedition/en/ECnews.html), which provides real-time monitoring of global news articles. The pipeline uses **semantic search** to find news relevant to each disaster event.  

### EMDAT

We rely on [EM-DAT](https://www.emdat.be/) as our historical source of disaster events. For each event, basic metadata such as disaster type, country, and date range are used to guide retrieval and constrain the search in EMM.  

> Note: The workflow could be applied to any other structured event list with sufficient metadata.

---
