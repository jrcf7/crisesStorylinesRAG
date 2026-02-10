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

### LLM Models

All LLM-based processing is powered via the [GPT@JRC service](https://gpt.jrc.ec.europa.eu/). The pipeline leverages multiple LLMs for:  

- **Generating storylines**: LLMs summarize the content of retrieved news chunks (via RAG) into coherent textual disaster storylines.  
- **Constructing knowledge graphs**: A second LLM call converts the storyline into a structured knowledge graph representing key relationships between hazards, drivers, impacts, and responses.  
- **Evaluation (triplets)**: Factual triplets extracted from news are used separately to assess precision and inter-annotator agreement among experts.  

This separation ensures that the main outputs (storylines and KGs) remain **faithful and interpretable**, while triplets provide a **quantitative benchmark** for validation.


## Pipeline Workflow

The `crisesStorylinesRAG` workflow integrates news retrieval, LLM-based summarization, and knowledge graph construction to enrich disaster event information at scale:

1. **News Retrieval (RAG)**  
   Historical disaster events from [EM-DAT](https://www.emdat.be/) (or any event list) are used as a query to search the [European Media Monitor (EMM)](https://emm.newsbrief.eu/NewsBrief/alertedition/en/ECnews.html) for relevant news articles. Queries are restricted by disaster type and event dates to improve relevance. Retrieved articles are chunked for downstream processing.

2. **Storyline Generation (LLM)**  
   Each news chunk is summarized into a **coherent storyline** using LLMs via the [GPT@JRC service](https://gpt.jrc.ec.europa.eu/). This produces a concise, human-readable narrative of the event, capturing the main hazards, drivers, impacts, and response actions.

3. **Knowledge Graph Construction (LLM)**  
   The storyline is then processed by another LLM call to extract structured relationships, forming a **knowledge graph**. Nodes represent entities such as hazards, affected populations, and response measures, while edges represent causal, preventive, or relational links.

4. **Validation (Triplets)**  
   To enable quantitative validation, **factual triplets** are separately extracted from the storylines. These triplets (e.g., “Node1 causes/prevents Node2”) are used in a controlled expert annotation task to assess **precision** and **inter-annotator agreement** (Krippendorff’s α, Cohen’s κ). This step does not feed back into the KG generation but provides a statistically meaningful benchmark of factual correctness.  

This dual approach ensures that the main outputs—storylines and knowledge graphs—remain high-quality and interpretable, while triplets allow **rigorous quantitative evaluation** without requiring full graph annotation for all events.



## Software Environment

The pipeline is implemented in **Python 3.12** and designed to run within a Conda environment. Development and experiments were conducted using GPU acceleration, though most components can also run on CPU for smaller-scale use.

Key software components include:

- **LLMs & Retrieval-Augmented Generation**
  - `transformers`, `accelerate`, `langchain`
  - `openai`, `tiktoken`
  - `torch` (with optional CUDA support), `deepspeed`, `bitsandbytes`

- **Data processing & evaluation**
  - `pandas`, `numpy`, `scikit-learn`, `scipy`
  - `datasets`, `evaluate`, `bert-score`

- **Knowledge graphs & NLP**
  - `networkx`, `rdflib`, `pykeen`
  - `nltk`, `sentencepiece`, `rapidfuzz`

- **Geospatial processing**
  - `geopandas`, `shapely`, `pyproj`, `rasterio`, `osmnx`

- **Visualization & interfaces**
  - `matplotlib`, `seaborn`, `plotly`, `pyvis`
  - `jupyterlab`, `gradio`, `fastapi`

The **complete list of dependencies and exact package versions** required to reproduce the environment is provided in the Conda environment file (`storylines-env.yml`) included in this repository and archived on Zenodo. This ensures full reproducibility of the pipeline and associated validation experiments.