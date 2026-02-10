# crisesStorylinesRAG

![Example Storyline and Knowledge Graph](AI4DRM_KG.png)

## Overview

`crisesStorylinesRAG` is a pipeline designed to **augment disaster event information with factual storylines and knowledge graphs** at scale. By combining **Large Language Models (LLMs)** with **Retrieval-Augmented Generation (RAG)** on news data from the **European Media Monitor (EMM)**, we generate structured representations of hazards, impacts, drivers, and responses for each event.

The pipeline produces:

- **Storylines**: coherent textual summaries extracted from multiple news sources  
- **Knowledge Graphs (KGs)**: condensed, structured representations of key causal or preventive relationships between entities in each disaster

This approach enables fast, interpretable insights into global disasters and supports research, situational awareness, and downstream analytical workflows.

---

## Data Sources

### European Media Monitor (EMM)

News articles are retrieved from the [European Media Monitor (EMM)](https://emm.newsbrief.eu/NewsBrief/alertedition/en/ECnews.html), which provides large-scale, near–real-time monitoring of global news. The pipeline uses **semantic search** to retrieve articles relevant to each disaster event.

### EM-DAT

Historical disaster events are sourced from [EM-DAT](https://www.emdat.be/). For each event, basic metadata (disaster type, country, affected location, and date range) are used to guide and constrain the search on EMM.

> **Note:** Although EM-DAT is used in this study, the pipeline is generic and can be applied to any structured list of events with minimal metadata (event type, location, and date).

---

## LLM Models and Access Requirements

All LLM-based processing is performed via the [GPT@JRC service](https://gpt.jrc.ec.europa.eu/).

### Model Used

In our experiments, all LLM calls were executed using:

- **`llama-3.3-70b-instruct`**

This model is used for:

- Generating disaster storylines from retrieved news (RAG)  
- Constructing causal knowledge graphs from the generated storylines

No fine-tuning is required; the pipeline relies entirely on **prompt-based inference**.

### Access Requirements

To run the full pipeline, users must have:

- Valid credentials for the **GPT@JRC** service  
- Authorization to query **`llama-3.3-70b-instruct`** (or a compatible instruction-following LLM)

---

## Conceptual Pipeline Overview

At a high level, `crisesStorylinesRAG` follows a two-stage LLM workflow driven by external evidence:

1. **Evidence retrieval**: Disaster metadata are used to retrieve relevant news articles from EMM via semantic search  
2. **Storyline synthesis (LLM + RAG)**: Retrieved news chunks are summarized into a single, coherent disaster storyline, grounded exclusively in the retrieved evidence  
3. **Knowledge graph construction (LLM)**: The generated storyline is converted into a compact **causal knowledge graph**, focusing on drivers, impacts, and preventive factors  
4. **Independent validation (triplets)**: Factual triplets are extracted and evaluated separately by experts to quantify precision and inter-annotator agreement; these triplets do **not** feed back into storylines or graphs

This ensures interpretability and faithfulness of the main outputs, while enabling rigorous quantitative validation.

---

## Pipeline Workflow (Implementation)

The workflow is implemented in the main script: emmRAG_pipeline.py


   
   