# LLM-LogHGC
LLM-LogHGC: Hypergraph Enhanced Unsupervised Log Parsing with Lightweight LLMs

This project combines local hypergraph representation learning, LLMs, and a template cache to replace variable log content with `<X>` and produce log templates. 

## Project Files

| File<br>          | Purpose<br>                                                  |
| ----------------- | ------------------------------------------------------------ |
| `main.py`         | Data preprocessing, neighborhood sampling, prompt construction, template validation, retry processing, and CSV export<br> |
| `MAHGCN.py`       | Token normalization, masked hypergraph representation learning, contrastive training, and similar-log selection<br>token |
| `llm.py`          | Connection checks and text generation requests               |
| `temple_cache.py` | Template storage, matching, structural grouping, and canonical template management<br> |



## Environment Setup

The main program requires Python and the following third-party packages:

```powershell
python -m pip install numpy pandas torch requests
```

Before parsing, start Ollama and make the model specified in `llm.py` available. 



## Input and Execution

Run from the project root:

在项目根目录运行：

```powershell
cd E:\LLM_LogHGC
python main.py
```

 Input files follow this naming convention:

```text
datasets/<dataset>/<dataset>_full.log_structured.csv
```



## Parsing Workflow

1. Load or build the log token ID cache and the inverted index mapping tokens to log indices.

2. Match each log against existing templates. Save the result immediately on a cache hit.

3. Retrieve neighbors for unmatched logs using shared tokens. 

4. Initialize and train a model on the local hypergraph, compute log embeddings and raw cosine similarities.

5. Include the target and reference logs in a prompt asking the LLM to replace variables with `<X>`.

6. Extract the output list, apply postprocessing, and validate its content and shared structure. Allow up to three attempts when output parsing or validation fails.

7. Export results in input row order. 

   

   Neighborhood sampling and model initialization do not use fixed random seeds, so selected reference logs may differ between runs.

   

## Output 

Results are saved to:

```text
datasets/<dataset>/<dataset>_parsed_results.csv
```

Results are written after parsing finishes. Running again with the same output path overwrites the previous file. Group IDs belong to the current run's cache: they need not equal ground-truth `EventId` values and are not guaranteed to remain stable across runs.

