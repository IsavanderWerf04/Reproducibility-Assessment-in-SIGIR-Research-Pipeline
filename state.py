"""
state.py
Defines the shared state object that flows through the LangGraph pipeline.
 
The state is a TypedDict — a dictionary where each field has a defined type.
Every node in the graph receives the full state and can add to or update it.
"""
 
from typing import TypedDict
 

class PaperState(TypedDict):
    """
    Shared state that flows through all nodes in the pipeline.

    Fields are added progressively as the paper moves through the graph:
    - file_path:          set by main.py before the graph runs
    - paper_text:         set by the Reader node
    - chunks:             set by the Chunker node
    - coverage_ledger:    set by the Extractor node
    - relevant_passages:  set by the Extractor node
    - results:            set by the Inspector nodes
    """
    file_path: str       # path to the .md file, e.g. "../data_pipeline/test_markdown/test1.md"
    paper_text: str      # full markdown text of the paper
    chunks: list         # list of paragraph strings split from paper_text [{"id": "c001", "text": "..."}]
    coverage_ledger: dict # {"processed": [...], "relevant": [...]}
    relevant_passages: dict # passages with IDs for inspector
    results: dict           # "Y", "N", or "U" + explanation of the Inspector's decision + list of all the evidence found for label and reasoning  [{"passage_id": ..., "verbatim_text": ..., "citation": ...}]
  