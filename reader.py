
"""
reader.py
The Reader node in the LangGraph pipeline.
 
Receives the state with file_path set, reads the markdown file,
and returns the paper text to be added to the state.
"""
 
from state import PaperState
 
 
def reader_node(state: PaperState) -> dict:
    """
    Reads the markdown file at state["file_path"] and returns the text.
    LangGraph will automatically merge {"paper_text": ...} into the state.
    """
    file_path = state["file_path"]
 
    print(f"[Reader] Reading {file_path}...")
 
    with open(file_path, "r", encoding="utf-8") as f:
        paper_text = f.read()
 
    print(f"[Reader] Loaded {len(paper_text):,} characters.")
 
    return {"paper_text": paper_text} 