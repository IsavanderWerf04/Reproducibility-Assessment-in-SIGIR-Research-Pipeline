"""
chunker.py
The Chunker node in the LangGraph pipeline.

Receives the state with paper_text set, splits it into fixed chunks
using LangChain's RecursiveCharacterTextSplitter, and assigns each
chunk a unique ID (c001, c002, ...).

Fixed chunking ensures same PDF always produces same chunks — 
this is the foundation of deterministic coverage.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from state import PaperState

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def chunker_node(state: PaperState) -> dict:
    """
    Splits the paper text into fixed chunks with unique IDs.
    LangGraph will automatically merge {"chunks": [...]} into the state.
    """
    print("[Chunker] Splitting paper into fixed chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    ) 

    raw_chunks = splitter.split_text(state["paper_text"])

    # merge short chunks into the next chunk
    # so location info is preserved
    merged = []
    pending = ""
    for chunk in raw_chunks:
        if len(chunk.strip()) < 50:
            pending = chunk.strip() + "\n"
        else:
            merged.append(pending + chunk)
            pending = ""
    if pending:
        merged.append(pending)

    # assign unique IDs to each chunk
    chunks = [
        {"id": f"c{str(i).zfill(3)}", "text": text}
        for i, text in enumerate(merged, 1)
    ]

    print(f"[Chunker] Created {len(chunks)} chunks with IDs c001-c{str(len(chunks)).zfill(3)}.")

    return {"chunks": chunks}