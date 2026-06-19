"""
extractor.py
The Extractor node in the LangGraph pipeline.

Receives all chunks with IDs, sends them in one API call to GPT-4o,
and finds relevant passages for ALL criteria at once using merged
extraction keys where criteria share the same evidence.

Merged keys:
  T_all   → T1, T2, T3
  C5_C6   → C5, C6
  D_all   → D4,D5,D6

Returns:
- coverage ledger proving every chunk was inspected
- relevant passages per criterion for the Inspector node
"""

import json
import re
import time
from openai import OpenAI
from dotenv import load_dotenv
from state import PaperState

load_dotenv()

# extraction keys — merged where criteria share the same evidence
CRITERIA = {
    "G2": """Extract all passages where the paper states its claims, contributions, or research questions, and passages that explain how the experiments, analyses, or evaluations connect to those claims.""",

    "G3": """Extract all passages where the paper discusses what the work cannot do, where it may fail, what it assumes, or where its scope is restricted — whether or not these are explicitly labeled as limitations.""",

    "G4": """Extract all passages describing how the proposed AI method works, including conceptual explanations, architecture descriptions, algorithm steps, training procedures, and implementation details.""",

    "T_all": """Extract all passages containing mathematical or formal content, including theorems, propositions, lemmas, proofs, formal guarantees, complexity analyses, convergence results, formal definitions, and mathematical derivations.""",

    "C2": """Extract all passages mentioning source code, implementations, repositories, or GitHub links, including any statements about code availability, access restrictions, or licensing.""",

    "C5_C6": """Extract all passages mentioning shared or released data, raw or unaggregated experimental data, data repositories, GitHub links where data is stored, data licenses, or statements about whether data can be reused or accessed.""",

    "C7": """Extract all passages describing the experimental setup, training procedure, or evaluation, including any mentions of randomness, seeds, repeated runs, stochastic components, or deterministic settings.""",

    "C9": """Extract all passages mentioning evaluation metrics, including what metrics are used, how they are defined or computed, why they were chosen, or citations to standard metric definitions.""",

    "C12": """Extract all passages reporting experimental results, including result tables, performance comparisons, measures of variation, confidence, significance, distributional information, inter-annotator agreement, and any table captions describing how results were evaluated.""",

    "C15": """Extract all passages describing the experimental setup or results that involve randomness, noise, or stochastic components, including any mentions of statistical significance tests or p-values.""",

    "D_all": """Extract all passages that mention, cite, describe, or discuss datasets used in the paper, including dataset names, sources, citations, availability, statistics, and collection procedures.""",
}

# maps merged extraction keys to the criteria that use them
PASSAGE_MAP = {
    "G2":     ["G2"],
    "G3":     ["G3"],
    "G4":     ["G4"],
    "T_all":  ["T1", "T2", "T3"],
    "C2":     ["C2"],
    "C5_C6":  ["C5", "C6"],
    "C7":     ["C7"],
    "C9":     ["C9"],
    "C12":    ["C12"],
    "C15":    ["C15"],
    "D_all": ["D4", "D5", "D6"]
}

ALL_CRITERIA = ["G2", "G3", "G4", "T1", "T2", "T3", "C2", "C5", "C6",
                "C7", "C9", "C12", "C15", "D4", "D5", "D6"]

SYSTEM_PROMPT = """You are an evidence extractor for reproducibility assessment of SIGIR papers.

You will receive numbered chunks of a paper. Your task is to:
1. Inspect every chunk.
2. Extract verbatim passages about given subjects.
3. A passage may belong to multiple criteria simultaneously.
4. If a passage contains information relevant to several criteria, duplicate the passage under all applicable criteria.
5. Do not prioritize one criterion over another.
6. Do not summarize or rewrite passages.
7. Merge adjacent relevant chunks only when they form one coherent passage.



Return only valid JSON with exactly these fields:
{
  "processed_chunks": ["all chunk IDs inspected"],
  "missed_chunks": ["any skipped chunk IDs, normally empty"],
  "relevant_passages": {
    "G2": [],
    "G3": [],
    "G4": [],
    "T_all": [],
    "C2": [],
    "C5_C6": [],
    "C7": [],
    "C9": [],
    "C12": [],
    "C15": [],
    "D_all": []
  } 
}

Each passage should follow this structure:
{
  "passage_id": "G2_1",
  "source_chunk_ids": ["chunk id"],
  "verbatim_text": "exact copied text"
}

If no relevant passage is found for a criterion, return an empty list for that criterion.
Do not include markdown fences, comments, or extra text."""

USER_TEMPLATE = """Criteria to check:
{criteria}

Inspect every chunk below and identify all passages relevant to any of the descriptions above.

{chunks_text}"""


def extractor_node(state: PaperState) -> dict:
    print("[Extractor] Sending all chunks to GPT-4o for full coverage inspection...")

    # format chunks as labeled text blocks
    chunks_text = ""
    for chunk in state["chunks"]:
        chunks_text += f"\nCHUNK {chunk['id']}:\n{chunk['text']}\n"

    criteria_text = "\n".join([f"- {k}: {v}" for k, v in CRITERIA.items()])

    client = OpenAI()

    # retry up to 3 times if JSON parsing fails
    for attempt in range(3):
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(
                    criteria=criteria_text,
                    chunks_text=chunks_text
                )}
            ],
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            result = json.loads(raw)
            break 
        except json.JSONDecodeError as e:
            print(f"[Extractor] JSON parse failed on attempt {attempt + 1}: {e}")
            if attempt < 2:
                print(f"[Extractor] Retrying in 2 seconds...")
                time.sleep(2)
            else:
                raise  

    coverage_ledger = {
        "processed_chunks": result.get("processed_chunks", []),
        "missed_chunks": result.get("missed_chunks", [])
    }

    
    raw_passages = result.get("relevant_passages", {})

    relevant_passages = {c: [] for c in ALL_CRITERIA}

    for extraction_key, criteria_list in PASSAGE_MAP.items():

        passages = raw_passages.get(extraction_key, [])
        for criterion in criteria_list:
            relevant_passages[criterion] = passages  

    print(f"[Extractor] Processed {len(coverage_ledger['processed_chunks'])} chunks.")
    print(f"[Extractor] Missed {len(coverage_ledger['missed_chunks'])} chunks.")
    total_passages = sum(len(v) for v in raw_passages.values())
    print(f"[Extractor] Found {total_passages} unique passage group(s).")
    for key, passages in raw_passages.items():
        print(f"[Extractor] {key}: {len(passages)} passage(s) → {PASSAGE_MAP[key]}")

    return {
        "coverage_ledger": coverage_ledger,
        "relevant_passages": relevant_passages
    }