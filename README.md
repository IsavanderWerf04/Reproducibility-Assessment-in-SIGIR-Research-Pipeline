# Reproducibility Assessment in SIGIR Research Pipeline

This repository contains the code accompanying the bachelor thesis:

> **Automated Reproducibility Assessment of Scientific Papers using a Multi-Agent Large Language Model Pipeline**

The project investigates whether large language models can automatically assess reproducibility criteria in scientific papers. The pipeline operationalizes a subset of the JAIR reproducibility checklist and applies it to papers from the SIGIR conference.

## Overview

The pipeline consists of five components:

1. **Reader**
   Reads a Markdown version of a scientific paper.

2. **Chunker**
   Splits the paper into overlapping chunks while preserving traceability through chunk identifiers.

3. **Extraction Agent**
   Uses GPT-4o to inspect all chunks and extract criterion-relevant passages.

4. **Repository Verification Component**
   Enriches extracted evidence with license information obtained from linked GitHub repositories.

5. **Classification Agent**
   Assigns reproducibility labels together with supporting evidence and reasoning.

The architecture is implemented using LangGraph.

## Installation

Clone the repository:

```bash
git clone https://github.com/IsavanderWerf04/Reproducibility-Assessment-in-SIGIR-Research-Pipeline.git
cd Reproducibility-Assessment-in-SIGIR-Research-Pipeline
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file based on `.env.example`:

```text
OPENAI_API_KEY=your_openai_api_key
GITHUB_TOKEN=your_github_token
```

## Input Format

The pipeline operates on scientific papers converted to Markdown format.

PDF files can be converted using:

```bash
python convert.py
```

which uses the `pymupdf4llm` library.

## Running the Pipeline

Example:

```python
from pipeline import pipeline

result = pipeline.invoke({
    "file_path": "paper.md"
})
```

The output consists of criterion-level assessments together with supporting evidence and reasoning.

## Implemented Criteria

* G2, G3, G4
* T1, T2, T3
* C2, C5, C6, C7, C9, C12, C15
* D4, D5, D6

Labels:

* **Y** – Yes
* **N** – No
* **U** – Uncertain
* **NA** – Not Applicable

## Data Availability

Copyright restrictions prevent redistribution of the ACM papers used in the experiments. Consequently, PDFs and generated Markdown files are not included in this repository. Users should obtain the papers separately before running the pipeline.

