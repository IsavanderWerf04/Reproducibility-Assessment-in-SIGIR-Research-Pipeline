"""
inspector.py
The Inspector node in the LangGraph pipeline.

Receives the state with relevant_chunks set, sends them to GPT-4o,
and judges whether all criteria iares satisfied.
Returns a label (Y/N/U) and reasoning.
"""

import json
import re
import time
from openai import OpenAI
from dotenv import load_dotenv
from state import PaperState

load_dotenv()

# rules per criterion
RULES = {
"G2": """
    Main question: "Does the paper clearly explain how the presented evidence supports the claims being investigated?"

    Y if:
    The paper presents evidence (experiments, analyses, proofs, evaluations, etc.) supporting the claims being investigated
    AND
    The papers contains an explanation of how this evidence supports those claims.

    Important:
    - Evidence does not have to be experimental. Analyses, evaluations, proofs, 
      observations, case studies, deployment experiences, or lessons learned 
      all count as evidence.

    N if:
    The paper presents results or analyses but does not explain how they support the claims
    OR
    leaves the connection between evidence and claims implicit or unclear.""",

"G3": """
    Main question: "Are the limitations or technical assumptions underlying the work explicitly stated and described clearly enough to understand their meaning and scope?"

    Y if:
    The paper explicitly communicates limitations or technical assumptions
    AND
    describes them clearly enough to understand their meaning and scope.
    important: 
    - Limitations do not have to be explicitly labeled as limitation, as long as they are clearly communicated and their meaning and scope is understandable.

    N if:
    The paper discusses limitations or technical assumptions, but they are vague, unclear, or only partially explained;
    OR
    important technical assumptions are only implied.

    UNCERTAIN if:
    No limitations or technical assumptions can be identified. """,

"G4": """
    Main question: "Are the Conceptual outlines and/or pseudo-code descriptions of the AI method provided and are important implementation details discussed?"

    1. Determine whether the paper proposes or implements any AI methods.
    
    Important: this includes:
    - novel architectures or algorithms
    - fine-tuning or adapting existing models with new configurations or objectives
    - combining existing components in a novel way
    
    Only return NA if the paper purely evaluates or discusses existing methods 
    without any implementation or adaptation of its own, otherwise continue.

    2 
    Y if:
    For the proposed AI method(s) in the paper:
    a conceptual explanation, algorithm description, architecture explanation, or pseudo-code is provided
    AND
    implementation-related details necessary to understand or reproduce the method are provided.
    
    N if:
    The paper proposes one or more AI methods, but neither a conceptual explanation nor implementation-related details are provided.
    
    U if:
    Only one of the two required elements is present.
    Examples include:
    * a conceptual explanation is provided but implementation details are insufficient;
    * implementation details are provided but the method explanation is vague;
    * diagrams or figures are provided without sufficient textual explanation;
    * relevant details are only partially available or deferred to supplementary material.

    NA if:
    The paper does not propose any AI method and only applies, evaluates, or discusses existing methods.""",

"T1": """
    Main question: "Does the paper contain a theoretical contribution, and if so, are all assumptions and restrictions stated clearly and formally?"

    1. Determine if the paper contains a theoretical contribution. 
    
    IMPORTANT: A mathematical definition, formula, metric, loss function, or 
    evaluation measure does NOT constitute a theoretical contribution, even if 
    it involves equations or formal notation.

    A theoretical contribution 
    requires at least one of the following:
    - an explicitly labeled theorem, proposition, lemma, or corollary
    - a formal proof or proof sketch
    - a convergence guarantee or regret bound with mathematical derivation
    - a complexity analysis with formal claims
    if NO theoretical contribution is present → return NA, otherwise continue.
    
    2. 
    Y if:
    The paper contains a theoretical contribution
    AND
    all assumptions and restrictions underlying the theoretical contribution are:
    explicitly communicated
    AND
    described with sufficient clarity and formality to understand their meaning and scope.
   
    Note: concepts that are mathematically defined, operationalized, or directly justified 
    may themselves constitute the formal communication of an assumption and should not 
    automatically be treated as unstated assumptions.
   
    N if:
    The paper contains a theoretical contribution, but:
    assumptions or restrictions are communicated vaguely, ambiguously, unclearly, or informally;
    OR
    assumptions or restrictions are implied but not explained.

    NA if: 
    No theoretical contribution
   
    Important:
    - Do NOT assign NA simply because no theoretical passages were extracted — if there is 
      evidence of mathematical or formal content, the label must be Y or N, not NA.
    - A single theorem or formal result is sufficient to constitute a theoretical contribution.""",

"T2": """
    Main question: "Does the paper contain a theoretical contribution? If yes, are all novel theoretical claims stated through explicit formal statements rather than being described only in ordinary text?"

    1. Determine if the paper contains a theoretical contribution. 
    
    IMPORTANT: A mathematical definition, formula, metric, loss function, or 
    evaluation measure does NOT constitute a theoretical contribution, even if 
    it involves equations or formal notation.

    A theoretical contribution 
    requires at least one of the following:
    - an explicitly labeled theorem, proposition, lemma, or corollary
    - a formal proof or proof sketch
    - a convergence guarantee or regret bound with mathematical derivation
    - a complexity analysis with formal claims
    if NO theoretical contribution is present → return NA, otherwise continue.
    
    2. 
    Y if:
    The paper contains a theoretical contribution and:
    every novel theoretical claim is stated formally
    AND
    claims are expressed in a sufficiently precise manner to unambiguously identify the statement being established
    AND
    claims appear as theorem statements, propositions, lemmas, corollaries, definitions, or equivalent formal statements.

    N if:
    The paper contains a theoretical contribution but:
    one or more important novel theoretical claims are stated only in ordinary text
    OR
    formal statements are absent or too vague to clearly identify the claim being established

    NA if: 
    No theoretical contribution""",

"T3": """
    Main question: "If the paper contains a theoretical contribution, are all non-trivial claims accompanied by proofs or proof sketches that provide sufficient detail to verify the claims?"

    1. Determine if the paper contains a theoretical contribution. 
    
    IMPORTANT: A mathematical definition, formula, metric, loss function, or 
    evaluation measure does NOT constitute a theoretical contribution, even if 
    it involves equations or formal notation.

    A theoretical contribution 
    requires at least one of the following:
    - an explicitly labeled theorem, proposition, lemma, or corollary
    - a formal proof or proof sketch
    - a convergence guarantee or regret bound with mathematical derivation
    - a complexity analysis with formal claims
    if NO theoretical contribution is present → return NA, otherwise continue.
    
    2. 
    Y if: 
    The paper contains a theoretical contribution and:
    every non-trivial claim has proof or proof sketch
    AND 
    the reasoning is detailed enough for a knowledgeable reader to verify the claim.
    
    note: Proofs may be presented through formal derivations, mathematical notation, diagrams, figures, or appendices and do not need to consist solely of prose. 

    N if: 
    The paper contains a theoretical contributions, but:
    proof of non-trivial claims consists only of reference to intuition 
    OR 
    major proof steps are omitted

    U if: 
    The paper contains a theoretical contributions, but:
    important theoretical claims have no proof

    NA if: 
    No theoretical contribution""",

"C2": """
    Main question: “Can another researcher legally access and reuse the source code for reproducibility purposes?”

    1. Determine if source code is explicitly provided or linked.
   A GitHub link should be accompanied by a statement that the code is available at that link.

    2. 
    Y if:
      Source code is accompanied by an explicit license permitting free use for reproducibility purposes.
    For public repositories, count the code as licensed only if the repository contains:
    * a LICENSE file; OR
    * an explicit licensing statement in the repository or README.
    
    Examples of acceptable licenses include:
    * MIT
    * Apache 2.0
    * GPL
    * BSD
    * other licenses that explicitly permit free reproducibility use.
    
    N if:
    Source code is mentioned or available, but:
    no license is provided;
    OR
    licensing restrictions prevent free reproducibility use;
    OR
    access to the code is restricted.
    
    Examples include:
    
    * "available upon request";
    * private repository;
    * commercial-only license;
    * non-redistributable access;
    * repository link without any licensing information.
    
    NA if:
    No source code is mentioned, linked, provided, or referenced anywhere in the paper.""",

"C5": """
    Main Question: "If raw, unaggregated experimental data is made available, can another researcher legally use the raw data to reproduce the results reported in the paper?"
    
    1. Determine if raw, unaggregated data is shared or mentioned
    if NOT return NA, otherwise continue

    2.
    Y if: 
    raw, unaggregated experimental data is made available;
    AND
    the data comes with an explicit license permitting free use for reproducibility purposes.

    N if:
    raw, unaggregated experimental data is available but no license is provided
    OR
    the license does not permit free use for reproducibility purposes.

    NA if:
    no raw, unaggregated experimental data is shared or mentioned. """,

"C6": """
    Main question: "If raw, unaggregated experimental data is made available, can researchers use the raw data for other scientific studies beyond reproducing this paper?"

    1. Determine if raw, unaggregated data is shared or mentioned
    if NOT return NA, otherwise continue

    2. 
    Y if:
    raw, unaggregated experimental data is made available;
    AND
    the data comes with an explicit license permitting free use for research purposes in general.

    N if:
    raw, unaggregated experimental data is available but no license is provided
    OR
    the license does not permit free use for research purposes in general.

    NA if:
    no raw, unaggregated experimental data is shared or mentioned. """,

"C7": """
    Main question: "When randomness influences the algorithm or experiments, does the paper describe the method for controlling random numbers and seeds sufficiently for reproducibility?"

    1. Determine if the algorithm or experimental setup depends on randomness: 
    if randomness is NOT relevant to the described method or experiments → return NA, 
    otherwise continue.
   
    2. 
    Y if:
    The algorithm or experimental setup depends on randomness for reproducibility
    AND
    the paper provides a sufficiently clear description of how randomness is handled,
    such that another scientist could reproduce the experiment in terms of randomness.
    Exact numerical seed values are not required. The following are considered sufficient:
    - stating the number of independent runs with averaged results
    - stating the number of runs with different random seeds and averaged results
    - describing a fixed seed procedure (even without the exact value)
    - describing deterministic settings or framework-level controls

    N if:
    The algorithm or experimental setup depends on randomness for reproducibility,
    BUT
    the method for generating, controlling, or managing randomness is missing, vague, 
    or insufficiently described. 

    The following are NOT sufficient:
    - a vague acknowledgement that randomness was controlled without any description
    - no mention of runs, seeds, or randomness control at all

   Important:
   - Implicit randomness common to all neural network implementations (e.g. standard weight 
     initialization, data shuffling) does NOT count as explicitly relevant unless the paper 
     itself describes it as such.
   - Do NOT assign NA simply because no randomness control is described — if randomness is 
     explicitly relevant, the label must be N, not NA.""",

"C9": """
    Main question: "Are the evaluation metrics clearly explained, and is their choice explicitly justified?"

    Y if:
    The paper explains what the metrics measure or how they are computed or cites a standard definition
    AND
    the paper provides an explanation for why these metrics are appropriate for the task or claims being evaluated.

    N if:
    the paper does not provide an explanation or citation of what the metric measures
    AND 
    no explicit motivation for choosing the metric is provided.

    UNCERTAIN if:
    There is an explanation of the metrics, but no motivation 
    OR
    There is a motivation, but no explanation.

    IMPORTANT:
    - for the category UNCERTAIN there HAS to be one of the two: a clear explanation or a clear motivation. 

    NA if: No computational experiments """,

"C12": """
    Main question: "Does the analysis of results go beyond single-dimensional summaries of performance by including measures of variation, confidence, statistical significance, or other distributional information?"

    1. Determine if the design calls for variation, confidence or distributional reporting.
    if NOT return NA, otherwise continue

    2.
    Y if:
    the experimental design calls for variation reporting
    AND
    the paper includes explicit measures of variation such as standard deviations,
    confidence intervals, error bars, or results from multiple runs showing variability.
    
    Important:
    - Significance tests (t-tests, p-values) alone do NOT satisfy this criterion.
    - Y requires actual variation measures alongside results, not just significance markers.

    N if:
    the experimental design calls for variation, confidence, or distributional reporting.
    AND
    the paper reports only single-dimensional summaries of performance;
    AND
    provides no information about variation, uncertainty, or distribution

    NA if: 
    the experimental design does not call for variation, confidence, or distributional reporting.
    
    Important:
    - Fixed benchmark evaluation alone is not sufficient to conclude NA,  the training process also needs to be considered """,

"C15": """
    Main question: "When experiments are affected by randomness or noise, are statistical hypothesis tests or equivalent significance analyses used to establish significance?"

    1. Determine if the experimental setup is affected by randomness or noise.
    if NOT return NA, otherwise continue.

    2. 
    Y if:
    the experiments are affected by randomness or noise;
    AND
    the paper uses statistical hypothesis tests or equivalent significance analyses to assess differences between results. 
    
    Examples:
    
    t-test
    Wilcoxon test
    Mann-Whitney test
    permutation tests
    bootstrap significance tests
    
    IMPORTANT: 
    - stating results of significance testing is sufficient

    N if:
    experiments are affected by randomness or noise;
    AND
    no statistical hypothesis tests or significance analyses are reported.
    
    Examples:
    - reporting only average scores;
    - claiming improvements without significance analysis."

    "NA if:  The experimental setup does not involve the kind of randomness or noise that statistical hypothesis tests are designed to address.
    
    important:
    - Fixed benchmark evaluation alone is not sufficient to conclude NA: the training process also needs to be considered """,

"D4": """
    Main question: "When datasets originate from external or public sources, does the paper properly cite those datasets?"
    
    1. Determine if any datasets are used and drawn from literature or public sources.
    if NOT return NA, otherwise continue

    2.
    Y if:
    One or more datasets from the literature or public sources are used
    AND
    the paper provides explicit citations or references identifying the dataset source appropriately.

    note : A dataset name followed by a reference number or citation should satisfy this criterion.
    
    N if:
    Datasets from the literature or public sources are used,
    BUT
    citations or references identifying the dataset source are missing, vague, or insufficient.
    
    NA if:
    No datasets from the literature or public sources are used or mentioned.""",

"D5": """
    Main question: "If the paper uses datasets from the existing literature, are all such datasets publicly available?"

    1. determine if any datasets are used and drawn from existing literature.
    if NOT return NA, otherwise continue

    2.  
    Y if:
    the paper uses one or more datasets from the existing literature;
    AND
    all such datasets are publicly available.

    N if:
    the paper uses one or more datasets from the existing literature;
    AND
    at least one dataset is private, proprietary, or otherwise inaccessible to the public.

    Important:
    - Public availability does not need to be explicitly stated.
    - Well-known benchmark datasets may be treated as publicly available if they 
      are cited or clearly identified by name.
    - Only label N if there is explicit evidence that a dataset is private or restricted.

    NA if:
    the paper does not use datasets from the existing literature. """,

"D6": """
    Main question: "If the paper introduces new datasets or uses non-public datasets, are these datasets described in sufficient detail, including relevant statistics, the data collection process, and the annotation process when applicable?"

    1. Determine if the paper introduces new datasets or uses datasets that are not publicly available
    if NOT return NA, otherwise continue

    2. 
    Y if:
    the paper introduces new datasets or uses datasets that are not publicly available;
    AND
    describes them sufficiently, including relevant statistics and the data collection process;
    AND
    describes the annotation process when annotation is relevant.

    N if:
    the paper introduces new datasets or uses datasets that are not publicly available;
    and
    important information regarding statistics, collection, or annotation is missing.
 
    NA if:
    the paper introduces no new datasets and uses no non-public datasets. """
}

SYSTEM_PROMPT = """You are a scientific reproducibility assessor for SIGIR research papers.

You will receive extracted passages from one paper, grouped by reproducibility criterion.
Your task is to assign a label for each criterion independently.

Consistency rules between criteria — apply these after judging individually:
- C7 determines whether randomness is relevant to the paper.
- If C7 = NA: C12 and C15 must also be NA.
- If C7 = Y or N: C12 and C15 cannot be NA.

Important rules:
1. Judge only from the provided extracted passages.
2. Do not use outside knowledge about the paper, dataset, method, venue, or GitHub repository.
3. Evaluate each criterion independently.
Only use passages listed under that criterion.
Ignore passages listed under other criteria..
5. If no relevant passages are provided for a criterion, do not automatically return NA.
Return NA only when there is explicit evidence that the criterion does not apply.
Exception: the C7/C12/C15 consistency rules above override this.Return NA only when there is explicit evidence that the criterion does not apply and return this evidence. 
6. Evidence must be copied verbatim from the provided passages.

Return only valid JSON.
Do not include markdown fences, comments, or extra text.

The JSON object must contain one key for every criterion provided.

For each criterion return:

{
  "label": "Y" | "N" | "U" | "NA",
  "reasoning": "...",
  "evidence": [...]
}

If no supporting evidence is available, use an empty evidence list.


"""

USER_TEMPLATE = """Judge all criteria below based only on the extracted passages.

For each criterion:
- First decide whether the criterion applies.
- Then decide whether the required information is present and sufficiently clear.

Extracted passages and rules:

{criteria_and_passages}
"""

# inspector node calling gpt-4o. 
# input is formatted passages and output is json format
# with label Y/N/U/NA and reasoning field. 
def inspector_node(state: PaperState) -> dict:
    print("[Inspector] Judging all criteria in one call...")

    # build criteria and passages block
    criteria_and_passages = ""
    for criterion_id, rules in RULES.items():
        passages = state["relevant_passages"].get(criterion_id, [])

        criteria_and_passages += f"\n{'='*40}\n"
        criteria_and_passages += f"CRITERION {criterion_id}:\n{rules}\n"
        criteria_and_passages += f"\nRELEVANT PASSAGES:\n"

        if passages:
            for p in passages:
                criteria_and_passages += f"\n[{p.get('passage_id', '?')}]\n{p.get('verbatim_text', '')}\n"
        else:
            criteria_and_passages += "(No relevant passages found.)\n"

    client = OpenAI()

     # retry up to 3 times if JSON parsing fails
    for attempt in range(3):
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(
                    criteria_and_passages=criteria_and_passages
                )}
            ],
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            results = json.loads(raw)
            break  
        except json.JSONDecodeError as e:
            print(f"[Inspector] JSON parse failed on attempt {attempt + 1}: {e}")
            if attempt < 2:
                print(f"[Inspector] Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print("[Inspector] All attempts failed, returning U for all criteria")
                results = {c: {"label": "U", "reasoning": "JSON parse error after 3 attempts", "evidence": []} for c in RULES}

    for criterion_id, res in results.items():
        print(f"[Inspector] {criterion_id}: {res.get('label')} — {res.get('reasoning')}")

    return {"results": results}