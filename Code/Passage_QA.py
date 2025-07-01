import json
import re
import os
from openai import AzureOpenAI
from together import Together
import spacy
import sys

from testner import extract_spans_from_text

try:
     nlp = spacy.load("en_core_web_sm")
except OSError:
     from spacy.cli import download
     download("en_core_web_sm")
     nlp = spacy.load("en_core_web_sm")

# --------------------------------------------------------------------------
# --------------------------  AZURE OPENAI CLIENT  -------------------------
# --------------------------------------------------------------------------
# Azure OpenAI Credentials and Deployment
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "your-azure-endpoint-here")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "your-azure-api-key-here")
AZURE_MODEL_NAME = "gpt-4o-2"

# Initialize Azure OpenAI Client - This client will be used for Span Extraction and Q&A Generation
azure_client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    api_version="2024-05-01-preview",
)

# --------------------------------------------------------------------------
# --------------------------  TOGETHER CLIENT FOR CLASSIFICATION  ----------
# --------------------------------------------------------------------------
together_api_key = os.getenv("TOGETHER_API_KEY", "your-together-api-key-here")

# Use the Llama 3.3 model for classification
LLAMA_CLASSIFICATION_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

# Initialize Together Client - This client will be used ONLY for Classification
together_client = Together(api_key=together_api_key)


# --------------------------------------------------------------------------
# ---------------------------  JSON PARSING HELPER (Improved) ---------------
# --------------------------------------------------------------------------
def extract_first_json_block(text: str) -> str:
    """
    Extract the first valid top-level JSON object or array from a string by manually parsing braces/brackets.
    Handles nested structures, strings, and escaped quotes.

    Args:
        text (str): The input string potentially containing JSON.
    Returns:
        str: The first JSON object/array substring, or an empty string if none found.
    """
    first_brace = text.find('{')
    first_bracket = text.find('[')

    if first_brace == -1 and first_bracket == -1:
        return ""

    # Determine the starting position and type ('{' or '[')
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        start = first_brace
        start_char = '{'
        end_char = '}'
    elif first_bracket != -1:
        start = first_bracket
        start_char = '['
        end_char = ']'
    else:
         return ""

    level = 0
    in_string = False
    escaped = False
    i = start
    while i < len(text):
        char = text[i]

        if char == '"' and not escaped:
            in_string = not in_string
        elif char == '\\':
            escaped = not escaped
        elif not in_string:
            if char == start_char:
                level += 1
            elif char == end_char:
                level -= 1
                if level == 0:
                    return text[start:i+1]

        if char != '\\':
           escaped = False

        i += 1
    return ""


def safe_json_parse(raw_text: str):
    """
    1. Strip typical code fences/triple backticks and language identifiers.
    2. Extract only the first valid JSON object or array block.
    3. Parse it into a Python object.

    Returns either the parsed object (dict or list) or:
      {"error": "...", "raw": raw_text}
    on failure.
    """
    text = raw_text.strip()
    match = re.match(r"^```(?:json)?\s*(.*)\s*```$", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    json_block = extract_first_json_block(text)
    if not json_block:
        if (text.startswith("{") and text.endswith("}")) or \
           (text.startswith("[") and text.endswith("]")):
             json_block = text
        else:
             return {"error": "No JSON object or array block found in output.", "raw": raw_text}

    try:
        json_block_corrected = json_block.replace(r'\_', '_')
        return json.loads(json_block_corrected)
    except json.JSONDecodeError as e:
        error_context_start = max(0, e.pos - 30)
        error_context_end = min(len(json_block), e.pos + 30)
        context_snippet = json_block[error_context_start:error_context_end].replace('\n', ' ')
        return {
            "error": f"Failed to parse JSON: {str(e)} near '{context_snippet}' (position {e.pos})",
            "raw_block": json_block,
            "raw_full": raw_text
        }

# --------------------------------------------------------------------------
# -----------------------  CLASSIFICATION FUNCTION (Refined Prompt) --------
# --------------------------------------------------------------------------
def classify_with_llm(title, subtitle, content):
    """
    Classifies a chunk into one of: EU Taxonomy, ESG, Sustainability, or Unknown.
    Uses refined prompt for better distinction.
    """
    prompt = f"""
    Title: {title or 'N/A'}
    Subtitle: {subtitle or 'N/A'}
    Context: {content}
    
    Based primarily on the Context provided, classify this text chunk into one of the following categories. Use the Title and Subtitle for additional context if needed. Return *only* the single, most appropriate class label (no explanation, reasoning, or punctuation). If specific terms (e.g., EU Taxonomy regulations) are present, prioritize them over broader themes.
    
    Categories:
    - EU Taxonomy: Specifically mentions EU Taxonomy regulations, principles, alignment criteria, eligibility, screening criteria, DNSH (Do No Significant Harm), Minimum Safeguards, or related KPIs (CapEx, OpEx, Turnover alignment).
    - ESG: Focuses on Environmental, Social, or Governance factors, metrics, risks, opportunities, reporting frameworks (like GRI, SASB, TCFD, CSRD), materiality assessments, stakeholder engagement, policies, or specific ESG initiatives not solely confined to broad sustainability concepts.
    - Sustainability: Covers broader sustainability topics like circular economy, climate action goals (without specific ESG framework detail), biodiversity efforts, resource efficiency, sustainable products/practices, or general corporate responsibility themes not fitting neatly into ESG reporting structures or EU Taxonomy specifics.
    - Unknown: Does not clearly fit into any of the above categories or lacks sufficient information for classification.
    """

    response = together_client.chat.completions.create(
        model=LLAMA_CLASSIFICATION_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert assistant specializing in accurately classifying text chunks related to corporate reporting into predefined categories: EU Taxonomy, ESG, Sustainability, or Unknown. Your response must be only the single chosen category label."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=10,
        temperature=0.1,
        top_p=1,
        stop=None,
    )
    classification_raw = response.choices[0].message.content.strip()

    valid_classes = ["EU Taxonomy", "ESG", "Sustainability", "Unknown"]
    if classification_raw in valid_classes:
        return classification_raw
    else:
        print(f"Warning: LLM returned unexpected classification '{classification_raw}'. Defaulting to Unknown.")
        return "Unknown"

# -------------------  NEW HELPER FUNCTIONS --------------------------------
def _dedup_and_verify_spans(raw_groups, context):
    """
    Cleans and verifies spans from LLM output.
    1. Verifies each span exists verbatim (case-insensitive) in the context.
    2. Deduplicates spans case-insensitively *across all groups*, keeping the first-encountered canonical form.
    3. Performs intra-group refinement: removes shorter spans if they are substrings of longer spans *within the same group*.
    4. Formats group labels (Title Case).
    5. Ensures only non-empty groups are retained.
    """
    if not isinstance(raw_groups, list):
        return []

    ctx_lower = context.lower()
    globally_seen_spans_map = {}
    
    intermediate_cleaned_groups = []

    for g in raw_groups:
        label = g.get("label", "Ungrouped")
        spans_for_this_group_after_global_dedup = []

        for s_raw in g.get("spans", []):
            s_clean = s_raw.strip()
            if not s_clean:
                continue
            
            if s_clean.lower() not in ctx_lower:
                continue
            
            s_lower = s_clean.lower()
            if s_lower not in globally_seen_spans_map:
                globally_seen_spans_map[s_lower] = s_clean
                spans_for_this_group_after_global_dedup.append(s_clean)

        if spans_for_this_group_after_global_dedup:
            intermediate_cleaned_groups.append({
                "label": label,
                "spans": spans_for_this_group_after_global_dedup
            })
        
    final_verified_groups = []
    for group_data in intermediate_cleaned_groups:
        label = group_data["label"]
        current_spans_in_group = group_data["spans"]

        if not current_spans_in_group:
            continue

        spans_to_keep_in_group = list(current_spans_in_group)

        for s1 in current_spans_in_group:
            for s2 in current_spans_in_group:
                if s1 == s2:
                    continue
                
                if s1 in s2 and s1 in spans_to_keep_in_group:
                    spans_to_keep_in_group.remove(s1)
        
        final_spans_for_this_group = sorted(list(set(spans_to_keep_in_group)))

        if final_spans_for_this_group:
            final_verified_groups.append({
                "label": label.title(),
                "spans": final_spans_for_this_group
            })
           
    return final_verified_groups


# --------------------------------------------------------------------------
# -----------------------  SPAN EXTRACTION FUNCTION (Refined Prompt) -------
# --------------------------------------------------------------------------
def extract_spans(classification, content, ner_entities=None):
    """
    Two-pass span extraction pipeline. Focus of this change is Prompt B.
    """

    if ner_entities is None:
        raw_ner_output = extract_spans_from_text(content) 
        structured_hints_list = raw_ner_output.get("spans", []) if isinstance(raw_ner_output, dict) else []
    else:
        structured_hints_list = ner_entities if isinstance(ner_entities, list) else []
    rule_based_hints = [h["text"] for h in structured_hints_list if isinstance(h, dict) and "text" in h]

    prompt_pass_a = f"""
    You are an assistant trained to extract concise, meaningful spans relevant to a given classification.
    
    Context: "{content}"
    Classification: "{classification}"
    
    Instructions:
    *Extract Key Spans:**
       - Identify precise, meaningful spans strictly relevant to the Classification.
       - **Specifically include substantive details:**
         - Entities, criteria, activities, and technical concepts.
         - Quantitative data (e.g., '50% reduction', '10 MWh', '€5 million').
         - Specific dates/timeframes (e.g., 'by 2030', 'FY2023').
         - Named standards/initiatives (e.g., 'GRI Standards', 'TCFD recommendations').
         - Specific regulations/policies (e.g., 'CSRD Article 8', 'EU Taxonomy').
         - Clearly defined risks or targets (e.g., 'net-zero target').
       - **Explicitly exclude overly broad or trivial terms:**
         - Generic terms like:'company', 'organization', 'report', 'year', 'data', 'information', 'impacts', 'approach', 'process', 'management', 'performance', 'strategy', 'framework','article','page','section','annex'.
         - Generic adjectives like ('important', 'significant') unless part of a named entity or specific metric.
         - Common functional phrases like ('in order to', 'as well as', 'responsible for').
       - Extract spans verbatim. Most spans should be **1-5 words**. Critical named entities, regulatory terms, or specific metrics that are naturally slightly longer (e.g., 6-7 words) may be included if they cannot be shortened without losing precise meaning. Do **not** modify, rephrase, or summarize.
       - Prioritize the most complete and specific verbatim phrase that accurately represents the key concept, within the allowed length.
    
    Return format:
    {{ "spans": ["<span 1>", "<span 2>", ...] }}
    """
    response_a = azure_client.chat.completions.create(
        model=AZURE_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a span extractor that prioritizes recall and outputs results in JSON format. Your goal is to find as many valid spans as possible, adhering strictly to the formatting and content guidelines."},
            {"role": "user", "content": prompt_pass_a},
        ],
        temperature=0, max_tokens=512, top_p=1, response_format={"type": "json_object"},
    )
    llm_spans_raw = response_a.choices[0].message.content.strip()
    llm_spans_obj = safe_json_parse(llm_spans_raw)
    llm_spans = llm_spans_obj.get("spans", []) if isinstance(llm_spans_obj, dict) else []
    if not isinstance(llm_spans, list): llm_spans = []

    candidate_spans = sorted(list(set(s.strip() for s in (llm_spans + rule_based_hints) if isinstance(s, str) and s.strip())))
    if not candidate_spans: 
        return {"groups": []}

    candidate_lines = "\n".join(f'- "{s}"' for s in candidate_spans)
    print(candidate_lines)
    
    prompt_pass_b = f"""
    You are an expert grouping and filtering assistant. Your goal is to organize spans into meaningful, specific thematic groups, minimizing use of the "Individual" category.
    
   
    
    Instructions:
        Follow these steps sequentially:
            1.  **Filter & Consolidate:**
               *   Keep only spans found verbatim in Context (case-insensitive, original casing).
               *   Remove trivial/generic standalone spans.
               *   If spans are redundant (substrings, acronym/expansion, exact synonyms conveying the same core idea), keep only the single most complete and informative version. Discard others.
        
           2.  **Thematic Grouping (Primary Focus):**
               *   **Actively group spans** sharing a clear, specific, and meaningful common theme or semantic relationship.
               *   **Form groups even with 2-3 highly related spans.** Do not default to "Individual" if a small, specific group can be formed.
               *   Create a **concise (2-5 words), highly specific `label`** for each group capturing its core theme. Avoid generic labels.
               *   **Prioritize creating thematic groups.**
        
           3.  **"Individual" Category (Use Sparingly):**
               *   Only spans that are **truly disparate and cannot be thematically linked** to any other span (even in a small group) go into the group labeled `"Individual"`.
               *   Before assigning to "Individual", double-check if it could form a small thematic group with another span.
           4.  **JSON Output:** Return ONLY one valid JSON object in this exact format. No extra text.
                {{
                  "groups": [
                    {{
                      "label": "<Group Label>",
                      "spans": ["<Span 1>", "<Span 2>"]
                    }},
                    {{
                      "label": "Individual",
                      "spans": ["<Span>"]
                    }}
                  ]
                 }}
    Context: "{content}"
    Classification: "{classification}"            
    Candidate Spans:"{candidate_lines}"
        
    """

    response_b = azure_client.chat.completions.create(
        model=AZURE_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a highly precise data structuring assistant. Your SOLE task is to critically evaluate candidate spans, filter them based on context and rules, group the valid ones, and output ONLY a valid JSON object in the specified format. Adherence to all rules and the JSON format is critical."},
            {"role": "user", "content": prompt_pass_b},
        ],
        temperature=0.0,
        max_tokens=1024,
        top_p=1,
        response_format={"type": "json_object"},
    )

    raw_grouped_json = response_b.choices[0].message.content.strip()
    grouped_obj = safe_json_parse(raw_grouped_json)

    if isinstance(grouped_obj, dict) and "groups" in grouped_obj:
        verified_groups = _dedup_and_verify_spans(grouped_obj["groups"], content)
        grouped_obj["groups"] = verified_groups
    elif isinstance(grouped_obj, dict) and "error" in grouped_obj:
        return grouped_obj
    else:
        return {"error": "Failed to get valid grouped JSON from LLM Prompt B or parse it correctly.", "raw_llm_output": raw_grouped_json}
    
    return grouped_obj

# --------------------------------------------------------------------------
# ------  FACTOID Q&A GENERATION (Using AzureOpenAI) -------
# --------------------------------------------------------------------------
def filter_factoid_qas(factoid_obj, spans, *, context=None):
    if context is None:
        context = spans.get("_paragraph", "")
    ctx_lower = context.lower()
    filtered = []
    if not isinstance(factoid_obj, dict) or "qa_pairs" not in factoid_obj:
        return {"qa_pairs": []}

    all_spans = []
    combined_answers = []
    if isinstance(spans, dict) and "groups" in spans:
        for group in spans["groups"]:
            if group.get("label") != "Individual":
                combined_answers.append(", ".join(group["spans"]))
            all_spans.extend(group["spans"])

    for qa in factoid_obj.get("qa_pairs", []):
        if not isinstance(qa, dict) or "answer" not in qa or "question" not in qa:
            continue
        ans = qa["answer"].strip()

        if ans in all_spans:
            if ans.lower() not in ctx_lower:
                continue
        elif ans in combined_answers:
            group_spans = next((g["spans"] for g in spans["groups"] if ", ".join(g["spans"]) == ans), None)
            if not group_spans or not all(span.lower() in ctx_lower for span in group_spans):
                continue
        else:
            continue

        if ans.lower() in qa["question"].lower():
            continue

        filtered.append(qa)
    return {"qa_pairs": filtered}


def generate_factoid_qa(classification, content, spans):
    if not isinstance(spans, dict) or "groups" not in spans or not isinstance(spans["groups"], list):
        if isinstance(spans, dict) and "error" in spans:
            return {"error": f"Cannot generate Factoid Q&A due to span extraction error: {spans['error']}", "raw_spans_error": spans}
        else:
            print(f"Warning: Invalid span structure passed to generate_factoid_qa for classification '{classification}'. Spans: {spans}")
            return {"error": "Invalid span structure received for Factoid Q&A generation."}

    prompt_2 = f"""
    **TASK:** Generate high-quality, factoid Question-Answer pairs based **strictly and solely** on the provided Context.
    
    **INPUT DATA:**
    *   Context: "{content}" - The source text.
    *   Spans (JSON): {json.dumps(spans)} - Specific text excerpts from Context; these are the **ONLY** valid answers.
    *   Classification: {classification} - Required tag for output pairs.
    
    **CORE RULES (APPLY TO ALL GENERATION - NON-NEGOTIABLE):**
    1.  **Answer Validity:** The `answer` field MUST be **EXACT, VERBATIM SPAN(S) ONLY** from the input `Spans (JSON)`. **NO** additions, omissions, paraphrasing, or changes allowed.
    2.  **Contextual Accuracy:** The `answer` MUST be the **PRECISE, COMPLETE, CORRECT** response to the `question`, based **STRICTLY AND SOLELY** on the provided `Context`.
    3.  **Conditional Generation:** Generate a Q&A pair **ONLY IF** you can form a natural, clear question for which the span(s) are the perfect, complete, and verbatim answer as found in the Context, AND all Core Rules are met. If any rule is violated, **DO NOT generate the pair**. Prioritize accuracy over quantity.
    4.  **Metadata:** Each output object MUST include "type": "Factoid" and "tag": "{classification}".
    
    **GENERATION PROCESS:**
    *   Think step-by-step internally to identify valid Q&A opportunities based on the Spans and Context. Do not output intermediate thoughts.
    *   **Span Handling Strategy:**
        *   For each group in `Spans (JSON)` (excluding "Individual"):
            *   create **one question** answered by **ALL** spans in the group combined (e.g., "SpanA, SpanB"). The combined answer format should typically use comma+space if it fits the context. Must meet Core Rules.
            *   create **one separate question** for **EACH INDIVIDUAL span** *within* the group. Must meet Core Rules.
        *   For each individual span in `Spans (JSON)` ("Individual" category):
            *   create **one question** for that **single span**. Must meet Core Rules.
    *   **Question Formulation:**
        *   Questions must be grammatical, natural, standalone, and vary in structure (What, When, Which, Percentage, Value, etc.).
        *   Reference organization name from Context if applicable.
    
    **EXPLICITLY AVOID (DO NOT GENERATE THESE QUESTION TYPES):**
    *   Trivial questions (answer obvious from span alone).
    *   Questions embedding the answer text.
    *   "What is [span]?" questions unless Context gives an explicit definition.
    *   "Why" or "How" questions.
    
    **MANDATORY FINAL VALIDATION:**
    *   Before finalizing output, review EVERY potential pair generated against the **Core Rules** and **Explicitly Avoid** list:
        1.  Does `answer` meet Rule 1 (EXACT VERBATIM SPAN(S) ONLY)? (Y/N)
        2.  Does `answer` meet Rule 2 (PRECISE, COMPLETE, CORRECT per CONTEXT ONLY)? (Y/N)
        3.  Does `question` meet Rule 3 (natural/clear) and avoid all types listed under "Explicitly Avoid"? (Y/N)
    *   **DISCARD** any pair failing *any* check. Only include pairs where ALL checks pass.
    
    **OUTPUT FORMAT (JSON ONLY):**
    Output **ONLY** a valid JSON object `{{"qa_pairs": [...]}}`. If none qualify, output `{{"qa_pairs": []}}`.
    Each object in `qa_pairs` MUST have: `"question"` (string), `"answer"` (string), `"type": "Factoid"`, `"tag": "{classification}"`.

    
    Example JSON Structure:
        {{
          "qa_pairs": [
            {{
              "question": "<Your Well-Formed Question>",
              "answer": "<Exact Span(s) Required>",
              "type": "Factoid",
              "tag": "{classification}"
            }}
          ]
        }}
    """

    response = azure_client.chat.completions.create(
        model=AZURE_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert Q&A generator. You create factoid questions strictly based on provided text and spans, ensuring the spans are the exact verbatim answers. You output only valid JSON."},
            {"role": "user", "content": prompt_2},
        ],
        max_tokens=2048,
        temperature=0.0,
        top_p=1,
        stop=None,
        response_format={"type": "json_object"},
    )
    raw_text = response.choices[0].message.content.strip()
    factoid_obj = safe_json_parse(raw_text)
    return filter_factoid_qas(factoid_obj, spans, context=content)


# --------------------------------------------------------------------------
# ---  NON-FACTOID Q&A GENERATION (Using AzureOpenAI) ------
# --------------------------------------------------------------------------
def generate_non_factoid_qa(classification, content):
    prompt_3_shortened = f"""
    **Task:** Generate high-quality, distinct, non-factoid (descriptive/explanatory) Question-Answer pairs STRICTLY from the provided `Context`. Aim for 7-15 pairs, or more if the `Context` supports them without sacrificing quality or distinctness.

    **Inputs:**
    *   Context: "{content}"
    *   Classification: "{classification}"

    **Key Instructions & Constraints:**
    1.  **Strict Context Adherence:** ALL questions and answers MUST be derived SOLELY from information EXPLICITLY STATED or DIRECTLY INFERABLE within the `Context`. NO external knowledge or assumptions.
    2.  **Non-Factoid Focus:**
        *   Questions MUST require descriptive or explanatory answers (e.g., understanding relationships, purposes, definitions, processes, implications as presented in the text).
        *   AVOID simple factoid questions answerable by a single data point (date, number, short proper noun unless it's being defined/explained).
    3.  **Distinctness:** Each Q&A pair must offer a unique insight or cover a different descriptive aspect, relationship, process, or explanation from the `Context`. Avoid redundancy.
    4.  **Answerability:** Every question MUST be fully answerable using ONLY the `Context`.

    **Question Formulation Guidance:**
    *   Frame questions to elicit explanations, descriptions, or understanding of:
        *   Relationships (e.g., 'How does X relate to Y, according to the text?')
        *   Processes/Mechanisms (e.g., 'Explain the mechanism for X based on the context.')
        *   Reasons/Justifications (e.g., 'What reasons are provided for Z in the text?')
        *   Implications/Consequences (e.g., 'What are the implications of X as detailed?')
        *   Definitions/Clarifications of concepts within the text.
    *   For 'Why'/'How': Ask for explanations of purpose, reasoning, methods, or processes *as understood from the text* (e.g., 'Explain the rationale behind X...' or 'Describe the method for X...').
    *   If a complex topic has multiple facets (e.g., several reasons, steps in a process), generate distinct Q&A for substantial components.
    *   Use company names from `Context` for specificity if available.

    **Answer Formulation Guidance:**
    *   Answers must be comprehensive yet focused, accurately summarizing/explaining relevant `Context` information.
    *   Aim for 1-4 sentences and approximately 25 to 70 words long, but ensure the answer fully addresses the question based ONLY on the text, even if slightly longer. Avoid trivial/short-phrase answers.

    **Mandatory Self-Correction Checklist (Review EACH pair before output):**
    1.  Is the question genuinely non-factoid (descriptive/explanatory)? (Y/N)
    2.  Is the Q&A pair distinct from others? (Y/N)
    3.  Is the question fully answerable ONLY from `Context`? (Y/N)
    4.  Is the answer comprehensive, accurate, and derived ONLY from `Context`? (Y/N)
    *   DISCARD any pair failing ANY check.

    **Output Format (JSON ONLY):**
    Return ONLY one valid JSON object: `{{"qa_pairs": [...]}}`. If no pairs qualify, use `{{"qa_pairs": []}}`.
    Each object in `qa_pairs` MUST have:
    -   `"question"`: (string)
    -   `"answer"`: (string)
    -   `"type"`: "Descriptive, non-factoid"
    -   `"tag"`: "{classification}"
    
    Example:
    {{
      "qa_pairs": [
        {{"question": "...", "answer": "...", "type": "Descriptive, non-factoid", "tag": "{classification}"}}
      ]
    }}
    """

    response = azure_client.chat.completions.create(
        model=AZURE_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert Q&A generator. You create descriptive, non-factoid questions and concise answers based strictly on the provided text. You output only valid JSON and meticulously follow all formatting and content rules."},
            {"role": "user", "content": prompt_3_shortened},
        ],
        max_tokens=4096,
        temperature=0.3,
        top_p=.9,
        stop=None,
        response_format={"type": "json_object"},
    )
    raw_text = response.choices[0].message.content.strip()
    non_factoid_obj = safe_json_parse(raw_text)
    return non_factoid_obj


# --------------------------------------------------------------------------
# ------------------  TOP-LEVEL CHUNK PROCESSOR (Simple Structure) --------
# --------------------------------------------------------------------------
def process_chunk(titles, subtitles, content, chunk_number=None, page_number=None):
    """
    Processes a single content chunk to classify it, extract spans, and generate Q&A pairs.
    Includes page_number in the metadata.
    """
    title_str = " | ".join(titles) if titles else ""
    subtitle_str = " | ".join(subtitles) if subtitles else ""

    try:
        classification = classify_with_llm(title_str, subtitle_str, content).strip()
    except Exception as e:
        print(f"Error during classification for chunk {chunk_number}: {e}")
        return {
             "metadata": {
                "chunk_number": chunk_number,
                "page_number": page_number,
                "titles": titles,
                "subtitles": subtitles,
                "paragraph": content,
                "classification": "Error during classification",
            },
            "error": f"Classification API call failed: {str(e)}"
        }

    if classification.lower() == "unknown":
        return {
            "metadata": {
                "chunk_number": chunk_number,
                "page_number": page_number,
                "titles": titles,
                "subtitles": subtitles,
                "paragraph": content,
                "classification": classification,
            },
            "spans": None,
            "qa_pairs_factoid": None,
            "qa_pairs_non_factoid": None
        }

    spans_result = None
    factoid_result = None
    non_factoid_result = None

    try:
        spans_result = extract_spans(classification, content)
    except Exception as e:
        print(f"Error during span extraction for chunk {chunk_number}: {e}")
        spans_result = {"error": f"Span Extraction execution failed: {str(e)}"}

    try:
        factoid_result = generate_factoid_qa(classification, content, spans_result)
    except Exception as e:
        print(f"Error during factoid Q&A generation for chunk {chunk_number}: {e}")
        factoid_result = {"error": f"Factoid Q&A execution failed: {str(e)}"}

    try:
        non_factoid_result = generate_non_factoid_qa(classification, content)
    except Exception as e:
        print(f"Error during non-factoid Q&A generation for chunk {chunk_number}: {e}")
        non_factoid_result = {"error": f"Non-Factoid Q&A execution failed: {str(e)}"}

    return {
        "metadata": {
            "chunk_number": chunk_number,
            "page_number": page_number,
            "titles": titles,
            "subtitles": subtitles,
            "paragraph": content,
            "classification": classification,
        },
        "spans": spans_result,
        "qa_pairs_factoid": factoid_result,
        "qa_pairs_non_factoid": non_factoid_result
    }

# --------------------------------------------------------------------------
# -------------------------  MARKDOWN PARSER ------------------------------
# --------------------------------------------------------------------------
def parse_markdown_by_chunks(filepath):
    """
    Parses a markdown file into chunks based on specific delimiters.
    Extracts chunk number, page number, titles, subtitles, and content for each.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at '{filepath}'")
        return None
    except Exception as e:
        print(f"Error reading input file '{filepath}': {e}")
        return None
    
    delimiter_pattern = r"^-{3,}\s*Chunk\s+(\d+)\s+\(page\s+(\d+),\s*starts at line\s+\d+\)\s*-{3,}\s*$"
    delimiters = list(re.finditer(delimiter_pattern, text, flags=re.MULTILINE))
    results = []

    if not delimiters:
        print("Warning: No chunk delimiters found. Processing file as a single chunk (Chunk 1).")
        chunk_text = text.strip()
        if chunk_text:
            chunk_num = 1
            page_num = None 
            headers = re.findall(r"^(#{1,4})\s+(.*)", chunk_text, re.MULTILINE)
            title_list = [h_text.strip() for level, h_text in headers if len(level) <= 2]
            subtitle_list = [h_text.strip() for level, h_text in headers if len(level) > 2]
            content_body = re.sub(r"^#{1,4}\s+.*", "", chunk_text, flags=re.MULTILINE).strip()
            content_body = "\n".join(line for line in content_body.splitlines() if line.strip())
            if content_body:
                results.append({
                    "chunk": chunk_num,
                    "page_number": page_num,
                    "titles": title_list,
                    "subtitles": subtitle_list,
                    "content": content_body
                })
        return results
    
    for i, match in enumerate(delimiters):
        if i == 0:
            continue

        prev_delimiter_match = delimiters[i-1]
        prev_delimiter_end = prev_delimiter_match.span()[1]
        
        chunk_text = text[prev_delimiter_end:match.span()[0]].strip()
        
        prev_chunk_num = int(prev_delimiter_match.group(1))
        prev_page_num = int(prev_delimiter_match.group(2))

        if chunk_text:
            headers = re.findall(r"^(#{1,4})\s+(.*)", chunk_text, re.MULTILINE)
            title_list = [h_text.strip() for level, h_text in headers if len(level) <= 2]
            subtitle_list = [h_text.strip() for level, h_text in headers if len(level) > 2]
            content_body = re.sub(r"^#{1,4}\s+.*", "", chunk_text, flags=re.MULTILINE).strip()
            content_body = "\n".join(line for line in content_body.splitlines() if line.strip())
            if content_body:
                results.append({
                    "chunk": prev_chunk_num,
                    "page_number": prev_page_num,
                    "titles": title_list,
                    "subtitles": subtitle_list,
                    "content": content_body
                })
    
    if delimiters:
        last_delimiter_match = delimiters[-1]
        last_delimiter_end = last_delimiter_match.span()[1]
        last_chunk_num = int(last_delimiter_match.group(1))
        last_page_num = int(last_delimiter_match.group(2))

        chunk_text = text[last_delimiter_end:].strip()
        if chunk_text:
            headers = re.findall(r"^(#{1,4})\s+(.*)", chunk_text, re.MULTILINE)
            title_list = [h_text.strip() for level, h_text in headers if len(level) <= 2]
            subtitle_list = [h_text.strip() for level, h_text in headers if len(level) > 2]
            content_body = re.sub(r"^#{1,4}\s+.*", "", chunk_text, flags=re.MULTILINE).strip()
            content_body = "\n".join(line for line in content_body.splitlines() if line.strip())
            if content_body:
                results.append({
                    "chunk": last_chunk_num,
                    "page_number": last_page_num,
                    "titles": title_list,
                    "subtitles": subtitle_list,
                    "content": content_body
                })

    results.sort(key=lambda x: x['chunk'])
    return results

# --------------------------------------------------------------------------
# ----------------------  MAIN ENTRY POINT --------------------------------
# --------------------------------------------------------------------------
def process_chunks_main(input_file, output_file):
    print(f"Starting processing for input file: '{input_file}'")
    print(f"Using Azure LLM Model: '{AZURE_MODEL_NAME}' for spans and Q&A generation.")
    print(f"Using Together LLM Model: '{LLAMA_CLASSIFICATION_MODEL}' for classification.")
    parsed_chunks = parse_markdown_by_chunks(input_file)
    if parsed_chunks is None:
        print("Exiting due to error during file parsing.")
        return
    if not parsed_chunks:
        print("No processable content chunks found in the input file.")
        output_data = {"results": []}
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"Empty result file written to '{output_file}'.")
        except Exception as e:
            print(f"Error writing empty output JSON file: {e}")
        return
    print(f"Parsed {len(parsed_chunks)} chunks. Starting LLM processing...")
    all_results = []
    total_chunks_to_process = len(parsed_chunks)
    for i, chunk_info in enumerate(parsed_chunks):
        chunk_num = chunk_info["chunk"]
        titles = chunk_info["titles"]
        subtitles = chunk_info["subtitles"]
        content = chunk_info["content"]
        page_number = chunk_info.get("page_number")

        print(f"Processing Chunk {chunk_num} (Page {page_number if page_number is not None else 'N/A'}, {i+1}/{total_chunks_to_process})...")
        try:
            result = process_chunk(titles, subtitles, content, chunk_number=chunk_num, page_number=page_number)
            if result["metadata"]["classification"].strip().lower() != "unknown":
                all_results.append(result)
                if result.get("error"):
                    print(f"Chunk {chunk_num}: Included with top-level error - {result['error']}")
                elif result.get("metadata", {}).get("classification") == "Error during classification":
                     print(f"Chunk {chunk_num}: Included with classification error flag.")
                else:
                    span_error = isinstance(result.get('spans'), dict) and 'error' in result['spans']
                    factoid_error = isinstance(result.get('qa_pairs_factoid'), dict) and 'error' in result['qa_pairs_factoid']
                    nonfact_error = isinstance(result.get('qa_pairs_non_factoid'), dict) and 'error' in result['qa_pairs_non_factoid']
                    
                    error_messages = []
                    if span_error: error_messages.append(f"Span error: {result['spans']['error'][:100]}...")
                    if factoid_error: error_messages.append(f"FactoidQA error: {result['qa_pairs_factoid']['error'][:100]}...")
                    if nonfact_error: error_messages.append(f"NonFactoidQA error: {result['qa_pairs_non_factoid']['error'][:100]}...")

                    if error_messages:
                        print(f"Chunk {chunk_num}: Included (Class: {result['metadata']['classification']}) with errors: {'; '.join(error_messages)}")
                    else:
                        print(f"Chunk {chunk_num}: Included (Class: {result['metadata']['classification']}) - OK.")
            else:
                print(f"Chunk {chunk_num}: Skipped due to 'Unknown' classification.")
        except Exception as e:
            print(f"!! Critical Error processing chunk {chunk_num} in main loop: {e}")
            all_results.append({
                "metadata": {
                    "chunk_number": chunk_num,
                    "page_number": page_number,
                    "titles": titles,
                    "subtitles": subtitles,
                    "paragraph": content,
                    "classification": "Critical Main Loop Error",
                },
                "error": f"Unhandled exception in main processing loop for chunk {chunk_num}: {str(e)}",
                "spans": None,
                "qa_pairs_factoid": None,
                "qa_pairs_non_factoid": None
            })
    print(f"\nFinished processing all chunks. Writing results to '{output_file}'...")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            output_data = {"results": all_results}
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Successfully processed output written to '{output_file}'")
    except Exception as e:
        print(f"Error writing output JSON file: {e}")

if __name__ == "__main__":
    input_md = "chunked_output.md"
    output_json = "QA_output_mixed_llms.json"
    process_chunks_main(input_md, output_json)