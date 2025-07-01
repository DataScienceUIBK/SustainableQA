import json
import os
import re
from openai import AzureOpenAI

# --- Configuration for Azure OpenAI Client ---
# IMPORTANT: For production environments, it is strongly recommended to use
# environment variables for API keys and sensitive information.
#
# To run this script, set the following environment variables:
# AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
# AZURE_OPENAI_API_KEY="your_api_key_here"
#
# For demonstration, default values are provided below, but replace them with your actual keys.
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "your-azure-endpoint-here")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "your-azure-api-key-here")
AZURE_MODEL_NAME = "gpt-4o-2"

# Initialize Azure OpenAI Client
azure_client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    api_version="2024-05-01-preview",
)

# --- JSON PARSING HELPER (Improved for robustness) ---
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

def generate_qa_pairs(paragraph_text: str) -> list:
    """
    Generates a list of comprehensive Question-Answering (QA) pairs for a given paragraph
    using the configured Azure OpenAI GPT-4o model. The LLM is instructed to
    think step-by-step implicitly before generating QAs.

    Args:
        paragraph_text (str): The text of the paragraph to generate QAs from.

    Returns:
        list: A list of dictionaries, where each dictionary has "question" and "answer" keys.
              Returns an empty list if an error occurs or no QAs are generated.
    """
    system_message = """
You are an expert AI assistant (GPT-4o) specialized in analyzing financial reports, particularly those adhering to the EU Taxonomy framework. Your task is to generate a comprehensive set of Question-Answering (QA) pairs based *only* on the provided paragraph text.

**Internal Thought Process (Do Not Show in Output):**
1.  **Understand Context**: Thoroughly read and understand the entire paragraph, identifying all explicit numerical data (percentages, monetary values), key terms, classifications (Taxonomy-aligned, eligible, non-eligible, transitional, enabling), specific economic activities, and environmental objectives (CCM, CCA, WTR, CE, PPC, BIO).
2.  **Identify Implicit Information**: Identify any implicit relationships, definitions, reasons, or processes described within the text. For example, how different figures relate or what a certain classification implies in the given context.
3.  **Categorize Information**: Systematically categorize the identified information by relevant EU Taxonomy objective and alignment status (e.g., all turnover data, then all CAPEX, then all OPEX, then specific activities under each).
4.  **Formulate Diverse Questions**: Formulate questions to cover every distinct piece of information identified. Ensure a mix of:
    *   **Factoid questions**: Direct questions about specific numbers, percentages, names.
    *   **Explanatory questions**: Questions requiring a descriptive answer about relationships, purposes, implications, or definitions as presented in the text.
    *   **Holistic questions**: Questions that require synthesizing information from multiple parts of the paragraph to provide a complete answer or draw broader conclusions.
5.  **Focus on Activities & Statuses**: Pay special attention to asking specific questions about *which* activities are classified under each Taxonomy status (aligned, eligible, non-eligible, transitional, enabling), along with their associated financial figures or percentages. Clearly differentiate between aligned and eligible where relevant.
6.  **Self-Correction/Validation**: Before finalizing each Q&A pair, meticulously cross-reference it against the original paragraph to ensure accuracy and strict adherence to the 'source constraint'. Verify that:
    *   The question is non-trivial and specific.
    *   The answer is fully comprehensive, accurate, and directly supported by the text.
    *   The Q&A pair is distinct and does not overlap significantly with other generated pairs.
    *   Avoid generating questions about the table title or caption.

**Key Guidelines for QA Generation:**
1.  **Source Constraint**: ALL answers MUST be directly extracted or logically inferred SOLELY from the provided paragraph. Do NOT use any outside knowledge or make assumptions.
2.  **Comprehensiveness**: Generate questions that cover EVERY piece of information, both explicit (figures, percentages, categories, names, classifications) and implicit (relationships, purposes, implications, definitions as presented in the text).
3.  **Holistic Questions**: Include questions that require synthesizing information from multiple parts of the paragraph to form a complete answer or draw broader conclusions/summaries from the text. This should be the *last* question for each paragraph, summarizing the core essence.
4.  **Specific Activity Focus**: Ask specific questions about *which* activities are classified as Taxonomy-aligned, Taxonomy-eligible, or Taxonomy-non-eligible, including any designated as 'Transitional' or 'Enabling'. Clearly state their associated figures or percentages.
5.  **Quality & Quantity**: Generate high-quality, non-redundant, and non-trivial questions. Aim for at least 20 distinct QA pairs if the content of the paragraph allows for such diversity and depth. If the paragraph is shorter or less dense, generate as many high-quality, distinct questions as possible without being repetitive or trivial.
6.  **Answer Formulation**: Answers must be comprehensive yet concise. For **factoid questions**, answers should be as close to verbatim extraction as possible. For **explanatory or holistic questions**, answers should accurately summarize or explain relevant information directly from the paragraph. Avoid overly short or trivial answers.

**Output Format**: Provide the output as a single JSON object with a key "qa_pairs" whose value is a JSON array of objects, where each object has a "question" key and an "answer" key. Ensure the output is *only* this JSON object, with no preceding or trailing text.

Example of desired output structure:
{
  "qa_pairs": [
    {
      "question": "What is AGRANA Group's total turnover for the 2023/24 fiscal year?",
      "answer": "AGRANA Group's total turnover for the 2023/24 fiscal year amounted to 3,786,876 thousand euros."
    },
    {
      "question": "What percentage of AGRANA's total turnover in 2023/24 was classified as Taxonomy-aligned, and what specific activities contributed to this classification?",
      "answer": "1.7% (62,308 thousand euros) of total turnover was classified as Taxonomy-aligned. This includes 'Manufacture of plastics in primary form' (CCM 3.17), 'Manufacture of biogas and biofuels for use in transport and of bioliquids' (CCM 4.13), and 'Anaerobic digestion of bio-waste' (CCM 5.7 / CE 2.5)."
    },
    {
      "question": "Summarize AGRANA's overall EU Taxonomy turnover performance for 2023/24 based on the provided data, highlighting the main areas of alignment and non-alignment.",
      "answer": "AGRANA's EU Taxonomy turnover for 2023/24 shows minimal alignment, with only 1.7% of total turnover being Taxonomy-aligned and 5.5% Taxonomy-eligible, primarily driven by Climate Change Mitigation activities. The vast majority (94.5%) of their revenue falls outside the Taxonomy's scope, largely because their core food and beverage industry is not explicitly included."
    }
  ]
}
"""
    user_message = f"Generate QA pairs for the following paragraph:\n\n---\nParagraph:\n{paragraph_text}\n---"

    try:
        response = azure_client.chat.completions.create(
            model=AZURE_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0.5
        )
        
        content = response.choices[0].message.content
        
        qa_data_wrapper = safe_json_parse(content)
        
        if isinstance(qa_data_wrapper, dict) and "qa_pairs" in qa_data_wrapper and isinstance(qa_data_wrapper["qa_pairs"], list):
            if "error" in qa_data_wrapper:
                print(f"Warning: safe_json_parse reported error during parsing LLM response. Error: {qa_data_wrapper['error']}\nRaw LLM response: {qa_data_wrapper.get('raw_full', 'N/A')}")
            return qa_data_wrapper["qa_pairs"]
        else:
            print(f"Warning: Unexpected JSON structure from LLM or parsing error for paragraph:\n{paragraph_text}\nRaw LLM Response:\n{content}\nSafe Parse Result: {qa_data_wrapper}")
            return []
    except Exception as e:
        print(f"An error occurred during OpenAI API call for paragraph:\n{paragraph_text}\nError: {e}")
        return []

def process_tables_data(input_json_path: str, output_json_path: str):
    """
    Reads the input JSON file containing paragraph summaries, generates QA pairs
    for each paragraph, and writes the updated data to an output JSON file.

    Args:
        input_json_path (str): Path to the input JSON file.
        output_json_path (str): Path to save the output JSON file with QA pairs.
    """
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_json_path}. Please check file format.")
        return

    processed_data = []
    for i, item in enumerate(data):
        paragraph_text = item["meta_data"].get("paragraph", "")
        if not paragraph_text:
            print(f"Skipping item {i} due to missing paragraph text.")
            item_copy = item.copy()
            item_copy["qa_pairs"] = [] 
            processed_data.append(item_copy)
            continue

        item_copy = item.copy() 
        item_copy["meta_data"] = item["meta_data"].copy()

        page_info = item_copy['meta_data'].get('page_number', 'N/A')
        caption_info = item_copy['meta_data'].get('table_caption', 'No Caption')
        print(f"Processing paragraph {i+1} from page {page_info} - '{caption_info}'...")
        
        qa_pairs = generate_qa_pairs(paragraph_text)
        item_copy["qa_pairs"] = qa_pairs
        processed_data.append(item_copy)
        print(f"Generated {len(qa_pairs)} QA pairs for this paragraph.")

    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        print(f"\nSuccessfully processed data and saved to {output_json_path}")
    except IOError as e:
        print(f"Error writing output file to {output_json_path}: {e}")

if __name__ == "__main__":
    input_file_name = "tables.md"
    output_file_name = "tables_qa.json"

    process_tables_data(input_file_name, output_file_name)