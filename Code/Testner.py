# coding: utf-8

"""
esg_span_extractor.py
─────────────────────
Generic ESG / sustainability span extractor (updated 2025-05-01).

Key points
──────────
• Combines rule-based regexes, spaCy PhraseMatcher, and a fine-tuned ESG-NER model
(downloaded from Hugging Face: mohammed933/esg_ner).
• Memoises heavyweight resources with @lru_cache.
• CLI-less – import and call extract_spans_from_text(...).
• This revision returns spans with their labels (e.g., "EU_REG," "SUST") in a
structured format ([{"text": str, "label": str}, ...]) for LLM refinement.
• Tightens date handling, recognises full EU regulation cites, expands keyword
coverage (SDG/SBTN/DNSH, CapEx/OpEx/EBITDA, etc.), and suppresses generic
single-word spans like "environmental".
"""
from __future__ import annotations

import re
import logging
from functools import lru_cache
from typing import Iterable, List, Dict, Any

import spacy
from spacy.matcher import PhraseMatcher
from spacy.language import Language
from spacy.tokens import Span
from spacy.lang.en.stop_words import STOP_WORDS
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

###############################################################################
# Logging
###############################################################################
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

###############################################################################
# Keyword dictionaries (extend freely)
###############################################################################
SUSTAINABILITY_KEYWORDS: set[str] = {
    # Environmental focus
    "ecology", "ecosystem", "climate change", "net zero", "renewable energy",
    "green building", "carbon neutrality", "life-cycle assessment",
    "biodiversity", "circular economy", "resource efficiency",
    "water stewardship", "waste management", "pollution prevention",
    "carbon capture", "nature-positive",
    # Social & mixed context
    "social equity", "ethical sourcing", "community resilience", "human rights",
    "fair trade", "just transition",
}

EU_TAXONOMY_KEYWORDS: set[str] = {
    "taxonomy regulation", "technical screening criteria", "do no significant harm",
    "dnsh",  # acronym only
    "minimum social safeguards", "economic activity", "substantial contribution",
    "taxonomy-eligible", "taxonomy-aligned", "climate change mitigation",
    "climate change adaptation", "sustainable use of water", "circular economy",
    "pollution prevention", "protection of ecosystems", "biodiversity",
}

ESG_KPI_KEYWORDS: set[str] = {
    "carbon footprint", "scope 1 emissions", "scope 2 emissions",
    "renewable energy share", "energy consumption", "water intensity",
    "waste diversion rate", "biodiversity impact", "deforestation rate",
    "employee diversity", "turnover rate", "workplace injury rate",
    "training hours per employee", "community investment",
    "customer satisfaction", "living wage coverage", "board diversity",
    "executive compensation", "anti-corruption", "data privacy incidents",
    "esg compliance", "capex", "opex", "ebitda", "ebitda margin",
    "green revenue",
}

STANDARDS_FRAMEWORKS: set[str] = {
    "gri", "sasb", "tcfd", "sdg", "sdgs", "sbtn", "sbt", "sbti", "cdp",
    "un global compact", "iso 14001", "ifrs s1", "ifrs s2",
}

GENERIC_TERMS: set[str] = {
    "regulation", "directive", "policy", "agreement", "guideline",
    "framework", "standard",
}

TRIVIAL_TERMS: set[str] = {
    "company", "organisation", "organization", "report", "data", "information",
    "impact", "approach", "process", "performance", "strategy", "management",
    "framework", "objective", "environmental", "social",  # suppress overly generic hits
}

###############################################################################
# Regex patterns
###############################################################################
REGEX_PATTERNS: dict[str, re.Pattern[str]] = {
    "EU_REG": re.compile(r"Reg\s?(?:\(?EU\)?)\s?\d{4}/\d+\b", re.I),
    "ISO": re.compile(r"\bISO\s?\d{4,5}\b", re.I),
    "IFRS": re.compile(r"\bIFRS\s?S?\d{1,4}\b", re.I),
    "GRI": re.compile(r"\bGRI\s?\d{1,4}(?:-\d{4})?\b", re.I),
    "SASB": re.compile(r"\bSASB\b", re.I),
    "TCFD": re.compile(r"\bTCFD\b", re.I),
    "PERCENT": re.compile(r"\b\d{1,3}(?:[.,]\d+)?\s?%", re.I),
    "MONEY": re.compile(
        r"\b(?:€|$|£)\s?\d+(?:[.,]\d+)?(?:\s?(?:thousand|k|million|m|billion|bn))?\b",
        re.I,
    ),
    # Tightened: require FY/Q context or explicit 19xx/20xx range
    "DATE": re.compile(r"\b(?:FY\s?\d{2,4}|Q[1-4]\s?\d{4}|(?:19|20)\d{2})\b", re.I),
    "QUANTITY": re.compile(
        r"\b\d+(?:[.,]\d+)?\s?(?:MWh|kWh|GWh|t|tonnes?|kg|l|liters?|m3|GJ|TJ|PJ)\b",
        re.I,
    ),
    "REG_GENERIC": re.compile(
        r"\b(?:[A-Z][\w&'-–]+\s+){2,}(?:Regulation|Directive|Policy|Agreement|Guideline|Act|Framework)\b",
    ),
}

###############################################################################
# Config
###############################################################################
SPACY_MODEL = "en_core_web_lg"
NER_MODEL = "mohammed933/esg_ner"  # <─── your Hugging Face model repository
NER_THRESHOLD = 0.85
MIN_LEN = 3
BLACKLIST = TRIVIAL_TERMS | {"group", "level", "type"}

###############################################################################
# Helpers – resource loaders (cached)
###############################################################################
@lru_cache(maxsize=1)
def _get_nlp() -> Language:
    try:
        nlp = spacy.load(SPACY_MODEL, disable=("parser", "lemmatizer"))
        logger.info("spaCy model '%s' loaded", SPACY_MODEL)
        return nlp
    except OSError as exc:
        logger.error(
            "spaCy model '%s' not installed. Run: python -m spacy download %s",
            SPACY_MODEL, SPACY_MODEL,
        )
        raise exc

@lru_cache(maxsize=1)
def _get_phrase_matcher() -> PhraseMatcher:
    nlp = _get_nlp()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    groups = {
        "SUST": SUSTAINABILITY_KEYWORDS,
        "EU": EU_TAXONOMY_KEYWORDS,
        "KPI": ESG_KPI_KEYWORDS,
        "STD": STANDARDS_FRAMEWORKS,
        "GEN": GENERIC_TERMS,
    }
    for label, terms in groups.items():
        matcher.add(label, [nlp.make_doc(t) for t in terms])
    logger.info("PhraseMatcher initialized")
    return matcher

@lru_cache(maxsize=1)
def _get_esg_ner():
    """Load ESG-NER model from Hugging Face Hub."""
    try:
        logger.info("Downloading ESG-NER model from Hugging Face: %s", NER_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(NER_MODEL)
        model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)
        ner_pipe = pipeline(
            "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
        )
        logger.info("ESG-NER model '%s' loaded successfully", NER_MODEL)
        return ner_pipe
    except Exception as exc:
        logger.error("Failed to load ESG-NER model from Hugging Face: %s", exc)
        raise exc

###############################################################################
# Internal utilities
###############################################################################
def _clean_markdown(text: str) -> str:
    """Strip common markdown artefacts for cleaner char-spans."""
    text = re.sub(r"(?:\[|\[#\]){1,3}(.*?)(?:\]|\[#\]){1,3}", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # links
    text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"\1", text)  # images
    text = re.sub(r"([^`]+)`", r"\1", text)  # inline code
    text = re.sub(r"^\s*[-*]{3,}\s*$", "", text, flags=re.M)  # hrules
    text = re.sub(r"^\s*>\s?", "", text, flags=re.M)  # blockquotes
    text = re.sub(r"^\s*[+-]\s+", "", text, flags=re.M)  # bullets
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.M)  # numbered lists
    text = re.sub(r"^\s*#+\s+", "", text, flags=re.M)  # headings
    return text

def _squash_whitespace(txt: str) -> str:
    return re.sub(r"\s+", " ", txt).strip(" ,.;:!-()[]{}<>")

def _filter_trivial(spans: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove duplicates, stop-words, blacklist terms, bare years, very short."""
    seen, keep = set(), []
    for span_dict in spans:
        clean = _squash_whitespace(span_dict["text"])
        low = clean.lower()
        if clean.isdigit() and len(clean) == 4:
            # drop bare years like "2020"
            continue
        if (
            len(clean) >= MIN_LEN
            and low not in STOP_WORDS
            and low not in BLACKLIST
            and low not in seen
        ):
            keep.append({"text": clean, "label": span_dict["label"]})
            seen.add(low)
    return keep

def _dedup_overlaps(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    spans = sorted(spans, key=lambda s: (s["span"].start_char, -s["span"].end_char, -len(s["span"].text)))
    result: List[Dict[str, Any]] = []
    last_end = -1
    for sp_dict in spans:
        sp = sp_dict["span"]
        if sp.start_char >= last_end:
            result.append(sp_dict)
            last_end = sp.end_char
        elif result and len(sp.text) > len(result[-1]["span"].text):
            result[-1] = sp_dict  # Replace with longer span
            last_end = sp.end_char
    return result

###############################################################################
# Public API
###############################################################################
def extract_spans_from_text(text: str, *, use_ner: bool = True) -> Dict[str, List[Dict[str, str]]]:
    cleaned = _clean_markdown(text)
    if not cleaned.strip():
        logger.warning("Input text is empty after markdown cleaning.")
        return {"spans": []}

    nlp = _get_nlp()
    doc = nlp(cleaned)
    matcher = _get_phrase_matcher()
    raw_spans: List[Dict[str, Any]] = []

    # Regex-based matches
    for label, pat in REGEX_PATTERNS.items():
        for match in pat.finditer(cleaned):
            sp = doc.char_span(match.start(), match.end(), label=label, alignment_mode="expand")
            if sp:
                raw_spans.append({"span": sp, "label": label})

    # PhraseMatcher dict hits
    for match_id, start, end in matcher(doc):
        label = doc.vocab.strings[match_id]
        sp = doc[start:end]
        raw_spans.append({"span": sp, "label": label})

    # ESG-NER model hits
    if use_ner:
        try:
            for ent in _get_esg_ner()(cleaned):
                if ent["score"] >= NER_THRESHOLD:
                    sp = doc.char_span(ent["start"], ent["end"], label=ent["entity_group"], alignment_mode="expand")
                    if sp:
                        raw_spans.append({"span": sp, "label": ent["entity_group"]})
        except Exception as exc:
            logger.warning("ESG-NER model failed, continuing without NER: %s", exc)

    if not raw_spans:
        logger.info("No candidate spans found.")
        return {"spans": []}

    filtered = _dedup_overlaps(raw_spans)
    logger.info("Overlap filtering: %d → %d", len(raw_spans), len(filtered))

    candidate_spans = [{"text": sp["span"].text, "label": sp["label"]} for sp in filtered]
    meaningful = _filter_trivial(candidate_spans)
    logger.info("%d spans after triviality filter", len(meaningful))

    # Preserve original order of first appearance
    first_pos: Dict[str, int] = {}
    for sp_dict in filtered:
        txt = _squash_whitespace(sp_dict["span"].text)
        if txt in {d["text"] for d in meaningful} and txt not in first_pos:
            first_pos[txt] = sp_dict["span"].start_char
    meaningful.sort(key=lambda d: first_pos.get(d["text"], float("inf")))

    return {"spans": meaningful}

if __name__ == "__main__":
    demo = """
In line with its business activities (the industrial processing of agricultural raw materials into foods and intermediate products for various industries) and its sustainability priorities in the areas of climate change mitigation, complete raw material utilisation, attention to environmental and social criteria within the company and in the agricultural and non-agricultural supply chain as well as in terms of ethical business conduct, AGRANA supports especially the Sustainable Development Goals (SDGs) 8, 13, 15 and 16 of the United Nations. In addition,\nIn summer 2020, the European Union adopted the Taxonomy Regulation (Reg (EU) 2020/852), which defines criteria for reporting revenues, investments and operating expenses from or in sustainable economic activities. To be considered sustainable, economic activities must serve one of six EU environmental objectives - climate change mitigation, adaptation to climate change, sustainable use of water resources, transition to a circular economy, pollution prevention, or protection of ecosystems and biodiversity – without significantly compromising any of the other five. In addition, the economic activities must meet minimum social standards.\nThe determination of Taxonomy-eligible or -aligned revenue, investment and operating expenses was carried out through the assessment under the technical screening criteria as well as the DNSH ("do no significant harm") criteria and the minimum social safeguards (Article 18 of the Taxonomy Regulation), in collaboration with the persons responsible for technology at the respective production sites as well as the controllership, finance, compliance and sustainability functions at the site, segment and Group levels. The AGRANA Group avoids any type of double counting by assigning the data for a given key performance indicator (KPI) to a single economic activity only. In cases where an activity contributes to several environmental objectives, it was always counted entirely towards the AGRANA Group's most important environmental goal, climate change mitigation.\nAGRANA ensures adherence to minimum social safeguards through its compliance management system and due diligence processes. The content of the Group-wide guidelines is based on the International Bill of Human Rights, the standards of the International Labour Organization, the OECD Guidelines for Multinational Enterprises and the UN Guiding Principles for Business and Human Rights. The AGRANA Compliance Office performs an annual risk analysis for all sites and countries where business activities are conducted. This risk analysis is based on selected indicators such as Coface risk ratings, the Corruption Perceptions Index and the International Trade Union Confederation (ITUC) Index, as well as on internal sources such as the evaluation of tips from the AGRANA whistleblowing hotline. The due diligence processes include internal audits by the Internal Audit department, external social audits at many AGRANA and supplier sites, and use of the tools of the Sustainable Agriculture Initiative Platform (SAI) for the agricultural supply chain. For detailed information on social matters in the supply chain, labour issues and respect for human rights, see the non-financial information statement from page 51 and the GRI content index (GRI 407 to 409) from page 220.
    """
    print(extract_spans_from_text(demo))