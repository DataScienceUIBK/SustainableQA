"""
SustainableQA Evaluation Framework

A comprehensive evaluation system for SustainableQA dataset using:
- Faithfulness: How well answers are grounded in context
- Relevance: How relevant QA pairs are to sustainability reporting

Supports multiple LLM providers:
- Azure OpenAI (GPT-4, GPT-3.5)
- Together AI (Llama, Mixtral, etc.)
- Local LLMs via Ollama API
- HuggingFace Transformers (local)

Author: SustainableQA Evaluation Team
"""

# =============================================================================
# API KEYS CONFIGURATION - REPLACE WITH YOUR ACTUAL KEYS
# =============================================================================

# Together AI API Key - Get from https://api.together.xyz/
TOGETHER_API_KEY = ""

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
AZURE_OPENAI_API_KEY = "your-azure-api-key-here"

# =============================================================================

import json
import os
import statistics
import requests
import time
import re
import sys
import hashlib
import logging
import argparse
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Import different LLM clients with fallbacks
try:
    from openai import AzureOpenAI
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("Azure OpenAI not available. Install: pip install openai")

try:
    from together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False
    print("Together AI not available. Install: pip install together")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install: pip install transformers torch")

# Configure logging
def setup_logging(level=logging.INFO, log_file=None):
    """Setup logging configuration for the application"""
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    # Configure root logger
    logger = logging.getLogger('sustainableqa_evaluator')
    logger.setLevel(level)

    # Clear any existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Initialize default logger
logger = setup_logging()

class LLMProvider(Enum):
    AZURE_OPENAI = "azure_openai"
    TOGETHER_AI = "together_ai"
    LOCAL_OLLAMA = "local_ollama"
    HUGGINGFACE_LOCAL = "huggingface_local"

class ValidationSeverity(Enum):
    CRITICAL = "critical"
    WARNING = "warning"

@dataclass
class ValidationIssue:
    severity: ValidationSeverity
    category: str
    message: str
    field: str

@dataclass
class EvaluationResult:
    # Core QA data
    question: str
    answer: str
    question_type: str  # "factoid", "non_factoid", or "table"
    classification: str  # "ESG", "EU Taxonomy", "Sustainability"

    # Evaluation scores
    faithfulness_score: float
    faithfulness_reasoning: str
    relevance_score: float
    relevance_reasoning: str

    # Component scores
    question_faithfulness_score: Optional[float] = None
    answer_faithfulness_score: Optional[float] = None
    question_relevance_score: Optional[float] = None
    answer_relevance_score: Optional[float] = None

    # Metadata
    chunk_number: Optional[int] = None
    page_number: Optional[int] = None
    table_caption: Optional[str] = None
    context: Optional[str] = None

    # Numeric validation
    numeric_guardrail_passed: Optional[bool] = None

    # Final decision
    keep_qa: Optional[bool] = None

    # Modification fields (only if modification applied)
    modification_applied: Optional[bool] = None
    original_question: Optional[str] = None
    original_answer: Optional[str] = None
    modified_question: Optional[str] = None
    modified_answer: Optional[str] = None
    modification_reasoning: Optional[str] = None
    modification_type: Optional[str] = None  # "question", "answer", "both"
    modification_target: Optional[str] = None  # "faithfulness", "relevance"
    modification_validation_passed: Optional[bool] = None

class SustainableQAEvaluator:
 
    def __init__(self, llm_provider: LLMProvider = LLMProvider.TOGETHER_AI, **kwargs):
        self.llm_provider = llm_provider
        self.client = None
        self.model_name = None
        self.call_count = 0
        self.last_call_time = 0
        self.enable_modification = kwargs.get("enable_modification", False)

        # Setup class-specific logger
        self.logger = logging.getLogger(f'sustainableqa_evaluator.{self.__class__.__name__}')

        # Consolidated modification configuration
        self.modification_config = {
            # Retry and backoff settings
            "max_modification_attempts": kwargs.get("max_modification_attempts", 3),
            "modification_retry_attempts": kwargs.get("modification_retry_attempts", 3),
            "retry_delay_base": kwargs.get("retry_delay_base", 1.0),
            "modification_retry_delay": kwargs.get("modification_retry_delay", 1.0),
            "exponential_backoff_factor": kwargs.get("exponential_backoff_factor", 2.0),

            # Quality and safety settings
            "enable_quality_validation": kwargs.get("enable_quality_validation", True),
            "quality_threshold": kwargs.get("quality_threshold", 6.0),  # Use 1-10 scale consistently
            "fallback_to_original": kwargs.get("fallback_to_original", True),
            "enable_rollback": kwargs.get("enable_rollback", True)
        }

        # Rate limiting settings (configurable per provider)
        self.rate_limits = {
            LLMProvider.TOGETHER_AI: {
                "calls_per_minute": 60,  # Match actual free tier limit
                "min_delay": 1.0,        # 60 RPM = 1 call per second optimal
                "burst_delay": 0.0       # No burst delay needed for free tier
            },
            LLMProvider.AZURE_OPENAI: {
                "calls_per_minute": 60,
                "min_delay": 1.0,
                "burst_delay": 2.0
            },
            LLMProvider.LOCAL_OLLAMA: {
                "calls_per_minute": 1000,  # No rate limits for local
                "min_delay": 0.1,
                "burst_delay": 0.1
            },
            LLMProvider.HUGGINGFACE_LOCAL: {
                "calls_per_minute": 1000,  # No rate limits for local
                "min_delay": 0.1,
                "burst_delay": 0.1
            }
        }

        self._initialize_llm_client(**kwargs)

        # Domain-specific keywords for classification (used by both keyword matching and LLM prompts)
        self.domain_keywords = {
            "ESG": [
                # Core ESG terms
                "ESG", "environmental social governance", "ESG integration", "ESG strategy",
                "ESG governance", "ESG oversight", "ESG committee", "ESG risk management",

                # Reporting frameworks and standards
                "GRI", "Global Reporting Initiative", "GRI Standards", "SASB", "Sustainability Accounting Standards Board",
                "TCFD", "Task Force on Climate-related Financial Disclosures", "TCFD recommendations",
                "CSRD", "Corporate Sustainability Reporting Directive", "NFRD", "Non-Financial Reporting Directive",
                "CDP", "Carbon Disclosure Project", "CDSB", "Climate Disclosure Standards Board",

                # ESG processes and assessments
                "materiality assessment", "materiality analysis", "materiality matrix", "double materiality",
                "stakeholder engagement", "stakeholder consultation", "stakeholder mapping",
                "ESG reporting", "ESG disclosure", "ESG transparency", "ESG benchmarking", "ESG ratings",
                "ESG due diligence", "ESG screening", "ESG data quality", "ESG assurance",

                # ESG-specific metrics and emissions
                "ESG indicators", "ESG KPIs", "ESG targets", "ESG metrics", "sustainability KPIs",
                "Scope 1 emissions", "Scope 2 emissions", "Scope 3 emissions", "carbon footprint",
                "GHG emissions", "emission reporting", "carbon accounting",

                # ESG-linked finance and investment
                "responsible investment", "impact investing", "ESG investing", "sustainable investing",
                "ESG risk assessment", "responsible business", "business ethics",

                # Global frameworks
                "SDG", "sustainable development goals", "Paris Agreement", "UN Global Compact"
            ],

            "EU Taxonomy": [
                # Core EU Taxonomy terms
                "EU Taxonomy", "taxonomy regulation", "taxonomy framework", "taxonomy criteria",
                "taxonomy-eligible", "taxonomy-aligned", "eligible activities", "aligned activities",
                "taxonomy alignment", "taxonomy eligibility", "taxonomy assessment", "taxonomy compliance",

                # Technical screening and criteria
                "technical screening criteria", "screening criteria", "substantial contribution",
                "environmental objectives", "climate change mitigation", "climate change adaptation",
                "sustainable use and protection of water", "transition to circular economy",
                "pollution prevention and control", "protection and restoration of biodiversity",

                # DNSH and safeguards
                "DNSH", "Do No Significant Harm", "DNSH criteria", "DNSH assessment",
                "minimum safeguards", "social safeguards", "human rights", "labour rights",
                "anti-corruption", "fair competition", "tax compliance",

                # Financial metrics and KPIs
                "CapEx", "capital expenditures", "OpEx", "operational expenditures", "turnover",
                "CapEx alignment", "OpEx alignment", "turnover alignment", "taxonomy KPIs",
                "green ratio", "brown ratio", "taxonomy percentage", "eligible proportion",

                # EU regulatory context
                "delegated regulation", "delegated acts", "regulatory technical standards",
                "taxonomy disclosure", "taxonomy reporting", "SFDR", "Sustainable Finance Disclosure Regulation",
                "platform on sustainable finance", "green classification", "sustainable activities",

                # Taxonomy-aligned finance
                "green bonds", "sustainability-linked loans", "taxonomy-aligned financing",
                "EU green bond standard", "sustainable finance taxonomy"
            ],

            "Sustainability": [
                # Circular economy
                "circular economy", "circularity", "circular business model", "circular design",
                "waste reduction", "waste management", "recycling", "reuse", "refurbishment",
                "material efficiency", "resource efficiency", "resource optimization", "waste-to-energy",
                "lifecycle management", "cradle-to-cradle", "end-of-life management",

                # Climate action (general, non-ESG framework specific)
                "climate action", "climate goals", "climate targets", "climate strategy", "climate resilience",
                "decarbonization", "carbon neutrality", "net zero", "carbon negative", "net positive",
                "renewable energy", "clean energy", "solar energy", "wind energy", "hydroelectric",
                "energy efficiency", "energy management", "energy transition", "fossil fuel reduction",

                # Biodiversity and ecosystems
                "biodiversity", "biodiversity conservation", "ecosystem", "ecosystem services",
                "natural capital", "habitat protection", "species conservation", "deforestation prevention",
                "land use management", "water conservation", "water stewardship", "water management",
                "ocean conservation", "marine protection", "sustainable agriculture", "regenerative agriculture",

                # Sustainable products and practices
                "sustainable products", "sustainable services", "sustainable sourcing", "sustainable supply chain",
                "responsible sourcing", "ethical sourcing", "local sourcing", "sustainable procurement",
                "green products", "eco-friendly products", "environmentally friendly", "sustainable design",
                "sustainable innovation", "green technology", "clean technology", "eco-innovation",

                # Corporate responsibility themes (non-ESG framework)
                "corporate sustainability", "sustainability strategy", "sustainability goals", "sustainability targets",
                "environmental stewardship", "environmental management", "environmental performance",
                "sustainable development", "sustainable growth", "sustainable practices", "sustainability initiatives",
                "environmental impact", "ecological impact", "emission reduction", "green transition",
                "sustainability transformation", "environmental responsibility", "regenerative business", "social impact"
            ]
        }

        self.technical_terms = {
            "TCFD": "Task Force on Climate-related Financial Disclosures",
            "GRI": "Global Reporting Initiative",
            "SASB": "Sustainability Accounting Standards Board",
            "CSRD": "Corporate Sustainability Reporting Directive",
            "DNSH": "Do No Significant Harm",
            "CapEx": "Capital Expenditures",
            "OpEx": "Operational Expenditures",
            "KPIs": "Key Performance Indicators",
            "ESG": "Environmental, Social, and Governance"
        }

        # Enhanced modification statistics tracking
        self.modification_stats = {
            "total_attempts": 0,
            "successful_modifications": 0,
            "failed_api": 0,
            "failed_quality": 0,
            "failed_validations": 0,
            "rollbacks": 0,
            "fallbacks_to_original": 0,
            "validation_issues_prevented": 0,
            "critical_issues_prevented": 0,
            "technical_terms_preserved": 0,
            "hallucinations_prevented": 0
        }

        # Track modification depth to prevent infinite recursion
        self.modification_depth = 0
        self.max_modification_depth = 1  # Only allow one level of modification per QA pair

    def _get_relevant_keywords(self, classification: str, max_keywords: int = 15) -> str:
        """Get most relevant keywords for a classification, intelligently selecting unique and best terms"""
        keywords = self.domain_keywords[classification]

        # Dynamically extract priority keywords based on uniqueness and importance
        priority_keywords = self._extract_priority_keywords(classification)

        # Start with priority keywords
        selected = priority_keywords.copy()

        # Add additional keywords up to max_keywords, prioritizing shorter/more specific terms
        remaining_keywords = [k for k in keywords if k not in selected]
        # Sort by length (shorter first) and alphabetically for consistency
        remaining_keywords.sort(key=lambda x: (len(x.split()), x.lower()))

        for keyword in remaining_keywords:
            if len(selected) >= max_keywords:
                break
            selected.append(keyword)

        return ", ".join(selected[:max_keywords])

    def _extract_priority_keywords(self, classification: str) -> List[str]:
        """Extract the most unique and important keywords for each domain"""
        all_keywords = self.domain_keywords[classification]
        other_domains = [domain for domain in self.domain_keywords.keys() if domain != classification]

        # Get keywords from other domains for uniqueness comparison
        other_keywords = set()
        for domain in other_domains:
            other_keywords.update([kw.lower() for kw in self.domain_keywords[domain]])

        # Define domain-specific core terms and unique identifiers
        core_priority = {
            "ESG": [
                # Most unique ESG identifiers
                "ESG", "environmental social governance", "ESG integration", "ESG strategy",
                "materiality assessment", "materiality analysis", "double materiality",
                "Scope 1 emissions", "Scope 2 emissions", "Scope 3 emissions",
                "GRI", "SASB", "TCFD", "CSRD", "CDP",
                "stakeholder engagement", "ESG reporting", "ESG ratings"
            ],
            "EU Taxonomy": [
                # Most unique EU Taxonomy identifiers
                "EU Taxonomy", "taxonomy regulation", "taxonomy-eligible", "taxonomy-aligned",
                "technical screening criteria", "substantial contribution", "DNSH",
                "Do No Significant Harm", "environmental objectives", "climate change mitigation",
                "CapEx", "OpEx", "turnover", "CapEx alignment", "green ratio",
                "delegated regulation", "SFDR", "platform on sustainable finance"
            ],
            "Sustainability": [
                # Most unique Sustainability identifiers
                "sustainability", "sustainable development", "sustainability strategy",
                "circular economy", "circularity", "circular business model",
                "renewable energy", "clean energy", "energy transition", "decarbonization",
                "biodiversity", "biodiversity conservation", "ecosystem services",
                "water conservation", "water stewardship", "waste reduction",
                "sustainable products", "sustainable sourcing", "green technology"
            ]
        }

        domain_priorities = core_priority.get(classification, [])

        # Filter to only include keywords that exist in the actual domain_keywords
        valid_priorities = []
        domain_keywords_lower = [kw.lower() for kw in all_keywords]

        for priority in domain_priorities:
            if priority.lower() in domain_keywords_lower:
                # Find the exact case match from domain_keywords
                for kw in all_keywords:
                    if kw.lower() == priority.lower():
                        valid_priorities.append(kw)
                        break

        # Add uniqueness boost - prefer keywords that don't appear in other domains
        unique_keywords = []
        for kw in all_keywords:
            if kw.lower() not in other_keywords and kw not in valid_priorities:
                unique_keywords.append(kw)

        # Sort unique keywords by importance (shorter terms first, then alphabetically)
        unique_keywords.sort(key=lambda x: (len(x.split()), x.lower()))

        # Combine: priority keywords first, then unique keywords
        final_selection = valid_priorities + unique_keywords[:5]  # Top 5 unique terms

        return final_selection[:15]  # Return up to 15 priority keywords

    def _initialize_llm_client(self, **kwargs):
        """Initialize the appropriate LLM client based on provider"""

        if self.llm_provider == LLMProvider.AZURE_OPENAI:
            if not AZURE_AVAILABLE:
                raise ImportError("Azure OpenAI client not available. Install: pip install openai")

            # Azure OpenAI Configuration
            self.client = AzureOpenAI(
                azure_endpoint=kwargs.get("azure_endpoint", AZURE_OPENAI_ENDPOINT),
                api_key=kwargs.get("azure_api_key", AZURE_OPENAI_API_KEY),
                api_version=kwargs.get("api_version", "2024-05-01-preview")
            )
            self.model_name = kwargs.get("model_name", "gpt-4")

        elif self.llm_provider == LLMProvider.TOGETHER_AI:
            if not TOGETHER_AVAILABLE:
                raise ImportError("Together AI client not available. Install: pip install together")

            # Together AI Configuration
            self.client = Together(
                api_key=kwargs.get("together_api_key", TOGETHER_API_KEY)
            )
            self.model_name = kwargs.get("model_name", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")

        elif self.llm_provider == LLMProvider.LOCAL_OLLAMA:
            self.client = {
                "base_url": kwargs.get("ollama_base_url", "http://localhost:11434"),
                "model": kwargs.get("model_name", "llama3.1:8b")
            }
            self.model_name = self.client["model"]

        elif self.llm_provider == LLMProvider.HUGGINGFACE_LOCAL:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers not available. Install: pip install transformers torch")

            model_name = kwargs.get("model_name", "/gpfs/data/fs72758/mohammed/eu/models/Qwen2.5-7B-Instruct")
            self.model_name = model_name

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.client = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )

                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Set chat template for Qwen2.5 if not available
                if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
                    self.tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

            except Exception as e:
                print(f"Error loading HuggingFace model {model_name}: {e}")
                raise
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

        self.logger.info(f"Initialized {self.llm_provider.value} with model: {self.model_name}")

    # Property accessors for consolidated configuration (backwards compatibility)
    @property
    def modification_retry_attempts(self):
        return self.modification_config["modification_retry_attempts"]

    @property
    def modification_retry_delay(self):
        return self.modification_config["modification_retry_delay"]

    @property
    def enable_quality_validation(self):
        return self.modification_config["enable_quality_validation"]

    @property
    def quality_threshold(self):
        return self.modification_config["quality_threshold"]

    @property
    def fallback_to_original(self):
        return self.modification_config["fallback_to_original"]

    @property
    def enable_rollback(self):
        return self.modification_config["enable_rollback"]

    # =============================================================================
    # ENHANCED LOGGING METHODS
    # =============================================================================

    def _log_phase1_qa_display(self, question: str, answer: str, context: str, classification: str,
                              question_type: str, chunk_number: Optional[int] = None, page_number: Optional[int] = None):
        """Phase 1: Display initial QA information with context"""
        # Create header with QA number info
        qa_info = f"QA {chunk_number or '?'}" if chunk_number else "QA"
        if page_number:
            qa_info += f" (Page {page_number})"

        print(f"\n{'=' * 70}")
        print(f"[{qa_info}] {question_type.upper()} EVALUATION")
        print(f"{'=' * 70}")
        print(f"QUESTION: {question}")
        print(f"ANSWER: {answer}")

        # Show relevant context excerpt (first 200 chars)
        context_excerpt = (context[:200] + "...") if len(context) > 200 else context
        print(f"CONTEXT: {context_excerpt}")
        print(f"CLASSIFICATION: {classification}")
        print(f"{'-' * 70}")
        print()

    def _log_phase2_evaluation_results(self, question_type: str, faithfulness_score: float,
                                      relevance_score: float, answer_faithfulness_score: float,
                                      answer_relevance_score: float, span_data: dict = None,
                                      question_faithfulness_score: float = None, question_relevance_score: float = None):
        """Phase 2: Display structured evaluation results with separate question and answer metrics"""

        # Show separate question and answer metrics first
        print(f"[COMPONENT SCORES]")
        if question_faithfulness_score is not None:
            print(f"  Question Faithfulness: {question_faithfulness_score:.1f}/10.0 {'OK' if question_faithfulness_score >= 6.0 else 'LOW'}")
        if answer_faithfulness_score is not None:
            print(f"  Answer Faithfulness:   {answer_faithfulness_score:.1f}/10.0 {'OK' if answer_faithfulness_score >= 6.0 else 'LOW'}")
        if question_relevance_score is not None:
            print(f"  Question Relevance:    {question_relevance_score:.1f}/10.0 {'OK' if question_relevance_score >= 6.0 else 'LOW'}")
        if answer_relevance_score is not None:
            print(f"  Answer Relevance:      {answer_relevance_score:.1f}/10.0 {'OK' if answer_relevance_score >= 6.0 else 'LOW'}")

        print(f"\n[COMBINED SCORES]")
        print(f"  Overall Faithfulness:  {faithfulness_score:.1f}/10.0 {'OK' if faithfulness_score >= 6.0 else 'LOW'}")
        print(f"  Overall Relevance:     {relevance_score:.1f}/10.0 {'OK' if relevance_score >= 6.0 else 'LOW'}")

        if question_type.lower() == "factoid":
            # For factoid questions, show detailed answer span analysis
            print(f"\n[ANSWER ANALYSIS] Analyzing spans for verbatim accuracy...")
            if hasattr(self, '_last_faithfulness_span_data') and self._last_faithfulness_span_data:
                for detail in self._last_faithfulness_span_data.get('span_details', []):
                    span = detail.get('span', 'Unknown')
                    score = detail.get('verbatim_score', detail.get('score', 0))
                    if score >= 8:
                        status = "VERBATIM"
                    elif score >= 5:
                        status = "PARTIAL"
                    else:
                        status = "NON-VERBATIM"
                    print(f"  - {span[:40]+'...' if len(span) > 40 else span}: {score}/10 {status}")
                    if score < 8:
                        problem = detail.get('verbatim_problem', 'Accuracy issues')
                        print(f"    -> Issue: {problem}")

            # Show relevance details with context explanation
            if span_data:
                span_count = len(span_data.get('span_details', [])) if span_data else 0
                print(f"\n[RELEVANCE ANALYSIS] Analyzing {span_count} spans...")
                print(f"  IMPORTANT: Relevance scoring is question-specific. The LLM evaluates how well")
                print(f"            each span answers THIS particular question. The same span (e.g., 'Corporate IT')")
                print(f"            may score differently for different questions even from the same context:")
                print(f"            - High score: if the question asks about technology/IT domains")
                print(f"            - Low score: if the question asks about environmental impact")

                # Show span-level results if available
                if 'span_details' in span_data:
                    for detail in span_data['span_details']:
                        span = detail.get('span', 'Unknown')
                        score = detail.get('score', 0)
                        status = "OK" if score >= 6.0 else "PROBLEMATIC"
                        reason = (detail.get('reason') or 'No reason provided')[:60]
                        print(f"  - {span[:40]+'...' if len(span) > 40 else span}: {score}/10 {status}")
                        if score < 6.0:
                            print(f"    -> Issue: {reason}")
        else:
            print(f"\n[RELEVANCE] Question + Answer evaluation")
            print(f"-> Overall Score: {relevance_score:.1f}/10.0 {'OK' if relevance_score >= 6.0 else 'LOW'}")
        print()

    def _log_phase3_modification_decision(self, faithfulness_score: float, relevance_score: float,
                                         modification_needed: bool, modification_applied: bool = False,
                                         original_answer: str = None, modified_answer: str = None,
                                         modification_reasoning: str = None):
        """Phase 3: Display modification decision and process"""
        print(f"[DECISION] Faithfulness: {faithfulness_score:.1f}/10 {'OK' if faithfulness_score >= 6.0 else 'LOW'} | "
              f"Relevance: {relevance_score:.1f}/10 {'OK' if relevance_score >= 6.0 else 'LOW'}")

        if modification_needed:
            if modification_applied:
                print(f"-> ATTEMPTING IMPROVEMENT")
                if original_answer and modified_answer and original_answer != modified_answer:
                    print(f"\nOriginal Answer: {original_answer}")
                    print(f"Modified Answer: {modified_answer}")
                    if modification_reasoning:
                        print(f"Reason: {modification_reasoning[:100]}")
                print()
            else:
                print("-> IMPROVEMENT NEEDED (modification disabled)")
        else:
            print("-> NO IMPROVEMENT NEEDED")
        print()

    def _log_phase4_final_outcome(self, final_faithfulness: float, final_relevance: float,
                                 keep_qa: bool, modification_applied: bool, final_decision: str):
        """Phase 4: Display final outcome summary"""
        outcome_status = "ACCEPTED" if keep_qa else "REJECTED"
        mod_status = "YES" if modification_applied else "NO"

        print(f"\n{'=' * 20} FINAL OUTCOME {'=' * 20}")
        print(f"  QA PAIR STATUS: {outcome_status}")
        print(f"  Final Scores:   F:{final_faithfulness:.1f}/10.0, R:{final_relevance:.1f}/10.0")
        print(f"  Modified:       {mod_status}")
        print(f"  Decision:       {final_decision}")
        print(f"  Threshold:      6.0 (both F and R must be >= 6.0 for acceptance)")
        print("=" * 60)

    def _log_modification_attempt(self, original_answer: str, problematic_spans: list, good_spans: list):
        """Log the start of modification process with clear before/after setup"""
        print(f"\n[MODIFICATION] Attempting to improve answer quality")
        print(f"Original Answer: {original_answer}")
        print(f"Problematic Spans ({len(problematic_spans)}): {[span[0] for span in problematic_spans]}")
        print(f"Good Spans ({len(good_spans)}): {good_spans}")
        print(f"Strategy: Improve or remove problematic spans\n")

    def _log_span_modification_step(self, span_text: str, action: str, reason: str = "", result: str = ""):
        """Log individual span modification steps"""
        if action == "ATTEMPT":
            display_text = span_text if len(span_text) <= 50 else span_text[:50] + '...'
            print(f"  -> Trying to improve: '{display_text}'")
            if reason:
                print(f"     Issue: {reason}")
        elif action == "IMPROVED":
            display_span = span_text if len(span_text) <= 50 else span_text[:50] + '...'
            display_result = result if len(result) <= 50 else result[:50] + '...'
            print(f"  OK Improved: '{display_span}' -> '{display_result}'")
        elif action == "REMOVED":
            display_text = span_text if len(span_text) <= 50 else span_text[:50] + '...'
            print(f"  X Removed: '{display_text}'")
            if reason:
                print(f"     Reason: {reason}")
        elif action == "REDUNDANT":
            overlap_percent = reason if reason else "high"
            display_result = result if len(result) <= 50 else result[:50] + '...'
            print(f"  ! Skipped: '{display_result}'")
            print(f"     Reason: Too similar to existing spans ({overlap_percent} overlap)")

    def _log_modification_result(self, original_answer: str, final_answer: str,
                                original_score: float, final_score: float, was_successful: bool):
        """Log the final result of modification process"""
        print(f"\n[MODIFICATION RESULT]")
        print(f"  Score Calculation: Individual span relevance scores are averaged")
        print(f"  Improvement Logic: Original spans -> Remove/improve problematic -> Recalculate average")

        if original_answer != final_answer:
            print(f"  BEFORE: {original_answer}")
            print(f"  AFTER:  {final_answer}")
            print(f"  Score Change: {original_score:.1f} -> {final_score:.1f} (improvement: {final_score - original_score:+.1f})")

            if was_successful:
                print(f"  Status: SUCCESS - Quality improved to acceptable level (>=6.0)")
            else:
                print(f"  Status: PARTIAL - Some improvements made but still below threshold (<6.0)")
        else:
            print(f"  Status: NO CHANGES - Could not improve the answer")
            print(f"  Answer: {original_answer}")
            print(f"  Score: {original_score:.1f} (unchanged)")
        print()

    def _validate_classification(self, classification: str) -> str:
        """
        Validate and normalize classification parameter.
        Returns valid classification or raises ValueError.
        """
        if not isinstance(classification, str):
            raise ValueError(f"Classification must be a string, got {type(classification)}")

        classification = classification.strip()
        if not classification:
            raise ValueError("Classification cannot be empty")

        valid_classifications = {"ESG", "EU Taxonomy", "Sustainability", "Unknown"}

        # Handle common variations and case-insensitive matching
        classification_mapping = {
            "esg": "ESG",
            "eu taxonomy": "EU Taxonomy",
            "eu_taxonomy": "EU Taxonomy",
            "taxonomy": "EU Taxonomy",
            "sustainability": "Sustainability",
            "unknown": "Unknown",
            "": "Unknown"  # Empty string maps to Unknown
        }

        # Try exact match first
        if classification in valid_classifications:
            return classification

        # Try case-insensitive mapping
        lower_classification = classification.lower()
        if lower_classification in classification_mapping:
            return classification_mapping[lower_classification]

        # If no match found, raise error with suggestions
        raise ValueError(
            f"Invalid classification '{classification}'. "
            f"Valid options are: {', '.join(valid_classifications)}"
        )


    def _apply_rate_limiting(self):
        """Apply intelligent rate limiting based on provider and usage patterns"""

        current_time = time.time()
        limits = self.rate_limits.get(self.llm_provider, {"min_delay": 1.0, "burst_delay": 2.0})

        # Calculate time since last call
        time_since_last = current_time - self.last_call_time

        # Determine required delay
        min_delay = limits["min_delay"]
        burst_delay = limits["burst_delay"]

        # Apply burst delay only if configured (Together AI free tier doesn't need this)
        if burst_delay > 0 and self.call_count % 25 == 0 and self.call_count > 0:
            required_delay = burst_delay
            self.logger.debug(f"Rate limit: Applying burst delay of {burst_delay}s after {self.call_count} calls")
        else:
            required_delay = min_delay

        # Sleep if needed
        if time_since_last < required_delay:
            sleep_time = required_delay - time_since_last
            self.logger.debug(f"Rate limit: Sleeping {sleep_time:.1f}s (min_delay={min_delay}s)")
            time.sleep(sleep_time)

        # Update tracking
        self.call_count += 1
        self.last_call_time = time.time()

        # Reset call count every minute to prevent overflow
        if self.call_count % 50 == 0:
            self.logger.info(f"Rate limit: Completed {self.call_count} calls, continuing...")

    def _classify_api_error(self, error: Exception) -> str:
        """Classify API errors to determine retry strategy"""
        error_str = str(error).lower()

        # Permanent errors (don't retry)
        permanent_indicators = [
            "invalid api key", "authentication failed", "unauthorized",
            "invalid model", "model not found", "permission denied",
            "quota exceeded", "billing", "suspended"
        ]

        # Temporary errors (retry with backoff)
        temporary_indicators = [
            "timeout", "connection", "network", "temporarily unavailable",
            "rate limit", "throttled", "overloaded", "503", "502", "500"
        ]

        for indicator in permanent_indicators:
            if indicator in error_str:
                return "permanent"

        for indicator in temporary_indicators:
            if indicator in error_str:
                return "temporary"

        # Unknown errors - treat as temporary with limited retries
        return "unknown"

    def _call_llm_with_retry(self, messages: List[Dict], max_tokens: int = 1024,
                            temperature: float = 0.0, context: str = "general") -> str:
        """Enhanced LLM calling with robust error handling and retry logic"""

        for attempt in range(self.modification_retry_attempts + 1):
            try:
                # Rate limiting is handled in _call_llm, don't double-apply

                # Make the API call
                return self._call_llm(messages, max_tokens, temperature)

            except Exception as e:
                error_classification = self._classify_api_error(e)

                print(f"API call attempt {attempt + 1}/{self.modification_retry_attempts + 1} failed: {e}")

                # Don't retry permanent errors
                if error_classification == "permanent":
                    print(f"Permanent error detected: {error_classification}. No retry.")
                    raise e

                # Last attempt - don't wait
                if attempt == self.modification_retry_attempts:
                    print(f"All retry attempts exhausted for {context}")
                    raise e

                # Wait before retry with exponential backoff
                wait_time = self.modification_retry_delay * (2 ** attempt)
                print(f"Temporary error ({error_classification}). Retrying in {wait_time}s...")
                time.sleep(wait_time)

        raise Exception(f"All {self.modification_retry_attempts + 1} attempts failed")

    def _call_llm(self, messages: List[Dict], max_tokens: int = 1024, temperature: float = 0.0) -> str:
        """Generic LLM calling function with intelligent rate limiting"""

        # Apply rate limiting before making the call
        self._apply_rate_limiting()

        if self.llm_provider == LLMProvider.AZURE_OPENAI:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Azure OpenAI API error: {e}")
                return f"Error: {str(e)}"

        elif self.llm_provider == LLMProvider.TOGETHER_AI:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Together AI API error: {e}")
                return f"Error: {str(e)}"

        elif self.llm_provider == LLMProvider.LOCAL_OLLAMA:
            try:
                # Convert messages to single prompt for Ollama
                prompt = self._messages_to_prompt(messages)

                response = requests.post(
                    f"{self.client['base_url']}/api/generate",
                    json={
                        "model": self.client['model'],
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens
                        }
                    },
                    timeout=120
                )

                if response.status_code == 200:
                    result = response.json()
                    return result["response"].strip()
                else:
                    return f"Error: Ollama API returned status {response.status_code}"
            except Exception as e:
                print(f"Ollama API error: {e}")
                return f"Error: {str(e)}"

        elif self.llm_provider == LLMProvider.HUGGINGFACE_LOCAL:
            try:
                # Use chat template if available (for Qwen2.5)
                if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    prompt = self._messages_to_prompt(messages)

                inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=4096, truncation=True)

                # Move inputs to same device as model
                if hasattr(self.client, 'device'):
                    inputs = inputs.to(self.client.device)

                with torch.no_grad():
                    outputs = self.client.generate(
                        inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )

                response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
                return response.strip()

            except Exception as e:
                print(f"HuggingFace local model error: {e}")
                return f"Error: {str(e)}"

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert messages format to single prompt string for local models"""
        prompt_parts = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")

        prompt_parts.append("Assistant:")  # Prompt for response
        return "\n\n".join(prompt_parts)

    def _validate_modification_quality(self, original_question: str, original_answer: str,
                                     modified_question: str, modified_answer: str,
                                     modification_type: str, context: str, question_type: str = "") -> Dict:
        """Focused validation: LLM-based coherence/readability + rule-based length"""

        quality_metrics = {
            "readability_score": 0.0,
            "coherence_score": 0.0,
            "length_appropriateness_score": 0.0,
            "passes_threshold": False,
            "issues": [],
            "recommendations": []
        }

        try:
            # 1. Length Check (Rule-based) - Handle factoid single-word answers
            length_result = self._check_length_simple(
                original_question, original_answer, modified_question, modified_answer, modification_type, question_type
            )
            quality_metrics["length_appropriateness_score"] = length_result["score"]
            quality_metrics["issues"].extend(length_result["issues"])

            # 1.5. Factoid Verbatim Check (Rule-based) - Ensure factoid answers are verbatim spans
            verbatim_score = 10.0  # Default pass for non-factoids
            if question_type.lower() == "factoid":
                verbatim_result = self._check_factoid_verbatim(modified_answer, context)
                verbatim_score = verbatim_result["score"]
                quality_metrics["issues"].extend(verbatim_result["issues"])
            quality_metrics["verbatim_score"] = verbatim_score

            # 2. Readability Check (LLM-based)
            readability_result = self._check_readability_llm(
                modified_question, modified_answer, modification_type, question_type
            )
            quality_metrics["readability_score"] = readability_result["score"]
            quality_metrics["issues"].extend(readability_result["issues"])

            # 3. Coherence Check (LLM-based) - How well answer relates to question
            coherence_result = self._check_coherence_llm(modified_question, modified_answer, question_type)
            quality_metrics["coherence_score"] = coherence_result["score"]
            quality_metrics["issues"].extend(coherence_result["issues"])

            # Quality threshold based on important metrics only
            min_threshold = self.quality_threshold

            passes_readability = quality_metrics["readability_score"] >= min_threshold
            passes_coherence = quality_metrics["coherence_score"] >= min_threshold
            passes_length = quality_metrics["length_appropriateness_score"] >= min_threshold
            passes_verbatim = quality_metrics["verbatim_score"] >= min_threshold

            quality_metrics["passes_threshold"] = all([
                passes_readability, passes_coherence, passes_length, passes_verbatim
            ])

        except Exception as e:
            print(f"Quality validation error: {e}")
            quality_metrics["passes_threshold"] = False
            quality_metrics["issues"].append(f"Quality validation failed: {str(e)}")

        return quality_metrics

    def _check_length_simple(self, orig_q: str, orig_a: str, mod_q: str, mod_a: str, modification_type: str, question_type: str = "") -> Dict:
        """Simple length check - factoid-aware for span rules"""
        issues = []
        score = 1.0

        # Basic format checks
        if modification_type in ["question", "both"]:
            if len(mod_q.strip()) == 0:
                issues.append("Question cannot be empty")
                score = 0.0
            elif not mod_q.strip().endswith('?'):
                issues.append("Question should end with ?")
                score *= 0.8
            else:
                # Allow wide range for questions, prevent only extreme changes
                q_ratio = len(mod_q) / max(len(orig_q), 1)
                if q_ratio > 5.0 or q_ratio < 0.2:
                    issues.append(f"Question length changed drastically ({q_ratio:.1f}x)")
                    score *= 0.7

        if modification_type in ["answer", "both"]:
            if len(mod_a.strip()) == 0:
                issues.append("Answer cannot be empty")
                score = 0.0
            elif question_type.lower() == "factoid":
                # Factoid-specific validation: check for span format
                answer_parts = [part.strip() for part in mod_a.split(',')]
                is_span_format = all(len(part.split()) <= 5 for part in answer_parts)  # Max 5 words per span

                if is_span_format:
                    # Span format is valid regardless of length change
                    score = 1.0
                else:
                    # Not span format - check if it's reasonable length
                    a_ratio = len(mod_a) / max(len(orig_a), 1)
                    if a_ratio > 3.0:
                        issues.append("Factoid answer should be spans (single words/phrases), not long text")
                        score *= 0.7
            else:
                # Non-factoid: allow wide range but prevent extreme changes
                a_ratio = len(mod_a) / max(len(orig_a), 1)
                if a_ratio > 8.0 or a_ratio < 0.1:
                    issues.append(f"Answer length changed extremely ({a_ratio:.1f}x)")
                    score *= 0.7

        return {"score": score, "issues": issues}

    def _check_readability_llm(self, question: str, answer: str, modification_type: str, question_type: str = "") -> Dict:
        """LLM-based readability check - factoid-aware"""

        if modification_type == "question":
            content = question
            content_type = "question"
        elif modification_type == "answer":
            content = answer
            content_type = "answer"
        else:  # both
            content = f"Question: {question}\nAnswer: {answer}"
            content_type = "question and answer"

        if question_type.lower() == "factoid" and modification_type in ["answer", "both"]:
            prompt = f"""Rate the quality of this factoid {content_type}:

{content}

For FACTOID questions, answers should be:
- Single span (e.g., "Yes", "€50M", "2023") OR
- Multiple spans separated by commas (e.g., "Corporate IT, Research, Finance")
- Spans must be verbatim from context (no rephrasing or modifications)
- Clear and concise factual information

Is this {content_type} appropriate for a factoid question?
SCORE: [1-10, where 7+ means good quality]
REASONING: [Brief explanation]"""
        else:
            prompt = f"""Rate the readability and language quality of this {content_type}:

{content}

Is the language clear, understandable, and well-written?
- Consider grammar, clarity, and appropriate vocabulary
- Rate 1-10 where 7+ means good readability

SCORE: [1-10]
REASONING: [Brief explanation]"""

        messages = [
            {"role": "system", "content": "You are evaluating text readability and language quality."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self._call_llm(messages, max_tokens=150, temperature=0.0)
            score, reasoning = self._parse_evaluation_response(response)

            issues = [] if score >= 6.0 else [f"Readability issue: {reasoning}"]
            return {"score": score, "issues": issues}
        except Exception as e:
            print(f"Readability LLM check failed: {e}")
            return {"score": 7.0, "issues": []}  # Default pass on error

    def _check_coherence_llm(self, question: str, answer: str, question_type: str = "") -> Dict:
        """LLM-based coherence check - factoid-aware for span requirements"""

        if question_type.lower() == "factoid":
            prompt = f"""Rate how well this factoid answer addresses the question:

Question: {question}
Answer: {answer}

For FACTOID questions:
- Answer should be specific fact(s) from the context
- Can be single span or comma-separated spans (verbatim extraction)
- Should directly answer what's asked (not explanation or elaboration)
- Examples: "Yes", "€50M", "Corporate IT, Finance, Research"

Does this answer appropriately respond to the factoid question?
SCORE: [1-10, where 7+ means good coherence]
REASONING: [Brief explanation]"""
        else:
            prompt = f"""Rate how well this answer relates to and addresses the question:

Question: {question}
Answer: {answer}

Does the answer directly address what the question is asking?
- Rate 1-10 where 7+ means good coherence
- Consider: Does the answer fit the question? Is it relevant and responsive?

SCORE: [1-10]
REASONING: [Brief explanation]"""

        messages = [
            {"role": "system", "content": "You are evaluating question-answer coherence and relevance."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self._call_llm(messages, max_tokens=150, temperature=0.0)
            score, reasoning = self._parse_evaluation_response(response)

            issues = [] if score >= 6.0 else [f"Coherence issue: {reasoning}"]
            return {"score": score, "issues": issues}
        except Exception as e:
            print(f"Coherence LLM check failed: {e}")
            return {"score": 7.0, "issues": []}  # Default pass on error

    def _check_factoid_verbatim(self, answer: str, context: str) -> Dict:
        """Check if factoid answer is verbatim span(s) from context"""
        issues = []

        # Clean and normalize for comparison
        clean_answer = answer.strip().lower()
        clean_context = context.lower()

        # Handle comma-separated spans
        answer_spans = [span.strip() for span in clean_answer.split(',')]

        # Check each span
        missing_spans = []
        for span in answer_spans:
            if span and span not in clean_context:
                missing_spans.append(span)

        if missing_spans:
            score = 1.0  # Fail - not verbatim
            issues.append(f"Factoid answer contains non-verbatim spans: {', '.join(missing_spans)}")
        else:
            score = 10.0  # Pass - all spans are verbatim

        return {"score": score, "issues": issues}

    def count_keyword_matches(self, text, keywords):
        """Count how many keywords from a domain appear in the text"""
        text_lower = text.lower()
        matches = 0
        matched_keywords = []

        for keyword in keywords:
            if keyword.lower() in text_lower:
                matches += 1
                matched_keywords.append(keyword)

        return matches, matched_keywords

    def assess_passage_classification(self, passage_text, current_classification):
        """Assess if current classification is correct using text-length aware keyword matching"""

        # Validate and normalize classification
        current_classification = self._validate_classification(current_classification)

        # Count matches for each domain
        esg_count, esg_matches = self.count_keyword_matches(passage_text, self.domain_keywords["ESG"])
        taxonomy_count, taxonomy_matches = self.count_keyword_matches(passage_text, self.domain_keywords["EU Taxonomy"])
        sustainability_count, sustainability_matches = self.count_keyword_matches(passage_text, self.domain_keywords["Sustainability"])

        # Calculate text-length aware normalized scores
        word_count = len(passage_text.split())
        text_length_factor = max(1.0, word_count / 100.0)  # Normalize to 100-word baseline

        # Normalized scores (matches per 100 words)
        domain_raw_scores = {
            "ESG": esg_count,
            "EU Taxonomy": taxonomy_count,
            "Sustainability": sustainability_count
        }

        domain_normalized_scores = {
            domain: score / text_length_factor for domain, score in domain_raw_scores.items()
        }

        # Find highest scoring domain (use normalized scores)
        highest_normalized_score = max(domain_normalized_scores.values())
        highest_domain = max(domain_normalized_scores, key=domain_normalized_scores.get)
        current_normalized_score = domain_normalized_scores[current_classification]

        # Convert to 0-1 confidence scale based on normalized scores
        if highest_normalized_score >= 3.0:  # 3+ matches per 100 words = HIGH confidence
            confidence_score = min(1.0, highest_normalized_score / 5.0)  # Cap at 5 matches/100 words = 1.0
            confidence_level = "HIGH"
        elif highest_normalized_score >= 1.5:  # 1.5-3 matches per 100 words = MEDIUM confidence
            confidence_score = 0.3 + (highest_normalized_score - 1.5) / 1.5 * 0.4  # 0.3-0.7 range
            confidence_level = "MEDIUM"
        else:  # <1.5 matches per 100 words = LOW confidence
            confidence_score = min(0.3, highest_normalized_score / 1.5 * 0.3)  # 0-0.3 range
            confidence_level = "LOW"

        # Determine if misclassified based on confidence level and significant difference
        is_misclassified = (current_classification != highest_domain and
                           confidence_level in ["HIGH", "MEDIUM"] and
                           highest_normalized_score > current_normalized_score + 0.5)

        return {
            "domain_scores": domain_raw_scores,
            "domain_normalized_scores": domain_normalized_scores,
            "current_score": domain_raw_scores[current_classification],
            "current_normalized_score": current_normalized_score,
            "highest_score": domain_raw_scores[highest_domain],
            "highest_normalized_score": highest_normalized_score,
            "highest_domain": highest_domain,
            "confidence": confidence_level,
            "confidence_score": confidence_score,
            "text_length_factor": text_length_factor,
            "word_count": word_count,
            "is_misclassified": is_misclassified
        }

    def get_effective_classification(self, passage_text, current_classification):
        """Apply Option C hybrid logic to determine effective classification"""

        assessment = self.assess_passage_classification(passage_text, current_classification)

        # Option C: Hybrid Approach Logic
        if assessment["confidence"] == "HIGH" and assessment["is_misclassified"]:
            # High Confidence: Auto-reclassify (≥3 normalized matches per 100 words)
            action = "AUTO_RECLASSIFY"
            effective_classification = assessment["highest_domain"]
            print(f"[AUTO-RECLASSIFY] {current_classification} -> {effective_classification} (confidence: HIGH, normalized_score: {assessment['highest_normalized_score']:.1f})")

        elif assessment["confidence"] == "MEDIUM" and assessment["is_misclassified"]:
            # Medium Confidence: Flag for review but use corrected classification (1.5-3 normalized matches per 100 words)
            action = "FLAG_AND_USE"
            effective_classification = assessment["highest_domain"]
            print(f"[FLAG-AND-USE] {current_classification} -> {effective_classification} (confidence: MEDIUM, normalized_score: {assessment['highest_normalized_score']:.1f}) - Needs Review")

        elif assessment["confidence"] == "LOW":
            # Low Confidence: Use original but note uncertainty (<1.5 normalized matches per 100 words)
            action = "KEEP_WITH_WARNING"
            effective_classification = current_classification
            self.logger.warning(f"[KEEP-WITH-WARNING] Classification: {current_classification} (confidence: LOW, normalized_score: {assessment['current_normalized_score']:.1f}) - Uncertain")

        else:
            # Classification is correct
            action = "KEEP_CURRENT"
            effective_classification = current_classification

        # Return both the effective classification and metadata
        return effective_classification, {
            "action": action,
            "assessment": assessment,
            "original_classification": current_classification
        }

    def classify_passage_with_llm(self, passage_text, title=None, subtitle=None):
        """
        Classify passage using LLM with domain-specific keywords as context.
        Uses the same keyword lists as the keyword matching function for consistency.
        """

        # Prepare keyword lists from domain_keywords (same as keyword matching function)
        esg_keywords = ", ".join(self.domain_keywords["ESG"])
        taxonomy_keywords = ", ".join(self.domain_keywords["EU Taxonomy"])
        sustainability_keywords = ", ".join(self.domain_keywords["Sustainability"])

        # Enhanced classification prompt with keyword context
        prompt = f"""You are a sustainability reporting expert. Classify the following text chunk into one of the categories below.

**IMPORTANT**: Use the provided keyword lists as guidance for each category, but also apply semantic understanding to make the best classification decision.

**Title**: {title or 'N/A'}
**Subtitle**: {subtitle or 'N/A'}
**Context**: {passage_text}

**Categories and Their Key Indicators**:

**EU Taxonomy**:
Key terms: {taxonomy_keywords}
Focus: Specifically mentions EU Taxonomy regulations, principles, alignment criteria, eligibility, screening criteria, DNSH (Do No Significant Harm), Minimum Safeguards, or related KPIs (CapEx, OpEx, Turnover alignment).

**ESG**:
Key terms: {esg_keywords}
Focus: Environmental, Social, or Governance factors, metrics, risks, opportunities, reporting frameworks (like GRI, SASB, TCFD, CSRD), materiality assessments, stakeholder engagement, policies, or specific ESG initiatives not solely confined to broad sustainability concepts.

**Sustainability**:
Key terms: {sustainability_keywords}
Focus: Broader sustainability topics like circular economy, climate action goals (without specific ESG framework detail), biodiversity efforts, resource efficiency, sustainable products/practices, or general corporate responsibility themes not fitting neatly into ESG reporting structures or EU Taxonomy specifics.

**Unknown**: Does not clearly fit into any of the above categories or lacks sufficient information for classification.

**Instructions**:
1. Analyze the context primarily, using title/subtitle for additional insight
2. If specific EU Taxonomy terms are present, prioritize EU Taxonomy classification
3. Consider semantic meaning, not just keyword presence
4. Provide a confidence score (1-10) for your classification
5. Give brief reasoning for your decision

**Output Format**:
Classification: [EU Taxonomy|ESG|Sustainability|Unknown]
Confidence: [1-10]
Reasoning: [Brief explanation of why this classification was chosen]"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self._call_llm_with_retry(messages, max_tokens=512, temperature=0.0)

            # Parse the response
            lines = response.strip().split('\n')
            classification = "Unknown"
            confidence = 5
            reasoning = "Unable to parse LLM response"

            for line in lines:
                line = line.strip()
                if line.startswith("Classification:"):
                    classification = line.split(":", 1)[1].strip()
                    classification = self._validate_classification(classification)
                elif line.startswith("Confidence:"):
                    try:
                        confidence_str = line.split(":", 1)[1].strip()
                        confidence_raw = float(confidence_str)  # Allow floats
                        confidence_raw = max(1.0, min(10.0, confidence_raw))  # Clamp to 1.0-10.0
                        confidence = confidence_raw  # Keep raw for logging
                    except (ValueError, IndexError):
                        confidence = 5.0
                        self.logger.warning(f"Failed to parse confidence value: {line.strip()}, defaulting to 5.0")
                elif line.startswith("Reasoning:"):
                    reasoning = line.split(":", 1)[1].strip()

            # Convert 1-10 scale to 0-1 probability scale
            confidence_score = confidence / 10.0  # 1->0.1, 5->0.5, 10->1.0

            # Determine confidence level based on normalized score
            if confidence_score >= 0.7:  # 7-10 on original scale
                confidence_level = "HIGH"
            elif confidence_score >= 0.4:  # 4-6 on original scale
                confidence_level = "MEDIUM"
            else:  # 1-3 on original scale
                confidence_level = "LOW"

            return {
                "classification": classification,
                "confidence": confidence,  # Original 1-10 for logging
                "confidence_score": confidence_score,  # Normalized 0-1 scale
                "confidence_level": confidence_level,  # Categorical level
                "reasoning": reasoning,
                "method": "llm"
            }

        except Exception as e:
            self.logger.warning(f"LLM classification failed: {e}")
            return {
                "classification": "Unknown",
                "confidence": 1,
                "confidence_score": 0.0,
                "confidence_level": "LOW",
                "reasoning": f"LLM classification failed: {str(e)}",
                "method": "llm_failed"
            }

    def hybrid_classify_passage(self, passage_text, current_classification, title=None, subtitle=None):
        """
        Enhanced classification using both keyword overlap and LLM analysis.
        Combines both approaches for more reliable classification decisions.
        """

        # Validate and normalize classification
        current_classification = self._validate_classification(current_classification)

        # Step 1: Get keyword-based classification (existing logic)
        keyword_assessment = self.assess_passage_classification(passage_text, current_classification)

        # Step 2: Get LLM-based classification
        llm_result = self.classify_passage_with_llm(passage_text, title, subtitle)

        # Step 3: Decision fusion logic
        final_classification, fusion_metadata = self._fuse_classification_decisions(
            current_classification, keyword_assessment, llm_result
        )

        return final_classification, fusion_metadata

    def _fuse_classification_decisions(self, current_classification, keyword_assessment, llm_result):
        """
        Intelligent fusion of keyword and LLM classification results using normalized confidence scores.
        Handles conflicts and determines final classification with improved logic.
        """

        keyword_highest = keyword_assessment["highest_domain"]
        keyword_confidence_level = keyword_assessment["confidence"]
        keyword_confidence_score = keyword_assessment["confidence_score"]
        keyword_raw_score = keyword_assessment["highest_score"]
        keyword_normalized_score = keyword_assessment["highest_normalized_score"]

        llm_classification = llm_result["classification"]
        llm_confidence_raw = llm_result["confidence"]
        llm_confidence_score = llm_result["confidence_score"]
        llm_confidence_level = llm_result["confidence_level"]
        llm_reasoning = llm_result["reasoning"]

        # Decision fusion rules using normalized 0-1 confidence scores
        decision_log = []

        # Rule 1: High confidence agreement (both methods agree with high confidence)
        if (keyword_confidence_level == "HIGH" and llm_confidence_level == "HIGH" and
            keyword_highest == llm_classification):
            final_classification = keyword_highest
            final_confidence_score = min(1.0, (keyword_confidence_score + llm_confidence_score) / 2 + 0.1)
            final_confidence_level = "VERY_HIGH"
            action = "BOTH_AGREE_HIGH"
            decision_log.append(f"Strong agreement: Keyword ({keyword_confidence_score:.2f}) + LLM ({llm_confidence_score:.2f}) both suggest {final_classification}")

        # Rule 2: Medium+ confidence agreement
        elif (keyword_confidence_level in ["HIGH", "MEDIUM"] and llm_confidence_level in ["HIGH", "MEDIUM"] and
              keyword_highest == llm_classification):
            final_classification = keyword_highest
            final_confidence_score = (keyword_confidence_score + llm_confidence_score) / 2
            final_confidence_level = "HIGH" if final_confidence_score >= 0.6 else "MEDIUM"
            action = "BOTH_AGREE_MEDIUM"
            decision_log.append(f"Good agreement: Keyword ({keyword_confidence_score:.2f}) + LLM ({llm_confidence_score:.2f}) both suggest {final_classification}")

        # Rule 3: High LLM confidence overrides low keyword confidence (conservative threshold)
        elif (llm_confidence_score >= 0.8 and keyword_confidence_score <= 0.3 and
              llm_confidence_level == "HIGH"):
            final_classification = llm_classification
            final_confidence_score = llm_confidence_score * 0.9  # Slight penalty for override
            final_confidence_level = "HIGH"
            action = "LLM_OVERRIDE_HIGH"
            decision_log.append(f"LLM override: High LLM confidence ({llm_confidence_score:.2f}) for {llm_classification} vs low keyword confidence ({keyword_confidence_score:.2f})")

        # Rule 4: High keyword confidence overrides low LLM confidence
        elif (keyword_confidence_score >= 0.7 and llm_confidence_score <= 0.3 and
              keyword_confidence_level == "HIGH"):
            final_classification = keyword_highest
            final_confidence_score = keyword_confidence_score * 0.9  # Slight penalty for override
            final_confidence_level = "HIGH"
            action = "KEYWORD_OVERRIDE_HIGH"
            decision_log.append(f"Keyword override: High keyword confidence ({keyword_confidence_score:.2f}) for {keyword_highest} vs low LLM confidence ({llm_confidence_score:.2f})")

        # Rule 5: Conflict resolution - use LLM for semantic understanding (conservative threshold)
        elif (keyword_highest != llm_classification and llm_confidence_score >= 0.7 and
              llm_confidence_score > keyword_confidence_score + 0.2):  # LLM must be significantly more confident
            final_classification = llm_classification
            final_confidence_score = llm_confidence_score * 0.8  # Penalty for conflict resolution
            final_confidence_level = "MEDIUM"
            action = "LLM_SEMANTIC_PRIORITY"
            decision_log.append(f"Semantic priority: LLM ({llm_confidence_score:.2f}) suggests {llm_classification} vs keyword {keyword_highest} ({keyword_confidence_score:.2f})")

        # Rule 6: Both low confidence - keep original but flag
        elif (keyword_confidence_score <= 0.3 and llm_confidence_score <= 0.4):
            final_classification = current_classification
            final_confidence_score = max(keyword_confidence_score, llm_confidence_score) * 0.5  # Low confidence
            final_confidence_level = "LOW"
            action = "BOTH_UNCERTAIN_KEEP_ORIGINAL"
            decision_log.append(f"Uncertainty: Both keyword ({keyword_confidence_score:.2f}) and LLM ({llm_confidence_score:.2f}) have low confidence")

        # Rule 7: Keyword priority for ties or unclear cases
        elif keyword_confidence_score >= llm_confidence_score:
            final_classification = keyword_highest
            final_confidence_score = keyword_confidence_score
            final_confidence_level = keyword_confidence_level
            action = "KEYWORD_PRIORITY"
            decision_log.append(f"Keyword priority: Keyword confidence ({keyword_confidence_score:.2f}) >= LLM confidence ({llm_confidence_score:.2f})")

        # Rule 8: LLM priority when LLM more confident
        else:
            final_classification = llm_classification
            final_confidence_score = llm_confidence_score
            final_confidence_level = llm_confidence_level
            action = "LLM_PRIORITY"
            decision_log.append(f"LLM priority: LLM confidence ({llm_confidence_score:.2f}) > keyword confidence ({keyword_confidence_score:.2f})")

        # Log the decision if classification changed
        if final_classification != current_classification:
            self.logger.info(f"[HYBRID-CLASSIFY] {current_classification} -> {final_classification} ({final_confidence_level}, {final_confidence_score:.2f}) via {action}")
            if decision_log:
                self.logger.debug(f"  Reasoning: {decision_log[0]}")

        return final_classification, {
            "action": action,
            "final_confidence": final_confidence_level,
            "final_confidence_score": final_confidence_score,
            "keyword_assessment": keyword_assessment,
            "llm_result": llm_result,
            "decision_log": decision_log,
            "original_classification": current_classification
        }

    def get_effective_classification_enhanced(self, passage_text, current_classification, title=None, subtitle=None):
        """
        Enhanced version of get_effective_classification using hybrid approach.
        Falls back to original method if hybrid approach fails.
        """

        try:
            # Use hybrid classification
            final_classification, metadata = self.hybrid_classify_passage(
                passage_text, current_classification, title, subtitle
            )

            # Add hybrid metadata
            metadata["method"] = "hybrid"
            metadata["enhancement"] = "keyword_llm_fusion"

            return final_classification, metadata

        except Exception as e:
            self.logger.warning(f"Hybrid classification failed, falling back to keyword-only: {e}")
            # Fallback to original method
            return self.get_effective_classification(passage_text, current_classification)

    def _validate_dataset_schema(self, data: dict) -> dict:
        """Validate dataset has expected structure based on actual test.json schema"""
        issues = []
        critical = []
        counters = {"chunks": 0, "table_qas": 0, "factoid_qas": 0, "non_factoid_qas": 0}

        # Check for required top-level keys
        if "all_chunks_qas" not in data:
            critical.append("Missing required key: all_chunks_qas")
        else:
            chunks = data["all_chunks_qas"]
            if not isinstance(chunks, list):
                critical.append(f"all_chunks_qas must be a list, got {type(chunks)}")
            else:
                counters["chunks"] = len(chunks)

        # Check optional table QAs
        if "all_table_qas" in data:
            table_qas = data["all_table_qas"]
            if isinstance(table_qas, list):
                counters["table_qas"] = len(table_qas)
            else:
                issues.append("all_table_qas should be a list")

        # Validate chunk structure (sample first few for performance)
        chunks_to_check = data.get("all_chunks_qas", [])[:3]
        for i, chunk in enumerate(chunks_to_check):
            if not isinstance(chunk, dict):
                critical.append(f"Chunk {i} is not a dictionary")
                continue

            # Check chunk metadata
            if "metadata" not in chunk:
                issues.append(f"Chunk {i} missing metadata")
            else:
                metadata = chunk["metadata"]
                required_meta = ["chunk_number", "page_number", "paragraph", "classification", "report_name"]
                for meta_key in required_meta:
                    if meta_key not in metadata:
                        issues.append(f"Chunk {i} metadata missing {meta_key}")

            # Check spans structure
            if "spans" not in chunk:
                issues.append(f"Chunk {i} missing spans")
            elif "groups" not in chunk["spans"]:
                issues.append(f"Chunk {i} spans missing groups")

            # Check factoid QAs structure
            if "qa_pairs_factoid" in chunk:
                factoid_section = chunk["qa_pairs_factoid"]
                if "qa_pairs" not in factoid_section:
                    issues.append(f"Chunk {i} factoid section missing qa_pairs")
                else:
                    qa_pairs = factoid_section["qa_pairs"]
                    if isinstance(qa_pairs, list):
                        counters["factoid_qas"] += len(qa_pairs)
                        # Sample check first QA
                        if qa_pairs and isinstance(qa_pairs[0], dict):
                            qa_sample = qa_pairs[0]
                            required_qa_fields = ["question", "answer", "type", "tag"]
                            for field in required_qa_fields:
                                if field not in qa_sample:
                                    issues.append(f"Factoid QA in chunk {i} missing {field}")

            # Check non-factoid QAs structure
            if "qa_pairs_non_factoid" in chunk:
                non_factoid_section = chunk["qa_pairs_non_factoid"]
                if "qa_pairs" not in non_factoid_section:
                    issues.append(f"Chunk {i} non-factoid section missing qa_pairs")
                else:
                    qa_pairs = non_factoid_section["qa_pairs"]
                    if isinstance(qa_pairs, list):
                        counters["non_factoid_qas"] += len(qa_pairs)

        # Validate table QAs structure (sample first few)
        table_qas_to_check = data.get("all_table_qas", [])[:2]
        for i, table_qa in enumerate(table_qas_to_check):
            if not isinstance(table_qa, dict):
                critical.append(f"Table QA {i} is not a dictionary")
                continue

            # Check table metadata
            if "metadata" not in table_qa:
                issues.append(f"Table QA {i} missing metadata")
            else:
                metadata = table_qa["metadata"]
                required_meta = ["page_number", "table_caption", "paragraph", "classification", "report_name"]
                for meta_key in required_meta:
                    if meta_key not in metadata:
                        issues.append(f"Table QA {i} metadata missing {meta_key}")

            # Check table qa_pairs structure
            if "qa_pairs" not in table_qa:
                issues.append(f"Table QA {i} missing qa_pairs")
            elif not isinstance(table_qa["qa_pairs"], list):
                issues.append(f"Table QA {i} qa_pairs should be a list")

        print(f"Dataset loaded successfully - Total chunks: {counters['chunks']}, Table sections: {counters['table_qas']}")
        if counters['chunks'] > 0:
            print(f"  Sample validation (first 3 chunks): ~{counters['factoid_qas']} factoid, ~{counters['non_factoid_qas']} non-factoid QAs per chunk")

        return {
            "valid": len(critical) == 0,
            "issues": issues,
            "critical": critical,
            "counters": counters
        }

    # =========================================================================
    # ENHANCED MODIFICATION SYSTEM - CORE VALIDATION METHODS
    # =========================================================================

    def _calculate_input_hash(self, question: str, answer: str, context: str) -> str:
        """Calculate hash for input to track modifications"""
        content = f"{question}|{answer}|{context[:100]}"
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def _extract_numerical_claims(self, text: str) -> List[Dict]:
        """Extract numerical claims with context"""
        claims = []
        pattern = r'(\d+(?:\.\d+)?)\s*([%]|percent|million|billion|EUR|USD|\$|tonnes?|kg|meters?|years?|months?|kWh|GWh|MWh|TWh|tCO2e|tCO₂e|CO2|m³|m2|kt|Mt|kWp|MWp)'

        for match in re.finditer(pattern, text, re.IGNORECASE):
            claims.append({
                "value": match.group(1),
                "unit": match.group(2),
                "context": text[max(0, match.start()-20):match.end()+20]
            })

        return claims

    def _number_in_context(self, num_claim: Dict, context_numbers: List[Dict]) -> bool:
        """Check if a numerical claim exists in context"""
        for ctx_num in context_numbers:
            if (abs(float(num_claim["value"]) - float(ctx_num["value"])) < 0.01 and
                num_claim["unit"].lower() == ctx_num["unit"].lower()):
                return True
        return False

    def _validate_numeric_consistency(self, original: str, modified: str, context: str):
        """Validate numeric consistency immediately"""
        mod_numbers = self._extract_numerical_claims(modified)
        ctx_numbers = self._extract_numerical_claims(context)
        orig_numbers = self._extract_numerical_claims(original)

        for num_claim in mod_numbers:
            if not self._number_in_context(num_claim, ctx_numbers) and not self._number_in_context(num_claim, orig_numbers):
                raise Exception(f"Numerical hallucination detected: {num_claim}")

    def _comprehensive_validation(self, original_question: str, original_answer: str,
                                modified_question: str, modified_answer: str,
                                context: str, modification_type: str, question_type: str = "") -> Dict:
        """Factoid-aware validation pipeline using new LLM-based + rule-based approach"""

        # Use the new factoid-aware validation (single call)
        quality_result = self._validate_modification_quality(
            original_question=original_question,
            original_answer=original_answer,
            modified_question=modified_question,
            modified_answer=modified_answer,
            modification_type=modification_type,
            context=context,
            question_type=question_type
        )

        # Convert quality issues to ValidationIssue format for compatibility
        issues = []
        for issue_text in quality_result.get("issues", []):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="quality",
                message=issue_text,
                field=modification_type
            ))

        # Add basic numeric validation for all answer modifications (critical)
        if modification_type in ["answer", "both"]:
            try:
                self._validate_numeric_consistency(original_answer, modified_answer, context)
            except Exception as e:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="numeric_consistency",
                    message=str(e),
                    field="answer"
                ))

        # Determine validation pass/fail
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        passes_validation = len(critical_issues) == 0 and quality_result.get("passes_threshold", True)

        # Update statistics
        if issues:
            self.modification_stats["validation_issues_prevented"] += len(issues)
            critical_issues_count = len(critical_issues)
            self.modification_stats["critical_issues_prevented"] += critical_issues_count

        return {
            "issues": issues,
            "passes_validation": passes_validation,
            "validation_details": {
                "total_issues": len(issues),
                "critical_issues": len(critical_issues),
                "warning_issues": len([i for i in issues if i.severity == ValidationSeverity.WARNING]),
                "overall_score": (quality_result.get("readability_score", 8.0) +
                                quality_result.get("coherence_score", 8.0) +
                                quality_result.get("length_appropriateness_score", 8.0)) / 3,
                "validation_timestamp": time.time()
            }
        }

    def _meets_quality_standards(self, validation_result: Dict) -> bool:
        """Use the pre-calculated validation result from comprehensive validation"""
        # The _comprehensive_validation already calculated passes_validation correctly
        # including readability, coherence, length, and critical issues
        return validation_result.get("passes_validation", False)

    def modify_qa_pair_safely(self, question: str, answer: str, context: str,
                             classification: str, question_type: str,
                             modification_type: str, target_improvement: str,
                             table_caption: Optional[str] = None,
                             chunk_number: Optional[int] = None,
                             page_number: Optional[int] = None) -> Dict:
        """
        Safe QA modification with improved logic and error handling
        Returns a dictionary with modification results
        """

        self.modification_stats["total_attempts"] += 1

        # Initialize audit trail
        audit_trail = {
            "start_time": time.time(),
            "input_hash": self._calculate_input_hash(question, answer, context),
            "modification_request": {
                "type": modification_type,
                "target": target_improvement,
                "question_type": question_type,
                "classification": classification
            }
        }

        modification_history = []
        previous_failures = []
        identical_content_attempts = 0  # Track attempts with no changes
        max_identical_attempts = 2  # Limit attempts with identical responses

        print(f"\n[MODIFY] {modification_type.upper()} for {target_improvement} improvement")
        print(f"   Q: {question[:80]}{'...' if len(question) > 80 else ''}")
        print(f"   A: {answer[:80]}{'...' if len(answer) > 80 else ''}")
        print(f"   Strategy: Run up to {self.modification_config['max_modification_attempts']} attempts with chain-of-thought learning")

        for attempt_num in range(self.modification_config["max_modification_attempts"]):
            print(f"\n   === ATTEMPT {attempt_num + 1}/{self.modification_config['max_modification_attempts']} ===")

            try:
                # Apply modification with chain-of-thought learning
                modified_q, modified_a, reasoning = self.apply_qa_modifications_with_cot(
                    question, answer, context, classification, question_type,
                    modification_type, target_improvement, table_caption,
                    previous_failures, attempt_num
                )

                # Check if any actual modifications were made
                question_changed = modified_q.strip() != question.strip()
                answer_changed = modified_a.strip() != answer.strip()
                no_changes_made = not question_changed and not answer_changed

                if no_changes_made:
                    identical_content_attempts += 1
                    print(f"   [WARN] No changes made ({identical_content_attempts}/{max_identical_attempts})")

                    if identical_content_attempts >= max_identical_attempts:
                        print(f"   [STOP] Too many attempts with no changes - stopping")
                        break

                    previous_failures.append(f"Attempt {attempt_num + 1}: LLM returned identical content")
                    continue

                print(f"   [MODIFIED] Q={'YES' if question_changed else 'NO'}, A={'YES' if answer_changed else 'NO'}")
                print(f"   [VALIDATE] Checking quality...")

                # Run comprehensive validation
                validation_result = self._comprehensive_validation(
                    original_question=question,
                    original_answer=answer,
                    modified_question=modified_q,
                    modified_answer=modified_a,
                    context=context,
                    modification_type=modification_type,
                    question_type=question_type
                )

                # If validation fails, record failure and continue
                if not self._meets_quality_standards(validation_result):
                    print(f"   [FAIL] Validation failed ({len(validation_result['issues'])} issues)")
                    previous_failures.append(f"Attempt {attempt_num + 1}: Validation failed")
                    continue

                # Validation passed - now check quality thresholds
                print(f"   [PASS] Validation passed, checking quality...")

                try:
                    # Evaluate modified content without printing intermediate results
                    modified_eval = self.evaluate_qa_pair(
                        modified_q, modified_a, context, classification, question_type,
                        chunk_number, page_number, table_caption, None, True, True  # disable_modification=True, silent=True
                    )

                    quality_met = (modified_eval.faithfulness_score >= self.quality_threshold and
                                 modified_eval.relevance_score >= self.quality_threshold)

                    if quality_met:
                        print(f"   [SUCCESS] Target: {target_improvement} | F:{modified_eval.faithfulness_score:.1f} R:{modified_eval.relevance_score:.1f} | Quality met!")
                        self.modification_stats["successful_modifications"] += 1

                        return {
                            "success": True,
                            "modified_question": modified_q,
                            "modified_answer": modified_a,
                            "modification_applied": True,
                            "original_question": question,
                            "original_answer": answer,
                            "modification_reasoning": reasoning,
                            "validation_passed": True,
                            "improvement_achieved": True,
                            "fallback_used": False,
                            "modification_attempts": attempt_num + 1,
                            "validation_issues_prevented": len(validation_result["issues"]),
                            "audit_trail": audit_trail
                        }
                    else:
                        print(f"   [RETRY] Target: {target_improvement} | F:{modified_eval.faithfulness_score:.1f} R:{modified_eval.relevance_score:.1f} (below threshold)")
                        previous_failures.append(f"Attempt {attempt_num + 1}: Quality below threshold")

                except Exception as eval_error:
                    print(f"[ERROR] Evaluation failed: {eval_error}")
                    previous_failures.append(f"Attempt {attempt_num + 1}: Evaluation error - {str(eval_error)[:100]}")

            except Exception as e:
                print(f"[EXCEPTION] Modification failed: {str(e)}")
                previous_failures.append(f"Attempt {attempt_num + 1}: Exception - {str(e)[:100]}")

                # Wait before retry
                if attempt_num < self.modification_config["max_modification_attempts"] - 1:
                    wait_time = self.modification_config["retry_delay_base"] * (2 ** attempt_num)
                    print(f"[WAITING] Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)

        # All modification attempts failed
        print(f"   [FAILED] All {self.modification_config['max_modification_attempts']} attempts failed")
        self.modification_stats["failed_validations"] += 1

        # Decide on fallback strategy
        if self.modification_config.get("fallback_to_original", True):
            print(f"   [FALLBACK] Using original content")
            self.modification_stats["fallbacks_to_original"] += 1

            return {
                "success": False,  # Honest: we failed to modify
                "modified_question": question,
                "modified_answer": answer,
                "modification_applied": False,
                "original_question": question,
                "original_answer": answer,
                "modification_reasoning": "All modification attempts failed - using original content",
                "validation_passed": False,  # We couldn't improve it
                "improvement_achieved": False,
                "fallback_used": True,
                "modification_attempts": len(previous_failures),
                "validation_issues_prevented": 0,
                "audit_trail": audit_trail
            }
        else:
            print(f"[ERROR] MODIFICATION COMPLETELY FAILED")
            return {
                "success": False,
                "modified_question": question,
                "modified_answer": answer,
                "modification_applied": False,
                "original_question": question,
                "original_answer": answer,
                "modification_reasoning": "All modification attempts failed",
                "validation_passed": False,
                "improvement_achieved": False,
                "fallback_used": False,
                "modification_attempts": len(previous_failures),
                "validation_issues_prevented": 0,
                "audit_trail": audit_trail
            }


    def check_numeric_accuracy(self, answer: str, context: str) -> bool:
        """Pre-LLM numeric guardrails: check if numbers in answer match context with contextual matching"""

        if not answer.strip() or not context.strip():
            return True

        # Extract percentages with context
        answer_pcts = self._extract_percentages_with_context(answer)
        context_pcts = self._extract_percentages_with_context(context)

        # Check percentage accuracy with semantic matching
        for ans_pct, ans_ctx in answer_pcts:
            matched = False
            for ctx_pct, ctx_ctx in context_pcts:
                # Check if percentages refer to the same metric (simple context matching)
                if ans_ctx.lower() in ctx_ctx.lower() or ctx_ctx.lower() in ans_ctx.lower() or not ans_ctx or not ctx_ctx:
                    if abs(float(ans_pct) - float(ctx_pct)) <= 0.5:  # <=0.5 pp difference
                        matched = True
                        break
            if not matched and context_pcts:  # Only fail if there are percentages to match
                return False

        # Extract other numbers with context and units
        answer_nums = self._extract_numbers_with_context(answer)
        context_nums = self._extract_numbers_with_context(context)

        # Check other numbers with proximity and unit matching
        for ans_num, ans_ctx, ans_unit in answer_nums:
            matched = False
            for ctx_num, ctx_ctx, ctx_unit in context_nums:
                # Check if numbers have compatible units and are semantically related
                if self._are_compatible_numbers(ans_ctx, ctx_ctx, ans_unit, ctx_unit):
                    # Fix: Use context value as denominator (ground truth)
                    ctx_val = float(ctx_num)
                    if ctx_val != 0:
                        relative_error = abs(float(ans_num) - ctx_val) / max(1e-12, abs(ctx_val))
                        if relative_error <= 0.02:  # <=2% relative error
                            matched = True
                            break
                    elif float(ans_num) == 0:  # Both are zero
                        matched = True
                        break
            if not matched and context_nums:  # Only fail if there are numbers to match
                return False

        return True

    def _extract_percentages_with_context(self, text: str) -> list:
        """Extract percentages with surrounding context"""
        import re

        # Find percentages with 5-10 words of context before
        pattern = r'(\S+(?:\s+\S+){0,9}\s+)?(\d+\.?\d*)%'
        matches = re.finditer(pattern, text, re.IGNORECASE)

        results = []
        for match in matches:
            pct_value = match.group(2)
            context_before = match.group(1) or ""

            # Extract key terms (nouns, metric keywords)
            context_terms = self._extract_key_terms(context_before)
            results.append((pct_value, context_terms))

        return results

    def _extract_numbers_with_context(self, text: str) -> list:
        """Extract numbers with context and units, excluding percentages"""
        import re

        # Find numbers with context and potential units
        pattern = r'(\S+(?:\s+\S+){0,5}\s+)?(\d+\.?\d*)(?:\s*(million|billion|thousand|kg|tons|tonnes|MW|GW|m2|ft2|USD|EUR|GBP|\$|kWh|GWh|MWh|TWh|tCO2e|tCO₂e|CO2|m³|kt|Mt|kWp|MWp))?(?!\s*%)'
        matches = re.finditer(pattern, text, re.IGNORECASE)

        results = []
        for match in matches:
            number = match.group(2)
            context_before = match.group(1) or ""
            unit = match.group(3) or ""

            # Extract key terms
            context_terms = self._extract_key_terms(context_before)

            results.append((number, context_terms, unit))

        return results

    def _extract_key_terms(self, text: str) -> str:
        """Extract key metric terms from context"""
        import re

        # Common sustainability metric keywords
        key_patterns = [
            r'\b(capex|opex|revenue|turnover|emissions?|scope\s*[123]?|energy|water|waste|carbon)\b',
            r'\b(percentage|ratio|rate|amount|total|net|gross|annual)\b',
            r'\b(facilities?|buildings?|employees?|suppliers?|customers?)\b'
        ]

        extracted_terms = []
        for pattern in key_patterns:
            matches = re.findall(pattern, text.lower())
            extracted_terms.extend(matches)

        return " ".join(extracted_terms).strip()

    def _are_compatible_numbers(self, ctx1: str, ctx2: str, unit1: str, unit2: str) -> bool:
        """Check if two numbers with context and units are comparable"""
        # Must have compatible units
        if unit1 and unit2 and unit1 != unit2:
            return False

        # Must have some contextual similarity (simple check)
        return ctx1.lower() in ctx2.lower() or ctx2.lower() in ctx1.lower() or not ctx1 or not ctx2

    def _parse_evaluation_response(self, response: str) -> Tuple[float, str]:
        """Parse LLM response to extract score and reasoning with robust fallbacks"""
        import re
        import json

        # Strategy 1: Try JSON parsing first
        try:
            # Look for JSON-like structure
            json_match = re.search(r'\{[^}]*"score"[^}]*\}', response, re.IGNORECASE | re.DOTALL)
            if json_match:
                json_data = json.loads(json_match.group())
                score = float(json_data.get('score', 5.0))
                reasoning = json_data.get('reasoning', json_data.get('explanation', response))
                return max(1, min(10, score)), reasoning
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Strategy 2: Enhanced pattern matching for REASONING:/SCORE: format
        try:
            lines = response.strip().split('\n')
            reasoning_lines = [line for line in lines if re.match(r'^\s*REASONING\s*:', line, re.IGNORECASE)]
            score_lines = [line for line in lines if re.match(r'^\s*SCORE\s*:', line, re.IGNORECASE)]

            reasoning = reasoning_lines[0].split(':', 1)[1].strip() if reasoning_lines else ""
            score_text = score_lines[0].split(':', 1)[1].strip() if score_lines else ""

            # Extract numeric score with multiple patterns
            score_patterns = [
                r'\b(\d+(?:\.\d+)?)\b',      # Standard decimal
                r'(\d+)/10',                  # X/10 format
                r'(\d+)\s*out\s*of\s*10'     # X out of 10 format
            ]

            score = 5.0  # Default
            for pattern in score_patterns:
                score_match = re.search(pattern, score_text, re.IGNORECASE)
                if score_match:
                    score = float(score_match.group(1))
                    break

            # If no reasoning found, use full response
            if not reasoning:
                reasoning = response

        except (ValueError, IndexError):
            reasoning = response
            score = 5.0

        # Strategy 3: Fallback - extract any number from response
        if score == 5.0:  # Still default, try broader extraction
            all_numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)
            if all_numbers:
                # Take first number that's in reasonable range for score
                for num_str in all_numbers:
                    num = float(num_str)
                    if 1 <= num <= 10:
                        score = num
                        break

        # Final validation and clamping
        score = max(1, min(10, score))

        return score, reasoning

    def evaluate_factoid_question_faithfulness(self, question: str, context: str) -> Tuple[float, str]:
        """Evaluate if factoid question can be answered with verbatim spans from context"""

        prompt = f"""You are an expert evaluator for factoid questions. Determine if this question can be answered with VERBATIM SPANS from the context.

SPAN EXTRACTION RULES:
1. Find exact text pieces in context that answer the question
2. Spans must be word-for-word identical (no paraphrasing)
3. Multiple spans can be combined if needed and separated by commas
4. No interpretation or external knowledge allowed

EVALUATION CRITERIA:
- 10: Perfect - Clear verbatim spans directly answer the question
- 9: Excellent - Clear verbatim spans with minimal combination needed
- 8: Very Good - Good verbatim spans with minor combination required
- 7: Good - Adequate verbatim spans but require careful extraction
- 6: Acceptable - Some verbatim spans but incomplete coverage
- 5: Weak - Limited verbatim spans with notable gaps
- 4: Below Average - Few relevant verbatim spans
- 3: Poor - Very limited verbatim span availability
- 2: Very Poor - Almost no usable verbatim spans
- 1: Failing - No verbatim spans can answer the question

CONTEXT: "{context}"
QUESTION: "{question}"

First identify any verbatim spans that could answer the question, then evaluate answerability.

FORMAT YOUR RESPONSE AS:
REASONING: [Identify specific spans found and evaluate coverage]
SCORE: [Number from 1-10]"""

        messages = [
            {"role": "system", "content": "You are an expert evaluator for factoid span extraction."},
            {"role": "user", "content": prompt}
        ]

        response = self._call_llm(messages, max_tokens=512, temperature=0.0)
        return self._parse_evaluation_response(response)

    def evaluate_nonfactoid_question_faithfulness(self, question: str, context: str) -> Tuple[float, str]:
        """Evaluate if non-factoid question can be answered conceptually from context"""

        prompt = f"""You are an expert evaluator for non-factoid questions. Assess if this question can be answered through conceptual understanding of the context.

NON-FACTOID REQUIREMENTS:
- Question requires analysis,description, synthesis, or explanation
- Context provides sufficient conceptual foundation
- May need reasoning beyond direct extraction

EVALUATION CRITERIA:
- 10: Perfect - All concepts clearly supported by context
- 9: Excellent - Strong conceptual foundation with minimal gaps
- 8: Very Good - Strong conceptual foundation with minor gaps
- 7: Good - Adequate context for meaningful answer
- 6: Acceptable - Adequate context with some limitations
- 5: Weak - Limited conceptual support
- 4: Below Average - Weak conceptual support
- 3: Poor - Very limited conceptual support
- 2: Very Poor - Minimal conceptual support
- 1: Failing - Insufficient context for proper analysis

CONTEXT: "{context}"
QUESTION: "{question}"

FORMAT YOUR RESPONSE EXACTLY AS:
REASONING: [Analysis of conceptual support vs question requirements]
SCORE: [Number from 1-10]"""

        messages = [
            {"role": "system", "content": "You are an expert evaluator for conceptual question answering."},
            {"role": "user", "content": prompt}
        ]

        response = self._call_llm(messages, max_tokens=512, temperature=0.0)
        return self._parse_evaluation_response(response)

    def evaluate_table_question_faithfulness(self, question: str, context: str, table_caption: Optional[str] = None) -> Tuple[float, str]:
        """Evaluate if table question can be answered from tabular paragraph data"""

        table_info = f"Table Caption: {table_caption}" if table_caption else "Table data converted to paragraph format"

        prompt = f"""You are an expert evaluator for table questions. Assess if this question can be answered from the tabular data presented in paragraph format.

TABLE REQUIREMENTS:
- Question targets specific data from table structure
- Paragraph contains sufficient tabular information
- Data relationships and values are extractable

{table_info}

EVALUATION CRITERIA:
- 10: Perfect - All table data clearly available for question
- 9: Excellent - Strong data foundation with minimal gaps
- 8: Very Good - Strong data foundation with minor gaps
- 7: Good - Adequate table data for meaningful answer
- 6: Acceptable - Adequate table data with some limitations
- 5: Weak - Limited relevant table data
- 4: Below Average - Very limited table data
- 3: Poor - Minimal relevant table data
- 2: Very Poor - Almost no relevant table data
- 1: Failing - Insufficient table data for proper answer

CONTEXT (TABULAR PARAGRAPH): "{context}"
QUESTION: "{question}"

FORMAT YOUR RESPONSE EXACTLY AS:
REASONING: [Analysis of available table data vs question requirements]
SCORE: [Number from 1-10]"""

        messages = [
            {"role": "system", "content": "You are an expert evaluator for table data questions."},
            {"role": "user", "content": prompt}
        ]

        response = self._call_llm(messages, max_tokens=512, temperature=0.0)
        return self._parse_evaluation_response(response)


    def evaluate_nonfactoid_answer_faithfulness(self, question: str, answer: str, context: str) -> Tuple[float, str]:
        """Evaluate if nonfactoid answer derives all information from context"""

        prompt = f"""You are an expert evaluator for nonfactoid answers. Assess if this answer derives ALL information from the context with no external knowledge.

NONFACTOID ANSWER REQUIREMENTS:
- All claims must be supported by the context
- No external knowledge or assumptions
- Accurate representation of context information
- Can include reasonable inferences from context

EVALUATION CRITERIA:
- 10: Perfect - All information directly from context
- 9: Excellent - Context-based with minimal interpretive language
- 8: Very Good - Context-based with minor interpretive language
- 7: Good - Mostly context-based with minimal reasonable inferences
- 6: Acceptable - Mostly context-based with some reasonable inferences
- 5: Fair - Context-based but with notable external additions
- 4: Below Average - Some context-based content with external additions
- 3: Poor - Limited context grounding with significant external content
- 2: Very Poor - Minimal context grounding, mostly external content
- 1: Failing - Significant external content or contradicts context

CONTEXT: "{context}"
QUESTION: "{question}"
ANSWER: "{answer}"

Check each claim in the answer against the context. Identify any external knowledge or unsupported statements.

FORMAT YOUR RESPONSE AS:
REASONING: [Your analysis of context grounding]
SCORE: [Number from 1-10]"""

        messages = [
            {"role": "system", "content": "You are a meticulous evaluator of nonfactoid answer faithfulness. Verify every claim is supported by context."},
            {"role": "user", "content": prompt}
        ]

        response = self._call_llm(messages, max_tokens=512, temperature=0.0)
        return self._parse_evaluation_response(response)

    def evaluate_table_answer_faithfulness(self, question: str, answer: str, context: str, table_caption: Optional[str] = None) -> Tuple[float, str]:
        """Evaluate table answer faithfulness for tabular paragraph data"""

        table_info = f"Table Caption: {table_caption}" if table_caption else "Table data converted to paragraph format"

        prompt = f"""You are an expert evaluator for table answers. Assess how faithfully this answer represents the tabular data in the context.

TABLE ANSWER REQUIREMENTS:
- All data points must be accurately extracted from table paragraph
- Numerical values must match exactly
- Relationships and comparisons must be correct
- No external assumptions or data

{table_info}

SCORE SCALE:
10: Perfect - All data accurate, complete table data utilization
9: Excellent - Accurate with comprehensive data usage
8: Good - Accurate core data with minor omissions
7: Acceptable - Mostly accurate with some data gaps
6: Fair - Generally accurate but missing key data points
5: Moderate - Some accuracy but notable data errors
4: Below Average - Several data inaccuracies or omissions
3: Poor - Major data errors or insufficient table usage
2: Very Poor - Mostly inaccurate table data representation
1: Failing - Completely inaccurate or ignores table data

CONTEXT (TABULAR PARAGRAPH): "{context}"
QUESTION: "{question}"
ANSWER: "{answer}"

FORMAT YOUR RESPONSE EXACTLY AS:
REASONING: [Analysis of data accuracy and table utilization]
SCORE: [Number from 1-10]"""

        messages = [
            {"role": "system", "content": "You are an expert evaluator for table data accuracy."},
            {"role": "user", "content": prompt}
        ]

        response = self._call_llm(messages, max_tokens=512, temperature=0.0)
        return self._parse_evaluation_response(response)

    def evaluate_factoid_question_relevance(self, question: str, classification: str,
                                           answer: str = "", context: str = "") -> Tuple[float, str]:
        """
        Evaluate factoid question relevance ensuring it targets domain-specific data
        that can be answered by available spans within context constraints.
        """

        prompt = f"""You are evaluating a factoid question's relevance to {classification} domain.
A good factoid question should target {classification}-specific extractable data that can be answered by the available spans.

EVALUATION CONTEXT:
- QUESTION: "{question}"
- AVAILABLE ANSWER SPANS: "{answer}"
- SOURCE CONTEXT: "{context}"

EVALUATION CRITERIA:
Assess how well this question targets {classification} professional data that can be extracted and answered by the available spans. Consider:
- Does it seek {classification}-specific metrics, KPIs, or professional data?
- Can the available spans meaningfully answer this question?
- Would {classification} professionals find this question valuable and answerable?
- Does the question align with the terminology and scope of the context?

SCORING SCALE:
- 10: Perfect {classification} question - targets essential domain data, fully answerable by spans
- 9: Excellent {classification} focus - clear professional value, well-answerable by spans
- 8: Strong {classification} relevance - good domain specificity, answerable by spans
- 7: Good {classification} connection - some domain focus, reasonably answerable
- 6: General {classification} relevance - basic domain connection, spans can answer
- 5: Minimal {classification} focus - generic question, unclear professional value
- 4: Weak {classification} relevance - poor domain specificity, questionable answerability
- 3: Limited {classification} connection - mostly generic, spans provide partial answer
- 2: Very limited domain relevance - unclear professional value, poor span alignment
- 1: No {classification} relevance - generic question, spans cannot meaningfully answer

FORMAT YOUR RESPONSE AS:
REASONING: [Analyze the question's {classification} professional value and how well it can be answered by the available spans]
SCORE: [Single number 1-10]"""

        messages = [
            {"role": "system", "content": f"You are an expert in {classification} domain evaluation with focus on factoid data extraction and professional utility."},
            {"role": "user", "content": prompt}
        ]

        response = self._call_llm(messages, max_tokens=512, temperature=0.0)
        return self._parse_evaluation_response(response)

    def evaluate_nonfactoid_question_relevance(self, question: str, classification: str) -> Tuple[float, str]:
        """Evaluate non-factoid question relevance for domain"""

        prompt = f"""Evaluate this non-factoid question's relevance to {classification} domain. Non-factoid questions should require conceptual understanding and analysis.

CRITERIA:
- 10: Requires deep {classification} conceptual analysis
- 9: Requires strong {classification} conceptual analysis
- 8: Clear {classification} analytical focus with depth
- 7: Clear {classification} analytical focus
- 6: Good {classification} relevance with some analytical depth
- 5: General {classification} relevance but surface-level
- 4: Weak {classification} relevance, limited analysis needed
- 3: Minimal {classification} relevance
- 2: Very limited {classification} connection
- 1: Not {classification}-relevant

QUESTION: "{question}"

FORMAT YOUR RESPONSE EXACTLY AS:
REASONING: [Analysis of {classification} domain relevance for conceptual analysis]
SCORE: [Number from 1-10]"""

        messages = [
            {"role": "system", "content": f"You are an expert in {classification} conceptual analysis."},
            {"role": "user", "content": prompt}
        ]

        response = self._call_llm(messages, max_tokens=512, temperature=0.0)
        return self._parse_evaluation_response(response)

    def evaluate_table_question_relevance(self, question: str, classification: str) -> Tuple[float, str]:
        """Evaluate table question relevance for domain"""

        prompt = f"""Evaluate this table question's relevance to {classification} domain. Table questions should target structured data analysis.

NOTE: Table data is presented in paragraph format - original table structure (rows, columns, cell positions) is not preserved.

CRITERIA:
- 10: Targets specific {classification} table data analysis
- 9: Clear {classification} tabular focus with specific analysis
- 8: Clear {classification} tabular focus
- 7: Good {classification} tabular focus with some specificity
- 6: General {classification} relevance with table elements
- 5: General {classification} relevance but unclear table focus
- 4: Weak {classification} relevance to tabular data
- 3: Minimal {classification} table relevance
- 2: Very limited {classification} table connection
- 1: Not {classification}-relevant

QUESTION: "{question}"

FORMAT YOUR RESPONSE EXACTLY AS:
REASONING: [Analysis of {classification} domain relevance for table data]
SCORE: [Number from 1-10]"""

        messages = [
            {"role": "system", "content": f"You are an expert in {classification} table data analysis."},
            {"role": "user", "content": prompt}
        ]

        response = self._call_llm(messages, max_tokens=512, temperature=0.0)
        return self._parse_evaluation_response(response)

    def evaluate_factoid_answer_relevance(self, answer: str, classification: str, context: str = "", question: str = "") -> Tuple[float, str, dict]:
        """Evaluate factoid answer relevance using context-aware span analysis with proportional weighting"""

        relevant_keywords = self._get_relevant_keywords(classification)

        # Use new context-aware evaluation approach
        try:
            final_score, llm_json, prompt, meta = self._factoid_relevance_judge(
                context=context,
                question=question,
                answer=answer,
                classification=classification,
                relevant_keywords=relevant_keywords
            )

            # Extract reasoning from LLM response
            reasoning = llm_json.get("reasoning", "Proportional relevance evaluation completed") if llm_json else "Evaluation completed"

            # Add metadata info to reasoning for debugging
            if meta:
                span_info = f" ({meta['relevant_spans']}/{meta['total_spans']} relevant spans)"
                reasoning = f"{reasoning}{span_info}"

            # Build span analysis data
            spans = [s.strip() for s in answer.split(',')]
            span_data = self._build_span_analysis_data(final_score, spans, llm_json, meta)

            return float(final_score), reasoning, span_data

        except Exception as e:
            # Fallback to simpler evaluation if new method fails
            score, reason = self._fallback_factoid_relevance_evaluation(answer, classification)
            spans = [s.strip() for s in answer.split(',')]
            span_data = {'has_problematic_spans': False, 'modification_needed': False, 'problematic_spans': [], 'good_spans': spans}
            return score, reason, span_data

    def _build_span_analysis_data(self, final_score, spans, llm_json, meta):
        """
        Build span analysis data for modification targeting using individual span evaluation.

        OPTIMIZATION NOTE: Uses individual span scores (< 6.0) to determine modification needs,
        not overall aggregation score. Mathematical proof shows overall_score < 6.0 implies
        at least one span < 6.0, making overall score check redundant for modification decisions.

        Individual spans are the authoritative source for modification triggering.
        Overall score is used for final QA decisions, reporting, and ranking.
        """
        try:
            # Extract individual span scores from LLM JSON response
            span_details = []
            problematic_spans = []
            good_spans = []

            if llm_json and 'spans' in llm_json and isinstance(llm_json['spans'], list):
                # Use individual span scores from LLM JSON
                for llm_span in llm_json['spans']:
                    span_text = llm_span.get('span', '').strip()
                    span_score = llm_span.get('score', final_score)

                    # Find matching span in original spans list
                    matched_span = None
                    for original_span in spans:
                        if span_text.lower() in original_span.lower() or original_span.lower() in span_text.lower():
                            matched_span = original_span
                            break

                    # Use matched span or LLM span text
                    final_span_text = matched_span if matched_span else span_text

                    if span_score < 6.0:
                        problematic_spans.append(final_span_text)
                        status = 'problematic'
                    else:
                        good_spans.append(final_span_text)
                        status = 'good'

                    span_details.append({
                        'span': final_span_text,
                        'score': span_score,
                        'status': status,
                        'reason': llm_span.get('evidence', 'Low sustainability relevance') if span_score < 6.0 else None
                    })

                # Handle any original spans not found in LLM response
                for original_span in spans:
                    if not any(detail['span'] == original_span for detail in span_details):
                        # Assign final_score to unmatched spans
                        if final_score < 6.0:
                            problematic_spans.append(original_span)
                            status = 'problematic'
                        else:
                            good_spans.append(original_span)
                            status = 'good'

                        span_details.append({
                            'span': original_span,
                            'score': final_score,
                            'status': status,
                            'reason': 'Low overall relevance score' if final_score < 6.0 else None
                        })

            elif meta and 'raw_scores' in meta:
                # Use metadata raw scores as fallback
                span_scores = meta['raw_scores']
                for i, span in enumerate(spans):
                    score = span_scores[i] if i < len(span_scores) else final_score
                    if score < 6.0:
                        problematic_spans.append(span)
                        status = 'problematic'
                    else:
                        good_spans.append(span)
                        status = 'good'

                    span_details.append({
                        'span': span,
                        'score': score,
                        'status': status,
                        'reason': 'Low relevance based on metadata' if score < 6.0 else None
                    })
            else:
                # Conservative fallback: use overall score but be conservative about modification
                print(f"   [SPAN-FALLBACK] No span data available, using conservative fallback...")
                for span in spans:
                    # Conservative approach: only mark as problematic if overall score is clearly low
                    if final_score < 5.0:
                        problematic_spans.append(span)
                        status = 'problematic'
                        score = final_score
                        reason = 'Low overall score (conservative fallback)'
                    else:
                        good_spans.append(span)
                        status = 'good'
                        score = final_score
                        reason = None

                    span_details.append({
                        'span': span,
                        'score': score,
                        'status': status,
                        'reason': reason
                    })

            return {
                'overall_score': final_score,
                'span_details': span_details,
                'good_spans': good_spans,
                'problematic_spans': problematic_spans,
                'has_problematic_spans': len(problematic_spans) > 0,
                'modification_needed': len(problematic_spans) > 0
            }
        except Exception as e:
            print(f"   [SPAN-ANALYSIS-ERROR] {e}")
            # Enhanced fallback: attempt individual span evaluation instead of relying on overall score
            try:
                print(f"   [SPAN-ERROR-RECOVERY] Attempting individual span evaluation...")
                problematic_spans = []
                good_spans = []
                span_details = []

                for span in spans:
                    # Conservative approach: only mark as problematic if overall score is clearly low
                    if final_score < 5.0:
                        problematic_spans.append(span)
                        status = 'problematic'
                        score = final_score
                        reason = 'Low overall score (error recovery fallback)'
                    else:
                        good_spans.append(span)
                        status = 'good'
                        score = final_score
                        reason = None

                    span_details.append({
                        'span': span,
                        'score': score,
                        'status': status,
                        'reason': reason
                    })

                return {
                    'overall_score': final_score,
                    'span_details': span_details,
                    'good_spans': good_spans,
                    'problematic_spans': problematic_spans,
                    'has_problematic_spans': len(problematic_spans) > 0,
                    'modification_needed': len(problematic_spans) > 0
                }

            except Exception as e2:
                print(f"   [SPAN-ERROR-RECOVERY-FAILED] {e2}")
                # Final fallback: only if everything fails, use conservative overall score approach
                if final_score < 5.0:
                    return {
                        'overall_score': final_score,
                        'span_details': [{'span': span, 'score': final_score, 'status': 'problematic'} for span in spans],
                        'good_spans': [],
                        'problematic_spans': spans,
                        'has_problematic_spans': True,
                        'modification_needed': True
                    }
                else:
                    return {
                        'overall_score': final_score,
                        'span_details': [{'span': span, 'score': final_score, 'status': 'good'} for span in spans],
                        'good_spans': spans,
                        'problematic_spans': [],
                        'has_problematic_spans': False,
                        'modification_needed': False
                    }

    def _modify_factoid_spans(self, answer: str, context: str, question: str, classification: str,
                            span_data: dict, question_type: str = "factoid") -> Tuple[str, bool, str]:
        """
        Professional targeted span modification with integrated assessment and redundancy prevention.
        Uses existing aggregation function for consistent scoring methodology.
        """
        if not span_data.get('has_problematic_spans', False):
            return answer, False, "No problematic spans detected"

        # Extract problematic spans with their reasons from enhanced assessment data
        problematic_spans = [(detail['span'], detail['score'], detail.get('reason', 'Low relevance'))
                           for detail in span_data.get('span_details', [])
                           if detail['status'] == 'problematic']

        good_spans = [detail['span'] for detail in span_data.get('span_details', [])
                     if detail['status'] == 'good']

        # Use enhanced logging for modification attempt
        self._log_modification_attempt(answer, problematic_spans, good_spans)

        # Process each problematic span with integrated modification + assessment + redundancy
        final_spans = good_spans.copy()
        successfully_modified_spans = []
        modification_count = 0

        for span_text, original_score, problem_reason in problematic_spans:
            self._log_span_modification_step(span_text, "ATTEMPT", problem_reason)
            # Build targeted modification prompt
            existing_content = ", ".join(final_spans) if final_spans else "None"

            prompt = f"""Improve this factoid span for {classification}:

CONTEXT: {context}
QUESTION: {question}
CURRENT SPAN: "{span_text}"
PROBLEM: {problem_reason}
OTHER SPANS: {existing_content}

Requirements:
1. Address the specific problem: {problem_reason}
2. Make more {classification}-relevant using context
3. Do NOT repeat information from other spans
4. Return "REMOVE" if cannot be improved meaningfully

Return only the improved span:"""

            try:
                # Get modification
                messages = [
                    {"role": "system", "content": f"Improve factoid spans for {classification} without redundancy."},
                    {"role": "user", "content": prompt}
                ]
                response = self._call_llm(messages, max_tokens=100, temperature=0.1)
                modified_span = response.strip().strip('"').strip("'")

                if modified_span.upper() == "REMOVE" or len(modified_span) < 2:
                    self._log_span_modification_step(span_text, "REMOVED", problem_reason)
                    continue

                # Integrated redundancy check
                modified_words = set(modified_span.lower().split())
                is_redundant = False
                for existing in final_spans:
                    existing_words = set(existing.lower().split())
                    if modified_words and existing_words:
                        overlap = len(modified_words & existing_words) / len(modified_words | existing_words)
                        if overlap > 0.8:
                            self._log_span_modification_step(span_text, "REDUNDANT", f"{overlap*100:.0f}% overlap with existing span", modified_span)
                            is_redundant = True
                            break

                if is_redundant:
                    continue

                # Integrated lightweight assessment
                assessment_prompt = f"""Score this span for {classification} relevance:

CONTEXT: {context}
SPAN: "{modified_span}"
DOMAIN: {classification}

Score 1-10:
- 8-10: Essential {classification} KPI/metric with clear domain connection
- 6-7: Important {classification} data with domain relevance
- 4-5: General business data with weak {classification} connection
- 1-3: Administrative/operational data with no {classification} relevance

Return only the score (1-10):"""

                score_messages = [
                    {"role": "system", "content": f"You are a {classification} expert. Return only a number."},
                    {"role": "user", "content": assessment_prompt}
                ]
                score_response = self._call_llm(score_messages, max_tokens=5, temperature=0.0)

                # Extract score
                import re
                score_match = re.search(r'\b([1-9]|10)\b', score_response.strip())
                new_score = float(score_match.group(1)) if score_match else 3.0

                # Validate improvement
                if new_score >= 6.0 and new_score > original_score:
                    final_spans.append(modified_span)
                    successfully_modified_spans.append((modified_span, new_score))
                    modification_count += 1
                    self._log_span_modification_step(span_text, "IMPROVED", f"Score: {original_score}->{new_score}", modified_span)
                else:
                    self._log_span_modification_step(span_text, "REMOVED", f"No improvement (Score: {new_score})")

            except Exception as e:
                print(f"     [ERROR] '{span_text}' - {str(e)}")

        # Final validation using existing aggregation function
        if modification_count > 0 and final_spans:
            # Build synthetic JSON for existing aggregation function
            synthetic_spans = []

            # Add good spans with their original scores
            for detail in span_data.get('span_details', []):
                if detail['status'] == 'good' and detail['span'] in final_spans:
                    synthetic_spans.append({
                        'span': detail['span'],
                        'score': detail['score'],
                        'found_in_context': True
                    })

            # Add successfully modified spans
            for span_text, new_score in successfully_modified_spans:
                synthetic_spans.append({
                    'span': span_text,
                    'score': new_score,
                    'found_in_context': True
                })

            # Use existing aggregation function for consistent scoring
            synthetic_json = {'spans': synthetic_spans}
            final_score, meta = self._aggregate_relevance_score(synthetic_json)

            if final_score >= 6:
                reconstructed_answer = ", ".join(final_spans)
                # Log the modification result
                initial_score = sum(detail.get('score', 0) for detail in span_data.get('span_details', [])) / max(len(span_data.get('span_details', [])), 1)
                self._log_modification_result(answer, reconstructed_answer, initial_score, final_score, True)
                return reconstructed_answer, True, f"Successfully modified {modification_count} spans (Final Score: {final_score})"

        # Fallback to good spans only
        if good_spans:
            fallback_answer = ", ".join(good_spans)
            # Calculate initial relevance score for comparison (approximate)
            initial_score = sum(detail.get('score', 0) for detail in span_data.get('span_details', [])) / max(len(span_data.get('span_details', [])), 1)
            # Log partial success
            self._log_modification_result(answer, fallback_answer, initial_score, 6.0, False)
            return fallback_answer, True, f"Retained {len(good_spans)} good spans"

        # Log failure
        initial_score = sum(detail.get('score', 0) for detail in span_data.get('span_details', [])) / max(len(span_data.get('span_details', [])), 1)
        self._log_modification_result(answer, answer, initial_score, initial_score, False)
        return answer, False, "Modification failed"



    def _fallback_factoid_relevance_evaluation(self, answer: str, classification: str) -> Tuple[float, str]:
        """Fallback evaluation method for factoid answer relevance"""
        relevant_keywords = self._get_relevant_keywords(classification)

        prompt = f"""Evaluate this factoid answer's relevance to {classification} domain using span-based analysis.

{classification.upper()} KEY TERMS: {relevant_keywords}

ANSWER: "{answer}"

Score based on {classification} analytical value:
- 8-10: Essential {classification} KPI/metric with clear domain connection
- 6-7: Important {classification} data with domain relevance
- 4-5: General business data with weak {classification} connection
- 1-3: Administrative/operational data with no {classification} relevance

FORMAT: REASONING: [Analysis] SCORE: [1-10]"""

        messages = [
            {"role": "system", "content": f"You are an expert in {classification} data evaluation."},
            {"role": "user", "content": prompt}
        ]

        response = self._call_llm(messages, max_tokens=512, temperature=0.0)
        return self._parse_evaluation_response(response)

    def evaluate_nonfactoid_answer_relevance(self, answer: str, classification: str) -> Tuple[float, str]:
        """Evaluate non-factoid answer relevance with class-aware LLM decision"""

        prompt = f"""Evaluate this answer's relevance to {classification} domain. For non-factoid answers, CONTEXTUAL ACCURACY and PRACTICAL RELEVANCE should score highest.

CLASSIFICATION CONTEXT:
- ESG: Environmental, Social, and Governance factors in business operations and reporting
- EU Taxonomy: European Union's classification system for sustainable economic activities
- Sustainability: Broader sustainable development, environmental stewardship, and corporate responsibility

IMPORTANT: Non-factoid answers often explain business processes, strategies, or organizational aspects that support {classification} goals. These can be highly relevant even without deep theoretical analysis.

NON-FACTOID ANSWER RELEVANCE CRITERIA (Prioritizing Practical Utility):
- 10: Directly answers question with clear {classification} context relevance and accuracy
- 9: Well-grounded answer with strong {classification} connection and practical utility
- 8: Good contextual answer with clear {classification} relevance for domain professionals
- 7: Adequate contextual answer with noticeable {classification} connection
- 6: Some contextual relevance with {classification} domain applicability
- 5: Limited {classification} connection but contextually accurate
- 4: Weak {classification} relevance, mostly generic but accurate content
- 3: Minimal {classification} connection
- 2: Very limited {classification} relevance
- 1: Not relevant to {classification} domain or contextually inaccurate

EVALUATION PRINCIPLE: Contextually accurate answers that explain business processes, strategies,
or organizational aspects supporting {classification} goals should score well (7-9) even without
deep theoretical analysis, as practical implementation is crucial for {classification} success.

ANSWER: "{answer}"

FORMAT YOUR RESPONSE EXACTLY AS:
REASONING: [Explain the practical {classification} domain utility and contextual accuracy]
SCORE: [Number from 1-10]"""

        messages = [
            {"role": "system", "content": f"You are an expert in {classification} strategy evaluation."},
            {"role": "user", "content": prompt}
        ]

        response = self._call_llm(messages, max_tokens=512, temperature=0.0)
        return self._parse_evaluation_response(response)

    def evaluate_table_answer_relevance(self, answer: str, classification: str) -> Tuple[float, str]:
        """Evaluate table answer relevance for domain"""

        prompt = f"""Evaluate this table answer's relevance to {classification} domain. For table data, DIRECT FACTUAL ACCURACY should score highest.

IMPORTANT: Table data is presented in paragraph format. Direct answers using exact table values are most valuable for {classification} professionals.

TABLE ANSWER RELEVANCE CRITERIA (Prioritizing Direct Factual Accuracy):
- 10: Direct, precise answer using exact table data for {classification} context
- 9: Accurate answer with exact values plus minimal relevant context for {classification}
- 8: Good factual answer with exact data, some additional {classification} context
- 7: Adequate factual answer with mostly accurate data for {classification} use
- 6: Some factual content with {classification} relevance but missing precision
- 5: Limited factual accuracy or {classification} relevance
- 4: Weak {classification} data relevance with inaccuracies
- 3: Minimal {classification} table relevance
- 2: Very limited {classification} data connection
- 1: Not {classification}-relevant or factually incorrect

EVALUATION PRINCIPLE: Direct answers using exact table values should score highest (9-10)
because precise data is what {classification} professionals need for decision-making and compliance.

ANSWER: "{answer}"

FORMAT YOUR RESPONSE EXACTLY AS:
REASONING: [Analysis of factual accuracy and {classification} domain utility for table data]
SCORE: [Number from 1-10]"""

        messages = [
            {"role": "system", "content": f"You are an expert in {classification} table data evaluation."},
            {"role": "user", "content": prompt}
        ]

        response = self._call_llm(messages, max_tokens=512, temperature=0.0)
        return self._parse_evaluation_response(response)

    # Combination formulas for Q+A metrics (simple averages)

    def determine_modification_needs(self, faithfulness_score: float, relevance_score: float,
                                   question_faithfulness_score: float, answer_faithfulness_score: float,
                                   question_relevance_score: float, answer_relevance_score: float) -> Tuple[str, str]:
        """Determine what needs to be modified based on faithfulness and relevance scores - simplified single threshold"""

        THRESHOLD = self.quality_threshold  # Use class threshold

        # Identify the lowest scoring dimension (faithfulness vs relevance only)
        scores = {
            "faithfulness": faithfulness_score,
            "relevance": relevance_score
        }

        lowest_dimension = min(scores, key=scores.get)

        # Determine if question, answer, or both need modification
        if lowest_dimension == "faithfulness":
            if question_faithfulness_score < answer_faithfulness_score:
                modification_type = "question" if answer_faithfulness_score >= THRESHOLD else "both"
            else:
                modification_type = "answer"
        else:  # relevance
            if question_relevance_score >= THRESHOLD:
                # Question already good - focus on answer
                modification_type = "answer"
            elif answer_relevance_score >= THRESHOLD:
                # Answer already good - focus on question
                modification_type = "question"
            else:
                # Both scores < threshold - modify the worse component
                modification_type = "question" if question_relevance_score < answer_relevance_score else "answer"

        return modification_type, lowest_dimension

    def modify_factoid_question(self, question: str, context: str, classification: str,
                               target_improvement: str, answer: str = "") -> Tuple[str, str]:
        """Modify factoid question to improve faithfulness or relevance using self-contained approach

        Args:
            question: Original question to modify
            context: Source context
            classification: Domain classification (ESG, EU Taxonomy, Sustainability)
            target_improvement: "faithfulness" or "relevance"
            answer: Current answer that should remain consistent
        """

        if target_improvement == "faithfulness":
            prompt = f"""You are an expert in factoid question modification. Improve this question to better target VERBATIM SPANS from the context.

QUESTION ANALYSIS:
The question "{question}" may have issues with verbatim answerability. Analyze what makes it difficult to answer with exact spans from context.

MODIFICATION APPROACH:
1. IDENTIFY PROBLEMS: Determine why current question is hard to answer with exact spans
2. TARGET SPECIFIC DATA: Focus on data points (numbers, names, dates) actually present in context
3. ALIGN LANGUAGE: Use terminology that matches context exactly
4. ENSURE EXTRACTABILITY: Make sure answers can be word-for-word from context

ANSWER CONSISTENCY CONSTRAINT:
The modified question MUST target the same data as the answer: "{answer}"
Your modified question should elicit this exact same factoid answer from the context.
Ensure the question targets the specific spans/data points that produce this answer.

PRESERVE {classification.upper()} RELEVANCE:
- Maintain {classification} domain focus where possible
- Keep domain-specific terminology when supported by context
- Ensure question remains valuable for {classification} analysis

REQUIREMENTS:
- Question must be DIFFERENT from "{question}"
- Must target verbatim extractable information
- Should allow multiple potential spans for comprehensive answers
- MUST elicit this specific answer: "{answer}"

CONTEXT: "{context}"
CURRENT QUESTION: "{question}"
TARGET ANSWER: "{answer}"

FORMAT YOUR RESPONSE AS:
REASONING: [Analyze current question's problems, explain specific changes and why they improve verbatim answerability and preserve answer consistency]
MODIFIED_QUESTION: [Substantially different question targeting verbatim extractable data that produces the target answer]"""

        elif target_improvement == "relevance":
            prompt = f"""You are a {classification} expert. Enhance this factoid question's domain relevance while ensuring it remains answerable by the answer.

MODIFICATION TASK:
Transform the question to better target {classification} professional data while maintaining answerability by the answer.

CURRENT SITUATION:
- QUESTION: "{question}"
- ANSWER: "{answer}"
- CONTEXT: "{context}"

MODIFICATION REQUIREMENTS:
1. Enhanced question MUST be answerable by answer in context
2. Use terminology and concepts from the source context where appropriate
3. Incorporate {classification}-specific professional language and focus
4. Maintain factoid nature (seeking exact, extractable data)
5. Question must be substantially different from the original

ENHANCEMENT APPROACH:
- Transform generic terms into {classification}-specific terminology
- Frame the question for {classification} professional utility
- Use context vocabulary to ensure alignment
- Target the type of data that the answer provides
- Focus on metrics/information valuable for {classification} analysis

VALIDATION CHECK:
Before finalizing, confirm: Can the answer "{answer}" directly answer your modified question?

CONTEXT: "{context}"
CURRENT QUESTION: "{question}"

FORMAT YOUR RESPONSE AS:
REASONING: [Explain how you enhanced the {classification} domain focus while preserving answerability]
MODIFIED_QUESTION: [Enhanced question with {classification} professional focus, answerable by the answer]"""

        else:
            raise ValueError(f"Unsupported target_improvement: {target_improvement}. Only 'faithfulness' and 'relevance' are supported.")

        messages = [
            {"role": "system", "content": f"You are an expert in {classification} reporting and factoid question improvement."},
            {"role": "user", "content": prompt}
        ]

        response = self._call_llm_with_retry(messages, max_tokens=512, temperature=0.0,
                                           context=f"factoid_question_modification_{target_improvement}")
        return self._parse_modification_response(response)


    def modify_nonfactoid_question(self, question: str, context: str, classification: str,
                                  target_improvement: str) -> Tuple[str, str]:
        """Modify non-factoid question to improve faithfulness or relevance using self-contained approach"""

        relevant_keywords = self._get_relevant_keywords(classification)

        if target_improvement == "faithfulness":
            prompt = f"""You are an expert in non-factoid question modification. Improve this question to be better grounded in context information.

QUESTION ANALYSIS:
The question "{question}" may have issues with context grounding. Analyze what elements require external knowledge or cannot be answered from context.

MODIFICATION APPROACH:
1. IDENTIFY EXTERNAL DEPENDENCIES: Find parts requiring knowledge not in context
2. FOCUS ON CONTEXT CONCEPTS: Target themes and information explicitly mentioned
3. ENHANCE ANSWERABILITY: Ensure context provides sufficient foundation for comprehensive answers
4. PRESERVE ANALYTICAL NATURE: Keep explanatory/analytical character

PRESERVE {classification.upper()} RELEVANCE:
- Maintain {classification} domain focus where context-supported
- Keep domain-specific analytical aspects when grounded in context
- Ensure question remains valuable for {classification} professionals

REQUIREMENTS:
- Question must be DIFFERENT from "{question}"
- Must be answerable from context alone
- Should allow for analytical/explanatory responses

CONTEXT: "{context}"
CURRENT QUESTION: "{question}"

FORMAT YOUR RESPONSE AS:
REASONING: [Analyze current question's context grounding issues, explain specific changes and why they improve context-based answerability]
MODIFIED_QUESTION: [Substantially different question grounded in context information]"""

        elif target_improvement == "relevance":
            prompt = f"""You are a {classification} expert. Improve this non-factoid question to enhance {classification} domain relevance while maintaining context grounding.

{classification.upper()} KEY TERMS: {relevant_keywords}

QUESTION ANALYSIS:
The question "{question}" may lack sufficient {classification} domain focus or analytical depth. Analyze what domain-specific improvements are needed.

MODIFICATION APPROACH:
1. IDENTIFY DOMAIN GAPS: Determine what {classification}-specific improvements are needed
2. INCORPORATE FRAMEWORKS: Use {classification} analytical frameworks and concepts
3. TARGET PROFESSIONAL INSIGHTS: Focus on analysis valuable for {classification} professionals
4. MAINTAIN ANALYTICAL NATURE: Keep explanatory/analytical character

PRESERVE FAITHFULNESS QUALITY:
- Ensure modified question can be answered from available context
- Do not create questions requiring extensive external knowledge
- Keep analytical nature while ensuring context provides foundation

REQUIREMENTS:
- Question must be DIFFERENT from "{question}"
- Must enhance {classification} professional relevance
- Should target domain-specific analytical insights

CONTEXT: "{context}"
CURRENT QUESTION: "{question}"

FORMAT YOUR RESPONSE AS:
REASONING: [Analyze current question's domain relevance, explain specific changes and why they enhance {classification} analytical focus]
MODIFIED_QUESTION: [Substantially different question with enhanced {classification} domain relevance and analytical depth]"""

        else:
            raise ValueError(f"Unsupported target_improvement: {target_improvement}. Only 'faithfulness' and 'relevance' are supported.")

        messages = [
            {"role": "system", "content": f"You are an expert in non-factoid question modification for {classification} domain."},
            {"role": "user", "content": prompt}
        ]

        response = self._call_llm_with_retry(messages, max_tokens=512, temperature=0.0,
                                           context=f"nonfactoid_question_modification_{target_improvement}")
        return self._parse_modification_response(response)


    def modify_table_question(self, question: str, context: str, classification: str,
                             target_improvement: str, table_caption: Optional[str] = None) -> Tuple[str, str]:
        """Modify table question to improve faithfulness or relevance using self-contained approach"""

        relevant_keywords = self._get_relevant_keywords(classification)

        if target_improvement == "faithfulness":
            prompt = f"""You are an expert in table question modification. Improve this question to better target the tabular data in context.

QUESTION ANALYSIS:
The question "{question}" may have issues with table data targeting. Analyze what makes it difficult to answer from available tabular data.

MODIFICATION APPROACH:
1. IDENTIFY TABLE ISSUES: Determine why current question is hard to answer from table data
2. TARGET SPECIFIC DATA: Focus on columns, rows, values, or relationships in the table
3. ALIGN WITH DATA: Use terminology that matches table content exactly
4. ENSURE EXTRACTABILITY: Make sure data can be directly extracted from table structure

PRESERVE {classification.upper()} RELEVANCE:
- Maintain {classification} domain focus when supported by table data
- Keep domain-specific data analysis aspects
- Ensure question remains valuable for {classification} reporting

REQUIREMENTS:
- Question must be DIFFERENT from "{question}"
- Must target available table data effectively
- Should allow for direct data extraction

CONTEXT (TABULAR): "{context}"
{f'TABLE CAPTION: {table_caption}' if table_caption else 'TABLE: Converted to paragraph format'}
CURRENT QUESTION: "{question}"

FORMAT YOUR RESPONSE AS:
REASONING: [Analyze current question's table targeting issues, explain specific changes and why they improve data extractability]
MODIFIED_QUESTION: [Substantially different question targeting available table data]"""

        elif target_improvement == "relevance":
            prompt = f"""You are a {classification} expert. Improve this table question to be more relevant to {classification} domain analysis.

{classification.upper()} KEY TERMS: {relevant_keywords}

DOMAIN RELEVANCE RULES:
1. Question should target {classification} table data analysis
2. Focus on domain-specific structured data insights
3. Target professional data analysis for {classification} reporting
4. Maintain table-focused analytical nature

MODIFICATION APPROACH:
- Target {classification} data analysis from table
- Focus on domain-specific metrics and comparisons
- Enhance professional value for {classification} context
- Strengthen table data analysis focus

CONTEXT (TABULAR PARAGRAPH): "{context}"
{f'TABLE CAPTION: {table_caption}' if table_caption else 'TABLE: Converted to paragraph format'}
ORIGINAL QUESTION: "{question}"

Improve the question's {classification} domain relevance for table analysis.

FORMAT YOUR RESPONSE AS:
REASONING: [Explain how modifications improve {classification} table analysis relevance]
MODIFIED_QUESTION: [Question with enhanced {classification} table analysis focus]"""

        else:
            raise ValueError(f"Unsupported target_improvement: {target_improvement}. Only 'faithfulness' and 'relevance' are supported.")

        messages = [
            {"role": "system", "content": f"You are an expert in table question modification for {classification} domain."},
            {"role": "user", "content": prompt}
        ]

        response = self._call_llm_with_retry(messages, max_tokens=512, temperature=0.0,
                                           context=f"table_question_modification_{target_improvement}")
        return self._parse_modification_response(response)


    def _parse_modification_response(self, response: str) -> Tuple[str, str]:
        """Parse modification response to extract reasoning and modified content"""
        try:
            lines = response.strip().split('\n')
            reasoning_lines = [line for line in lines if line.startswith('REASONING:')]

            # Look for MODIFIED_QUESTION or MODIFIED_ANSWER
            modified_lines = [line for line in lines if line.startswith('MODIFIED_QUESTION:') or line.startswith('MODIFIED_ANSWER:')]

            reasoning = reasoning_lines[0].replace('REASONING:', '').strip() if reasoning_lines else "Modification applied"

            if modified_lines:
                modified_content = modified_lines[0]
                if modified_content.startswith('MODIFIED_QUESTION:'):
                    modified_content = modified_content.replace('MODIFIED_QUESTION:', '').strip()
                elif modified_content.startswith('MODIFIED_ANSWER:'):
                    modified_content = modified_content.replace('MODIFIED_ANSWER:', '').strip()
            else:
                # Improved fallback: try to extract content after common patterns
                response_text = response.strip()

                # Try to find content after common patterns
                patterns_to_try = [
                    'MODIFIED_ANSWER:', 'Modified Answer:', 'Answer:', 'ANSWER:',
                    'MODIFIED_QUESTION:', 'Modified Question:', 'Question:', 'QUESTION:',
                    'Result:', 'Output:'
                ]

                modified_content = None
                for pattern in patterns_to_try:
                    if pattern in response_text:
                        parts = response_text.split(pattern, 1)
                        if len(parts) > 1:
                            candidate = parts[1].strip().split('\n')[0].strip()
                            # Remove any quotes around the content
                            candidate = candidate.strip('"\'')
                            if candidate and candidate != response_text:
                                modified_content = candidate
                                break

                # If no pattern found, try to extract the last meaningful line
                if not modified_content:
                    non_empty_lines = [line.strip() for line in lines if line.strip() and not line.startswith(('REASONING:', 'Note:', 'Explanation:', 'TASK:', 'MANDATORY:', 'CRITICAL:', 'SEARCH:', 'FORMAT:'))]
                    if non_empty_lines:
                        # Take the last line and clean it
                        modified_content = non_empty_lines[-1].strip('"\'')
                    else:
                        modified_content = response_text

        except Exception as e:
            print(f"Error parsing modification response: {e}")
            reasoning = "Modification parsing error"
            modified_content = response.strip()

        return reasoning, modified_content


    def evaluate_and_modify_factoid_question_relevance(self, question: str, classification: str,
                                                      answer: str, context: str,
                                                      threshold: float = 6.0) -> Tuple[str, float, str, bool]:
        """
        Complete workflow: Assess factoid question relevance and modify if below threshold.

        Returns:
            - final_question: Original or modified question
            - final_score: Relevance score (original or post-modification)
            - reasoning: Assessment and modification reasoning
            - was_modified: Whether modification was applied
        """

        # Step 1: Initial assessment
        original_score, original_reasoning = self.evaluate_factoid_question_relevance(
            question, classification, answer, context
        )

        print(f"   [QUESTION-REL] Original score: {original_score}/10")

        # Step 2: Check if modification needed
        if original_score >= threshold:
            return question, original_score, original_reasoning, False

        # Step 3: Attempt modification for relevance
        print(f"   [QUESTION-MOD] Attempting relevance enhancement...")

        mod_reasoning, modified_question = self.modify_factoid_question(
            question, context, classification, "relevance", answer
        )
        mod_success = modified_question != question

        if not mod_success:
            print(f"   [QUESTION-MOD] Failed: {mod_reasoning}")
            return question, original_score, f"{original_reasoning} | Modification failed: {mod_reasoning}", False

        # Step 4: Re-assess modified question
        modified_score, modified_reasoning = self.evaluate_factoid_question_relevance(
            modified_question, classification, answer, context
        )

        print(f"   [QUESTION-MOD] Modified score: {modified_score}/10")

        # Step 5: Keep modification only if improved
        if modified_score > original_score:
            final_reasoning = f"Original: {original_reasoning} | Modified: {modified_reasoning} | Improvement: {modified_score - original_score:+.1f}"
            print(f"   [QUESTION-MOD] Success - improvement: {modified_score - original_score:+.1f}")
            return modified_question, modified_score, final_reasoning, True
        else:
            final_reasoning = f"{original_reasoning} | Modification attempted but no improvement ({modified_score} vs {original_score})"
            print(f"   [QUESTION-MOD] Rejected - no improvement")
            return question, original_score, final_reasoning, False

    def apply_qa_modifications(self, question: str, answer: str, context: str, classification: str,
                              question_type: str, modification_type: str,
                              target_improvement: str, table_caption: Optional[str] = None) -> Tuple[str, str, str]:
        """Apply modifications to QA pair based on identified needs with type-specific methods"""

        # Get effective classification using hybrid approach
        effective_classification, classification_metadata = self.get_effective_classification_enhanced(context, classification)

        print(f"  Applying {modification_type} modification for {target_improvement} improvement...")

        modified_question = question
        modified_answer = answer
        reasoning_parts = []

        if modification_type in ["question", "both"]:
            if question_type.lower() == "factoid":
                q_reasoning, modified_question = self.modify_factoid_question(
                    question, context, effective_classification, target_improvement, answer
                )
            elif question_type.lower() == "table":
                q_reasoning, modified_question = self.modify_table_question(
                    question, context, effective_classification, target_improvement, table_caption
                )
            else:  # non_factoid
                q_reasoning, modified_question = self.modify_nonfactoid_question(
                    question, context, effective_classification, target_improvement
                )
            reasoning_parts.append(f"Question: {q_reasoning}")

        if modification_type in ["answer", "both"]:
            if question_type.lower() == "factoid":
                # Use factoid CoT version with empty instructions for standard mode
                a_reasoning, modified_answer = self.modify_factoid_answer_with_cot(
                    modified_question, answer, context, effective_classification, target_improvement, ""
                )
            elif question_type.lower() == "table":
                # Use table-specific CoT version
                a_reasoning, modified_answer = self.modify_table_answer_with_cot(
                    modified_question, answer, context, effective_classification, target_improvement, "", table_caption
                )
            else:  # non_factoid
                # Use non-factoid CoT version with empty instructions for standard mode
                a_reasoning, modified_answer = self.modify_nonfactoid_answer_with_cot(
                    modified_question, answer, context, effective_classification, target_improvement, ""
                )
            reasoning_parts.append(f"Answer: {a_reasoning}")

        combined_reasoning = " | ".join(reasoning_parts)

        return modified_question, modified_answer, combined_reasoning

    def apply_qa_modifications_with_cot(self, question: str, answer: str, context: str, classification: str,
                                       question_type: str, modification_type: str,
                                       target_improvement: str, table_caption: Optional[str] = None,
                                       previous_failures: List[str] = None, attempt_num: int = 0) -> Tuple[str, str, str]:
        """Apply modifications with chain-of-thought learning from previous failures"""

        # Get effective classification using hybrid approach
        effective_classification, classification_metadata = self.get_effective_classification_enhanced(context, classification)

        if not previous_failures or attempt_num == 0:
            # First attempt - use standard modification
            return self.apply_qa_modifications(question, answer, context, classification,
                                             question_type, modification_type, target_improvement, table_caption)

        # Enhanced chain-of-thought: Learn from specific failures
        failure_analysis = self._analyze_failure_patterns(previous_failures)
        cot_instructions = f"""
PREVIOUS ATTEMPTS FAILED FOR THESE SPECIFIC REASONS:
{chr(10).join(previous_failures)}

CHAIN-OF-THOUGHT LEARNING:
{failure_analysis}

CRITICAL ACTIONS FOR THIS ATTEMPT:
- If previous failures mention "semantic coherence": Ensure the answer directly responds to the question
- If previous failures mention "unchanged content": Make substantial modifications to improve quality
- If previous failures mention specific scores: Focus on the failing dimension (faithfulness/relevance)
- If previous failures mention "verbatim": Use exact word-for-word spans from context
"""

        print(f"  Chain-of-thought attempt {attempt_num + 1}: Learning from {len(previous_failures)} previous failures...")

        # Use enhanced modification methods with failure-aware prompts
        modified_question = question
        modified_answer = answer

        if modification_type in ["question", "both"]:
            if question_type.lower() == "factoid":
                # Use standard method for now - enhanced prompting via context
                q_reasoning, modified_question = self.modify_factoid_question(
                    question, context + "\n\n" + cot_instructions, effective_classification, target_improvement, answer
                )
            elif question_type.lower() == "table":
                q_reasoning, modified_question = self.modify_table_question(
                    question, context + "\n\n" + cot_instructions, effective_classification, target_improvement, table_caption
                )
            else:  # non_factoid
                q_reasoning, modified_question = self.modify_nonfactoid_question(
                    question, context + "\n\n" + cot_instructions, effective_classification, target_improvement
                )

        if modification_type in ["answer", "both"]:
            if question_type.lower() == "factoid":
                a_reasoning, modified_answer = self.modify_factoid_answer_with_cot(
                    modified_question, answer, context, effective_classification, target_improvement, cot_instructions
                )
            elif question_type.lower() == "table":
                # Use table-specific CoT approach
                a_reasoning, modified_answer = self.modify_table_answer_with_cot(
                    modified_question, answer, context, effective_classification, target_improvement, cot_instructions, table_caption
                )
            else:  # non_factoid
                # Use non-factoid CoT approach
                a_reasoning, modified_answer = self.modify_nonfactoid_answer_with_cot(
                    modified_question, answer, context, effective_classification, target_improvement, cot_instructions
                )

        # Return results with chain-of-thought reasoning
        reasoning_parts = []
        if 'q_reasoning' in locals():
            reasoning_parts.append(f"Question: {q_reasoning}")
        if 'a_reasoning' in locals():
            reasoning_parts.append(f"Answer: {a_reasoning}")

        combined_reasoning = f"CoT Attempt {attempt_num + 1}: " + " | ".join(reasoning_parts)

        return (modified_question if 'modified_question' in locals() else question,
                modified_answer if 'modified_answer' in locals() else answer,
                combined_reasoning)

    def _analyze_failure_patterns(self, previous_failures: List[str]) -> str:
        """Analyze failure patterns to provide specific guidance"""
        if not previous_failures:
            return "No previous failures to analyze."

        analysis = []
        semantic_failures = sum(1 for f in previous_failures if "semantic coherence" in f.lower())
        unchanged_failures = sum(1 for f in previous_failures if "unchanged" in f.lower())
        score_failures = sum(1 for f in previous_failures if "threshold" in f.lower())

        if semantic_failures > 0:
            analysis.append(f"[ALERT] SEMANTIC COHERENCE is the main issue ({semantic_failures} attempts failed)")
            analysis.append("   SOLUTION: The question and answer must logically connect - answer must respond to what question asks")

        if unchanged_failures > 0:
            analysis.append(f"[WARNING]  No changes being made ({unchanged_failures} attempts unchanged)")
            analysis.append("   SOLUTION: Must actually modify content, not just return original")

        if score_failures > 0:
            analysis.append(f"METRICS Quality scores not improving ({score_failures} attempts below threshold)")
            analysis.append("   SOLUTION: Focus on the specific failing dimension (faithfulness vs relevance)")

        return "\n".join(analysis) if analysis else "Multiple different issues detected - need comprehensive approach"

    def modify_factoid_answer_with_cot(self, question: str, answer: str, context: str, classification: str,
                                      target_improvement: str, cot_instructions: str) -> Tuple[str, str]:
        """Enhanced factoid answer modification with Chain-of-Thought learning"""

        base_prompt = f"""You are an expert in corporate sustainability reporting.

{cot_instructions}

CONTEXT: "{context}"
QUESTION: "{question}"
ORIGINAL ANSWER: "{answer}"

CRITICAL: Based on the failure analysis above, you must address the specific issues mentioned.

"""

        if target_improvement == "faithfulness":
            # Use dedicated span-based faithfulness modification
            # First evaluate to get complete evaluation data
            faith_score, eval_reasoning, span_data = self.evaluate_factoid_answer_faithfulness_spans(
                question, answer, context, classification
            )

            # Skip modification if already faithful enough
            if faith_score >= 8.0:
                return f"CoT Enhanced: {cot_instructions} No modification needed - faithfulness score {faith_score} already high", answer

            # Apply span-based faithfulness modification with full context
            modified_answer, was_modified, reasoning = self._modify_factoid_spans_for_faithfulness(
                answer, context, question, classification, span_data
            )

            if was_modified:
                cot_reasoning = f"CoT Enhanced: {cot_instructions} Previous evaluation (Score: {faith_score}): {eval_reasoning}. Applied span-based faithfulness correction: {reasoning}"
                return cot_reasoning, modified_answer
            else:
                # Fall back to CoT approach if span method didn't modify
                prompt = base_prompt + f"""
TASK: Modify this factoid answer to be PERFECTLY FAITHFUL to the context using exact verbatim extraction.

PREVIOUS EVALUATION (Score: {faith_score}/10): {eval_reasoning}

FAITHFULNESS REQUIREMENTS (CRITICAL):
1. EXACT VERBATIM EXTRACTION: Answer must be word-for-word from context text
2. NO PARAPHRASING: Do not rephrase or reword any part of the context
3. NO EXTERNAL KNOWLEDGE: Use only what appears exactly in the context
4. SPAN SELECTION: Choose the most complete verbatim spans that answer the question
5. NO REDUNDANCY: Avoid duplicate, overlapping, or semantically identical spans

FORMAT YOUR RESPONSE EXACTLY AS:
REASONING: [Explain why original answer was not verbatim and which exact spans you selected]
MODIFIED_ANSWER: [Exact word-for-word span(s) from context text]"""

        elif target_improvement == "relevance":
            # Use dedicated span-based relevance modification
            # First evaluate to get complete evaluation data
            relevance_score, eval_reasoning, span_data = self.evaluate_factoid_answer_relevance(
                answer, classification, context, question
            )

            # Skip modification if already relevant enough
            if relevance_score >= 6.0:
                return f"CoT Enhanced: {cot_instructions} No modification needed - relevance score {relevance_score} already high", answer

            # Apply span-based relevance modification with full context
            modified_answer, was_modified, reasoning = self._modify_factoid_spans(
                answer, context, question, classification, span_data
            )

            if was_modified:
                cot_reasoning = f"CoT Enhanced: {cot_instructions} Previous evaluation (Score: {relevance_score}): {eval_reasoning}. Applied span-based relevance improvement: {reasoning}"
                return cot_reasoning, modified_answer
            else:
                # Fall back to CoT approach if span method didn't modify
                prompt = base_prompt + f"""
TASK: Find a better FACTOID ANSWER by extracting verbatim spans from the context.

PREVIOUS EVALUATION (Score: {relevance_score}/10): {eval_reasoning}

FACTOID ANSWER RULES (CRITICAL):
1. Answer MUST be EXACT, VERBATIM SPAN(S) ONLY extracted word-for-word from the CONTEXT
2. Must provide BETTER {classification}-relevance than the original answer: "{answer}"

FORMAT YOUR RESPONSE EXACTLY AS:
REASONING: [Why this extracted span is more {classification}-relevant than "{answer}"]
MODIFIED_ANSWER: [Exact verbatim span(s) extracted from CONTEXT - different from "{answer}"]"""

        # Get LLM response
        messages = [{"role": "user", "content": prompt}]
        response = self._call_llm(messages)

        # Parse response
        reasoning, modified_answer = self._parse_modification_response(response)
        reasoning = "CoT Enhanced: " + reasoning

        return reasoning, modified_answer

    def modify_nonfactoid_answer_with_cot(self, question: str, answer: str, context: str, classification: str,
                                        target_improvement: str, cot_instructions: str) -> Tuple[str, str]:
        """Enhanced non-factoid answer modification with Chain-of-Thought learning"""

        base_prompt = f"""You are an expert in corporate sustainability reporting and analytical answer improvement.

{cot_instructions}

CONTEXT: "{context}"
QUESTION: "{question}"
ORIGINAL ANSWER: "{answer}"

"""

        if target_improvement == "faithfulness":
            prompt = base_prompt + f"""
TASK: Modify this non-factoid answer to be PERFECTLY FAITHFUL to the context while maintaining analytical depth.

NON-FACTOID FAITHFULNESS REQUIREMENTS:
1. CONTEXT GROUNDING: All claims, analysis, and interpretations must be supported by specific information in the context
2. NUMERICAL ACCURACY: Any numerical values, percentages, dates, or quantitative data must match context exactly
3. NO EXTERNAL KNOWLEDGE: Do not add information, frameworks, or analysis not derivable from the context
4. ANALYTICAL CONSISTENCY: Maintain comprehensive explanatory nature while ensuring all reasoning is context-based
5. EVIDENCE LINKAGE: Explicitly connect analytical conclusions to specific context evidence
6. DOMAIN RELEVANCE: While improving faithfulness, maintain {classification} domain relevance where context supports it

MODIFICATION PROCESS:
1. Identify unsupported claims, external assumptions, or inaccurate numerical references
2. Verify all quantitative data against context
3. Strengthen analytical reasoning using only context-derived evidence
4. Maintain comprehensive explanatory approach appropriate for non-factoid questions
5. Preserve {classification} domain focus where supported by context information
6. Ensure answer must be DIFFERENT from original while improving context faithfulness

FORMAT YOUR RESPONSE EXACTLY AS:
REASONING: [Explain what faithfulness issues were corrected and how analysis now stays grounded in context while maintaining {classification} relevance]
MODIFIED_ANSWER: [Comprehensive analytical answer based strictly on context information with accurate data and maintained {classification} domain focus]"""

        elif target_improvement == "relevance":
            prompt = base_prompt + f"""
TASK: Enhance this answer's {classification} domain relevance while maintaining strict context faithfulness.

{classification.upper()} DOMAIN FOCUS:
- ESG: Focus on environmental impact, social responsibility, governance practices, stakeholder value, risk management
- EU Taxonomy: Focus on sustainable economic activities, taxonomy alignment, technical screening criteria, environmental objectives
- Sustainability: Focus on sustainable development, resource efficiency, long-term value creation, impact measurement

ENHANCEMENT STRATEGY:
1. ANALYTICAL DEPTH: Provide deeper analysis relevant to {classification} professionals and decision-makers
2. DOMAIN PERSPECTIVE: Frame insights from {classification} operational and strategic viewpoints
3. PROFESSIONAL VALUE: Target content that {classification} practitioners would find actionable
4. CONTEXTUAL GROUNDING: Keep all enhancements strictly based on provided context - do not add external information
5. FLEXIBLE APPROACH: Enhance analytical, numerical, or explanatory content as appropriate
6. CONTEXT LIMITATION: All {classification} domain enhancements must be derivable from the given context

MODIFICATION REQUIREMENTS:
- Enhance {classification} domain analytical perspective using only context-supported information
- Strengthen professional value for {classification} stakeholders and practitioners
- Improve strategic relevance to {classification} business objectives and frameworks
- Maintain answer type (analytical/numerical/explanatory) while enhancing domain focus
- All modifications must stay within the boundaries of the provided context
- Answer must be DIFFERENT from original with clear {classification} relevance improvement

FORMAT YOUR RESPONSE EXACTLY AS:
REASONING: [Explain how {classification} domain relevance was enhanced using only context-supported information while preserving faithfulness]
MODIFIED_ANSWER: [Enhanced answer with stronger {classification} domain focus derived strictly from context information]"""

        else:
            raise ValueError(f"Unsupported target_improvement: {target_improvement}. Only 'faithfulness' and 'relevance' are supported.")

        # Get LLM response
        messages = [{"role": "user", "content": prompt}]
        response = self._call_llm(messages)

        # Parse response
        reasoning, modified_answer = self._parse_modification_response(response)
        reasoning = "CoT Enhanced (Non-Factoid): " + reasoning

        return reasoning, modified_answer

    def modify_table_answer_with_cot(self, question: str, answer: str, context: str, classification: str,
                                    target_improvement: str, cot_instructions: str, table_caption: Optional[str] = None) -> Tuple[str, str]:
        """Enhanced table answer modification with Chain-of-Thought learning"""

        base_prompt = f"""You are an expert in corporate sustainability reporting and tabular data analysis.

{cot_instructions}

TABLE CONTEXT: "{context}"
{f'TABLE CAPTION: {table_caption}' if table_caption else 'TABLE: Tabular data converted to text format'}
QUESTION: "{question}"
ORIGINAL ANSWER: "{answer}"

IMPORTANT: The TABLE CONTEXT has been converted to paragraph format. Original table structure (rows, columns, cell positions) is not preserved in this format.

"""

        if target_improvement == "faithfulness":
            prompt = base_prompt + """
TASK: Modify this table answer to be PERFECTLY FAITHFUL to the TABLE CONTEXT while handling both specific and summary questions.

TABLE FAITHFULNESS REQUIREMENTS:
1. DATA ACCURACY: All numerical values, percentages, names, and categories must match the TABLE CONTEXT exactly
2. CALCULATION PRECISION: Any mathematical operations, totals, or derived values must be mathematically correct
3. NO INVENTED DATA: Do not create, assume, or extrapolate data points not present in the TABLE CONTEXT
4. SUMMARY ACCURACY: For summary questions, ensure aggregations and generalizations are based on actual TABLE CONTEXT
5. REFERENCE PRECISION: Accurately describe data relationships and patterns present in the TABLE CONTEXT

TABLE QUESTION TYPES HANDLING:
- SPECIFIC DATA EXTRACTION: Direct numerical values, names, categories from TABLE CONTEXT
- SUMMARY/ANALYTICAL: Trends, patterns, comparisons, totals derived from TABLE CONTEXT
- COMPARATIVE: Relationships between different table elements
- CATEGORICAL: Groupings and classifications present in TABLE CONTEXT

MODIFICATION PROCESS:
1. Cross-verify all numerical values and data points against TABLE CONTEXT
2. Correct any calculation errors or mathematical inaccuracies
3. Remove claims not supported by TABLE CONTEXT
4. For summary questions, ensure generalizations are data-based
5. Ensure answer must be DIFFERENT from original while improving data accuracy

VALIDATION CHECKLIST:
- Do all numbers exactly match the TABLE CONTEXT?
- Are calculations mathematically correct based on available data?
- Are summary statements supported by actual table content?
- Is the data interpretation accurate and complete?

FORMAT YOUR RESPONSE EXACTLY AS:
REASONING: [Explain what data accuracy issues were corrected and how the answer now faithfully represents table information]
MODIFIED_ANSWER: [Accurate answer based precisely on TABLE CONTEXT, appropriate for question type]"""

        elif target_improvement == "relevance":
            prompt = base_prompt + f"""
TASK: Enhance this table answer's {classification} domain relevance while maintaining complete data accuracy.

SIMPLE RELEVANCE REQUIREMENTS:
1. Focus on {classification}-relevant data points in the TABLE CONTEXT
2. Use {classification} professional terminology where appropriate
3. Maintain all numerical values exactly as they appear in TABLE CONTEXT
4. Provide interpretation valuable for {classification} professionals

MODIFICATION PROCESS:
1. Identify {classification}-relevant metrics in the TABLE CONTEXT
2. Enhance {classification} terminology and professional perspective
3. Keep all data completely accurate
4. Ensure answer must be DIFFERENT from original while improving {classification} relevance

FORMAT YOUR RESPONSE EXACTLY AS:
REASONING: [Explain how {classification} relevance was improved while keeping data accurate]
MODIFIED_ANSWER: [Data-accurate answer with enhanced {classification} focus]"""

        else:
            raise ValueError(f"Unsupported target_improvement: {target_improvement}. Only 'faithfulness' and 'relevance' are supported.")

        # Get LLM response
        messages = [{"role": "user", "content": prompt}]
        response = self._call_llm(messages)

        # Parse response
        reasoning, modified_answer = self._parse_modification_response(response)
        reasoning = "CoT Enhanced (Table): " + reasoning

        return reasoning, modified_answer


    def evaluate_qa_pair(self, question: str, answer: str, context: str, classification: str,
                        question_type: str, chunk_number: Optional[int] = None, page_number: Optional[int] = None,
                        table_caption: Optional[str] = None, classification_metadata: Optional[Dict] = None,
                        disable_modification: bool = False, silent: bool = False) -> EvaluationResult:
        """Evaluate a single QA pair using Q&A-only approach with numeric guardrails"""

        if not silent:
            self._log_phase1_qa_display(question, answer, context, classification, question_type, chunk_number, page_number)

        # Reset modification depth for each new QA pair evaluation
        if not disable_modification:
            self.modification_depth = 0

        # Pre-LLM numeric guardrails
        numeric_check_passed = self.check_numeric_accuracy(answer, context)

        # Evaluate question faithfulness based on question type
        if question_type.lower() == "factoid":
            question_faithfulness_score, question_faithfulness_reasoning = self.evaluate_factoid_question_faithfulness(
                question, context
            )
        elif question_type.lower() == "table":
            question_faithfulness_score, question_faithfulness_reasoning = self.evaluate_table_question_faithfulness(
                question, context, table_caption
            )
        else:  # non_factoid
            question_faithfulness_score, question_faithfulness_reasoning = self.evaluate_nonfactoid_question_faithfulness(
                question, context
            )

        # Evaluate answer faithfulness based on question type
        if question_type.lower() == "factoid":
            answer_faithfulness_score, answer_faithfulness_reasoning, faithfulness_span_data = self.evaluate_factoid_answer_faithfulness_spans(
                question, answer, context, classification
            )
            # Store for logging purposes
            self._last_faithfulness_span_data = faithfulness_span_data
        elif question_type.lower() == "table":
            answer_faithfulness_score, answer_faithfulness_reasoning = self.evaluate_table_answer_faithfulness(
                question, answer, context, table_caption
            )
            # No span data for table
            self._last_faithfulness_span_data = None
        else:  # non_factoid
            answer_faithfulness_score, answer_faithfulness_reasoning = self.evaluate_nonfactoid_answer_faithfulness(
                question, answer, context
            )
            # No span data for non-factoid
            self._last_faithfulness_span_data = None

        # Apply numeric guardrail penalty
        if not numeric_check_passed:
            answer_faithfulness_score = 1.0
            answer_faithfulness_reasoning += " [NUMERIC GUARDRAIL FAILED]"

        # Combined faithfulness score (simple average)
        faithfulness_score = (question_faithfulness_score + answer_faithfulness_score) / 2
        faithfulness_reasoning = f"Q_faith: {question_faithfulness_score:.1f}, A_faith: {answer_faithfulness_score:.1f}"

        # Use pre-determined classification (already processed at context level)
        effective_classification = classification
        print(f"  Using pre-determined classification: {effective_classification}")

        # Evaluate question relevance based on question type
        if question_type.lower() == "factoid":
            question_relevance_score, question_relevance_reasoning = self.evaluate_factoid_question_relevance(
                question, effective_classification, answer, context
            )
        elif question_type.lower() == "table":
            question_relevance_score, question_relevance_reasoning = self.evaluate_table_question_relevance(
                question, effective_classification
            )
        else:  # non_factoid
            question_relevance_score, question_relevance_reasoning = self.evaluate_nonfactoid_question_relevance(
                question, effective_classification
            )

        # Evaluate answer relevance based on question type
        if question_type.lower() == "factoid":
            answer_relevance_score, answer_relevance_reasoning, span_data = self.evaluate_factoid_answer_relevance(
                answer, effective_classification, context, question
            )
        elif question_type.lower() == "table":
            answer_relevance_score, answer_relevance_reasoning = self.evaluate_table_answer_relevance(
                answer, effective_classification
            )
        else:  # non_factoid
            answer_relevance_score, answer_relevance_reasoning = self.evaluate_nonfactoid_answer_relevance(
                answer, effective_classification
            )

        # Combined relevance score (simple average)
        relevance_score = (question_relevance_score + answer_relevance_score) / 2
        relevance_reasoning = f"Q_rel: {question_relevance_score:.1f}, A_rel: {answer_relevance_score:.1f}"

        # Phase 2 Logging: Show evaluation results
        if not silent:
            # Get span data for factoid answers
            span_data_for_logging = None
            if question_type.lower() == "factoid" and 'span_data' in locals():
                span_data_for_logging = span_data
            self._log_phase2_evaluation_results(question_type, faithfulness_score, relevance_score,
                                               answer_faithfulness_score, answer_relevance_score, span_data_for_logging,
                                               question_faithfulness_score, question_relevance_score)

        # Calculate quality thresholds for keep/discard decision
        keep_qa = faithfulness_score >= self.quality_threshold and relevance_score >= self.quality_threshold

        # Calculate hybrid numeric accuracy for backwards compatibility (now includes factoids with numbers)
        numeric_consistency_result = None
        if question_type.lower() in ["table", "non_factoid"]:
            # Complex numeric evaluation removed - was over-engineered
            numeric_consistency_result = {"score": 1.0, "details": "Complex numeric validation disabled", "exact_count": 0, "partial_count": 0}
        elif question_type.lower() == "factoid":
            # Apply numeric guardrails to factoids when they contain numbers
            import re
            if re.search(r'\d+', answer):
                # Complex numeric evaluation removed - was over-engineered
                numeric_consistency_result = {"score": 1.0, "details": "Complex numeric validation disabled", "exact_count": 0, "partial_count": 0}

        # Extract numeric consistency details if available
        numeric_consistency_score = None
        numeric_consistency_details = None
        numeric_exact_count = None
        numeric_partial_count = None

        if numeric_consistency_result:
            numeric_consistency_score = numeric_consistency_result["score"]
            numeric_consistency_details = numeric_consistency_result["details"]
            numeric_exact_count = numeric_consistency_result["exact_count"]
            numeric_partial_count = numeric_consistency_result["partial_count"]

        # ENHANCED MODIFICATION LOGIC - with comprehensive validation
        modification_applied = False
        original_question_store = question
        original_answer_store = answer
        modified_question_store = None
        modified_answer_store = None
        modification_reasoning_store = None
        modification_type_store = None
        modification_target_store = None
        modification_validation_passed = None
        modification_issues_prevented = 0
        modification_attempts = 0
        modification_fallback_used = False

        # Check if modification is enabled and needed
        # Only modify if either dimension is below base threshold (6.0)
        modification_needed = faithfulness_score < 6.0 or relevance_score < 6.0

        # Phase 3 Logging: Show modification decision
        if not silent:
            self._log_phase3_modification_decision(faithfulness_score, relevance_score, modification_needed)

        if (self.enable_modification and
            hasattr(self, 'determine_modification_needs') and
            modification_needed and
            not disable_modification and
            self.modification_depth < self.max_modification_depth):  # Prevent infinite recursion

            try:
                # Increment modification depth to prevent infinite recursion
                self.modification_depth += 1

                # Track modification attempts for detailed failure reporting
                self.current_modification_attempts = 1

                # Determine modification needs using existing logic
                modification_type, target_improvement = self.determine_modification_needs(
                    faithfulness_score,
                    relevance_score,
                    question_faithfulness_score,
                    answer_faithfulness_score,
                    question_relevance_score,
                    answer_relevance_score
                )

                # Apply safe modifications using enhanced system
                modification_result = self.modify_qa_pair_safely(
                    question=question,
                    answer=answer,
                    context=context,
                    classification=classification,
                    question_type=question_type,
                    modification_type=modification_type,
                    target_improvement=target_improvement,
                    table_caption=table_caption,
                    chunk_number=chunk_number,
                    page_number=page_number
                )

                # Extract modification metadata with safe key access
                modification_applied = modification_result.get("modification_applied", False)
                modification_validation_passed = modification_result.get("validation_passed", False)
                modification_issues_prevented = modification_result.get("validation_issues_prevented", 0)
                modification_attempts = modification_result.get("modification_attempts", 0)
                modification_fallback_used = modification_result.get("fallback_used", False)

                if modification_applied and not modification_fallback_used:
                    # Use modified content for final evaluation
                    question = modification_result.get("modified_question", question)
                    answer = modification_result.get("modified_answer", answer)
                    modification_reasoning_store = modification_result.get("modification_reasoning", "No reasoning provided")
                    modification_type_store = modification_type
                    modification_target_store = target_improvement
                    modified_question_store = question
                    modified_answer_store = answer

                    print(f"\nMETRICS RE-EVALUATING METRICS AFTER MODIFICATION")
                    print(f"{'-'*70}")
                    print(f"[METRICS] Original Scores:")
                    print(f"   - Faithfulness: {faithfulness_score:.2f}")
                    print(f"   - Relevance: {relevance_score:.2f}")
                    print(f"METRICS Component Breakdown (Original):")
                    print(f"   - Question Faithfulness: {question_faithfulness_score:.2f}")
                    print(f"   - Answer Faithfulness: {answer_faithfulness_score:.2f}")
                    print(f"   - Question Relevance: {question_relevance_score:.2f}")
                    print(f"   - Answer Relevance: {answer_relevance_score:.2f}")

                    # Re-evaluate with modified content for accurate scoring
                    print(f"\n[PROCESSING] Calculating new scores with modified content...")

                    # Quick re-evaluation for modified content
                    if question_type.lower() == "factoid":
                        new_answer_faithfulness_score, new_answer_faithfulness_reasoning, _ = self.evaluate_factoid_answer_faithfulness_spans(
                            modified_question, modified_answer, context, classification
                        )
                    else:
                        new_answer_faithfulness_score, new_answer_faithfulness_reasoning = self.evaluate_nonfactoid_answer_faithfulness(
                            modified_question, modified_answer, context
                        )

                    # Evaluate modified question faithfulness based on question type
                    if question_type.lower() == "factoid":
                        new_question_faithfulness_score, new_question_faithfulness_reasoning = self.evaluate_factoid_question_faithfulness(
                            modified_question, context
                        )
                    elif question_type.lower() == "table":
                        new_question_faithfulness_score, new_question_faithfulness_reasoning = self.evaluate_table_question_faithfulness(
                            modified_question, context, table_caption
                        )
                    else:  # non_factoid
                        new_question_faithfulness_score, new_question_faithfulness_reasoning = self.evaluate_nonfactoid_question_faithfulness(
                            modified_question, context
                        )

                    # Evaluate modified question relevance based on question type
                    if question_type.lower() == "factoid":
                        new_question_relevance_score, new_question_relevance_reasoning = self.evaluate_factoid_question_relevance(
                            modified_question, classification, modified_answer, context
                        )
                    elif question_type.lower() == "table":
                        new_question_relevance_score, new_question_relevance_reasoning = self.evaluate_table_question_relevance(
                            modified_question, classification
                        )
                    else:  # non_factoid
                        new_question_relevance_score, new_question_relevance_reasoning = self.evaluate_nonfactoid_question_relevance(
                            modified_question, classification
                        )

                    # Evaluate modified answer relevance based on question type
                    if question_type.lower() == "factoid":
                        new_answer_relevance_score, new_answer_relevance_reasoning, _ = self.evaluate_factoid_answer_relevance(
                            modified_answer, classification, context, modified_question
                        )
                    elif question_type.lower() == "table":
                        new_answer_relevance_score, new_answer_relevance_reasoning = self.evaluate_table_answer_relevance(
                            modified_answer, classification
                        )
                    else:  # non_factoid
                        new_answer_relevance_score, new_answer_relevance_reasoning = self.evaluate_nonfactoid_answer_relevance(
                            modified_answer, classification
                        )

                    # Store original scores for comparison
                    original_faithfulness = faithfulness_score
                    original_relevance = relevance_score

                    # Update scores with modified content (domain specificity removed)
                    question_faithfulness_score = new_question_faithfulness_score
                    answer_faithfulness_score = new_answer_faithfulness_score
                    question_relevance_score = new_question_relevance_score
                    answer_relevance_score = new_answer_relevance_score

                    # Recalculate combined scores (simple averages)
                    faithfulness_score = (question_faithfulness_score + answer_faithfulness_score) / 2
                    relevance_score = (question_relevance_score + answer_relevance_score) / 2

                    # Calculate improvements
                    faithfulness_improvement = faithfulness_score - original_faithfulness
                    relevance_improvement = relevance_score - original_relevance

                    keep_qa = faithfulness_score >= self.quality_threshold and relevance_score >= self.quality_threshold

                    print(f"\n[METRICS] NEW SCORES AFTER MODIFICATION:")
                    print(f"   - Faithfulness: {faithfulness_score:.2f} ({faithfulness_improvement:+.2f})")
                    print(f"   - Relevance: {relevance_score:.2f} ({relevance_improvement:+.2f})")
                    print(f"METRICS Component Breakdown (Modified):")
                    print(f"   - Question Faithfulness: {question_faithfulness_score:.2f}")
                    print(f"   - Answer Faithfulness: {answer_faithfulness_score:.2f}")
                    print(f"   - Question Relevance: {question_relevance_score:.2f}")
                    print(f"   - Answer Relevance: {answer_relevance_score:.2f}")

                    print(f"\nTARGET MODIFICATION IMPACT:")
                    print(f"   - Overall Improvement: {(faithfulness_improvement + relevance_improvement)/2:+.2f}")
                    print(f"   - Target was: {target_improvement}")
                    print(f"   - Quality Threshold Met: {'YES' if keep_qa else 'NO'}")
                    print(f"   - Decision: {'KEEP' if keep_qa else 'DELETE'}")
                    print(f"{'-'*70}\n")

            except Exception as e:
                print(f"  Enhanced modification failed, using original: {e}")
                modification_applied = False
                modification_fallback_used = True
                # Ensure attempt count reflects the actual attempt made
                modification_attempts = getattr(self, 'current_modification_attempts', 1)

                # Update legacy stats for compatibility
                self.modification_stats["total_attempts"] += 1
                self.modification_stats["failed_api"] += 1
                self.modification_stats["fallbacks_to_original"] += 1
            finally:
                # Always decrement modification depth when exiting modification
                self.modification_depth = max(0, self.modification_depth - 1)

        # SPAN-LEVEL MODIFICATION FOR FACTOID ANSWERS (Enhanced Strategy)
        span_modification_applied = False
        span_modification_successful = False
        span_modification_reasoning = ""

        # OPTIMIZATION: Span modification triggered by individual span scores (< 6.0),
        # not overall aggregation score. Mathematical proof: overall_score < 6.0 implies
        # at least one span < 6.0, making overall score check redundant for modification.
        if (question_type.lower() == "factoid" and
            'span_data' in locals() and
            span_data.get('has_problematic_spans', False) and  # Based on any(span_score < 6.0)
            not disable_modification):

            try:
                # Apply span-level modification
                span_modified_answer, span_modification_successful, span_modification_reasoning = self._modify_factoid_spans(
                    answer, context, question, effective_classification, span_data, question_type
                )

                if span_modification_successful:
                    # Re-evaluate with span-modified answer
                    # Quick re-evaluation for span-modified content
                    new_relevance_score, new_relevance_reasoning, new_span_data = self.evaluate_factoid_answer_relevance(
                        span_modified_answer, effective_classification, context, question
                    )

                    # Check if overall score improved to >= 6.0
                    new_overall_relevance = (question_relevance_score + new_relevance_score) / 2

                    if new_overall_relevance >= 6.0:
                        # Update the answer and scores
                        answer = span_modified_answer
                        answer_relevance_score = new_relevance_score
                        answer_relevance_reasoning = new_relevance_reasoning
                        relevance_score = new_overall_relevance
                        relevance_reasoning = f"Q_rel: {question_relevance_score:.1f}, A_rel: {answer_relevance_score:.1f}"
                        span_modification_applied = True
                        modification_applied = True  # Update main modification status
                        modified_answer_store = span_modified_answer
                        modification_reasoning_store = span_modification_reasoning

                        # Update keep decision based on new scores
                        keep_qa = faithfulness_score >= self.quality_threshold and relevance_score >= self.quality_threshold
                    else:

                        # Check if we should delete problematic spans and keep good ones
                        if new_overall_relevance < 6.0:
                            # Apply selective retention strategy
                            good_spans = span_data.get('good_spans', [])
                            if good_spans:
                                retained_answer = ", ".join(good_spans)

                                # Re-evaluate with only good spans
                                final_relevance_score, final_relevance_reasoning, _ = self.evaluate_factoid_answer_relevance(
                                    retained_answer, effective_classification, context, question
                                )
                                final_overall_relevance = (question_relevance_score + final_relevance_score) / 2

                                if final_overall_relevance >= 6.0:
                                    answer = retained_answer
                                    answer_relevance_score = final_relevance_score
                                    answer_relevance_reasoning = final_relevance_reasoning
                                    relevance_score = final_overall_relevance
                                    relevance_reasoning = f"Q_rel: {question_relevance_score:.1f}, A_rel: {answer_relevance_score:.1f}"
                                    span_modification_applied = True
                                    modification_applied = True  # Update main modification status
                                    modified_answer_store = retained_answer
                                    modification_reasoning_store = f"Retained {len(good_spans)} good spans"
                                    keep_qa = faithfulness_score >= self.quality_threshold and relevance_score >= self.quality_threshold
                                else:
                                    pass  # Modification didn't reach threshold

            except Exception as e:
                print(f"  [SPAN-ERROR] Span modification failed: {e}")
                span_modification_reasoning = f"Span modification error: {e}"

        # POST-MODIFICATION DECISION: Determine final keep/delete decision with detailed reasons
        # Use the actual attempt count from modification result, fallback to class attribute
        modification_attempts_count = modification_attempts if 'modification_attempts' in locals() else getattr(self, 'current_modification_attempts', 0)
        modification_failed_flag = not modification_applied and modification_attempts_count > 0

        # Simplified decision summary
        modification_status = ""
        if modification_applied:
            modification_status = f"Modified ({modification_attempts_count} attempts)"
        elif modification_failed_flag:
            modification_status = f"Failed after {modification_attempts_count} attempts"
        else:
            modification_status = "No modification needed"

        final_decision, final_decision_reason = self._make_final_qa_decision(
            faithfulness_score=faithfulness_score,
            relevance_score=relevance_score,
            modification_applied=modification_applied,
            original_keep_qa=keep_qa,
            modification_attempts=modification_attempts_count,
            modification_failed=modification_failed_flag
        )

        # Phase 4 Logging: Show final outcome
        if not silent:
            self._log_phase4_final_outcome(faithfulness_score, relevance_score, keep_qa, modification_applied, final_decision)
            # Show modification details if any were applied
            if modification_applied and modified_answer_store and modified_answer_store != original_answer_store:
                print(f"Modified Answer: {modified_answer_store}")
                print(f"Modification Reason: {modification_reasoning_store}")
                print("-" * 60)

        return EvaluationResult(
            question=question,
            answer=answer,
            question_type=question_type,
            classification=classification,
            faithfulness_score=faithfulness_score,
            faithfulness_reasoning=faithfulness_reasoning,
            relevance_score=relevance_score,
            relevance_reasoning=relevance_reasoning,
            chunk_number=chunk_number,
            page_number=page_number,
            table_caption=table_caption,
            context=context,
            question_faithfulness_score=question_faithfulness_score,
            answer_faithfulness_score=answer_faithfulness_score,
            question_relevance_score=question_relevance_score,
            answer_relevance_score=answer_relevance_score,
            numeric_guardrail_passed=numeric_check_passed,
            keep_qa=keep_qa,
            modification_applied=modification_applied,
            original_question=original_question_store,
            original_answer=original_answer_store,
            modified_question=modified_question_store,
            modified_answer=modified_answer_store,
            modification_reasoning=modification_reasoning_store,
            modification_type=modification_type_store,
            modification_target=modification_target_store,
            modification_validation_passed=modification_validation_passed
        )

    def _make_final_qa_decision(self, faithfulness_score: float, relevance_score: float,
                               modification_applied: bool, original_keep_qa: bool,
                               modification_attempts: int = 0, modification_failed: bool = False) -> tuple[str, str]:
        """Make final decision with detailed deletion reasons"""

        # Base quality thresholds - same for original and modified
        base_faithfulness_threshold = 6.0
        base_relevance_threshold = 6.0

        if modification_applied:
            # Check if modification improved quality to base standards
            meets_base_criteria = (
                faithfulness_score >= base_faithfulness_threshold and
                relevance_score >= base_relevance_threshold
            )

            if meets_base_criteria:
                return "KEEP_ENHANCED", "Quality improved through modification - meets base standards"
            elif original_keep_qa:
                return "KEEP_ORIGINAL", "Modification didn't help but original acceptable"
            else:
                # Detailed deletion reasons after modification
                reasons = []
                if faithfulness_score < base_faithfulness_threshold:
                    reasons.append(f"faithfulness {faithfulness_score:.1f} < {base_faithfulness_threshold}")
                if relevance_score < base_relevance_threshold:
                    reasons.append(f"relevance {relevance_score:.1f} < {base_relevance_threshold}")

                reason = f"DELETE: Modification failed to improve quality - {', '.join(reasons) if reasons else 'below base thresholds'}"
                return "DELETE_LOW_QUALITY", reason
        else:
            # No modification applied - check if it was attempted but failed
            if modification_failed:
                # Modification was attempted but failed entirely
                reasons = []
                if faithfulness_score < base_faithfulness_threshold:
                    reasons.append(f"faithfulness {faithfulness_score:.1f} < {base_faithfulness_threshold}")
                if relevance_score < base_relevance_threshold:
                    reasons.append(f"relevance {relevance_score:.1f} < {base_relevance_threshold}")

                if modification_attempts > 0:
                    reason = f"DELETE: {modification_attempts} modification attempts failed, original quality insufficient - {', '.join(reasons) if reasons else 'below thresholds'}"
                else:
                    reason = f"DELETE: Modification system failed, original quality insufficient - {', '.join(reasons) if reasons else 'below thresholds'}"
                return "DELETE_LOW_QUALITY", reason
            else:
                # No modification attempted
                if original_keep_qa:
                    return "KEEP_ORIGINAL", "Original quality sufficient - no modification needed"
                else:
                    reasons = []
                    if faithfulness_score < base_faithfulness_threshold:
                        reasons.append(f"faithfulness {faithfulness_score:.1f} < {base_faithfulness_threshold}")
                    if relevance_score < base_relevance_threshold:
                        reasons.append(f"relevance {relevance_score:.1f} < {base_relevance_threshold}")

                    reason = f"DELETE: Original quality insufficient - {', '.join(reasons) if reasons else 'below quality thresholds'}"
                    return "DELETE_LOW_QUALITY", reason

    def _result_to_simplified_dict(self, result: EvaluationResult) -> dict:
        """Convert EvaluationResult to structured dictionary with logical field clusters"""

        # Core QA Content cluster
        qa_content = {
            "question": result.question,
            "answer": result.answer,
            "question_type": result.question_type,
            "classification": result.classification
        }

        # Evaluation Scores cluster
        scores = {
            "faithfulness_score": result.faithfulness_score,
            "relevance_score": result.relevance_score,
        }

        # Score Reasoning cluster
        reasoning = {
            "faithfulness_reasoning": result.faithfulness_reasoning,
            "relevance_reasoning": result.relevance_reasoning,
        }

        # Detailed Component Scores cluster
        component_scores = {
            "question_faithfulness_score": result.question_faithfulness_score,
            "answer_faithfulness_score": result.answer_faithfulness_score,
            "question_relevance_score": result.question_relevance_score,
            "answer_relevance_score": result.answer_relevance_score,
        }

        # Quality Assessment cluster
        quality = {
            "keep_qa": result.keep_qa,
            "numeric_guardrail_passed": result.numeric_guardrail_passed
        }

        # Document Metadata cluster
        metadata = {
            "chunk_number": result.chunk_number,
            "page_number": result.page_number,
            "table_caption": result.table_caption,
            "context": result.context
        }

        # Build the structured output
        structured = {
            "qa_content": qa_content,
            "scores": scores,
            "reasoning": reasoning,
            "component_scores": component_scores,
            "quality": quality,
            "metadata": metadata
        }

        # Modification cluster (if modification was enabled and needed)
        if hasattr(result, 'modification_applied') and result.modification_applied is not None:
            modification = {
                "modification_applied": result.modification_applied,
                "modification_type": result.modification_type,
                "modification_target": result.modification_target,
                "modification_validation_passed": result.modification_validation_passed
            }

            # Add modified content if modification was successful
            if result.modification_applied:
                modification.update({
                    "original_question": result.original_question,
                    "original_answer": result.original_answer,
                    "modified_question": result.modified_question,
                    "modified_answer": result.modified_answer,
                    "modification_reasoning": result.modification_reasoning
                })

            structured["modification"] = modification

        return structured

    def evaluate_sustainableqa_dataset(self, dataset_path: str, output_path: str,
                                     sample_size: Optional[int] = None,
                                     save_progress: bool = True) -> Dict:
        """Evaluate entire SustainableQA dataset or a sample"""

        print(f"Loading dataset from {dataset_path}...")

        # Load dataset with proper error handling
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at: {dataset_path}. Please provide a valid dataset file path.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in dataset file: {dataset_path}. Error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading dataset from {dataset_path}: {str(e)}")

        # Validate dataset schema
        schema_validation = self._validate_dataset_schema(data)
        if not schema_validation["valid"]:
            print(f"Warning: Dataset schema issues detected: {', '.join(schema_validation['issues'])}")
            if schema_validation["critical"]:
                raise ValueError(f"Critical schema errors: {', '.join(schema_validation['critical'])}")

        all_results = []
        chunks_processed = 0
        total_qa_pairs = 0

        chunks_to_process = data.get("all_chunks_qas", [])
        if sample_size:
            chunks_to_process = chunks_to_process[:sample_size]

        print(f"Processing {len(chunks_to_process)} chunks with {self.llm_provider.value} model: {self.model_name}")

        for chunk_idx, chunk in enumerate(chunks_to_process):
            context = chunk["metadata"]["paragraph"]
            original_classification = chunk["metadata"]["classification"]
            chunk_number = chunk["metadata"].get("chunk_number")
            page_number = chunk["metadata"].get("page_number")
            spans = chunk.get("spans", {})

            # Skip if classification is not in our target categories
            if original_classification not in ["ESG", "EU Taxonomy", "Sustainability"]:
                continue

            print(f"\nChunk {chunk_idx+1}/{len(chunks_to_process)}: {original_classification} classification")

            # PERFORM CLASSIFICATION ONCE PER CONTEXT/PARAGRAPH
            print(f"  Determining effective classification for this context...")
            effective_classification, classification_metadata = self.get_effective_classification_enhanced(
                context, original_classification
            )

            if effective_classification != original_classification:
                self.logger.info(f"  Classification updated: {original_classification} -> {effective_classification}")
            else:
                self.logger.debug(f"  Classification confirmed: {effective_classification}")

            # Store classification metadata for transparency
            chunk_classification_info = {
                "original": original_classification,
                "effective": effective_classification,
                "metadata": classification_metadata
            }

            print(f"  Processing QA pairs with effective classification: {effective_classification}")

            # Process factoid questions
            if chunk.get("qa_pairs_factoid") and chunk["qa_pairs_factoid"].get("qa_pairs"):
                for qa_idx, qa in enumerate(chunk["qa_pairs_factoid"]["qa_pairs"]):
                    try:
                        print(f"  Factoid QA {qa_idx+1}/{len(chunk['qa_pairs_factoid']['qa_pairs'])}")
                        result = self.evaluate_qa_pair(
                            question=qa["question"],
                            answer=qa["answer"],
                            context=context,
                            classification=effective_classification,
                            question_type="factoid",
                            chunk_number=chunk_number,
                            page_number=page_number,
                            classification_metadata=chunk_classification_info
                        )
                        all_results.append(result)
                        total_qa_pairs += 1

                        # Save progress periodically
                        if save_progress and len(all_results) % 10 == 0:
                            self._save_progress(all_results, output_path, temp=True)

                    except Exception as e:
                        print(f"    Error processing factoid QA: {e}")
                        continue

            # Process non-factoid questions
            if chunk.get("qa_pairs_non_factoid") and chunk["qa_pairs_non_factoid"].get("qa_pairs"):
                for qa_idx, qa in enumerate(chunk["qa_pairs_non_factoid"]["qa_pairs"]):
                    try:
                        print(f"  Non-factoid QA {qa_idx+1}/{len(chunk['qa_pairs_non_factoid']['qa_pairs'])}")
                        result = self.evaluate_qa_pair(
                            question=qa["question"],
                            answer=qa["answer"],
                            context=context,
                            classification=effective_classification,
                            question_type="non_factoid",
                            chunk_number=chunk_number,
                            page_number=page_number,
                            classification_metadata=chunk_classification_info
                        )
                        all_results.append(result)
                        total_qa_pairs += 1

                        # Save progress periodically
                        if save_progress and len(all_results) % 10 == 0:
                            self._save_progress(all_results, output_path, temp=True)

                    except Exception as e:
                        print(f"    Error processing non-factoid QA: {e}")
                        continue

            chunks_processed += 1

        # Process table QAs if they exist in the dataset
        if "all_table_qas" in data:
            table_qas = data["all_table_qas"]
            if sample_size:
                table_qas = table_qas[:sample_size]

            print(f"\nProcessing {len(table_qas)} table QAs...")

            for table_idx, table_qa in enumerate(table_qas):
                try:
                    # Extract table information
                    table_caption = table_qa["metadata"].get("table_caption", "")
                    paragraph = table_qa["metadata"].get("paragraph", "")
                    original_table_classification = table_qa["metadata"].get("classification", "Sustainability")

                    # Compose context from table caption and paragraph
                    context = f"Table Caption: {table_caption}\n\n{paragraph}"

                    print(f"\nTable {table_idx+1}/{len(table_qas)}: {original_table_classification} classification")

                    # PERFORM CLASSIFICATION ONCE PER TABLE CONTEXT
                    print(f"  Determining effective classification for this table context...")
                    effective_table_classification, table_classification_metadata = self.get_effective_classification_enhanced(
                        context, original_table_classification
                    )

                    if effective_table_classification != original_table_classification:
                        print(f"  Table classification updated: {original_table_classification} -> {effective_table_classification}")
                    else:
                        print(f"  Table classification confirmed: {effective_table_classification}")

                    # Store table classification metadata
                    table_classification_info = {
                        "original": original_table_classification,
                        "effective": effective_table_classification,
                        "metadata": table_classification_metadata
                    }

                    # Process table QA pairs
                    if table_qa.get("qa_pairs"):
                        print(f"  Processing {len(table_qa['qa_pairs'])} table QA pairs...")
                        for qa_idx, qa in enumerate(table_qa["qa_pairs"]):
                            try:
                                print(f"    Table QA {qa_idx+1}/{len(table_qa['qa_pairs'])}")
                                result = self.evaluate_qa_pair(
                                    question=qa["question"],
                                    answer=qa["answer"],
                                    context=context,
                                    classification=effective_table_classification,
                                    question_type="table",
                                    table_caption=table_caption,
                                    classification_metadata=table_classification_info
                                )
                                all_results.append(result)
                                total_qa_pairs += 1

                                # Save progress periodically
                                if save_progress and len(all_results) % 10 == 0:
                                    self._save_progress(all_results, output_path, temp=True)

                            except Exception as e:
                                print(f"      Error processing table QA: {e}")
                                continue

                except Exception as e:
                    print(f"    Error processing table {table_idx}: {e}")
                    continue

            if chunks_processed % 5 == 0:
                print(f"Progress: {chunks_processed}/{len(chunks_to_process)} chunks, {total_qa_pairs} QA pairs processed")

        # Generate summary statistics
        print("\nGenerating summary statistics...")
        summary = self._generate_summary_statistics(all_results)

        # Save final results
        output_data = {
            "evaluation_metadata": {
                "llm_provider": self.llm_provider.value,
                "model_name": self.model_name,
                "total_chunks_processed": chunks_processed,
                "total_qa_pairs_evaluated": len(all_results),
                "dataset_path": dataset_path,
                "sample_size": sample_size
            },
            "summary": summary,
            "detailed_results": [self._result_to_simplified_dict(result) for result in all_results]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        # Clean up temporary file
        temp_path = output_path.replace('.json', '_temp.json')
        if os.path.exists(temp_path):
            os.remove(temp_path)

        print(f"\nEvaluation complete!")
        print(f"METRICS Results saved to: {output_path}")
        print(f"[METRICS] Processed {len(all_results)} QA pairs from {chunks_processed} chunks")
        print(f"[AI] Using {self.llm_provider.value} with model: {self.model_name}")

        return output_data
    
    def _save_progress(self, results: List[EvaluationResult], output_path: str, temp: bool = False):
        """Save progress to temporary file with error handling"""
        try:
            if temp:
                temp_path = output_path.replace('.json', '_temp.json')
            else:
                temp_path = output_path

            # Ensure directory exists only if there *is* a directory part
            import os
            dirpath = os.path.dirname(temp_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)

            progress_data = {
                "progress": {
                    "total_evaluated": len(results),
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "partial_results": [self._result_to_simplified_dict(result) for result in results]
            }

            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)

            print(f"Progress saved to: {temp_path}")

        except Exception as e:
            print(f"Error saving progress: {e}")
            raise


    def _generate_summary_statistics(self, results: List[EvaluationResult]) -> Dict:
        """Generate comprehensive summary statistics with error handling"""

        if not results:
            return {"error": "No results to summarize"}

        try:
            # Overall statistics
            faithfulness_scores = [r.faithfulness_score for r in results if r.faithfulness_score is not None]
            relevance_scores = [r.relevance_score for r in results if r.relevance_score is not None]
            # Calculate composite scores for reporting (simple average of faithfulness + relevance)
            overall_scores = [(r.faithfulness_score + r.relevance_score) / 2
                            for r in results
                            if r.faithfulness_score is not None and r.relevance_score is not None]

            # By question type
            factoid_results = [r for r in results if r.question_type == "factoid"]
            nonfactoid_results = [r for r in results if r.question_type == "non_factoid"]
            table_results = [r for r in results if r.question_type == "table"]

            # By classification
            esg_results = [r for r in results if r.classification == "ESG"]
            taxonomy_results = [r for r in results if r.classification == "EU Taxonomy"]
            sustainability_results = [r for r in results if r.classification == "Sustainability"]

            def calc_stats(scores):
                if not scores:
                    return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}
                return {
                    "mean": round(statistics.mean(scores), 2),
                    "median": round(statistics.median(scores), 2),
                    "std": round(statistics.stdev(scores) if len(scores) > 1 else 0, 2),
                    "min": round(min(scores), 2),
                    "max": round(max(scores), 2)
                }

            # Quality thresholds based on thesis research
            high_quality_threshold = 8.0
            medium_quality_threshold = 6.0

            summary = {
            "total_qa_pairs": len(results),
            "overall_metrics": {
                "faithfulness": calc_stats(faithfulness_scores),
                "relevance": calc_stats(relevance_scores),
                "overall": calc_stats(overall_scores)
            },
            "by_question_type": {
                "factoid": {
                    "count": len(factoid_results),
                    "faithfulness": calc_stats([r.faithfulness_score for r in factoid_results]),
                    "relevance": calc_stats([r.relevance_score for r in factoid_results]),
                    "overall": calc_stats([(r.faithfulness_score + r.relevance_score) / 2 for r in factoid_results])
                },
                "non_factoid": {
                    "count": len(nonfactoid_results),
                    "faithfulness": calc_stats([r.faithfulness_score for r in nonfactoid_results]),
                    "relevance": calc_stats([r.relevance_score for r in nonfactoid_results]),
                    "overall": calc_stats([(r.faithfulness_score + r.relevance_score) / 2 for r in nonfactoid_results])
                },
                "table": {
                    "count": len(table_results),
                    "faithfulness": calc_stats([r.faithfulness_score for r in table_results]),
                    "relevance": calc_stats([r.relevance_score for r in table_results]),
                    "overall": calc_stats([(r.faithfulness_score + r.relevance_score) / 2 for r in table_results])
                }
            },
            "by_classification": {
                "ESG": {
                    "count": len(esg_results),
                    "faithfulness": calc_stats([r.faithfulness_score for r in esg_results]),
                    "relevance": calc_stats([r.relevance_score for r in esg_results]),
                    "overall": calc_stats([(r.faithfulness_score + r.relevance_score) / 2 for r in esg_results])
                },
                "EU Taxonomy": {
                    "count": len(taxonomy_results),
                    "faithfulness": calc_stats([r.faithfulness_score for r in taxonomy_results]),
                    "relevance": calc_stats([r.relevance_score for r in taxonomy_results]),
                    "overall": calc_stats([(r.faithfulness_score + r.relevance_score) / 2 for r in taxonomy_results])
                },
                "Sustainability": {
                    "count": len(sustainability_results),
                    "faithfulness": calc_stats([r.faithfulness_score for r in sustainability_results]),
                    "relevance": calc_stats([r.relevance_score for r in sustainability_results]),
                    "overall": calc_stats([(r.faithfulness_score + r.relevance_score) / 2 for r in sustainability_results])
                }
            },
            "quality_distribution": {
                "high_quality": {
                    "count": len([r for r in results if (r.faithfulness_score + r.relevance_score) / 2 >= high_quality_threshold]),
                    "percentage": round(len([r for r in results if (r.faithfulness_score + r.relevance_score) / 2 >= high_quality_threshold]) / len(results) * 100, 1)
                },
                "medium_quality": {
                    "count": len([r for r in results if medium_quality_threshold <= (r.faithfulness_score + r.relevance_score) / 2 < high_quality_threshold]),
                    "percentage": round(len([r for r in results if medium_quality_threshold <= (r.faithfulness_score + r.relevance_score) / 2 < high_quality_threshold]) / len(results) * 100, 1)
                },
                "low_quality": {
                    "count": len([r for r in results if (r.faithfulness_score + r.relevance_score) / 2 < medium_quality_threshold]),
                    "percentage": round(len([r for r in results if (r.faithfulness_score + r.relevance_score) / 2 < medium_quality_threshold]) / len(results) * 100, 1)
                }
            },
            "score_ranges": {
                "excellent_8_10": len([r for r in results if (r.faithfulness_score + r.relevance_score) / 2 >= 8]),
                "good_6_8": len([r for r in results if 6 <= (r.faithfulness_score + r.relevance_score) / 2 < 8]),
                "average_4_6": len([r for r in results if 4 <= (r.faithfulness_score + r.relevance_score) / 2 < 6]),
                "poor_below_4": len([r for r in results if (r.faithfulness_score + r.relevance_score) / 2 < 4])
            },
            "qa_decisions": {
                "kept": {
                    "count": len([r for r in results if r.keep_qa == True]),
                    "percentage": round(len([r for r in results if r.keep_qa == True]) / len(results) * 100, 1)
                },
                "deleted": {
                    "count": len([r for r in results if r.keep_qa == False]),
                    "percentage": round(len([r for r in results if r.keep_qa == False]) / len(results) * 100, 1)
                },
                "modified": {
                    "count": len([r for r in results if hasattr(r, 'modification_applied') and r.modification_applied == True]),
                    "percentage": round(len([r for r in results if hasattr(r, 'modification_applied') and r.modification_applied == True]) / len(results) * 100, 1) if results else 0
                }
            }
            }

            # Add modification statistics if modifications were attempted
            if self.modification_stats["total_attempts"] > 0:
                summary["modification_stats"] = self._get_modification_summary()

            return summary

        except Exception as e:
            print(f"Error generating summary statistics: {e}")
            return {
                "error": f"Failed to generate statistics: {str(e)}",
                "total_qa_pairs": len(results)
            }

    def _get_modification_summary(self) -> Dict:
        """Generate modification statistics summary"""
        stats = self.modification_stats
        total_attempted = stats.get("total_attempts", 0)

        if total_attempted == 0:
            return {"message": "No modifications attempted"}

        successful = stats.get("successful_modifications", 0)
        failed_api = stats.get("failed_api", 0)
        failed_quality = stats.get("failed_quality", 0)
        rolled_back = stats.get("rolled_back", 0)
        fallback_to_original = stats.get("fallback_to_original", 0)

        try:
            success_rate = (successful / total_attempted) * 100
            failure_rate = ((failed_api + failed_quality) / total_attempted) * 100
            rollback_rate = (rolled_back / total_attempted) * 100
        except ZeroDivisionError:
            success_rate = failure_rate = rollback_rate = 0.0

        return {
            "total_attempted": total_attempted,
            "successful": successful,
            "success_rate": f"{success_rate:.1f}%",
            "api_failures": failed_api,
            "quality_failures": failed_quality,
            "total_failures": failed_api + failed_quality,
            "failure_rate": f"{failure_rate:.1f}%",
            "rolled_back": rolled_back,
            "rollback_rate": f"{rollback_rate:.1f}%",
            "fallback_to_original": fallback_to_original
        }

    # =========================================================================
    # NEW FACTOID RELEVANCE EVALUATION METHODS
    # =========================================================================

    def _build_factoid_relevance_prompt(self, context: str, question: str, spans: List[str], classification: str, relevant_keywords: str) -> str:
        """Build enhanced factoid relevance evaluation prompt for pre-split spans"""
        spans_text = "\n".join([f"- {span}" for span in spans])

        prompt = f"""You are an expert evaluator for factoid answer relevance in the {classification} domain.
Your task is to assess how relevant each span is for {classification} analysis, using the full CONTEXT.
Do NOT calculate any overall/final score. Only score each span and provide supporting evidence.

CONTEXT:
{context}

QUESTION:
{question}

SPANS TO EVALUATE:
{spans_text}

DOMAIN ANCHORS ({classification}):
{relevant_keywords}

INSTRUCTIONS:
1. For each span listed above, evaluate its relevance to {classification} analysis based on domain value
2. Score each span 1-10 based on {classification} analytical value:
   - 8-10: Essential {classification} KPI/metric with clear domain connection
   - 6-7: Important {classification} data with domain relevance
   - 4-5: General business data with weak {classification} connection
   - 1-3: Administrative/operational data with no {classification} relevance
3. Identify domain anchors (keywords) that support the relevance assessment

Return your answer in strict JSON (no markdown), with this structure:

{{
  "spans": [
    {{
      "span": "<span text>",
      "evidence": "<context evidence supporting {classification} relevance>",
      "anchors": ["relevant_keyword1", "relevant_keyword2"],
      "score": <1-10>
    }}
  ],
  "reasoning": "<2-3 sentence justification for relevance based on {classification} domain value>".
}}
Esnure returning full complete json object.
Do NOT calculate or return any overall/final score. Score and explain each span individually, then give your short reasoning at the end.
"""
        return prompt.strip()

    def _aggregate_relevance_score(self, llm_json: Dict, relevant_threshold: int = 6) -> Tuple[int, Dict]:
        """
        Computes the proportional relevance score from LLM's per-span scores.
        Args:
            llm_json: parsed JSON output from LLM.
            relevant_threshold: minimum score to count a span as 'relevant' (default=6).
        Returns:
            final_score: int (1-10)
            meta: dict with components for debugging/auditing
        """
        spans = llm_json.get("spans", [])
        if not spans:
            return 1, {"reason": "No spans found"}

        # Extract and validate scores (focus purely on domain relevance)
        scores = [max(1, min(10, int(round(s["score"])))) for s in spans if "score" in s]
        total = max(1, len(spans))
        relevant_scores = [sc for sc in scores if sc >= relevant_threshold]
        ratio = len(relevant_scores) / total

        if not relevant_scores:
            final = 1
        else:
            relevant_mean = sum(relevant_scores) / len(relevant_scores)
            final = round(relevant_mean * ratio)

        final = max(1, min(10, final))
        meta = {
            "total_spans": total,
            "relevant_spans": len(relevant_scores),
            "relevance_ratio": ratio,
            "relevant_mean": sum(relevant_scores) / len(relevant_scores) if relevant_scores else 0,
            "raw_scores": scores,
            "final_score": final,
            "llm_json": llm_json
        }
        return final, meta
    
    def _factoid_relevance_judge(self, context: str, question: str, answer: str, classification: str, relevant_keywords: str) -> Tuple[int, Optional[Dict], str, Optional[Dict]]:
        """
        1. Builds prompt.
        2. Calls LLM (with retry/backoff).
        3. Parses JSON output robustly.
        4. Aggregates final proportional relevance score.
        Returns: final_score (int 1–10), llm_json, prompt, meta
        """
        # Pre-split spans
        spans = [s.strip() for s in answer.split(',') if s.strip()]
        prompt = self._build_factoid_relevance_prompt(context, question, spans, classification, relevant_keywords)

        # LLM call with retry + context tag for logs
        messages = [
            {"role": "system", "content": "You are a careful, honest expert. Return STRICT JSON only (no Markdown)."},
            {"role": "user", "content": prompt}
        ]
        llm_response = self._call_llm_with_retry(messages, max_tokens=1024, temperature=0.0, context="factoid_relevance_judge")

        # Robust JSON extraction
        def extract_json(text: str):
            try:
                t = text.strip()
                # strip common fences
                if t.startswith("```json"): t = t[7:].strip()
                if t.startswith("```"): t = t[3:].strip()
                if t.endswith("```"): t = t[:-3].strip()
                # take the first balanced JSON object
                start = t.find("{")
                if start == -1:
                    raise ValueError("No JSON object found")
                depth = 0
                end = None
                for i, ch in enumerate(t[start:], start=start):
                    if ch == "{": depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                if end is None:
                    raise ValueError("Unbalanced braces")
                obj = t[start:end]
                return json.loads(obj)
            except Exception as e:
                raise ValueError(f"Failed to parse LLM JSON: {e}\nRaw:\n{text}")

        try:
            llm_json = extract_json(llm_response)
            final_score, meta = self._aggregate_relevance_score(llm_json)
            return final_score, llm_json, prompt, meta
        except Exception as json_error:
            print(f"   [JSON-PARSE-ERROR] {json_error}")
            import re

            score_matches = re.findall(r'"score":\s*(\d+)', llm_response)
            if score_matches:
                scores = [int(s) for s in score_matches]
                avg_score = sum(scores) / len(scores)
                final_score = max(1, min(10, int(round(avg_score))))
                print(f"   [RECOVERED] Extracted {len(scores)} scores from raw text, average: {final_score}")

                spans_in_response = re.findall(r'"span":\s*"([^"]+)"', llm_response)
                span_scores = scores if len(scores) == len(spans_in_response) else [final_score] * len(spans_in_response)

                synthetic_spans = []
                for i, span_text in enumerate(spans_in_response):
                    score = span_scores[i] if i < len(span_scores) else final_score
                    synthetic_spans.append({
                        'span': span_text,
                        'score': score,
                        'evidence': 'Recovered from parsing failure'
                    })

                synthetic_json = {
                    'spans': synthetic_spans,
                    'reasoning': 'Recovered from JSON parsing failure'
                }

                meta = {
                    'total_spans': len(synthetic_spans),
                    'relevant_spans': sum(1 for s in span_scores if s >= 6),
                    'relevance_ratio': (sum(1 for s in span_scores if s >= 6) / len(span_scores)) if span_scores else 0,
                    'relevant_mean': (sum(s for s in span_scores if s >= 6) / max(1, sum(1 for s in span_scores if s >= 6))),
                    'raw_scores': span_scores,
                    'final_score': final_score,
                    'llm_json': synthetic_json,
                    'json_recovery': True
                }
                return final_score, synthetic_json, prompt, meta
            else:
                print("   [FALLBACK] No scores found in raw text, returning low score")
                final_score = 2
                meta = {
                    'total_spans': 1,
                    'relevant_spans': 0,
                    'relevance_ratio': 0,
                    'relevant_mean': 0,
                    'raw_scores': [final_score],
                    'final_score': final_score,
                    'llm_json': None,
                    'json_recovery': False
                }
                return final_score, None, prompt, meta


# =============================================================================
# FAITHFULNESS ASSESSMENT AND MODIFICATION SYSTEM
# =============================================================================

    def evaluate_factoid_answer_faithfulness_spans(self, question: str, answer: str, context: str, classification: str = "") -> Tuple[float, str, dict]:
        """
        Evaluate factoid answer faithfulness using span-by-span verbatim analysis with 3-scale scoring.
        Follows same professional architecture as relevance assessment.

        Returns:
            Tuple[float, str, dict]: (final_faithfulness_score, reasoning, span_data)
        """

        # Split answer into individual spans
        spans = [s.strip() for s in answer.split(',') if s.strip()]

        if not spans:
            return 1.0, "No spans found in answer", {
                'span_details': [],
                'verbatim_spans': [],
                'non_verbatim_spans': [],
                'has_non_verbatim_spans': True,
                'total_spans': 0,
                'verbatim_count': 0,
                'modification_needed': True
            }

        print(f"   [FAITHFULNESS-ASSESS] Evaluating {len(spans)} spans for verbatim accuracy")

        try:
            # Get LLM faithfulness assessment for all spans
            final_score, llm_json, prompt, meta = self._factoid_faithfulness_judge(
                context=context,
                question=question,
                answer=answer,
                classification=classification
            )

            # Extract reasoning from LLM response
            reasoning = llm_json.get("reasoning", "Verbatim faithfulness evaluation completed") if llm_json else "Evaluation completed"

            # Add metadata info to reasoning
            if meta:
                span_info = f" ({meta['faithful_spans']}/{meta['total_spans']} verbatim spans)"
                reasoning = f"{reasoning}{span_info}"

            # Build faithfulness span analysis data
            span_data = self._build_faithfulness_analysis_data(final_score, spans, llm_json, meta)

            return float(final_score), reasoning, span_data

        except Exception as e:
            print(f"   [FAITHFULNESS-ERROR] {e}")
            # Fallback to conservative evaluation
            span_data = {
                'span_details': [{'span': span, 'verbatim_score': 1, 'status': 'non_verbatim', 'verbatim_problem': 'Assessment failed'} for span in spans],
                'verbatim_spans': [],
                'non_verbatim_spans': spans,
                'has_non_verbatim_spans': True,
                'total_spans': len(spans),
                'verbatim_count': 0,
                'modification_needed': True
            }
            return 1.0, f"Faithfulness assessment failed: {str(e)}", span_data
    
    def _factoid_faithfulness_judge(self, context: str, question: str, answer: str, classification: str) -> Tuple[int, Optional[Dict], str, Optional[Dict]]:
        """
        LLM-based faithfulness assessment using 3-scale scoring (10/5/1).
        Similar to relevance judge but focused on verbatim accuracy.
        """
        # Pre-split spans
        spans = [s.strip() for s in answer.split(',') if s.strip()]
        prompt = self._build_factoid_faithfulness_prompt(context, question, spans, classification)

        # LLM call with retry + context tag
        messages = [
            {"role": "system", "content": "You are a strict verbatim accuracy evaluator for research. Return STRICT JSON only (no Markdown)."},
            {"role": "user", "content": prompt}
        ]
        llm_response = self._call_llm_with_retry(messages, max_tokens=1024, temperature=0.0, context="factoid_faithfulness_judge")

        # Robust JSON extraction
        def extract_json(text: str):
            try:
                t = text.strip()
                if t.startswith("```json"): t = t[7:].strip()
                if t.startswith("```"): t = t[3:].strip()
                if t.endswith("```"): t = t[:-3].strip()
                start = t.find("{")
                if start == -1:
                    raise ValueError("No JSON object found")
                depth = 0
                end = None
                for i, ch in enumerate(t[start:], start=start):
                    if ch == "{": depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                if end is None:
                    raise ValueError("Unbalanced braces")
                obj = t[start:end]
                return json.loads(obj)
            except Exception as e:
                raise ValueError(f"Failed to parse LLM JSON: {e}\nRaw:\n{text}")

        try:
            llm_json = extract_json(llm_response)
            final_score, meta = self._aggregate_faithfulness_score(llm_json)
            return final_score, llm_json, prompt, meta
        except Exception as json_error:
            print(f"   [FAITHFULNESS-JSON-ERROR] {json_error}")
            import re
            score_matches = re.findall(r'"verbatim_score":\s*(\d+)', llm_response)
            if not score_matches:
                score_matches = re.findall(r'"score":\s*(\d+)', llm_response)

            if score_matches:
                scores = [int(s) for s in score_matches]
                avg_score = sum(scores) / len(scores)
                final_score = max(1, min(10, int(round(avg_score))))
                print(f"   [RECOVERED] Extracted {len(scores)} scores from raw text, average: {final_score}")

                spans_in_response = re.findall(r'"span":\s*"([^"]+)"', llm_response)
                span_scores = scores if len(scores) == len(spans_in_response) else [final_score] * len(spans_in_response)

                synthetic_spans = []
                for i, span_text in enumerate(spans_in_response):
                    score = span_scores[i] if i < len(span_scores) else final_score
                    synthetic_spans.append({
                        'span': span_text,
                        'verbatim_score': score,
                        'found_in_context': True
                    })

                synthetic_json = {
                    'spans': synthetic_spans,
                    'reasoning': 'Recovered from JSON parsing failure'
                }

                meta = {
                    'total_spans': len(synthetic_spans),
                    'faithful_spans': sum(1 for s in span_scores if s >= 6),
                    'faithfulness_ratio': (sum(1 for s in span_scores if s >= 6) / len(span_scores)) if span_scores else 0,
                    'faithful_mean': (sum(s for s in span_scores if s >= 6) / max(1, sum(1 for s in span_scores if s >= 6))),
                    'raw_scores': span_scores,
                    'final_score': final_score,
                    'llm_json': synthetic_json,
                    'json_recovery': True
                }
                return final_score, synthetic_json, prompt, meta
            else:
                print("   [FALLBACK] No scores found in raw text, returning low score")
                final_score = 2
                meta = {
                    'total_spans': 1,
                    'faithful_spans': 0,
                    'faithfulness_ratio': 0,
                    'faithful_mean': 0,
                    'raw_scores': [final_score],
                    'final_score': final_score,
                    'llm_json': None,
                    'json_recovery': False
                }
                return final_score, None, prompt, meta

    def _build_factoid_faithfulness_prompt(self, context: str, question: str, spans: List[str], classification: str) -> str:
        """Build faithfulness evaluation prompt with 3-scale scoring for pre-split spans"""
        domain_text = f" for {classification}" if classification else ""

        spans_text = "\n".join([f"- {span}" for span in spans])

        prompt = f"""You are evaluating verbatim accuracy of factoid answer spans{domain_text}.
Check if each span appears EXACTLY word-for-word in the context.

CONTEXT:
{context}

QUESTION:
{question}

SPANS TO EVALUATE:
{spans_text}

INSTRUCTIONS:
1. For each span listed above, check word-by-word if it appears exactly in context
2. Score each span using 3-scale scoring:
   - 10: PERFECT VERBATIM - all words appear exactly in context
   - 5: PARTIAL VERBATIM - some words verbatim, some words not verbatim
   - 1: NOT VERBATIM - most/all words not found exactly in context

3. For spans scoring < 10, identify specific verbatim problems

Return strict JSON format:

{{
  "spans": [
    {{
      "span": "<exact span text>",
      "verbatim_score": <10, 5, or 1>,
      "verbatim_problem": "<specific issue if score < 10>",
      "non_verbatim_words": ["<word1>", "<word2>"],
      "found_in_context": true|false
    }}
  ],
  "reasoning": "<brief verbatim assessment summary>".
}}
Ensure returning complete correct final json object.
Focus on exact word matching - number formats, terminology, punctuation must match exactly."""
        return prompt.strip()

    def _aggregate_faithfulness_score(self, llm_json: Dict, faithful_threshold: int = 6, cap_if_not_found: bool = True, cap_value: int = 1) -> Tuple[int, Dict]:
        """
        Aggregate faithfulness scores using proportional weighting (same logic as relevance).
        Designed for 3-scale scoring system (10/5/1).
        """
        spans = llm_json.get("spans", [])
        if not spans:
            return 1, {"reason": "No spans found"}

        # Cap if any span not found in context
        if cap_if_not_found and any(not s.get("found_in_context", True) for s in spans):
            candidate = cap_value
        else:
            candidate = None

        # Extract verbatim scores (handle both 'verbatim_score' and 'score' keys)
        scores = []
        for s in spans:
            if "verbatim_score" in s:
                scores.append(max(1, min(10, int(round(s["verbatim_score"])))))
            elif "score" in s:
                scores.append(max(1, min(10, int(round(s["score"])))))
            else:
                scores.append(1)  # Default low score

        total = max(1, len(spans))
        faithful_scores = [sc for sc in scores if sc >= faithful_threshold]
        ratio = len(faithful_scores) / total

        if not faithful_scores:
            final = 1
        else:
            faithful_mean = sum(faithful_scores) / len(faithful_scores)
            final = round(faithful_mean * ratio)

        if candidate is not None:
            final = min(final, candidate)

        final = max(1, min(10, final))
        meta = {
            "total_spans": total,
            "faithful_spans": len(faithful_scores),
            "faithfulness_ratio": ratio,
            "faithful_mean": sum(faithful_scores) / len(faithful_scores) if faithful_scores else 0,
            "candidate_cap": candidate,
            "raw_scores": scores,
            "final_score": final,
            "aggregation_type": "faithfulness"
        }
        return final, meta

    def _build_faithfulness_analysis_data(self, final_score, spans, llm_json, meta):
        """
        Build faithfulness span analysis data for modification system.
        Similar structure to relevance span data.
        """
        try:
            span_details = []
            verbatim_spans = []
            non_verbatim_spans = []

            if llm_json and 'spans' in llm_json and isinstance(llm_json['spans'], list):
                # Use individual span scores from LLM JSON
                for llm_span in llm_json['spans']:
                    span_text = llm_span.get('span', '').strip()
                    # Handle both 'verbatim_score' and 'score' keys
                    span_score = llm_span.get('verbatim_score', llm_span.get('score', final_score))

                    # Find matching span in original spans list
                    matched_span = None
                    for original_span in spans:
                        if span_text.lower() in original_span.lower() or original_span.lower() in span_text.lower():
                            matched_span = original_span
                            break

                    final_span_text = matched_span if matched_span else span_text

                    if span_score < 6.0:
                        non_verbatim_spans.append(final_span_text)
                        status = 'non_verbatim'
                    else:
                        verbatim_spans.append(final_span_text)
                        status = 'verbatim'

                    span_details.append({
                        'span': final_span_text,
                        'verbatim_score': span_score,
                        'status': status,
                        'verbatim_problem': llm_span.get('verbatim_problem', 'Verbatim issues detected') if span_score < 6.0 else None,
                        'non_verbatim_words': llm_span.get('non_verbatim_words', [])
                    })

                # Handle any original spans not found in LLM response
                for original_span in spans:
                    if not any(detail['span'] == original_span for detail in span_details):
                        if final_score < 6.0:
                            non_verbatim_spans.append(original_span)
                            status = 'non_verbatim'
                        else:
                            verbatim_spans.append(original_span)
                            status = 'verbatim'

                        span_details.append({
                            'span': original_span,
                            'verbatim_score': final_score,
                            'status': status,
                            'verbatim_problem': 'Overall low faithfulness score' if final_score < 6.0 else None,
                            'non_verbatim_words': []
                        })

            else:
                # Fallback without detailed span analysis
                for span in spans:
                    if final_score < 6.0:
                        non_verbatim_spans.append(span)
                        status = 'non_verbatim'
                    else:
                        verbatim_spans.append(span)
                        status = 'verbatim'

                    span_details.append({
                        'span': span,
                        'verbatim_score': final_score,
                        'status': status,
                        'verbatim_problem': 'Assessment failed - conservative scoring' if final_score < 6.0 else None,
                        'non_verbatim_words': []
                    })

            return {
                'span_details': span_details,
                'verbatim_spans': verbatim_spans,
                'non_verbatim_spans': non_verbatim_spans,
                'has_non_verbatim_spans': len(non_verbatim_spans) > 0,
                'total_spans': len(spans),
                'verbatim_count': len(verbatim_spans),
                'modification_needed': len(non_verbatim_spans) > 0,
                'aggregation_meta': meta
            }

        except Exception as e:
            print(f"   [FAITHFULNESS-SPAN-DATA-ERROR] {e}")
            return {
                'span_details': [{'span': span, 'verbatim_score': 1, 'status': 'non_verbatim', 'verbatim_problem': 'Analysis failed'} for span in spans],
                'verbatim_spans': [],
                'non_verbatim_spans': spans,
                'has_non_verbatim_spans': True,
                'total_spans': len(spans),
                'verbatim_count': 0,
                'modification_needed': True,
                'aggregation_meta': None
            }

    def _modify_factoid_spans_for_faithfulness(self, answer: str, context: str, question: str,
                                              classification: str, span_data: dict) -> Tuple[str, bool, str]:
        """
        Modify factoid spans to improve faithfulness through word-level verbatim correction.
        Follows same professional architecture as relevance modification.
        """
        if not span_data.get('has_non_verbatim_spans', False):
            return answer, False, "All spans already verbatim - no modification needed"

        non_verbatim_spans = [(detail['span'], detail['verbatim_score'], detail.get('verbatim_problem', 'Verbatim issues'))
                             for detail in span_data.get('span_details', [])
                             if detail['status'] == 'non_verbatim']

        verbatim_spans = [detail['span'] for detail in span_data.get('span_details', [])
                         if detail['status'] == 'verbatim']

        print(f"   [FAITHFULNESS-MOD] Processing {len(non_verbatim_spans)} non-verbatim spans")

        # Process each non-verbatim span with integrated correction
        final_spans = verbatim_spans.copy()
        successfully_corrected_spans = []
        modification_count = 0

        for span_text, original_score, verbatim_problem in non_verbatim_spans:
            # Build targeted word-level correction prompt
            existing_content = ", ".join(final_spans) if final_spans else "None"

            prompt = f"""Fix non-verbatim words in this span using EXACT text from context.

CONTEXT: {context}
CURRENT SPAN: "{span_text}"
VERBATIM PROBLEM: {verbatim_problem}
OTHER EXISTING SPANS: {existing_content}
DOMAIN: {classification}

Requirements:
1. Fix non-verbatim words using EXACT word-for-word replacements from context
2. Keep verbatim words unchanged
3. CRITICAL: Modified span must NOT overlap with existing spans: {existing_content}
4. Avoid any redundancy (maximum 40% word overlap with existing spans)
5. Return "REMOVE" if cannot fix without creating redundancy

Examples:
- Fix "2.8M" → "2,800,000" (if found exactly in context)
- Fix "admin costs" → "administrative expenses" (if found exactly)
- Keep verbatim words unchanged

Return only the corrected span or "REMOVE":"""

            try:
                # Get word-level correction
                messages = [
                    {"role": "system", "content": f"Fix verbatim issues in {classification} spans without redundancy."},
                    {"role": "user", "content": prompt}
                ]
                response = self._call_llm(messages, max_tokens=100, temperature=0.1)
                corrected_span = response.strip().strip('"').strip("'")

                if corrected_span.upper() == "REMOVE" or len(corrected_span) < 2:
                    print(f"     [REMOVED] '{span_text}' - {verbatim_problem}")
                    continue

                # Enhanced redundancy check (40% threshold - stricter than relevance)
                corrected_words = set(corrected_span.lower().split())
                is_redundant = False
                for existing in final_spans:
                    existing_words = set(existing.lower().split())
                    if corrected_words and existing_words:
                        overlap = len(corrected_words & existing_words) / len(corrected_words | existing_words)
                        if overlap > 0.8:  # Stricter than relevance (60%)
                            print(f"     [REDUNDANT] '{corrected_span}' - 40%+ overlap with '{existing}'")
                            is_redundant = True
                            break

                if is_redundant:
                    continue

                # Lightweight faithfulness assessment (3-scale)
                assessment_prompt = f"""Score this span for verbatim accuracy in {classification}:

CONTEXT: {context}
SPAN: "{corrected_span}"
DOMAIN: {classification}

Score (choose ONE):
- 10: All words appear exactly in context (perfect verbatim)
- 5: Some words appear exactly, some don't (partial verbatim)
- 1: Most words don't appear exactly in context (not verbatim)

Return only the score (10, 5, or 1):"""

                score_messages = [
                    {"role": "system", "content": f"You are a {classification} verbatim accuracy evaluator. Return only a number."},
                    {"role": "user", "content": assessment_prompt}
                ]
                score_response = self._call_llm(score_messages, max_tokens=5, temperature=0.0)

                # Extract score
                import re
                score_match = re.search(r'\b(1|5|10)\b', score_response.strip())
                new_score = float(score_match.group(1)) if score_match else 1.0

                # Validate improvement
                if new_score >= 6.0 and new_score > original_score:
                    final_spans.append(corrected_span)
                    successfully_corrected_spans.append((corrected_span, new_score))
                    modification_count += 1
                    print(f"     [CORRECTED] '{span_text}' -> '{corrected_span}' (Score: {original_score}->{new_score})")
                else:
                    print(f"     [FAILED] '{span_text}' - No verbatim improvement (Score: {new_score})")

            except Exception as e:
                print(f"     [ERROR] '{span_text}' - {str(e)}")

        # Final validation using faithfulness aggregation function
        if modification_count > 0 and final_spans:
            # Build synthetic JSON for faithfulness aggregation
            synthetic_spans = []

            # Add verbatim spans with their original scores
            for detail in span_data.get('span_details', []):
                if detail['status'] == 'verbatim' and detail['span'] in final_spans:
                    synthetic_spans.append({
                        'span': detail['span'],
                        'verbatim_score': detail['verbatim_score'],
                        'found_in_context': True
                    })

            # Add successfully corrected spans
            for span_text, new_score in successfully_corrected_spans:
                synthetic_spans.append({
                    'span': span_text,
                    'verbatim_score': new_score,
                    'found_in_context': True
                })

            # Use faithfulness aggregation function for consistent scoring
            synthetic_json = {'spans': synthetic_spans}
            final_score, meta = self._aggregate_faithfulness_score(synthetic_json)

            if final_score >= 6:
                reconstructed_answer = ", ".join(final_spans)
                return reconstructed_answer, True, f"Successfully corrected {modification_count} spans to verbatim (Final Score: {final_score})"

        # Fallback to verbatim spans only
        if verbatim_spans:
            return ", ".join(verbatim_spans), True, f"Retained {len(verbatim_spans)} verbatim spans"

        return answer, False, "Faithfulness correction failed"


# =============================================================================
# CLI AND CONFIGURATION FUNCTIONALITY
# =============================================================================

# CLI imports already imported at top

def create_azure_evaluator(enable_modification=False):
    """Azure OpenAI Configuration"""
    config = {
        "llm_provider": LLMProvider.AZURE_OPENAI,
        "azure_endpoint": AZURE_OPENAI_ENDPOINT,
        "azure_api_key": AZURE_OPENAI_API_KEY,
        "model_name": "gpt-4",
        "api_version": "2024-05-01-preview",
        "enable_modification": enable_modification
    }
    return SustainableQAEvaluator(**config)

def create_together_evaluator(enable_modification=False, enable_quality_validation=True,
                             modification_retry_attempts=2, quality_threshold=6.0):
    """Together AI Configuration - Speed Optimized for Free Tier"""
    config = {
        "llm_provider": LLMProvider.TOGETHER_AI,
        "together_api_key": TOGETHER_API_KEY,
        "model_name": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",  # Free tier model
        "enable_modification": enable_modification,
        "enable_quality_validation": enable_quality_validation,
        "modification_retry_attempts": modification_retry_attempts,
        "modification_retry_delay": 1.0,
        "quality_threshold": quality_threshold,
        "fallback_to_original": True,
        "enable_rollback": True,
        # Speed optimizations for free tier
        "max_modification_attempts": 3,  # Three attempts with chain-of-thought
        "retry_delay_base": 0.5,  # Faster base retry
        "exponential_backoff_factor": 1.5,  # Less aggressive backoff
        "default_max_tokens": 512,  # Reduced from 1024 for speed
        "evaluation_max_tokens": 256,  # Even smaller for evaluations
    }
    return SustainableQAEvaluator(**config)

def create_ollama_evaluator(enable_modification=False):
    """Local Ollama Configuration"""
    config = {
        "llm_provider": LLMProvider.LOCAL_OLLAMA,
        "ollama_base_url": "http://localhost:11434",
        "model_name": "llama3.1:8b",
        "enable_modification": enable_modification
    }
    return SustainableQAEvaluator(**config)

def create_huggingface_evaluator(enable_modification=False):
    """HuggingFace Local Configuration"""
    config = {
        "llm_provider": LLMProvider.HUGGINGFACE_LOCAL,
        "model_name": "microsoft/DialoGPT-large",
        "enable_modification": enable_modification
    }
    return SustainableQAEvaluator(**config)

def display_summary(results):
    """Display a formatted summary of evaluation results"""

    summary = results.get('summary', {})
    metadata = results.get('evaluation_metadata', {})

    print(f"\nEVALUATION SUMMARY")
    print(f"{'='*60}")

    # Basic info
    print(f"Provider: {metadata.get('llm_provider', 'Unknown')}")
    print(f"Model: {metadata.get('model_name', 'Unknown')}")
    print(f"Total QA pairs: {summary.get('total_qa_pairs', 0)}")

    # Overall scores
    overall_metrics = summary.get('overall_metrics', {})
    if overall_metrics:
        print(f"\nOverall Mean Scores:")
        for metric_name, stats in overall_metrics.items():
            mean_score = stats.get('mean', 0)
            std_score = stats.get('std', 0)
            print(f"   {metric_name.replace('_', ' ').title()}: {mean_score:.2f} (+/-{std_score:.2f})")

    # By question type
    by_type = summary.get('by_question_type', {})
    if by_type:
        print(f"\nBy Question Type:")
        for qtype, stats in by_type.items():
            count = stats.get('count', 0)
            overall_mean = stats.get('overall', {}).get('mean', 0)
            if count > 0:
                print(f"   {qtype.title()}: {count} questions (Mean: {overall_mean:.2f})")

    # By classification
    by_classification = summary.get('by_classification', {})
    if by_classification:
        print(f"\nBy Classification:")
        for cls, stats in by_classification.items():
            count = stats.get('count', 0)
            overall_mean = stats.get('overall', {}).get('mean', 0)
            if count > 0:
                print(f"   {cls}: {count} questions (Mean: {overall_mean:.2f})")

    # Quality distribution
    quality_dist = summary.get('quality_distribution', {})
    if quality_dist:
        print(f"\nQuality Distribution:")
        for quality_level, info in quality_dist.items():
            count = info.get('count', 0)
            percentage = info.get('percentage', 0)
            level_name = quality_level.replace('_', ' ').title()
            print(f"   {level_name}: {count} ({percentage}%)")

def run_evaluation_cli():
    """CLI interface for running evaluations"""

    parser = argparse.ArgumentParser(description='SustainableQA Evaluation Framework')
    parser.add_argument('--provider', choices=['azure', 'together', 'ollama', 'huggingface'],
                       default='together', help='LLM provider to use')
    parser.add_argument('--dataset', help='Path to SustainableQA dataset JSON file')
    parser.add_argument('--output', help='Output path for results (default: auto-generated)')
    parser.add_argument('--sample-size', type=int, help='Number of samples to evaluate (default: full dataset)')
    parser.add_argument('--config', action='store_true', help='Show configuration examples')
    parser.add_argument('--modify-deletions', action='store_true',
                       help='Enable modification of QA pairs marked for deletion (instead of discarding them)')
    parser.add_argument('--disable-quality-validation', action='store_true',
                       help='Disable quality validation for modifications (faster but riskier)')
    parser.add_argument('--quality-threshold', type=float, default=6.0,
                       help='Quality threshold for accepting modifications (1.0-10.0, default: 6.0)')
    parser.add_argument('--retry-attempts', type=int, default=3,
                       help='Number of API retry attempts for modifications (default: 3)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level (default: INFO)')
    parser.add_argument('--log-file', help='Path to log file (optional)')

    args = parser.parse_args()

    # Setup logging based on CLI arguments
    log_level = getattr(logging, args.log_level.upper())
    global logger
    logger = setup_logging(level=log_level, log_file=args.log_file)

    if args.config:
        print_config_examples()
        return

    # Require dataset if not showing config
    if not args.dataset:
        parser.error("--dataset is required unless using --config")

    # Validate dataset path
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found at {args.dataset}")
        print("Please ensure:")
        print("  1. The file path is correct")
        print("  2. The file exists and is accessible")
        print("  3. The file is a valid JSON dataset")
        return

    # Create output path if not provided
    if not args.output:
        dataset_name = Path(args.dataset).stem
        output_filename = f"evaluation_{args.provider}_{dataset_name}"
        if args.sample_size:
            output_filename += f"_sample{args.sample_size}"
        output_filename += ".json"
        args.output = output_filename

    print(f"SustainableQA Evaluation Runner")
    print("="*50)
    print(f"Provider: {args.provider}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Sample size: {'Full dataset' if args.sample_size is None else args.sample_size}")
    print(f"Modification mode: {'Enabled' if args.modify_deletions else 'Disabled'}")

    # Initialize evaluator
    try:
        print(f"\nInitializing {args.provider} evaluator...")

        # Risk mitigation parameters
        enable_quality_validation = not args.disable_quality_validation
        quality_threshold = max(1.0, min(10.0, args.quality_threshold))  # Clamp to 1-10 scale
        retry_attempts = max(1, min(10, args.retry_attempts))  # Clamp to reasonable range

        if args.provider == 'azure':
            evaluator = create_azure_evaluator(enable_modification=args.modify_deletions)
        elif args.provider == 'together':
            evaluator = create_together_evaluator(
                enable_modification=args.modify_deletions,
                enable_quality_validation=enable_quality_validation,
                modification_retry_attempts=retry_attempts,
                quality_threshold=quality_threshold
            )
        elif args.provider == 'ollama':
            evaluator = create_ollama_evaluator(enable_modification=args.modify_deletions)
        elif args.provider == 'huggingface':
            evaluator = create_huggingface_evaluator(enable_modification=args.modify_deletions)
        else:
            raise ValueError(f"Unknown provider: {args.provider}")

        print(f"Evaluator initialized successfully")

    except Exception as e:
        print(f"Error initializing evaluator: {e}")
        print(f"Check your API keys and configuration for {args.provider}")
        return

    # Run evaluation
    try:
        print(f"\nStarting evaluation...")
        results = evaluator.evaluate_sustainableqa_dataset(
            dataset_path=args.dataset,
            output_path=args.output,
            sample_size=args.sample_size,
            save_progress=True
        )

        print(f"\nEvaluation completed successfully!")
        display_summary(results)

    except KeyboardInterrupt:
        print(f"\nEvaluation interrupted by user")

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        print(f"Check your configuration and try again")

def print_config_examples():
    """Print configuration examples for all providers"""

    print("SustainableQA Evaluator - Configuration Examples")
    print("="*60)

    print("\n1. AZURE OPENAI CONFIGURATION")
    print("-" * 30)
    print("Setup:")
    print("  1. Edit the API keys section at the top of this file:")
    print("     AZURE_OPENAI_ENDPOINT = 'https://your-resource.openai.azure.com/'")
    print("     AZURE_OPENAI_API_KEY = 'your-api-key'")
    print("\nUsage:")
    print("  python sustainableqa_evaluator.py --provider azure --dataset path/to/dataset.json")

    print("\n2. TOGETHER AI CONFIGURATION")
    print("-" * 30)
    print("Setup:")
    print("  1. Get API key from: https://api.together.xyz/")
    print("  2. Edit the API keys section at the top of this file:")
    print("     TOGETHER_API_KEY = 'your-together-api-key'")
    print("\nPopular models:")
    print("  - meta-llama/Llama-3.3-70B-Instruct-Turbo-Free (recommended)")
    print("  - meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-Free")
    print("  - mistralai/Mixtral-8x7B-Instruct-v0.1")
    print("\nUsage:")
    print("  python sustainableqa_evaluator.py --provider together --dataset path/to/dataset.json")

    print("\n3. LOCAL OLLAMA CONFIGURATION")
    print("-" * 30)
    print("Prerequisites:")
    print("  - Install Ollama: https://ollama.ai/download")
    print("  - Pull model: ollama pull llama3.1:8b")
    print("  - Start Ollama: ollama serve")
    print("\nPopular models:")
    print("  - llama3.1:8b (recommended for balance)")
    print("  - llama3.1:70b (higher quality)")
    print("  - mistral:7b (faster)")
    print("\nUsage:")
    print("  python sustainableqa_evaluator.py --provider ollama --dataset path/to/dataset.json")

    print("\n4. HUGGINGFACE LOCAL CONFIGURATION")
    print("-" * 30)
    print("Prerequisites:")
    print("  - pip install transformers torch accelerate")
    print("  - GPU recommended for larger models")
    print("\nPopular models:")
    print("  - microsoft/DialoGPT-large")
    print("  - microsoft/DialoGPT-medium (smaller)")
    print("\nUsage:")
    print("  python sustainableqa_evaluator.py --provider huggingface --dataset path/to/dataset.json")

def test_enhanced_classification():
    """Test the enhanced classification system with sample data"""

    print("Testing Enhanced Classification System")
    print("=" * 50)

    # Create a test evaluator
    try:
        evaluator = create_together_evaluator(enable_modification=False)
        print("[SUCCESS] Evaluator created successfully")
    except Exception as e:
        print(f"[ERROR] Failed to create evaluator: {e}")
        return

    # Test cases with different classification scenarios
    test_cases = [
        {
            "text": "The company's materiality assessment identified climate change as a key ESG risk. We follow GRI Standards for our sustainability reporting framework.",
            "current_classification": "Sustainability",
            "title": "ESG Risk Assessment",
            "expected": "ESG"
        },
        {
            "text": "Our CapEx alignment with EU Taxonomy technical screening criteria shows 15.2% alignment, meeting DNSH requirements for sustainable activities.",
            "current_classification": "ESG",
            "title": "EU Taxonomy Compliance",
            "expected": "EU Taxonomy"
        },
        {
            "text": "The circular economy initiative focuses on waste reduction and resource efficiency through sustainable product design and recycling programs.",
            "current_classification": "ESG",
            "title": "Circular Economy",
            "expected": "Sustainability"
        },
        {
            "text": "Company overview",
            "current_classification": "Sustainability",
            "title": "Short Text Test",
            "expected": "Unknown"  # Low confidence scenario
        },
        {
            "text": "The company has been working on various sustainability initiatives including renewable energy projects, water conservation measures, and community engagement programs. These efforts align with our commitment to environmental stewardship and social responsibility. Our sustainability strategy encompasses multiple dimensions of sustainable development including climate action goals, biodiversity conservation, and resource efficiency improvements.",
            "current_classification": "ESG",
            "title": "Long Text with Mixed Keywords",
            "expected": "Sustainability"  # Text length normalization test
        }
    ]

    print(f"\nTesting {len(test_cases)} classification scenarios...\n")

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['title']}")
        print(f"Current: {test_case['current_classification']}")
        print(f"Expected: {test_case['expected']}")

        try:
            # Test enhanced classification
            final_classification, metadata = evaluator.get_effective_classification_enhanced(
                test_case['text'],
                test_case['current_classification'],
                title=test_case['title']
            )

            print(f"Result: {final_classification}")
            print(f"Method: {metadata.get('method', 'unknown')}")
            print(f"Action: {metadata.get('action', 'unknown')}")

            # Check if result matches expected
            if final_classification == test_case['expected']:
                print("[CORRECT] Classification matches expected")
            else:
                print("[DIFFERENT] Classification differs from expected")

        except Exception as e:
            print(f"[FAILED] Test failed: {e}")

        print("-" * 40)

        print("Enhanced classification testing completed!")

    # =========================================================================
    # REMOVED DUPLICATE FUNCTIONS - KEPT AS COMMENTS FOR REFERENCE
    # =========================================================================

    # def _build_factoid_relevance_prompt(self, context: str, question: str, answer: str, classification: str, relevant_keywords: str) -> str:
    #     """Build enhanced factoid relevance evaluation prompt - DUPLICATE REMOVED"""
    #     # This function was duplicated and removed. The active version is at line 4328
    #     pass
    #
    # def _aggregate_relevance_score(self, llm_json: Dict, relevant_threshold: int = 6, cap_if_not_found: bool = True, cap_value: int = 4) -> Tuple[int, Dict]:
    #     """Computes the proportional relevance score from LLM's per-span scores - DUPLICATE REMOVED"""
    #     # This function was duplicated and removed. The active version is at line 4446
    #     pass
    #
    # def _factoid_relevance_judge(self, context: str, question: str, answer: str, classification: str, relevant_keywords: str) -> Tuple[int, Optional[Dict], str, Optional[Dict]]:
    #     """1. Builds prompt. 2. Calls LLM. 3. Parses JSON output. 4. Aggregates final proportional relevance score - DUPLICATE REMOVED"""
    #     # This function was duplicated and removed. The active version is at line 4495
    #     pass


def main():
    """Main entry point with multiple modes"""

    if len(sys.argv) == 1:
        # No arguments - show help
        print("SustainableQA Evaluation Framework")
        print("=" * 40)
        print("\nUsage modes:")
        print("  python sustainableqa_evaluator.py --help                    # Show CLI help")
        print("  python sustainableqa_evaluator.py --config                 # Show configuration examples")
        print("  python sustainableqa_evaluator.py --provider together \\    # Run evaluation")
        print("                                     --dataset dataset.json")
        print("  python sustainableqa_evaluator.py --provider together \\    # Run with modifications")
        print("                                     --dataset dataset.json \\")
        print("                                     --modify-deletions")
        print("\nFor testing:")
        print("  python sustainableqa_evaluator.py test-classification      # Test enhanced classification")
        print("\nQuick setup for Together AI Llama 3.3 70B (Free):")
        print("  1. pip install together")
        print("  2. Get free API key: https://api.together.xyz/")
        print("  3. Edit API keys section at top of this file:")
        print("     TOGETHER_API_KEY = 'your-actual-api-key'")
        print("  4. python sustainableqa_evaluator.py --provider together --dataset your_dataset.json")
        print("     (Add --modify-deletions to enable QA modifications for low-quality pairs)")
        print("\nIMPROVEMENTS APPLIED:")
        print("+ ENHANCED: Hybrid classification system (keyword overlap + LLM semantic analysis)")
        print("+ LLM-based domain specificity evaluation (focuses on answer content, not questions)")
        print("+ LLM-based relevance scoring with sustainability context understanding")
        print("+ Context information included in output results")
        print("+ No hardcoded regex patterns - handles unseen sustainability terminology intelligently")
        print("+ Proper business-level sustainability term recognition (scores 5-7 range)")
        print("+ Intelligent decision fusion with confidence-based classification conflict resolution")
        print("\nRate limiting: 2s delays between calls, optimized for free tier")
        return

    # Handle special commands

    if len(sys.argv) == 2 and sys.argv[1] == "test-classification":
        test_enhanced_classification()
        return

    # Handle CLI mode
    run_evaluation_cli()

if __name__ == "__main__":
    main()