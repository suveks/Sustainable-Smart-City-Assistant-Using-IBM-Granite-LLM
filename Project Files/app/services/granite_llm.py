"""
granite_llm.py
==========================
Local-only replacement for the former Watsonx gateway.
Uses open-source Hugging Face models so the project works
without any external API keys or internet access once models
are cached locally.

Required pip packages (add to requirements.txt):
    transformers>=4.40.0
    torch>=2.0.0           # or the CPU-only torch wheel
    sentencepiece          # needed by many HF models

Default models (change via env-vars if you wish):
    • TEXT_GEN_MODEL   = google/flan-t5-base   (general Q&A / tips / reports)
    • SUMMARY_MODEL    = facebook/bart-large-cnn (policy summarisation)
Both run happily on CPU; first run will download weights (~250 MB + 400 MB).
"""

from __future__ import annotations

import os
import random
from functools import lru_cache
from typing import Dict, Any

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
    Pipeline,
)


# ---------------------------------------------------------------------#
# 1. Model / pipeline helpers
# ---------------------------------------------------------------------#
TEXT_GEN_MODEL = os.getenv("TEXT_GEN_MODEL", "google/flan-t5-base")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "facebook/bart-large-cnn")


@lru_cache(maxsize=1)
def _get_text_pipeline() -> Pipeline:
    """
    Lazy-load the text-generation / instruction-following model.
    By default we use FLAN-T5 (seq2seq) which works for Q&A, tips, reports.
    """
    tokenizer = AutoTokenizer.from_pretrained(TEXT_GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(TEXT_GEN_MODEL)
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1,
    )


@lru_cache(maxsize=1)
def _get_summary_pipeline() -> Pipeline:
    """Load separate summarisation-optimised model."""
    tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARY_MODEL)
    return pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        max_length=120,
        min_length=30,
        do_sample=False,
    )


def _generate(text_prompt: str) -> str:
    """Utility – run text prompt through the generation pipeline."""
    generator = _get_text_pipeline()
    output = generator(text_prompt, num_return_sequences=1)[0]["generated_text"]
    # FLAN tends to echo prompt – strip if duplicated
    return output.replace(text_prompt, "").strip()


# ---------------------------------------------------------------------#
# 2. Public functions used across the project
# ---------------------------------------------------------------------#
_GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
_CITY_KEYWORDS = {
    "city",
    "urban",
    "municipal",
    "transport",
    "infrastructure",
    "energy",
    "waste",
    "water",
    "air quality",
    "public transport",
    "policy",
    "sustainability",
    "eco",
    "green",
    "renewable",
    "smart city",
    "environment",
    "climate",
    "building",
    "planning",
    "development",
    "governance",
}


def ask_city_question(prompt: str) -> str:
    """
    Handles chat requests focused on smart-city / sustainability topics.
    If prompt is greeting → returns greeting.
    If prompt unrelated to city topics → politely declines.
    Otherwise → passes to local LLM.
    """
    prompt_lc = prompt.strip().lower()

    if any(greet in prompt_lc for greet in _GREETINGS) and len(prompt_lc.split()) <= 3:
        return random.choice(
            [
                "Hello! How can I assist you with smart-city or sustainability topics today?",
                "Hi there! Ask me anything about sustainable cities or urban living.",
                "Hey! Ready to help with your city or sustainability questions.",
            ]
        )

    if not any(word in prompt_lc for word in _CITY_KEYWORDS):
        return (
            "I'm specialised in sustainable cities, smart-city governance, and urban living. "
            "Please ask something related to these topics!"
        )

    system_hint = (
        "You are a helpful assistant specialised in sustainable cities, smart-city governance, "
        "urban planning, and eco-friendly living. Provide practical, concise answers."
    )
    full_prompt = f"{system_hint}\n\nQ: {prompt}\nA:"
    return _generate(full_prompt)


def get_sustainability_tips(category: str) -> str:
    """
    Returns three actionable sustainability tips for the given category.
    """
    prompt = (
        f"Give three practical sustainability tips for '{category}'. "
        "Each tip should be 1-2 sentences."
    )
    return _generate(prompt)


def generate_eco_tip(topic: str) -> str:
    """
    Returns one concise eco-friendly tip focused on city/community level.
    """
    prompt = (
        f"Provide one concise, actionable eco-friendly tip for making a city more sustainable "
        f"regarding: {topic}. Aim for a single sentence."
    )
    return _generate(prompt)


def generate_summary(text: str) -> str:
    """
    Summarise long policy text (≈3–4 sentences).
    Uses a dedicated summarisation pipeline.
    """
    summariser = _get_summary_pipeline()
    summary = summariser(text)[0]["summary_text"]
    return summary.strip()


def generate_city_report(kpi_data: Dict[str, Any]) -> str:
    """
    Creates a structured sustainability report (markdown) from KPI data.
    """
    city = kpi_data.get("city_name", "the city")
    year = kpi_data.get("year", "the specified year")
    kpi_json_str = repr(kpi_data)[:1500]  # truncate very long input

    prompt = f"""
You are an expert sustainability analyst. Write a well-structured markdown report 
for **{city}** covering the year **{year}** using the KPI data below.

KPI_DATA:
{kpi_json_str}

Structure:
1. Air Quality
2. Energy Usage
3. Water Usage
4. Waste Management
5. Public Transport
6. Environmental Protection
7. Key Insights & Recommendations

Bullet points and clear headings preferred. Keep it under 500 words.
"""
    return _generate(prompt)
