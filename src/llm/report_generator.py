import os
import numpy as np
from src.rag.retriever import retrieve
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client=Groq(api_key=os.environ["GROQ_API_KEY"])

def summarize_risk_grid(risk_grid):
    """
    Turns the raw grid into plain numeric facts the LLM can quote directly -
    this keeps the 'data' part of the report deterministic, not LLM-guessed.
    """
    flat=risk_grid.flatten()
    total=len(flat)
    counts={level: int(np.sum(flat==level)) for level in ["low","moderate","high","severe"]}
    percentages={level:round(100*c/total,1)for level,c in counts.items()}
    return percentages

def build_prompt(query,risk_summary,doctrine_chunks):
    context_text="\n\n".join(
        f"[Source {i+1}]: {chunk}" for i,(chunk,score) in enumerate(doctrine_chunks)

    )

    system_prompt = (
        "You are a terrain intelligence analyst generating operational SITREP reports. "
        "Use ONLY the terrain risk data and reference material provided below. "
        "If something isn't covered by the provided data or references, explicitly say so "
        "rather than guessing. Cite which source supports each claim using [Source N]."
    )

    user_prompt = f"""
TERRAIN RISK DATA (from slope analysis):
- Low risk: {risk_summary['low']}%
- Moderate risk: {risk_summary['moderate']}%
- High risk: {risk_summary['high']}%
- Severe risk: {risk_summary['severe']}%

REFERENCE MATERIAL:
{context_text}

TASK: {query}

Write the report with exactly these sections:
1. Terrain Summary
2. Risk Zones
3. Recommended Route Considerations
4. Supply Feasibility
"""
    return system_prompt, user_prompt


def generate_report(query, risk_grid_path="data/raw/risk_grid.npy"):
    risk_grid = np.load(risk_grid_path, allow_pickle=True)
    risk_summary = summarize_risk_grid(risk_grid)
    doctrine_chunks = retrieve(query, k=3)

    system_prompt, user_prompt = build_prompt(query, risk_summary, doctrine_chunks)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # check console.groq.com for current available free models
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,   # low — factual consistency over creativity, per the concept notes
        max_tokens=800,
    )

    report = response.choices[0].message.content
    return report


if __name__ == "__main__":
    report = generate_report("Assess this region for a foot-supply route under difficult terrain conditions")
    print(report)

    os.makedirs("docs", exist_ok=True)
    with open("docs/sample_report.md", "w", encoding="utf-8") as f:
        f.write(report)
