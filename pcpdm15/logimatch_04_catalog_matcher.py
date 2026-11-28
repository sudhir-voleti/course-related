# logimatch_04_catalog_matcher.py
# Cell 5: AI Catalog Matching, Transparent Prompt Inspector & Human Gatekeeper Toggle

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import List, Literal

import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


class SegmentVerdict(BaseModel):
    extracted_operations: List[str] = Field(
        description="Verbatim products, certifications, or capabilities found in Extracted Evidence."
    )
    matched_items: List[str] = Field(
        description="1 to 2 best-fitting 'Item ID: Item Name' pairs from the uploaded catalog."
    )
    rejected_items: List[str] = Field(
        description="At least 2 disqualified catalog items with a concise 1-line reason why each does not fit."
    )
    review_gate: Literal["READY_TO_PITCH", "HUMAN_REVIEW_NEEDED"] = Field(
        description="Set to READY_TO_PITCH if evidence clearly supports a catalog match; set to HUMAN_REVIEW_NEEDED if evidence is ambiguous, generic, or weak."
    )
    reasoning: str = Field(
        description="2 concise sentences explaining why the brain chose that specific segment/SKU for this row based on verified evidence."
    )


def _format_generic_catalog(cat_df: pd.DataFrame, cmap: dict) -> str:
    lines = []
    for _, r in cat_df.iterrows():
        item_id = str(r[cmap["sku_id"]])
        item_name = str(r[cmap["sku_name"]])
        signals = str(r[cmap["signals"]])
        disq = str(r[cmap["disqualifiers"]]) if cmap.get("disqualifiers") else "None specified"
        extra_cols = [
            c for c in cat_df.columns
            if c not in {cmap["sku_id"], cmap["sku_name"], cmap["signals"], cmap.get("disqualifiers")}
        ]
        extras = " | ".join([f"{c}: {r[c]}" for c in extra_cols if pd.notna(r[c])])
        lines.append(
            f"- [{item_id}] {item_name} | Signals: {signals} | Disqualifiers: {disq}"
            + (f" | {extras}" if extras else "")
        )
    return "\n".join(lines)


def _build_matcher_prompt(firm_name: str, evidence: str, catalog_text: str) -> str:
    return f"""You are a Senior B2B Solutions Strategist.
Evaluate prospect '{firm_name}' strictly from the Extracted Web Evidence below and tally it against the Offering Catalog.

RULES:
1. Never guess products from the company name alone. Rely ONLY on Extracted Web Evidence.
2. Select 1 to 2 best-fitting items from the catalog in `matched_items`.
3. List at least 2 disqualified catalog items with concise reasons in `rejected_items`.
4. Set `review_gate` to `READY_TO_PITCH` if concrete product/service evidence matches the catalog; set `HUMAN_REVIEW_NEEDED` if the snippet is too generic or ambiguous.
5. In `reasoning`, explain clearly in 1–2 sentences why that specific segment/SKU was chosen over alternatives.

=== UPLOADED OFFERING CATALOG ===
{catalog_text}

=== PROSPECT: {firm_name} ===
Extracted Web Evidence:
{evidence[:2200] if evidence else "[NO EVIDENCE GATHERED — HELD BEFORE LLM CALL]"}"""


def _single_gemini_call(client, prompt: str) -> dict:
    last_err = ""
    for model_name in ["gemini-2.5-flash", "gemini-2.5-flash-lite"]:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SegmentVerdict,
                    temperature=0.2,
                ),
            )
            res = json.loads(resp.text)
            res["model_used"] = model_name
            return res
        except Exception as e:
            last_err = str(e)
            if any(k in last_err.lower() for k in ["503", "unavailable", "429", "404", "not found"]):
                continue
            return {"error": f"API Error ({model_name}): {last_err[:110]}"}
    return {"error": f"API Busy/Unavailable ({last_err[:100]})"}


def _score_lead_hard_20s(client, prompt: str) -> dict:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_single_gemini_call, client, prompt)
    try:
        return future.result(timeout=20.0)
    except FuturesTimeout:
        return {"error": "Hard 20s timeout reached — skipped automatically"}
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


# --- UI Controls (Always blank so each student pastes their own API key) ---
api_key_c5 = widgets.Password(
    value="",
    description="API Key:",
    placeholder="Paste your Gemini API key (AIzaSy...)",
    layout=widgets.Layout(width="360px"),
)
btn_run_match = widgets.Button(
    description="Run AI Catalog Verdict",
    button_style="primary",
    icon="check-circle",
    layout=widgets.Layout(width="230px"),
)
lbl_match_status = widgets.HTML(
    value="<i>Paste your personal Gemini API key above and click <b>Run AI Catalog Verdict</b> (streams rows live; hard 20s cap per row).</i>"
)

out_verdict_tab = widgets.Output()
out_gatekeeper_tab = widgets.Output()
out_prompt_c5_tab = widgets.Output()

tabs_c5 = widgets.Tab(children=[out_verdict_tab, out_gatekeeper_tab, out_prompt_c5_tab])
tabs_c5.set_title(0, "1. Live AI Verdict & Reasoning")
tabs_c5.set_title(1, "2. Human Gatekeeper Toggle")
tabs_c5.set_title(2, "3. Inspect AI Brain Prompt")

DISPLAY_COLS = ["Firm_Name", "Matched_Items", "Rejected_Items", "Review_Gate", "Reasoning"]


def _format_verdict_table_html(df: pd.DataFrame) -> str:
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    rows_html = []
    for _, r in df[cols].iterrows():
        gate = str(r.get("Review_Gate", ""))
        if "READY_TO_PITCH" in gate:
            badge_bg, badge_fg = "#dcfce7", "#166534"
        elif "DEAD_LEAD_HOLD" in gate:
            badge_bg, badge_fg = "#fee2e2", "#991b1b"
        else:
            badge_bg, badge_fg = "#fef3c7", "#92400e"

        cells = []
        for c in cols:
            val = str(r[c]) if pd.notna(r[c]) else ""
            if c == "Review_Gate":
                cells.append(
                    f"<td style='padding:8px; border:1px solid #cbd5e1; vertical-align:top;'>"
                    f"<span style='background:{badge_bg}; color:{badge_fg}; font-weight:bold; padding:3px 8px; border-radius:4px; font-size:0.85em;'>{val}</span></td>"
                )
            else:
                cells.append(
                    f"<td style='padding:8px; border:1px solid #cbd5e1; vertical-align:top; font-size:0.9em;'>{val}</td>"
                )
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    headers = "".join(
        f"<th style='background:#002D72; color:#ffffff; padding:8px; border:1px solid #002D72; text-align:left;'>{c}</th>"
        for c in cols
    )
    return (
        "<table style='width:100%; border-collapse:collapse; margin-top:10px;'>"
        f"<thead><tr>{headers}</tr></thead><tbody>{''.join(rows_html)}</tbody></table>"
    )


def _render_verdict_tab():
    with out_verdict_tab:
        clear_output()
        if "pipeline_df" not in globals() or pipeline_df is None:
            display(HTML("<i>Upload data in Cell 2 and gather evidence in Cell 4 first.</i>"))
            return
        display(HTML(_format_verdict_table_html(pipeline_df)))


def _render_gatekeeper_tab():
    with out_gatekeeper_tab:
        clear_output()
        if "pipeline_df" not in globals() or pipeline_df is None or "Review_Gate" not in pipeline_df.columns:
            display(HTML("<i>Run <b>AI Catalog Verdict</b> first to populate gatekeeper controls.</i>"))
            return

        display(HTML(
            "<p style='margin-bottom:8px;'><b>Human-in-the-Loop Override:</b> Review ambiguous or held prospects below and toggle their gate status before running Stage 5 Outreach.</p>"
        ))

        row_widgets = []
        for idx, r in pipeline_df.iterrows():
            firm = str(
