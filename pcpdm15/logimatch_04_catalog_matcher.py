# logimatch_04_catalog_matcher.py
# Cell 5: AI Catalog Matching, Transparent Prompt Inspector & Human Gatekeeper Toggle

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
    for model_name in ["gemini-3.5-flash-lite", "gemini-3.8-flash"]:
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
            if any(k in last_err.lower() for k in ["503", "unavailable", "429"]):
                continue
            return {"error": f"API Error ({model_name}): {last_err[:90]}"}
    return {"error": f"API Busy ({last_err[:80]})"}

def _score_lead_hard_20s(client, prompt: str) -> dict:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_single_gemini_call, client, prompt)
    try:
        return future.result(timeout=20.0)
    except FuturesTimeout:
        return {"error": "Hard 20s timeout reached — skipped automatically"}
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

# --- UI Controls ---
api_key_c5 = widgets.Password(
    description="API Key:",
    placeholder="AIzaSy...",
    layout=widgets.Layout(width="340px"),
)
btn_run_match = widgets.Button(
    description="Run AI Catalog Verdict",
    button_style="primary",
    icon="check-circle",
    layout=widgets.Layout(width="230px"),
)
lbl_match_status = widgets.HTML(
    value="<i>Enter your Gemini API key and click <b>Run AI Catalog Verdict</b> (streams rows live; hard 20s cap per row).</i>"
)

out_verdict_tab = widgets.Output()
out_gatekeeper_tab = widgets.Output()
out_prompt_c5_tab = widgets.Output()

tabs_c5 = widgets.Tab(children=[out_verdict_tab, out_gatekeeper_tab, out_prompt_c5_tab])
tabs_c5.set_title(0, "1. Live AI Verdict & Reasoning")
tabs_c5.set_title(1, "2. Human Gatekeeper Toggle")
tabs_c5.set_title(2, "3. Inspect AI Brain Prompt")

DISPLAY_COLS = ["Firm_Name", "Matched_Items", "Rejected_Items", "Review_Gate", "Reasoning"]

def _render_prompt_inspector_c5():
    with out_prompt_c5_tab:
        clear_output()
        if "pipeline_df" not in globals() or pipeline_df is None or "catalog_df" not in globals() or catalog_df is None:
            display(HTML("<i>Upload data in Cell 2 and gather evidence in Cell 4 to preview rendered prompts.</i>"))
            return

        cat_text = _format_generic_catalog(catalog_df, col_map)
        dd_p_firm = widgets.Dropdown(
            options=list(pipeline_df["Firm_Name"]),
            description="Preview Lead:",
            style={"description_width": "100px"},
            layout=widgets.Layout(width="360px"),
        )
        prompt_box = widgets.HTML()

        def _update_prompt_view(change=None):
            row = pipeline_df[pipeline_df["Firm_Name"] == dd_p_firm.value].iloc[0]
            ev = row["Extracted_Evidence"] if "Extracted_Evidence" in pipeline_df.columns else ""
            rendered = _build_matcher_prompt(dd_p_firm.value, ev, cat_text)
            escaped = rendered.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            prompt_box.value = f"""
            <div style='margin-top:8px;'>
                <p><b>Exact Backend Prompt Sent to Gemini (JSON Schema Enforced: <code>SegmentVerdict</code>):</b></p>
                <pre style='background:#0f172a; color:#f8fafc; padding:12px; border-radius:6px; max-height:320px; overflow-y:auto; white-space:pre-wrap; font-size:0.86em;'>{escaped}</pre>
            </div>"""

        dd_p_firm.observe(_update_
