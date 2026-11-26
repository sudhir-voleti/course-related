# logimatch_04_catalog_matcher.py
# Cell 5: AI Catalog Matching & Human-in-the-Loop Review Gate (Hard 20s Cutoff + Live Row Streaming)

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

def _score_lead_hard_20s(client, firm_name: str, evidence: str, catalog_text: str) -> dict:
    prompt = f"""
You are a Senior B2B Solutions Strategist.
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
{evidence[:2200]}
"""
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
out_override_tab = widgets.Output()

tabs_c5 = widgets.Tab(children=[out_verdict_tab, out_override_tab])
tabs_c5.set_title(0, "1. Live AI Verdict & Reasoning Table")
tabs_c5.set_title(1, "2. Human Review & Override")

DISPLAY_COLS = ["Firm_Name", "Matched_Items", "Rejected_Items", "Review_Gate", "Reasoning"]

def _render_human_override():
    with out_override_tab:
        clear_output()
        if "pipeline_df" not in globals() or pipeline_df is None or "Matched_Items" not in pipeline_df.columns:
            display(HTML("<i>Run the AI Catalog Verdict in Tab 1 first.</i>"))
            return

        cat_choices = [
            f"{r[col_map['sku_id']]}: {r[col_map['sku_name']]}" for _, r in catalog_df.iterrows()
        ] + ["HELD — Do Not Pitch"]
        dd_firm = widgets.Dropdown(
            options=list(pipeline_df["Firm_Name"]),
            description="Inspect Lead:",
            style={"description_width": "100px"},
            layout=widgets.Layout(width="360px"),
        )
        dd_override_sku = widgets.Dropdown(
            options=cat_choices,
            description="Override Match:",
            style={"description_width": "100px"},
            layout=widgets.Layout(width="360px"),
        )
        dd_override_gate = widgets.Dropdown(
            options=["READY_TO_PITCH", "HUMAN_REVIEW_NEEDED"],
            description="Review Gate:",
            style={"description_width": "100px"},
            layout=widgets.Layout(width="360px"),
        )
        btn_save_override = widgets.Button(
            description="Apply Human Override",
            button_style="warning",
            layout=widgets.Layout(width="200px"),
        )
        lbl_override_msg = widgets.HTML()
        detail_box = widgets.HTML()

        def _load_firm_details(change=None):
            row = pipeline_df[pipeline_df["Firm_Name"] == dd_firm.value].iloc[0]
            first_match = row["Matched_Items"].split(" | ")[0]
            dd_override_sku.value = first_match if first_match in cat_choices else "HELD — Do Not Pitch"
            dd_override_gate.value = row["Review_Gate"]
            lbl_override_msg.value = ""
            detail_box.value = f"""
            <div style='border:1px solid #cbd5e1; border-radius:6px; padding:12px; margin-top:10px; background:#fff;'>
                <p><b>Current AI Match:</b> {row['Matched_Items']} &nbsp;|&nbsp; <b>Gate:</b> <code>{row['Review_Gate']}</code></p>
                <p><b>Rejected Offerings:</b> {row['Rejected_Items']}</p>
                <p><b>Brain Reasoning:</b> {row['Reasoning']}</p>
                <p><b>Extracted Evidence ({row['Evidence_Source']}):</b></p>
                <div style='background:#f8fafc; border:1px solid #e2e8f0; padding:8px; max-height:130px; overflow-y:auto; font-size:0.9em;'>{row['Extracted_Evidence'] or '— No evidence gathered —'}</div>
            </div>"""

        def _apply_override(_):
            idx = pipeline_df.index[pipeline_df["Firm_Name"] == dd_firm.value][0]
            pipeline_df.at[idx, "Matched_Items"] = dd_override_sku.value
            pipeline_df.at[idx, "Review_Gate"] = dd_override_gate.value
            pipeline_df.at[idx, "Reasoning"] = f"[Human Reviewer Override] {pipeline_df.at[idx, 'Reasoning']}"
            lbl_override_msg.value = (
                f"<span style='color:#15803d;'><b>✅ Updated {dd_firm.value} → {dd_override_sku.value} ({dd_override_gate.value})</b></span>"
            )
            _load_firm_details()
            with out_verdict_tab:
                clear_output(wait=True)
                display(HTML(pipeline_df[DISPLAY_COLS].to_html(index=False)))

        dd_firm.observe(_load_firm_details, names="value")
        btn_save_override.on_click(_apply_override)
        _load_firm_details()
        display(widgets.VBox([
            widgets.HTML("<b>Human-in-the-Loop Audit:</b> Inspect any prospect's evidence and override the AI segment match or review gate before Cell 6 drafts emails."),
            dd_firm,
            widgets.HBox([dd_override_sku, dd_override_gate, btn_save_override]),
            lbl_override_msg,
            detail_box,
        ]))

def on_run_match_clicked(_):
    global pipeline_df
    if "pipeline_df" not in globals() or pipeline_df is None or "Evidence_Status" not in pipeline_df.columns:
        lbl_match_status.value = "<span style='color:#b91c1c;'><b>⚠️ Please gather evidence in Cell 4 first.</b></span>"
        return
    if "catalog_df" not in globals() or catalog_df is None or not api_key_c5.value.strip():
        lbl_match_status.value = (
            "<span style='color:#b91c1c;'><b>⚠️ Ensure Catalog CSV is uploaded in Cell 2 and API Key is entered above.</b></span>"
        )
        return

    btn_run_match.disabled = True
    client = genai.Client(api_key=api_key_c5.value.strip())
    cat_text = _format_generic_catalog(catalog_df, col_map)

    matches, rejects, gates, reasonings, ops_list, live_rows = [], [], [], [], [], []
    total = len(pipeline_df)

    for idx, row in pipeline_df.iterrows():
        firm = row["Firm_Name"]
        if row["Evidence_Status"] != "EVIDENCE_READY":
            m_str = "HELD — Do Not Pitch"
            r_str = "Skipped (No verifiable web evidence)"
            g_str = "HUMAN_REVIEW_NEEDED"
            rsn_str = "Held automatically because both direct site scrape and web search returned empty."
            op_str = ""
        else:
            lbl_match_status.value = f"<i>⏳ Scoring ({idx + 1}/{total}): <b>{firm}</b> (20s hard cap)...</i>"
            res = _score_lead_hard_20s(client, firm, row["Extracted_Evidence"], cat_text)
            if "error" in res:
                m_str = "HELD — Timeout/Busy"
                r_str = res["error"]
                g_str = "HUMAN_REVIEW_NEEDED"
                rsn_str = res["error"]
                op_str = ""
            else:
                m_str = " | ".join(res.get("matched_items", []))
                r_str = " | ".join(res.get("rejected_items", []))
                g_str = res.get("review_gate", "HUMAN_REVIEW_NEEDED")
                rsn_str = res.get("reasoning", "")
                op_str = " | ".join(res.get("extracted_operations", []))
            time.sleep(0.5)

        matches.append(m_str)
        rejects.append(r_str)
        gates.append(g_str)
        reasonings.append(rsn_str)
        ops_list.append(op_str)

        live_rows.append({
            "Firm_Name": firm,
            "Matched_Items": m_str,
            "Rejected_Items": r_str,
            "Review_Gate": g_str,
            "Reasoning": rsn_str,
        })
        with out_verdict_tab:
            clear_output(wait=True)
            display(HTML(pd.DataFrame(live_rows).to_html(index=False)))

    pipeline_df["Extracted_Operations"] = ops_list
    pipeline_df["Matched_Items"] = matches
    pipeline_df["Rejected_Items"] = rejects
    pipeline_df["Review_Gate"] = gates
    pipeline_df["Reasoning"] = reasonings

    n_ready = (pipeline_df["Review_Gate"] == "READY_TO_PITCH").sum()
    lbl_match_status.value = (
        f"<span style='color:#15803d;'><b>✅ AI Verdict Complete: {n_ready}/{total} leads cleared as READY_TO_PITCH; "
        f"{total - n_ready} flagged for HUMAN_REVIEW_NEEDED.</b></span>"
    )
    btn_run_match.disabled = False
    _render_human_override()

btn_run_match.on_click(on_run_match_clicked)
display(widgets.VBox([widgets.HBox([api_key_c5, btn_run_match]), lbl_match_status, tabs_c5]))
