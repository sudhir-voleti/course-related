# logimatch_05_outreach_composer.py
# Cell 6: Grounded Outreach Composer, Live Audit Table & CSV/JSON Export

from google.colab import files as colab_files

class OutreachDraft(BaseModel):
    subject_line: str = Field(
        description="Concise, specific B2B email subject line referencing the prospect's industry or operation."
    )
    email_body: str = Field(
        description="3-sentence consultative outreach email referencing verified evidence and pitching ONLY the matched catalog item(s)."
    )

def _single_email_call(client, prompt: str) -> dict:
    last_err = ""
    for model_name in ["gemini-3.5-flash-lite", "gemini-3.8-flash"]:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=OutreachDraft,
                    temperature=0.3,
                ),
            )
            return json.loads(resp.text)
        except Exception as e:
            last_err = str(e)
            if any(k in last_err.lower() for k in ["503", "unavailable", "429"]):
                continue
            return {"subject_line": "API_ERROR", "email_body": f"API Error: {last_err[:90]}"}
    return {"subject_line": "API_BUSY", "email_body": f"Model busy ({last_err[:80]})"}

def _draft_email_hard_20s(client, firm_name: str, matched_items: str, evidence: str, reasoning: str) -> dict:
    prompt = f"""
You are a Senior B2B Solutions Consultant.
Write a crisp 3-sentence outreach email to '{firm_name}' pitching ONLY the approved offering(s) below.

RULES:
1. Reference 1–2 concrete operational facts from the Verified Evidence (never invent claims).
2. Connect those facts directly to the Approved Catalog Match and Reasoning.
3. Keep the tone consultative, peer-to-peer, and concise (3 sentences max, no fluff).

=== APPROVED CATALOG MATCH ===
{matched_items}

=== MATCH REASONING ===
{reasoning}

=== VERIFIED EVIDENCE ===
{evidence[:2200]}
"""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_single_email_call, client, prompt)
    try:
        return future.result(timeout=20.0)
    except FuturesTimeout:
        return {"subject_line": "TIMEOUT", "email_body": "Hard 20s timeout reached — retry row."}
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

# --- UI Controls ---
default_key = api_key_c5.value if "api_key_c5" in globals() and api_key_c5.value else ""
api_key_c6 = widgets.Password(
    value=default_key,
    description="API Key:",
    placeholder="AIzaSy...",
    layout=widgets.Layout(width="340px"),
)
btn_draft_all = widgets.Button(
    description="Draft Approved Outreach Emails",
    button_style="success",
    icon="envelope",
    layout=widgets.Layout(width="250px"),
)
btn_dl_csv = widgets.Button(
    description="Download Final CSV",
    button_style="info",
    disabled=True,
    layout=widgets.Layout(width="170px"),
)
btn_dl_json = widgets.Button(
    description="Download Final JSON",
    button_style="info",
    disabled=True,
    layout=widgets.Layout(width="170px"),
)
lbl_c6_status = widgets.HTML(
    value="<i>Click <b>Draft Approved Outreach Emails</b> to generate pitches ONLY for <code>READY_TO_PITCH</code> leads (streams live row by row).</i>"
)

out_c6_table = widgets.Output()
out_c6_reader = widgets.Output()

tabs_c6 = widgets.Tab(children=[out_c6_table, out_c6_reader])
tabs_c6.set_title(0, "1. Live Outreach Status Table")
tabs_c6.set_title(1, "2. Email Pitch Reader")

def _render_email_reader():
    with out_c6_reader:
        clear_output()
        if "pipeline_df" not in globals() or pipeline_df is None or "Email_Body" not in pipeline_df.columns:
            return
        dd_email_firm = widgets.Dropdown(
            options=list(pipeline_df["Firm_Name"]),
            description="Select Lead:",
            style={"description_width": "90px"},
            layout=widgets.Layout(width="360px"),
        )
        email_card = widgets.HTML()

        def _update_card(change=None):
            row = pipeline_df[pipeline_df["Firm_Name"] == dd_email_firm.value].iloc[0]
            is_ready = row["Review_Gate"] == "READY_TO_PITCH"
            badge_color = "#15803d" if is_ready else "#b45309"
            email_card.value = f"""
            <div style='border:1px solid #cbd5e1; border-radius:6px; padding:14px; margin-top:10px; background:#fff;'>
                <p><b>Gate Status:</b> <span style='color:{badge_color}; font-weight:bold;'>{row['Review_Gate']}</span> &nbsp;|&nbsp; <b>Matched SKU:</b> {row['Matched_Items']}</p>
                <p><b>Subject:</b> <code>{row['Email_Subject']}</code></p>
                <hr>
                <div style='background:#f8fafc; padding:12px; border-left:4px solid #002D72; white-space:pre-wrap; font-size:0.95em;'>{row['Email_Body']}</div>
            </div>"""

        dd_email_firm.observe(_update_card, names="value")
        _update_card()
        display(widgets.VBox([dd_email_firm, email_card]))

def on_draft_all_clicked(_):
    global pipeline_df
    if "pipeline_df" not in globals() or pipeline_df is None or "Review_Gate" not in pipeline_df.columns:
        lbl_c6_status.value = "<span style='color:#b91c1c;'><b>⚠️ Please run the AI Catalog Verdict in Cell 5 first.</b></span>"
        return
    if not api_key_c6.value.strip():
        lbl_c6_status.value = "<span style='color:#b91c1c;'><b>⚠️ Please enter your Gemini API Key above.</b></span>"
        return

    btn_draft_all.disabled = True
    client = genai.Client(api_key=api_key_c6.value.strip())
    subjects, bodies, live_preview = [], [], []
    total = len(pipeline_df)

    for idx, row in pipeline_df.iterrows():
        firm = row["Firm_Name"]
        gate = row["Review_Gate"]

        if gate != "READY_TO_PITCH":
            subj = "HELD — DO NOT SEND"
            body = "HELD FOR MANUAL REVIEW — Automated outreach locked until human reviewer clears this lead."
        else:
            lbl_c6_status.value = f"<i>⏳ Drafting pitch ({idx + 1}/{total}): <b>{firm}</b> (20s hard cap)...</i>"
            res = _draft_email_hard_20s(client, firm, row["Matched_Items"], row["Extracted_Evidence"], row["Reasoning"])
            subj = res.get("subject_line", "")
            body = res.get("email_body", "")
            time.sleep(0.5)

        subjects.append(subj)
        bodies.append(body)
        live_preview.append({
            "Firm_Name": firm,
            "Review_Gate": gate,
            "Matched_Items": row["Matched_Items"],
            "Email_Subject": subj,
            "Email_Preview": (body[:95] + "...") if len(body) > 95 else body,
        })

        with out_c6_table:
            clear_output(wait=True)
            display(HTML(pd.DataFrame(live_preview).to_html(index=False)))

    pipeline_df["Email_Subject"] = subjects
    pipeline_df["Email_Body"] = bodies

    export_cols = [
        "Firm_Name",
        "Website_URL",
        "Access_Status",
        "Evidence_Source",
        "Matched_Items",
        "Rejected_Items",
        "Review_Gate",
        "Reasoning",
        "Email_Subject",
        "Email_Body",
    ]
    final_df = pipeline_df[export_cols]
    final_df.to_csv("logimatch_final_output.csv", index=False)
    final_df.to_json("logimatch_final_output.json", orient="records", indent=2, force_ascii=False)

    n_sent = (pipeline_df["Review_Gate"] == "READY_TO_PITCH").sum()
    lbl_c6_status.value = (
        f"<span style='color:#15803d;'><b>✅ Complete! Drafted {n_sent} grounded pitches "
        f"({total - n_sent} held safely). Export files ready.</b></span>"
    )
    btn_draft_all.disabled = False
    btn_dl_csv.disabled = False
    btn_dl_json.disabled = False
    _render_email_reader()

btn_draft_all.on_click(on_draft_all_clicked)
btn_dl_csv.on_click(lambda _: colab_files.download("logimatch_final_output.csv"))
btn_dl_json.on_click(lambda _: colab_files.download("logimatch_final_output.json"))

display(widgets.VBox([
    widgets.HBox([api_key_c6, btn_draft_all, btn_dl_csv, btn_dl_json]),
    lbl_c6_status,
    tabs_c6,
]))
