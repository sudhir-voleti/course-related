# logimatch_05_outreach_composer.py
# Cell 6: Grounded Outreach Composer, Transparent Prompt Inspector & CSV/JSON Export

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.colab import files as colab_files


class OutreachDraft(BaseModel):
    subject_line: str = Field(
        description="Concise, specific B2B email subject line referencing the prospect's industry or operation."
    )
    email_body: str = Field(
        description="3-sentence consultative outreach email referencing verified evidence and pitching ONLY the matched catalog item(s)."
    )


def _build_outreach_prompt(firm_name: str, matched_items: str, evidence: str, reasoning: str) -> str:
    return f"""You are a Senior B2B Solutions Consultant.
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
{evidence[:2200] if evidence else "[NO EVIDENCE — ROW LOCKED]"}"""


def _single_email_call(client, prompt: str) -> dict:
    last_err = ""
    for model_name in ["gemini-2.5-flash", "gemini-2.5-pro"]:
        for _ in range(2):
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
                if any(k in last_err.lower() for k in ["503", "unavailable", "429", "resource_exhausted"]):
                    time.sleep(2.5)
                    continue
                if any(k in last_err.lower() for k in ["404", "not found", "no longer"]):
                    break
                return {"subject_line": "API_ERROR", "email_body": f"API Error: {last_err[:100]}"}
    return {"subject_line": "API_BUSY", "email_body": f"Model busy ({last_err[:90]})"}


def _draft_email_hard_40s(client, prompt: str) -> dict:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_single_email_call, client, prompt)
    try:
        return future.result(timeout=40.0)
    except FuturesTimeout:
        return {"subject_line": "TIMEOUT", "email_body": "Hard 40s timeout reached — retry row."}
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


# --- UI Controls (Carries over session widget value from Cell 5 if entered; never reads Colab Secrets) ---
default_key = api_key_c5.value if "api_key_c5" in globals() and api_key_c5.value else ""
api_key_c6 = widgets.Password(
    value=default_key,
    description="API Key:",
    placeholder="Paste your Gemini API key (AIzaSy...)",
    layout=widgets.Layout(width="360px"),
)
btn_draft_all = widgets.Button(
    description="Draft Approved Outreach Emails",
    button_style="success",
    icon="envelope",
    layout=widgets.Layout(width="255px"),
)
btn_dl_csv = widgets.Button(
    description="Download Final CSV",
    button_style="info",
    icon="download",
    disabled=True,
    layout=widgets.Layout(width="180px"),
)
btn_dl_json = widgets.Button(
    description="Download Final JSON",
    button_style="info",
    icon="download",
    disabled=True,
    layout=widgets.Layout(width="180px"),
)
lbl_c6_status = widgets.HTML(
    value="<i>Click <b>Draft Approved Outreach Emails</b> to generate pitches ONLY for <code>READY_TO_PITCH</code> leads (streams live row by row).</i>"
)

out_c6_table = widgets.Output()
out_c6_reader = widgets.Output()
out_c6_prompt = widgets.Output()

tabs_c6 = widgets.Tab(children=[out_c6_table, out_c6_reader, out_c6_prompt])
tabs_c6.set_title(0, "1. Live Outreach Status Table")
tabs_c6.set_title(1, "2. Email Pitch Reader")
tabs_c6.set_title(2, "3. Inspect AI Brain Prompt")


def _render_c6_status_table(rows_list: list):
    with out_c6_table:
        clear_output()
        if not rows_list:
            display(HTML("<i>Click <b>Draft Approved Outreach Emails</b> above to run Stage 5.</i>"))
            return

        html_rows = []
        for r in rows_list:
            gate = r["Review_Gate"]
            if gate == "READY_TO_PITCH":
                badge_bg, badge_fg = "#dcfce7", "#166534"
            elif gate == "DEAD_LEAD_HOLD":
                badge_bg, badge_fg = "#fee2e2", "#991b1b"
            else:
                badge_bg, badge_fg = "#fef3c7", "#92400e"

            html_rows.append(
                f"<tr>"
                f"<td style='padding:8px; border:1px solid #cbd5e1; vertical-align:top;'><b>{r['Firm_Name']}</b></td>"
                f"<td style='padding:8px; border:1px solid #cbd5e1; vertical-align:top;'>"
                f"<span style='background:{badge_bg}; color:{badge_fg}; font-weight:bold; padding:3px 8px; border-radius:4px; font-size:0.85em;'>{gate}</span></td>"
                f"<td style='padding:8px; border:1px solid #cbd5e1; vertical-align:top; font-size:0.9em;'>{r['Matched_Items']}</td>"
                f"<td style='padding:8px; border:1px solid #cbd5e1; vertical-align:top; font-size:0.9em;'><code>{r['Email_Subject']}</code></td>"
                f"<td style='padding:8px; border:1px solid #cbd5e1; vertical-align:top; font-size:0.9em;'>{r['Email_Body']}</td>"
                f"</tr>"
            )

        table_html = (
            "<table style='width:100%; border-collapse:collapse; margin-top:10px;'>"
            "<thead><tr>"
            "<th style='background:#002D72; color:#fff; padding:8px; border:1px solid #002D72; text-align:left;'>Firm_Name</th>"
            "<th style='background:#002D72; color:#fff; padding:8px; border:1px solid #002D72; text-align:left;'>Review_Gate</th>"
            "<th style='background:#002D72; color:#fff; padding:8px; border:1px solid #002D72; text-align:left;'>Matched_Items</th>"
            "<th style='background:#002D72; color:#fff; padding:8px; border:1px solid #002D72; text-align:left;'>Email_Subject</th>"
            "<th style='background:#002D72; color:#fff; padding:8px; border:1px solid #002D72; text-align:left;'>Email_Body</th>"
            "</tr></thead>"
            f"<tbody>{''.join(html_rows)}</tbody></table>"
        )
        display(HTML(table_html))


def _render_prompt_inspector_c6():
    with out_c6_prompt:
        clear_output()
        if "pipeline_df" not in globals() or pipeline_df is None or "Matched_Items" not in pipeline_df.columns:
            display(HTML("<i>Run Cell 5 first to preview the outreach composer prompt.</i>"))
            return

        dd_p_firm = widgets.Dropdown(
            options=list(pipeline_df["Firm_Name"]),
            description="Preview Lead:",
            style={"description_width": "100px"},
            layout=widgets.Layout(width="360px"),
        )
        prompt_box = widgets.HTML()

        def _update_prompt_view(change=None):
            row = pipeline_df[pipeline_df["Firm_Name"] == dd_p_firm.value].iloc[0]
            if row["Review_Gate"] != "READY_TO_PITCH":
                rendered = (
                    f"[GUARDRAIL LOCKED — Review_Gate = {row['Review_Gate']}]\n"
                    f"No prompt sent to Gemini. Row automatically quarantined as 'HELD — DO NOT SEND'."
                )
            else:
                ev = str(row["Extracted_Evidence"]) if "Extracted_Evidence" in pipeline_df.columns and pd.notna(row["Extracted_Evidence"]) else ""
                rendered = _build_outreach_prompt(
                    dd_p_firm.value,
                    str(row.get("Matched_Items", "")),
                    ev,
                    str(row.get("Reasoning", "")),
                )
            escaped = rendered.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            prompt_box.value = f"""
            <div style='margin-top:8px;'>
                <p><b>Exact Backend Prompt Sent to Gemini (JSON Schema Enforced: <code>OutreachDraft</code>):</b></p>
                <pre style='background:#0f172a; color:#f8fafc; padding:12px; border-radius:6px; max-height:320px; overflow-y:auto; white-space:pre-wrap; font-size:0.86em;'>{escaped}</pre>
            </div>"""

        dd_p_firm.observe(_update_prompt_view, names="value")
        _update_prompt_view()
        display(widgets.VBox([dd_p_firm, prompt_box]))


def _render_email_reader():
    with out_c6_reader:
        clear_output()
        if "pipeline_df" not in globals() or pipeline_df is None or "Email_Body" not in pipeline_df.columns:
            display(HTML("<i>Draft outreach emails first to inspect individual messages here.</i>"))
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
        lbl_c6_status.value = "<span style='color:#b91c1c;'><b>Missing upstream data:</b> Please run the AI Catalog Verdict in Cell 5 first.</span>"
        return
    if not api_key_c6.value.strip():
        lbl_c6_status.value = "<span style='color:#b91c1c;'><b>API Key required:</b> Please enter your Gemini API key above.</span>"
        return

    btn_draft_all.disabled = True
    tabs_c6.selected_index = 0
    client = genai.Client(api_key=api_key_c6.value.strip())
    subjects, bodies, live_preview = [], [], []
    total = len(pipeline_df)

    try:
        for i, (_, row) in enumerate(pipeline_df.iterrows(), start=1):
            firm = str(row["Firm_Name"])
            gate = str(row["Review_Gate"])
            matched = str(row.get("Matched_Items", ""))

            if gate != "READY_TO_PITCH":
                subj = "HELD — DO NOT SEND"
                body = "HELD FOR MANUAL REVIEW — Automated outreach locked until human reviewer clears this lead."
            else:
                lbl_c6_status.value = f"<i>Drafting pitch ({i}/{total}): <b>{firm}</b>...</i>"
                ev = str(row["Extracted_Evidence"]) if "Extracted_Evidence" in pipeline_df.columns and pd.notna(row["Extracted_Evidence"]) else ""
                prompt = _build_outreach_prompt(firm, matched, ev, str(row.get("Reasoning", "")))
                res = _draft_email_hard_40s(client, prompt)
                subj = res.get("subject_line", "")
                body = res.get("email_body", "")
                time.sleep(1.5)

            subjects.append(subj)
            bodies.append(body)
            live_preview.append({
                "Firm_Name": firm,
                "Review_Gate": gate,
                "Matched_Items": matched,
                "Email_Subject": subj,
                "Email_Body": body,
            })
            _render_c6_status_table(live_preview)

        pipeline_df["Email_Subject"] = subjects
        pipeline_df["Email_Body"] = bodies

        btn_dl_csv.disabled = False
        btn_dl_json.disabled = False
        lbl_c6_status.value = "<span style='color:#15803d;'><b>Done!</b> Approved pitches drafted and unverified rows safely locked. Use Tab 2 to inspect emails or download deliverables below.</span>"
        _render_email_reader()
        _render_prompt_inspector_c6()
    finally:
        btn_draft_all.disabled = False


def on_download_csv_clicked(_):
    if "pipeline_df" in globals() and pipeline_df is not None:
        out_path = "logimatch_pro_final_results.csv"
        pipeline_df.to_csv(out_path, index=False)
        colab_files.download(out_path)


def on_download_json_clicked(_):
    if "pipeline_df" in globals() and pipeline_df is not None:
        out_path = "logimatch_pro_final_results.json"
        pipeline_df.to_json(out_path, orient="records", indent=2)
        colab_files.download(out_path)


btn_draft_all.on_click(on_draft_all_clicked)
btn_dl_csv.on_click(on_download_csv_clicked)
btn_dl_json.on_click(on_download_json_clicked)

display(
    widgets.VBox([
        widgets.HBox([api_key_c6, btn_draft_all, btn_dl_csv, btn_dl_json]),
        lbl_c6_status,
        tabs_c6,
    ])
)

_render_c6_status_table([])
_render_email_reader()
_render_prompt_inspector_c6()
