# logimatch_02_site_probe.py
# Cell 3: Tool 1 — Interactive Website Accessibility & Bot-Check Probe

pipeline_df = None

def check_site_access(url: str) -> dict:
    """Checks if a prospect URL is live and accessible to an automated HTTP agent."""
    url = str(url).strip()
    if not url or not url.startswith("http"):
        return {
            "HTTP_Status": "N/A",
            "Access_Status": "BLOCKED_OR_UNREACHABLE",
            "Access_Note": "Invalid or missing URL",
        }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        code = resp.status_code
        if code == 200 and len(resp.text.strip()) >= 200:
            return {
                "HTTP_Status": code,
                "Access_Status": "ACCESSIBLE",
                "Access_Note": "OK — Ready for direct scraping",
            }
        elif code == 200:
            return {
                "HTTP_Status": code,
                "Access_Status": "BLOCKED_OR_UNREACHABLE",
                "Access_Note": "HTTP 200 but empty/JS-only shell",
            }
        else:
            return {
                "HTTP_Status": code,
                "Access_Status": "BLOCKED_OR_UNREACHABLE",
                "Access_Note": f"Blocked by server (HTTP {code})",
            }
    except requests.exceptions.Timeout:
        return {
            "HTTP_Status": "TIMEOUT",
            "Access_Status": "BLOCKED_OR_UNREACHABLE",
            "Access_Note": "Connection timed out (>10s)",
        }
    except Exception as e:
        return {
            "HTTP_Status": "ERR",
            "Access_Status": "BLOCKED_OR_UNREACHABLE",
            "Access_Note": f"Connection failed ({type(e).__name__})",
        }

access_btn = widgets.Button(
    description="Run Agentic Site Access Check",
    button_style="primary",
    icon="search",
    layout=widgets.Layout(width="270px", height="36px"),
)
access_status_lbl = widgets.HTML(
    value="<i>Click the button to probe which prospect websites allow automated agent access.</i>"
)
access_table_out = widgets.Output()

def on_access_check_clicked(_):
    global pipeline_df
    if "leads_df" not in globals() or leads_df is None:
        access_status_lbl.value = (
            "<span style='color:#b91c1c;'><b>⚠️ Please upload your Leads CSV in Cell 2 first.</b></span>"
        )
        return

    access_btn.disabled = True
    access_rows = []
    total = len(leads_df)

    for idx, row in leads_df.iterrows():
        firm = str(row[col_map["firm"]]).strip()
        url = str(row[col_map["url"]]).strip()
        access_status_lbl.value = f"<i>⏳ Probing ({idx + 1}/{total}): <b>{firm}</b>...</i>"
        probe = check_site_access(url)
        access_rows.append({
            "Firm_Name": firm,
            "Website_URL": url,
            "HTTP_Status": probe["HTTP_Status"],
            "Access_Status": probe["Access_Status"],
            "Access_Note": probe["Access_Note"],
        })

    pipeline_df = pd.DataFrame(access_rows)
    n_ok = (pipeline_df["Access_Status"] == "ACCESSIBLE").sum()
    access_status_lbl.value = (
        f"<span style='color:#15803d;'><b>✅ Access Check Complete: {n_ok}/{total} sites directly accessible; "
        f"{total - n_ok} flagged for fallback search.</b></span>"
    )
    access_btn.disabled = False

    with access_table_out:
        clear_output()
        display(HTML(pipeline_df.to_html(index=False)))

access_btn.on_click(on_access_check_clicked)
display(widgets.VBox([widgets.HBox([access_btn, access_status_lbl]), access_table_out]))
