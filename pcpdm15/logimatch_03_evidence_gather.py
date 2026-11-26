# logimatch_03_evidence_gather.py
# Cell 4: Tool 2 & Tool 3 — Single-Site Tool Demo (Tab 1) + Roster Evidence Fetch & Fallback Recovery (Tab 2)

def fetch_site_text(url: str) -> str:
    """Tool 2: Scrapes clean operational text directly from an accessible website."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(str(url).strip(), headers=headers, timeout=10, allow_redirects=True)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        blocks = [el.get_text(" ", strip=True) for el in soup.find_all(["h1", "h2", "h3", "p", "li"])]
        cleaned = " | ".join([b for b in blocks if len(b.split()) >= 4])
        if len(cleaned) < 80:
            # Fallback for div/span-heavy modern layouts (e.g., raghuvamsi.com)
            cleaned = " ".join(soup.get_text(" ", strip=True).split())
        return cleaned[:3000] if len(cleaned) >= 80 else ""
    except Exception:
        return ""

def search_web_fallback(company_name: str, city: str = "") -> str:
    """Tool 3: Fallback DuckDuckGo search when direct site fetch is blocked or empty."""
    query = f"{company_name} {city} products manufacturing services".strip()
    try:
        results = list(DDGS().text(query, max_results=3))
        if not results:
            return ""
        return " | ".join([f"{r.get('title', '')}: {r.get('body', '')}" for r in results])[:2000]
    except Exception:
        return ""

# --- Tab 1: Single-Site Tool Demo (Inspect what the AI Brain actually receives) ---
if "pipeline_df" in globals() and pipeline_df is not None:
    accessible_firms = list(pipeline_df.loc[pipeline_df["Access_Status"] == "ACCESSIBLE", "Firm_Name"])
    all_firms = list(pipeline_df["Firm_Name"])
    demo_choices = accessible_firms if accessible_firms else all_firms
else:
    demo_choices = ["(Run Cell 3 first)"]

dd_demo_firm = widgets.Dropdown(
    options=demo_choices,
    description="Demo Lead:",
    style={"description_width": "90px"},
    layout=widgets.Layout(width="350px"),
)
btn_demo_tool = widgets.Button(
    description="Test Tools on Selected Lead",
    button_style="info",
    icon="eye",
    layout=widgets.Layout(width="240px"),
)
out_demo_tool = widgets.HTML(
    value="<i>Select an accessible lead and click <b>Test Tools on Selected Lead</b> to inspect the exact text extracted for the AI brain.</i>"
)

# --- Tab 2: Batch Roster Fetch + Fallback Recovery ---
btn_gather_all = widgets.Button(
    description="Gather Evidence for All Leads",
    button_style="primary",
    icon="download",
    layout=widgets.Layout(width="260px"),
)
lbl_gather_status = widgets.HTML(
    value="<i>Click to scrape accessible sites and run backup web search on blocked sites.</i>"
)
out_gather_table = widgets.Output()

tabs_c4 = widgets.Tab(children=[
    widgets.VBox([widgets.HBox([dd_demo_firm, btn_demo_tool]), out_demo_tool]),
    widgets.VBox([widgets.HBox([btn_gather_all, lbl_gather_status]), out_gather_table]),
])
tabs_c4.set_title(0, "1. Single-Site Tool Demo")
tabs_c4.set_title(1, "2. Batch Evidence & Fallback")

def on_demo_tool(_):
    if "pipeline_df" not in globals() or pipeline_df is None:
        out_demo_tool.value = "<span style='color:#b91c1c;'><b>⚠️ Run Cell 3 first.</b></span>"
        return
    firm = dd_demo_firm.value
    row = pipeline_df[pipeline_df["Firm_Name"] == firm].iloc[0]
    url = row["Website_URL"]
    out_demo_tool.value = f"<i>⏳ Running Tool 2 (<code>fetch_site_text</code>) on <b>{firm}</b>...</i>"

    txt = fetch_site_text(url)
    tool_used = "Tool 2: fetch_site_text()"
    if not txt:
        txt = search_web_fallback(firm)
        tool_used = "Tool 2 (Blocked) → Tool 3: search_web_fallback()"

    if txt:
        out_demo_tool.value = f"""
        <div style='border:1px solid #cbd5e1; border-radius:6px; padding:12px; margin-top:8px; background:#fff;'>
            <p><b>Lead:</b> {firm} &nbsp;|&nbsp; <b>Tool Used:</b> <code>{tool_used}</code> &nbsp;|&nbsp; <b>Extracted Length:</b> {len(txt)} chars</p>
            <div style='background:#f8fafc; border:1px solid #e2e8f0; padding:10px; max-height:200px; overflow-y:auto; font-size:0.9em;'>{txt}</div>
        </div>"""
    else:
        out_demo_tool.value = (
            f"<span style='color:#b45309;'><b>⚠️ Both tools returned empty for {firm} (DEAD_LEAD_HOLD).</b></span>"
        )

def on_gather_all(_):
    global pipeline_df
    if "pipeline_df" not in globals() or pipeline_df is None:
        lbl_gather_status
