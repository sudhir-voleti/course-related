# logimatch_01_data_uploader.py
# Cell 2: Interactive Dual-CSV Uploader, Auto-Preview, Variable Summary & Column Mapper

catalog_df = None
leads_df = None
col_map = {
    "sku_id": None,
    "sku_name": None,
    "signals": None,
    "disqualifiers": None,
    "firm": None,
    "url": None,
}

def _read_uploaded_csv(upload_widget):
    """Extracts the uploaded CSV bytes from ipywidgets FileUpload across Colab widget versions."""
    val = upload_widget.value
    if not val:
        return None, ""
    if isinstance(val, tuple):  # ipywidgets >= 8
        entry = val[0]
        fname = entry["name"]
        content = bytes(entry["content"])
    elif isinstance(val, dict):  # ipywidgets 7
        fname = list(val.keys())[0]
        content = val[fname]["content"]
    else:
        return None, ""
    return pd.read_csv(io.BytesIO(content)), fname

def _guess_col(columns, keywords, fallback_idx=0):
    for kw in keywords:
        for col in columns:
            if kw in col.lower():
                return col
    return columns[min(fallback_idx, len(columns) - 1)]

def _build_var_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Column": df.columns,
        "Type": [str(t) for t in df.dtypes],
        "Non-Null": [int(df[c].notna().sum()) for c in df.columns],
        "Sample Value": [str(df[c].dropna().iloc[0])[:60] if df[c].notna().any() else "—" for c in df.columns],
    })

# --- Tab 1: Offering Catalog CSV ---
up_cat = widgets.FileUpload(accept=".csv", multiple=False, description="Choose Catalog CSV", layout=widgets.Layout(width="230px"))
lbl_cat_status = widgets.HTML(value="<i>Upload your offering matrix CSV (e.g., <code>catalog.csv</code>). Preview renders automatically.</i>")
out_cat_preview = widgets.Output()

# --- Tab 2: Prospect Leads CSV ---
up_leads = widgets.FileUpload(accept=".csv", multiple=False, description="Choose Leads CSV", layout=widgets.Layout(width="230px"))
lbl_leads_status = widgets.HTML(value="<i>Upload your target prospect list CSV (e.g., <code>leads_input.csv</code>). Preview renders automatically.</i>")
out_leads_preview = widgets.Output()

# --- Tab 3: Column Mapping Confirmation ---
out_mapper = widgets.Output()

tabs_c2 = widgets.Tab(children=[
    widgets.VBox([widgets.HBox([up_cat, lbl_cat_status]), out_cat_preview]),
    widgets.VBox([widgets.HBox([up_leads, lbl_leads_status]), out_leads_preview]),
    out_mapper,
])
tabs_c2.set_title(0, "1. Offering Catalog CSV")
tabs_c2.set_title(1, "2. Prospect Leads CSV")
tabs_c2.set_title(2, "3. Column Mapping")

def _refresh_mapper():
    with out_mapper:
        clear_output()
        if catalog_df is None or leads_df is None:
            display(HTML("<i>Upload both CSV files in Tabs 1 and 2 to confirm column mappings.</i>"))
            return

        cat_cols = list(catalog_df.columns)
        none_cat_cols = ["(None)"] + cat_cols
        lead_cols = list(leads_df.columns)

        col_map["sku_id"] = _guess_col(cat_cols, ["id", "sku", "code"], 0)
        col_map["sku_name"] = _guess_col(cat_cols, ["name", "title", "service", "offering", "product"], 1)
        col_map["signals"] = _guess_col(cat_cols, ["signal", "fit", "target", "use", "desc", "ideal"], 2)
        guessed_disq = _guess_col(none_cat_cols, ["disqual", "exclude", "avoid", "not", "contra"], 0)
        col_map["disqualifiers"] = None if guessed_disq == "(None)" else guessed_disq

        col_map["firm"] = _guess_col(lead_cols, ["firm", "company", "lead", "name", "client", "org"], 0)
        col_map["url"] = _guess_col(lead_cols, ["url", "website", "site", "web", "domain", "link"], 1)

        style = {"description_width": "145px"}
        w_layout = widgets.Layout(width="380px")

        dd_sku_id = widgets.Dropdown(options=cat_cols, value=col_map["sku_id"], description="Catalog Item ID:", style=style, layout=w_layout)
        dd_sku_name = widgets.Dropdown(options=cat_cols, value=col_map["sku_name"], description="Catalog Item Name:", style=style, layout=w_layout)
        dd_signals = widgets.Dropdown(options=cat_cols, value=col_map["signals"], description="Qualifying Signals:", style=style, layout=w_layout)
        dd_disq = widgets.Dropdown(options=none_cat_cols, value=guessed_disq, description="Disqualifiers:", style=style, layout=w_layout)
        dd_firm = widgets.Dropdown(options=lead_cols, value=col_map["firm"], description="Lead Firm Name:", style=style, layout=w_layout)
        dd_url = widgets.Dropdown(options=lead_cols, value=col_map["url"], description="Lead Website URL:", style=style, layout=w_layout)
        lbl_map_ok = widgets.HTML(value="<span style='color:#15803d;'><b>✅ Column mappings auto-detected and active. Adjust dropdowns only if needed.</b></span>")

        def _sync_map(_=None):
            col_map["sku_id"] = dd_sku_id.value
            col_map["sku_name"] = dd_sku_name.value
            col_map["signals"] = dd_signals.value
            col_map["disqualifiers"] = None if dd_disq.value == "(None)" else dd_disq.value
            col_map["firm"] = dd_firm.value
            col_map["url"] = dd_url.value
            lbl_map_ok.value = f"<span style='color:#15803d;'><b>✅ Mapping updated:</b> Catalog [{col_map['sku_id']} / {col_map['sku_name']}] &amp; Leads [{col_map['firm']} / {col_map['url']}]</span>"

        for w in [dd_sku_id, dd_sku_name, dd_signals, dd_disq, dd_firm, dd_url]:
            w.observe(_sync_map, names="value")

        display(widgets.VBox([
            widgets.HTML("<b>Catalog Schema Mapping:</b>"),
            dd_sku_id, dd_sku_name, dd_signals, dd_disq,
            widgets.HTML("<hr><b>Leads Schema Mapping:</b>"),
            dd_firm, dd_url,
            lbl_map_ok,
        ]))

def _on_cat_upload(change):
    global catalog_df
    df, fname = _read_uploaded_csv(up_cat)
    if df is None:
        return
    catalog_df = df
    lbl_cat_status.value = f"<span style='color:#15803d;'><b>✅ Loaded <code>{fname}</code></b> ({len(df)} rows × {len(df.columns)} cols)</span>"
    with out_cat_preview:
        clear_output()
        display(HTML("<b>Top 5 Rows Preview:</b>" + df.head(5).to_html(index=False)))
        display(HTML("<br><b>Variable Summary:</b>" + _build_var_summary(df).to_html(index=False)))
    _refresh_mapper()

def _on_leads_upload(change):
    global leads_df
    df, fname = _read_uploaded_csv(up_leads)
    if df is None:
        return
    leads_df = df
    lbl_leads_status.value = f"<span style='color:#15803d;'><b>✅ Loaded <code>{fname}</code></b> ({len(df)} rows × {len(df.columns)} cols)</span>"
    with out_leads_preview:
        clear_output()
        display(HTML("<b>Top 5 Rows Preview:</b>" + df.head(5).to_html(index=False)))
        display(HTML("<br><b>Variable Summary:</b>" + _build_var_summary(df).to_html(index=False)))
    _refresh_mapper()

up_cat.observe(_on_cat_upload, names="value")
up_leads.observe(_on_leads_upload, names="value")
_refresh_mapper()

display(tabs_c2)
