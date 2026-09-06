# ============================================================
# TARGETING DASHBOARD — Lec05 MKTG
# Self-contained Colab UI for binary logit targeting
# ============================================================

import pandas as pd
import numpy as np
import io
import base64
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
import statsmodels.api as sm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# --- GLOBALS ---
df = None
model_results = None
scored_df = None

# --- WIDGETS ---
upload_widget = widgets.FileUpload(accept='.csv', multiple=False, description='Upload CSV')

# Variable selection widgets
y_dropdown = widgets.Dropdown(description='Target (Y):', options=[], layout=widgets.Layout(width='60%'))
target_value_dropdown = widgets.Dropdown(description='Target Value = 1:', options=[], layout=widgets.Layout(width='60%'))
x_selector = widgets.SelectMultiple(description='Predictors (X):', options=[], layout=widgets.Layout(width='60%', height='150px'))
nonmetric_selector = widgets.SelectMultiple(description='Categorical X:', options=[], layout=widgets.Layout(width='60%', height='120px'))

confirm_vars_button = widgets.Button(description='Confirm Selections', button_style='info', layout=widgets.Layout(width='200px'))
var_status = widgets.HTML(value='<p style="color:#666;"><i>Upload a CSV first, then select variables here.</i></p>')

run_button = widgets.Button(description='Run Targeting Model', button_style='success', layout=widgets.Layout(width='200px'))
threshold_slider = widgets.FloatSlider(value=0.5, min=0.1, max=0.9, step=0.05, description='Threshold:', readout_format='.0%')
download_button = widgets.Button(description='Download Scored CSV', button_style='primary', layout=widgets.Layout(width='200px'))

output_upload = widgets.Output()
output_varselect = widgets.Output()
output_results = widgets.Output()
output_accuracy = widgets.Output()
output_scoring = widgets.Output()

tabs = widgets.Tab(children=[
    widgets.VBox([widgets.HTML('<h3>Step 1: Upload Data</h3>'), upload_widget, output_upload]),
    widgets.VBox([
        widgets.HTML('<h3>Step 2: Select Variables</h3>'),
        widgets.HTML('<p>Choose your target segment, predictors, and which predictors are categorical (non-metric). Categorical variables will be one-hot encoded automatically.</p>'),
        y_dropdown, target_value_dropdown, x_selector, nonmetric_selector,
        confirm_vars_button, var_status, output_varselect
    ]),
    widgets.VBox([widgets.HTML('<h3>Step 3: Predictor Results</h3>'), run_button, output_results]),
    widgets.VBox([widgets.HTML('<h3>Step 4: Model Accuracy</h3>'), output_accuracy]),
    widgets.VBox([widgets.HTML('<h3>Step 5: Score & Rank Prospects</h3>'), threshold_slider, download_button, output_scoring])
])
tabs.set_title(0, 'Upload')
tabs.set_title(1, 'Select Variables')
tabs.set_title(2, 'Predictors')
tabs.set_title(3, 'Accuracy')
tabs.set_title(4, 'Score & Rank')

# --- CALLBACKS ---

def on_upload(change):
    global df
    if not upload_widget.value:
        return
    key = list(upload_widget.value.keys())[0]
    content = upload_widget.value[key]['content']
    df = pd.read_csv(io.BytesIO(content))
    
    with output_upload:
        clear_output()
        print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        display(df.head())
        
        # Populate all selectors
        cols = list(df.columns)
        y_dropdown.options = cols
        x_selector.options = cols
        nonmetric_selector.options = cols
        
        if len(cols) > 1:
            y_dropdown.value = cols[0]
            x_selector.value = tuple(c for c in cols if c != cols[0])
        
        update_target_value_options(None)
        var_status.value = '<p style="color:green;"><b>Data loaded. Go to "Select Variables" tab to configure your model.</b></p>'

upload_widget.observe(on_upload, names='value')

def update_target_value_options(change):
    if df is None or not y_dropdown.value:
        return
    y_col = y_dropdown.value
    unique_vals = df[y_col].dropna().unique().tolist()
    
    if len(unique_vals) == 2 and all(isinstance(v, (int, float, np.integer, np.floating)) for v in unique_vals):
        target_value_dropdown.options = ['(auto — already 0/1)']
        target_value_dropdown.value = '(auto — already 0/1)'
        target_value_dropdown.disabled = True
    else:
        target_value_dropdown.options = [str(v) for v in unique_vals]
        target_value_dropdown.value = str(unique_vals[0])
        target_value_dropdown.disabled = False

y_dropdown.observe(update_target_value_options, names='value')

def confirm_selections(b):
    y_col = y_dropdown.value
    x_cols = list(x_selector.value)
    nonmetric_cols = list(nonmetric_selector.value)
    target_val = target_value_dropdown.value
    
    # Validation
    issues = []
    if y_col in x_cols:
        issues.append("Target cannot also be a predictor.")
    if len(x_cols) == 0:
        issues.append("Select at least one predictor.")
    if len(set(x_cols) & set(nonmetric_cols)) != len(nonmetric_cols):
        issues.append("Categorical selections must be a subset of predictors.")
    
    with output_varselect:
        clear_output()
        if issues:
            display(HTML('<p style="color:red;"><b>Please fix:</b><br>' + '<br>'.join(issues) + '</p>'))
        else:
            cont_cols = [c for c in x_cols if c not in nonmetric_cols]
            summary = f"""
            <p style="color:green;"><b>Selections confirmed. Ready to run model.</b></p>
            <ul>
            <li><b>Target (Y):</b> {y_col} = "{target_val}" → 1, others → 0</li>
            <li><b>Predictors (X):</b> {len(x_cols)} selected</li>
            <li><b>Categorical (one-hot):</b> {', '.join(nonmetric_cols) if nonmetric_cols else 'None'}</li>
            <li><b>Continuous:</b> {', '.join(cont_cols) if cont_cols else 'None'}</li>
            </ul>
            <p><i>Click "Run Targeting Model" in the Predictors tab.</i></p>
            """
            display(HTML(summary))
            tabs.selected_index = 2  # Auto-switch to Predictors tab

confirm_vars_button.on_click(confirm_selections)

def _prepare_X(dframe, x_cols, nonmetric_cols):
    X_parts = []
    cont_cols = [c for c in x_cols if c not in nonmetric_cols]
    
    if cont_cols:
        X_cont = dframe[cont_cols].copy()
        for c in cont_cols:
            X_cont[c] = pd.to_numeric(X_cont[c], errors='coerce')
        X_parts.append(X_cont)
    
    cat_cols = [c for c in x_cols if c in nonmetric_cols]
    if cat_cols:
        X_cat = pd.get_dummies(dframe[cat_cols], drop_first=True)
        X_parts.append(X_cat)
    
    if not X_parts:
        raise ValueError("No valid predictor columns.")
    
    X = pd.concat(X_parts, axis=1)
    X = X.astype(float)
    X = sm.add_constant(X, has_constant='add')
    return X

def run_model(b):
    global model_results, scored_df
    
    if df is None:
        with output_results:
            clear_output()
            print("Upload a CSV first.")
        return
    
    y_col = y_dropdown.value
    x_cols = list(x_selector.value)
    nonmetric_cols = list(nonmetric_selector.value)
    
    if not x_cols:
        with output_results:
            clear_output()
            print("Select variables in the 'Select Variables' tab first.")
        return
    
    # Prep Y
    dff = df[[y_col] + x_cols].dropna()
    
    if target_value_dropdown.disabled or target_value_dropdown.value == '(auto — already 0/1)':
        y = dff[y_col].astype(int)
    else:
        target_val = target_value_dropdown.value
        if dff[y_col].dtype != 'object':
            try:
                target_val = type(dff[y_col].iloc[0])(target_val)
            except:
                pass
        y = (dff[y_col] == target_val).astype(int)
    
    # Prep X
    try:
        X = _prepare_X(dff, x_cols, nonmetric_cols)
    except Exception as e:
        with output_results:
            clear_output()
            print(f"Error preparing predictors: {e}")
        return
    
    if y.nunique() < 2:
        with output_results:
            clear_output()
            print("Target has only one class. Cannot train.")
        return
    
    # Train/test
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit logit
    try:
        model = sm.Logit(y_train, X_train)
        result = model.fit(disp=0, maxiter=100)
    except Exception as e:
        with output_results:
            clear_output()
            print(f"Model failed: {e}")
        return
    
    model_results = result
    y_pred_prob = result.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # --- TAB 3: PREDICTORS ---
    with output_results:
        clear_output()
        display(HTML(f'<p><b>Target:</b> {y_col} = "{target_value_dropdown.value}" → 1 | <b>In segment:</b> {y.sum()} | <b>Not in segment:</b> {len(y)-y.sum()}</p>'))
        
        summ = pd.DataFrame({
            'Variable': result.params.index,
            'Coefficient': result.params.values,
            'Std_Error': result.bse.values,
            'P_Value': result.pvalues.values,
            'Odds_Ratio': np.exp(result.params.values),
            'Significant': ['***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else '' 
                           for p in result.pvalues.values]
        })
        display(HTML('<h4>Which observables predict segment membership?</h4>'))
        display(summ.style.format({
            'Coefficient': '{:.3f}', 'Std_Error': '{:.3f}', 'P_Value': '{:.4f}', 'Odds_Ratio': '{:.2f}'
        }).set_properties(**{'text-align': 'left'}))
        
        sig = summ[summ['P_Value'] < 0.1].copy()
        if len(sig) > 0:
            sig = sig[sig['Variable'] != 'const'].sort_values('Odds_Ratio', ascending=True)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, max(3, len(sig)*0.4)))
            colors = ['green' if c > 0 else 'red' for c in sig['Coefficient']]
            ax.barh(sig['Variable'], sig['Odds_Ratio'], color=colors, alpha=0.7)
            ax.axvline(x=1, color='black', linestyle='--', linewidth=1)
            ax.set_xlabel('Odds Ratio (>1 = increases likelihood)')
            ax.set_title('Strongest Predictors')
            plt.tight_layout()
            plt.show()
    
    # --- TAB 4: ACCURACY ---
    with output_accuracy:
        clear_output()
        display(HTML('<h4>How accurately can we identify target customers?</h4>'))
        
        report = classification_report(y_test, y_pred, output_dict=True, target_names=['Not Target', 'Target'])
        metrics_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1-Score', 'Support'],
            'Not Target': [f"{report['Not Target']['precision']:.2f}", f"{report['Not Target']['recall']:.2f}",
                          f"{report['Not Target']['f1-score']:.2f}", f"{int(report['Not Target']['support'])}"],
            'Target': [f"{report['Target']['precision']:.2f}", f"{report['Target']['recall']:.2f}",
                      f"{report['Target']['f1-score']:.2f}", f"{int(report['Target']['support'])}"]
        })
        display(metrics_df.style.hide(axis='index'))
        
        try:
            auc = roc_auc_score(y_test, y_pred_prob)
            display(HTML(f'<p><b>AUC:</b> {auc:.2f} <small>(0.5=random, 0.8=good, 0.9=excellent)</small></p>'))
        except:
            pass
        
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=['Actually Not Target', 'Actually Target'],
                             columns=['Predicted Not Target', 'Predicted Target'])
        display(HTML('<p><b>Confusion Matrix:</b></p>'))
        display(cm_df)
    
    # --- TAB 5: SCORE & RANK ---
    try:
        X_full = _prepare_X(df, x_cols, nonmetric_cols)
    except Exception as e:
        with output_scoring:
            clear_output()
            print(f"Error: {e}")
        return
    
    probs = result.predict(X_full)
    scored_df = df.copy()
    scored_df['Likelihood_Score'] = probs
    scored_df['Rank'] = scored_df['Likelihood_Score'].rank(ascending=False, method='dense').astype(int)
    update_scoring(None)

def update_scoring(change):
    if scored_df is None:
        return
    thresh = threshold_slider.value
    top_n = int(len(scored_df) * (1 - thresh)) + 1
    top_df = scored_df.nlargest(top_n, 'Likelihood_Score')[['Rank', 'Likelihood_Score'] + list(x_selector.value)]
    
    with output_scoring:
        clear_output()
        display(HTML(f'<h4>Top {top_n} prospects (Score ≥ {thresh:.0%})</h4>'))
        display(top_df.head(20).style.format({'Likelihood_Score': '{:.1%}'}))
        display(HTML(f'<p>Top 20 shown. Download for full list.</p>'))

threshold_slider.observe(update_scoring, names='value')

def download_scored(b):
    if scored_df is None:
        return
    csv = scored_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    payload = f'<a href="data:text/csv;base64,{b64}" download="targeting_scored.csv">Download scored prospects</a>'
    with output_scoring:
        display(HTML(payload))

run_button.on_click(run_model)
download_button.on_click(download_scored)

# --- RENDER ---
display(HTML('<h2>Targeting Dashboard: Find Your Segment in the Wild</h2>'))
display(tabs)
