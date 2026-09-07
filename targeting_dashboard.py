# ============================================================
# TARGETING DASHBOARD — Lec05 MKTG
# sklearn backend — robust to singular matrices
# ============================================================
import pandas as pd
import numpy as np
import io
import base64
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

# Sortable tables
try:
    from itables import show
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "itables==2.2.5", "-q"])
    from itables import show
    
# --- GLOBALS ---
df = None
model_obj = None
feature_names = None
scored_df = None

# --- WIDGETS ---
upload_widget = widgets.FileUpload(accept='.csv', multiple=False, description='Upload CSV')

y_dropdown = widgets.Dropdown(description='Target (Y):', options=[], layout=widgets.Layout(width='60%'))
target_value_dropdown = widgets.Dropdown(description='Target Value = 1:', options=[], layout=widgets.Layout(width='60%'))
x_selector = widgets.SelectMultiple(description='Predictors (X):', options=[], layout=widgets.Layout(width='60%', height='150px'))
nonmetric_selector = widgets.SelectMultiple(description='Categorical X (optional):', options=[], layout=widgets.Layout(width='60%', height='100px'))

confirm_vars_button = widgets.Button(description='Confirm Selections', button_style='info', layout=widgets.Layout(width='200px'))
var_status = widgets.HTML(value='<p style="color:#666;"><i>Upload a CSV first.</i></p>')

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
        widgets.HTML('<p>Choose target and predictors. <b>Categorical selection is optional</b> — leave empty if all Xs are numeric.</p>'),
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
        
        cols = list(df.columns)
        y_dropdown.options = cols
        x_selector.options = cols
        nonmetric_selector.options = cols
        
        if len(cols) > 1:
            y_dropdown.value = cols[0]
            x_selector.value = tuple(c for c in cols if c != cols[0])
        
        # KEY FIX: Categorical defaults to NONE selected
        nonmetric_selector.value = ()
        
        update_target_value_options(None)
        var_status.value = '<p style="color:green;"><b>Data loaded. Go to "Select Variables" tab.</b></p>'

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
    
    issues = []
    if y_col in x_cols:
        issues.append("Target cannot also be a predictor.")
    if len(x_cols) == 0:
        issues.append("Select at least one predictor.")
    bad_cats = set(nonmetric_cols) - set(x_cols)
    if len(bad_cats) > 0:
        issues.append(f"Categoricals not in predictors: {bad_cats}")
    
    with output_varselect:
        clear_output()
        if issues:
            display(HTML('<p style="color:red;"><b>Please fix:</b><br>' + '<br>'.join(issues) + '</p>'))
        else:
            cont_cols = [c for c in x_cols if c not in nonmetric_cols]
            summary = f"""
            <p style="color:green;"><b>Selections confirmed. Ready to run.</b></p>
            <ul>
            <li><b>Target:</b> {y_col} = "{target_val}" → 1</li>
            <li><b>Predictors:</b> {len(x_cols)}</li>
            <li><b>Categorical:</b> {', '.join(nonmetric_cols) if nonmetric_cols else '<i>None</i>'}</li>
            <li><b>Continuous:</b> {', '.join(cont_cols) if cont_cols else '<i>None</i>'}</li>
            </ul>"""
            display(HTML(summary))
            tabs.selected_index = 2

confirm_vars_button.on_click(confirm_selections)

def _build_design(dframe, x_cols, nonmetric_cols):
    """Return design matrix X (no constant) and feature name list."""
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
        raise ValueError("No valid predictors.")
    
    X = pd.concat(X_parts, axis=1)
    return X.astype(float)

def _fit_logit_sklearn(X_train, y_train):
    """Fit with sklearn — moderate L2 regularization for stable coefficients."""
    model = LogisticRegression(
        penalty='l2',
        C=1.0,  # moderate regularization
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def _get_p_values(model, X, y):
    """Approximate p-values via Wald test using Hessian approximation."""
    coef = model.coef_[0]
    proba = model.predict_proba(X)[:, 1]
    W = np.diag(proba * (1 - proba))
    # Hessian ≈ X.T @ W @ X + (1/C)*I  (with L2 penalty)
    hessian = X.T @ W @ X + (1 / 1e10) * np.eye(X.shape[1])
    try:
        cov = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(hessian)
    se = np.sqrt(np.diag(cov))
    z = coef / se
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return p, se

def run_model(b):
    global model_obj, feature_names, scored_df
    
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
            print("Select variables first.")
        return
    
    # Build Y
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
    
    if y.nunique() < 2:
        with output_results:
            clear_output()
            print("Target has only one class. Cannot train.")
        return
    
    # Build X
    try:
        X = _build_design(dff, x_cols, nonmetric_cols)
    except Exception as e:
        with output_results:
            clear_output()
            print(f"Error: {e}")
        return
    
    feature_names = list(X.columns)
    
    # Train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit model (sklearn — never crashes on singular matrix)
    model_obj = _fit_logit_sklearn(X_train, y_train)
    
    # Predictions
    y_pred_prob = model_obj.predict_proba(X_test)[:, 1]
    y_pred = model_obj.predict(X_test)
    
    # P-values
    p_values, std_errors = _get_p_values(model_obj, X_train.values, y_train.values)
    
    # --- TAB 3: PREDICTORS ---
    with output_results:
        clear_output()
        display(HTML(f'<p><b>Target:</b> {y_col} = "{target_value_dropdown.value}" → 1 | <b>In segment:</b> {y.sum()} | <b>Not in segment:</b> {len(y)-y.sum()}</p>'))
        
        summ = pd.DataFrame({
            'Variable': feature_names,
            'Coefficient': model_obj.coef_[0],
            'Std_Error': std_errors,
            'P_Value': p_values,
            'Odds_Ratio': np.exp(model_obj.coef_[0]),
            'Significant': ['***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else '' 
                           for p in p_values]
        })
        display(HTML('<h4>Which observables predict segment membership?</h4>'))
        try:
            show(summ, paging=False, buttons=['copy','csv'], classes="display compact")
        except Exception:
            display(summ.style.format({
                'Coefficient': '{:.3f}', 'Std_Error': '{:.3f}', 'P_Value': '{:.4f}', 'Odds_Ratio': '{:.2f}'
            }).set_properties(**{'text-align': 'left'}))
        
        sig = summ[summ['P_Value'] < 0.1].copy()
        if len(sig) > 0:
            sig = sig.sort_values('Odds_Ratio', ascending=True)
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
        display(HTML('<h4>Model Accuracy</h4>'))
        
        train_acc = model_obj.score(X_train, y_train) * 100
        test_acc = model_obj.score(X_test, y_test) * 100
        
        display(HTML(f"""
        <table style="width:55%; border-collapse:collapse; margin:12px 0; font-size:1.05em;">
          <tr style="background:#f2f2f2;">
            <th style="border:1px solid #ccc; padding:10px; text-align:left;">Dataset</th>
            <th style="border:1px solid #ccc; padding:10px; text-align:center;">Accuracy</th>
          </tr>
          <tr>
            <td style="border:1px solid #ccc; padding:10px;">Training (model learned from this)</td>
            <td style="border:1px solid #ccc; padding:10px; text-align:center; font-weight:bold; color:#003366;">{train_acc:.1f}%</td>
          </tr>
          <tr>
            <td style="border:1px solid #ccc; padding:10px;">Test (held-out, never seen before)</td>
            <td style="border:1px solid #ccc; padding:10px; text-align:center; font-weight:bold; color:#003366;">{test_acc:.1f}%</td>
          </tr>
        </table>
        <p style="color:#666; font-size:0.9em;"><i>Test accuracy is the number that matters for targeting new customers.</i></p>
        """))
        
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=['Actually Not Target', 'Actually Target'],
                             columns=['Predicted Not Target', 'Predicted Target'])
        display(HTML('<p><b>Confusion Matrix (Test Set):</b></p>'))
        display(cm_df)
    
    # --- TAB 5: SCORE & RANK ---
    try:
        X_full = _build_design(df, x_cols, nonmetric_cols)
    except Exception as e:
        with output_scoring:
            clear_output()
            print(f"Error: {e}")
        return
    
    probs = model_obj.predict_proba(X_full)[:, 1]
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
