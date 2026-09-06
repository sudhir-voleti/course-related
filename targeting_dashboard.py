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
y_dropdown = widgets.Dropdown(description='Target Column:', options=[], layout=widgets.Layout(width='50%'))
target_value_dropdown = widgets.Dropdown(description='Target Value (1):', options=[], layout=widgets.Layout(width='50%'))

x_selector = widgets.SelectMultiple(description='Predictors (X):', options=[], layout=widgets.Layout(width='50%', height='120px'))
nonmetric_selector = widgets.SelectMultiple(description='Non-Metric X:', options=[], layout=widgets.Layout(width='50%', height='100px'))

run_button = widgets.Button(description='Run Targeting Model', button_style='success', layout=widgets.Layout(width='200px'))
threshold_slider = widgets.FloatSlider(value=0.5, min=0.1, max=0.9, step=0.05, description='Threshold:', readout_format='.0%')
download_button = widgets.Button(description='Download Scored CSV', button_style='primary', layout=widgets.Layout(width='200px'))

output_upload = widgets.Output()
output_results = widgets.Output()
output_accuracy = widgets.Output()
output_scoring = widgets.Output()

tabs = widgets.Tab(children=[
    widgets.VBox([widgets.HTML('<h3>Step 1: Upload your discriminant data</h3>'), upload_widget, output_upload]),
    widgets.VBox([
        widgets.HTML('<h3>Step 2: Select Target, Predictors, and Non-Metric Flags</h3>'),
        widgets.HTML('<small>Select which X variables are categorical (non-metric). These will be one-hot encoded. Others treated as continuous.</small>'),
        y_dropdown, target_value_dropdown, x_selector, nonmetric_selector, run_button, output_results
    ]),
    widgets.VBox([widgets.HTML('<h3>Step 3: Model Accuracy</h3>'), output_accuracy]),
    widgets.VBox([widgets.HTML('<h3>Step 4: Score & Rank Prospects</h3>'), threshold_slider, download_button, output_scoring])
])
tabs.set_title(0, 'Upload')
tabs.set_title(1, 'Predictors')
tabs.set_title(2, 'Accuracy')
tabs.set_title(3, 'Score & Rank')

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
        
        # Populate selectors
        cols = list(df.columns)
        y_dropdown.options = cols
        x_selector.options = cols
        nonmetric_selector.options = cols
        if len(cols) > 1:
            y_dropdown.value = cols[0]
            x_selector.value = tuple(c for c in cols if c != cols[0])
        update_target_value_options(None)

upload_widget.observe(on_upload, names='value')

def update_target_value_options(change):
    """Populate target value dropdown based on Y column's unique values."""
    if df is None or not y_dropdown.value:
        return
    y_col = y_dropdown.value
    unique_vals = df[y_col].dropna().unique().tolist()
    
    if len(unique_vals) == 2 and all(isinstance(v, (int, float, np.integer, np.floating)) for v in unique_vals):
        target_value_dropdown.options = ['(auto — binary numeric)']
        target_value_dropdown.value = '(auto — binary numeric)'
        target_value_dropdown.disabled = True
    else:
        target_value_dropdown.options = [str(v) for v in unique_vals]
        target_value_dropdown.value = str(unique_vals[0])
        target_value_dropdown.disabled = False

y_dropdown.observe(update_target_value_options, names='value')

def _prepare_X(dframe, x_cols, nonmetric_cols):
    """Build design matrix: one-hot non-metric, keep continuous as-is."""
    X_parts = []
    
    # Continuous variables (not in nonmetric list)
    cont_cols = [c for c in x_cols if c not in nonmetric_cols]
    if cont_cols:
        X_cont = dframe[cont_cols].copy()
        # Force numeric
        for c in cont_cols:
            X_cont[c] = pd.to_numeric(X_cont[c], errors='coerce')
        X_parts.append(X_cont)
    
    # Non-metric (categorical) variables
    cat_cols = [c for c in x_cols if c in nonmetric_cols]
    if cat_cols:
        X_cat = pd.get_dummies(dframe[cat_cols], drop_first=True)
        X_parts.append(X_cat)
    
    if not X_parts:
        raise ValueError("No valid predictor columns selected.")
    
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
    
    if y_col in x_cols:
        with output_results:
            clear_output()
            print("Remove the target variable from predictors.")
        return
    
    if len(x_cols) == 0:
        with output_results:
            clear_output()
            print("Select at least one predictor.")
        return
    
    # Prep Y
    dff = df[[y_col] + x_cols].dropna()
    
    if target_value_dropdown.disabled or target_value_dropdown.value == '(auto — binary numeric)':
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
            print("Target variable has only one class after filtering. Cannot train model.")
        return
    
    # Train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError as e:
        # Fallback if stratify fails (e.g., too few in one class)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit logit
    try:
        model = sm.Logit(y_train, X_train)
        result = model.fit(disp=0, maxiter=100)
    except Exception as e:
        with output_results:
            clear_output()
            print(f"Model failed to converge: {e}")
        return
    
    model_results = result
    
    # Predictions
    y_pred_prob = result.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # --- TAB 2: PREDICTORS ---
    with output_results:
        clear_output()
        
        display(HTML(f'<p><b>Target:</b> {y_col} = "{target_value_dropdown.value}" → 1 | <b>N in segment:</b> {y.sum()} | <b>N not in segment:</b> {len(y)-y.sum()}</p>'))
        if nonmetric_cols:
            display(HTML(f'<p><small>One-hot encoded: {", ".join(nonmetric_cols)} | Continuous: {", ".join([c for c in x_cols if c not in nonmetric_cols])}</small></p>'))
        
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
            'Coefficient': '{:.3f}',
            'Std_Error': '{:.3f}',
            'P_Value': '{:.4f}',
            'Odds_Ratio': '{:.2f}'
        }).set_properties(**{'text-align': 'left'}))
        
        sig = summ[summ['P_Value'] < 0.1].copy()
        if len(sig) > 0:
            sig = sig[sig['Variable'] != 'const'].sort_values('Odds_Ratio', ascending=True)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, max(3, len(sig)*0.4)))
            colors = ['green' if c > 0 else 'red' for c in sig['Coefficient']]
            ax.barh(sig['Variable'], sig['Odds_Ratio'], color=colors, alpha=0.7)
            ax.axvline(x=1, color='black', linestyle='--', linewidth=1)
            ax.set_xlabel('Odds Ratio (>1 = increases likelihood of being in target segment)')
            ax.set_title('Strongest Predictors of Target Segment Membership')
            plt.tight_layout()
            plt.show()
    
    # --- TAB 3: ACCURACY ---
    with output_accuracy:
        clear_output()
        display(HTML('<h4>How accurately can we identify target customers?</h4>'))
        
        report = classification_report(y_test, y_pred, output_dict=True, target_names=['Not Target', 'Target'])
        metrics_df = pd.DataFrame({
            'Metric': ['Precision (of those we target, how many are real?)', 
                       'Recall (of real targets, how many do we find?)', 
                       'F1-Score', 'Support (count in test set)'],
            'Not Target': [f"{report['Not Target']['precision']:.2f}", 
                          f"{report['Not Target']['recall']:.2f}",
                          f"{report['Not Target']['f1-score']:.2f}",
                          f"{int(report['Not Target']['support'])}"],
            'Target': [f"{report['Target']['precision']:.2f}", 
                      f"{report['Target']['recall']:.2f}",
                      f"{report['Target']['f1-score']:.2f}",
                      f"{int(report['Target']['support'])}"]
        })
        display(metrics_df.style.hide(axis='index'))
        
        try:
            auc = roc_auc_score(y_test, y_pred_prob)
            display(HTML(f'<p><b>Overall Model Quality (AUC):</b> {auc:.2f} <br><small>(0.5 = random, 0.8 = good, 0.9 = excellent)</small></p>'))
        except:
            pass
        
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=['Actually Not Target', 'Actually Target'], 
                             columns=['Predicted Not Target', 'Predicted Target'])
        display(HTML('<p><b>Confusion Matrix:</b></p>'))
        display(cm_df)
    
    # --- TAB 4: SCORE & RANK (on full data) ---
    try:
        X_full = _prepare_X(df, x_cols, nonmetric_cols)
    except Exception as e:
        with output_scoring:
            clear_output()
            print(f"Error scoring full data: {e}")
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
        display(HTML(f'<h4>Top {top_n} prospects to target (Likelihood Score ≥ {thresh:.0%})</h4>'))
        display(top_df.head(20).style.format({'Likelihood_Score': '{:.1%}'}))
        display(HTML(f'<p>Showing top 20 of {top_n}. Click Download for full list.</p>'))

threshold_slider.observe(update_scoring, names='value')

def download_scored(b):
    if scored_df is None:
        return
    csv = scored_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    payload = f'<a href="data:text/csv;base64,{b64}" download="targeting_scored.csv">Click here to download scored prospects</a>'
    with output_scoring:
        display(HTML(payload))

run_button.on_click(run_model)
download_button.on_click(download_scored)

# --- RENDER ---
display(HTML('<h2>Targeting Dashboard: Find Your Segment in the Wild</h2>'))
display(tabs)
