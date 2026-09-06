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
from sklearn.preprocessing import StandardScaler
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
y_dropdown = widgets.Dropdown(description='Target (Y):', options=[], layout=widgets.Layout(width='50%'))
x_selector = widgets.SelectMultiple(description='Predictors (X):', options=[], layout=widgets.Layout(width='50%', height='150px'))
run_button = widgets.Button(description='Run Targeting Model', button_style='success', layout=widgets.Layout(width='200px'))
threshold_slider = widgets.FloatSlider(value=0.5, min=0.1, max=0.9, step=0.05, description='Threshold:', readout_format='.0%')
download_button = widgets.Button(description='Download Scored CSV', button_style='primary', layout=widgets.Layout(width='200px'))

output_upload = widgets.Output()
output_results = widgets.Output()
output_accuracy = widgets.Output()
output_scoring = widgets.Output()

tabs = widgets.Tab(children=[
    widgets.VBox([widgets.HTML('<h3>Step 1: Upload your discriminant data</h3>'), upload_widget, output_upload]),
    widgets.VBox([widgets.HTML('<h3>Step 2: Select Target Segment and Observables</h3>'), y_dropdown, x_selector, run_button, output_results]),
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
        if len(cols) > 1:
            y_dropdown.value = cols[0]
            x_selector.value = tuple(c for c in cols if c != cols[0])

upload_widget.observe(on_upload, names='value')

def run_model(b):
    global model_results, scored_df
    
    if df is None:
        with output_results:
            clear_output()
            print("Upload a CSV first.")
        return
    
    y_col = y_dropdown.value
    x_cols = list(x_selector.value)
    
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
    
    # Prep data
    dff = df[[y_col] + x_cols].dropna()
    y = dff[y_col].astype(int)
    
    # Handle categorical X
    X = pd.get_dummies(dff[x_cols], drop_first=True)
    X = sm.add_constant(X)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fit logit
    try:
        model = sm.Logit(y_train, X_train)
        result = model.fit(disp=0)
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
        
        # Summary table
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
        
        # Top predictors chart
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
        
        # Metrics
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
        
        # AUC
        try:
            auc = roc_auc_score(y_test, y_pred_prob)
            display(HTML(f'<p><b>Overall Model Quality (AUC):</b> {auc:.2f} <br><small>(0.5 = random, 0.8 = good, 0.9 = excellent)</small></p>'))
        except:
            pass
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=['Actually Not Target', 'Actually Target'], 
                             columns=['Predicted Not Target', 'Predicted Target'])
        display(HTML('<p><b>Confusion Matrix:</b></p>'))
        display(cm_df)
    
    # --- TAB 4: SCORE & RANK (on full data) ---
    X_full = pd.get_dummies(df[x_cols], drop_first=True)
    # Align columns with training
    for col in X.columns:
        if col not in X_full.columns and col != 'const':
            X_full[col] = 0
    X_full = X_full[X.columns.drop('const') if 'const' in X.columns else X.columns]
    X_full = sm.add_constant(X_full, has_constant='add')
    
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
