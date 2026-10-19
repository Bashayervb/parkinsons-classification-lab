"""Read-only explorer for saved experiment results; no patient input form."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title="Parkinson's Classification Lab", page_icon='📊', layout='wide')
st.title("Parkinson's Classification Lab")
st.caption('Educational dataset experiment · not a diagnostic or screening service')
metrics_path=ROOT/'reports/metrics.json'
if not metrics_path.exists():
    st.info('Run `python -m src.train` first to generate the experiment reports.')
    st.stop()
report=json.loads(metrics_path.read_text())
st.info('Synthetic educational data, as stated by the publisher. Model scores are not individual clinical risks.')
a,b,c=st.columns(3)
a.metric('Dataset records',f"{report['rows']:,}");b.metric('Training records',report['training_rows']);c.metric('Held-out records',report['test_rows'])
labels={'without_assessments_symptoms':'Without symptom / assessment columns','all_non_identifier_features':'All non-identifier features'}
scenario=st.selectbox('Predeclared feature scenario',list(labels),format_func=labels.get)
r=report['scenarios'][scenario]
st.write('Selected model:',r['selected_model'],'· Input columns:',r['feature_count'])
st.caption('Selection uses training-only five-fold average precision. Fixed threshold = 0.5.')
a,b,c,d=st.columns(4)
a.metric('Holdout average precision',f"{r['test']['average_precision']:.3f}")
b.metric('Baseline average precision',f"{r['baseline_test']['average_precision']:.3f}")
c.metric('Recall',f"{r['test']['recall']:.3f}");d.metric('Specificity',f"{r['test']['specificity']:.3f}")
cv=pd.read_csv(ROOT/'reports/cross_validation.csv');cv=cv[cv.scenario==scenario]
st.subheader('Training cross-validation results')
st.dataframe(cv.groupby('model')[['ap','roc','balanced']].agg(['mean','std']),use_container_width=True)
st.subheader('Holdout confusion matrix')
st.dataframe(pd.DataFrame(r['test']['confusion_matrix'],index=['Reference 0','Reference 1'],columns=['Predicted 0','Predicted 1']))
st.subheader('Inspect an anonymous held-out example')
pred=pd.read_csv(ROOT/'reports/holdout_predictions.csv');pred=pred[pred.scenario==scenario]
case=st.slider('Example index (not a patient identifier)',1,len(pred),1)
row=pred.iloc[case-1]
st.write({'dataset_label':int(row.dataset_label),'predicted_label':int(row.predicted_label),'model_score':round(float(row.model_score),3)})
st.caption('This example comes from the fixed test set; it is not a new patient assessment.')
with st.expander('Feature columns'):st.write(r['features'])
with st.expander('Methodology and limitations'):st.markdown((ROOT/'reports/summary.md').read_text())
st.download_button('Download experiment metrics',metrics_path.read_text(),file_name='metrics.json',mime='application/json')
