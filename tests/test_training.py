import unittest
import numpy as np
import pandas as pd
from src.train import feature_columns, make_pipeline, metric_values, load_data, ROOT
from sklearn.linear_model import LogisticRegression

class TrainingChecks(unittest.TestCase):
    def test_excluded_inputs(self):
        df=load_data(ROOT/'data/parkinsons_disease_data.csv')
        for scenario in ['without_assessments_symptoms','all_non_identifier_features']:
            cols=feature_columns(df,scenario)
            self.assertTrue({'Diagnosis','PatientID','DoctorInCharge'}.isdisjoint(cols))
        self.assertNotIn('UPDRS',feature_columns(df,'without_assessments_symptoms'))

    def test_metrics_against_hand_calculation(self):
        # TN=1, FP=1, FN=0, TP=2 -> recall=1, specificity=.5, balanced=.75
        m=metric_values([0,0,1,1],[.1,.9,.8,.7])
        self.assertEqual(m['confusion_matrix'],[[1,1],[0,2]])
        self.assertEqual(m['balanced_accuracy'],.75)
        self.assertEqual(m['recall'],1.)
        self.assertEqual(m['specificity'],.5)

    def test_scaler_does_not_use_holdout(self):
        train=pd.DataFrame({'Age':[50.,60.,70.,80.]})
        heldout=pd.DataFrame({'Age':[1000.]})
        pipe=make_pipeline(['Age'],LogisticRegression())
        pipe.fit(train,[0,0,1,1])
        scaler=pipe.named_steps['preprocess'].named_transformers_['numeric'].named_steps['scale']
        self.assertEqual(scaler.mean_[0],65.)
        pipe.predict_proba(heldout)
        self.assertEqual(scaler.mean_[0],65.)

if __name__=='__main__':unittest.main()
