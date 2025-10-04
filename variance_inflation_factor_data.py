from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd


'''
VIF (Variance Inflation Factor) Class
--------------------------------------
This class computes VIF values for features in a given DataFrame.
It helps detect multicollinearity in regression problems.

Usage:
------
from vif_module import VIF   # if saved as vif_module.py

vif_calc = VIF(X_train)      # pass training features
vif_df = vif_calc.get_vif()  # get DataFrame with VIF values

'''


class VIF:

    def __init__(self, X_train = pd.DataFrame()):

        self.X_train = X_train
        self.vif = pd.DataFrame(columns=['Features','VIF'])
        
    def get_vif_values(self):

        self.vif['Features'] = self.X_train.columns
        self.vif['VIF'] = [
            variance_inflation_factor(self.X_train.values,i) 
            for i in range (self.X_train.shape[1])
        ]
        self.vif ['VIF'] = round(self.vif['VIF'],2)
        self.vif = self.vif.sort_values(by=['VIF'],ascending=False)

        return self.vif
    