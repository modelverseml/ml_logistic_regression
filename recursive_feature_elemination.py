from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


class RfeClass:

    def __init__(self,X_train,y_train,number_of_features):

        self.X_train = X_train
        self.number_of_fetaures = number_of_features
        self.y_train = y_train

    def get_rfe_output(self):

        lm = LogisticRegression()
        
        self.lm = lm.fit(self.X_train,self.y_train)

        rfe = RFE(self.lm,n_features_to_select=self.number_of_fetaures)

        self.rfe = rfe.fit(self.X_train,self.y_train)

        self.top_columns = self.X_train.columns[self.rfe.support_]

        return None