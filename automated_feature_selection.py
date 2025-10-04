import statsmodels.api as sm
import variance_inflation_factor_data as vif_class
import pandas as pd



def final_data(X_train,y_train):

    features = list(X_train.columns)
    
    while(True):

        features_data = get_p_values(X_train[features],y_train)

        vif_values  =  get_vif_values(X_train[features])

        features_data = pd.merge(features_data,vif_values,on='Features',how='inner')
        
        
        
        if ((features_data['VIF'] > 10).any() or (features_data['p-value'] > 0.05).any()):

            
            if (features_data['VIF'] > 10).any() and (features_data['p-value'] > 0.05).any():
                
                worst_feature = features_data.sort_values(['VIF','p-value'], ascending=False).iloc[0]["Features"]
                
            elif (features_data['p-value'] > 0.05).any():
                
                worst_feature = features_data.sort_values('p-value', ascending=False).iloc[0]["Features"]
    
            elif (features_data['VIF'] > 10).any():
                
                worst_feature = features_data.sort_values('VIF', ascending=False).iloc[0]["Features"]
                
        

        else:
            print(features_data)
            return features
            

        features.remove(worst_feature)


def get_vif_values(X_train):

    vif = vif_class.VIF(X_train)
    
    return vif.get_vif_values()


def get_p_values(X_train,y_train):

    X_train_sm = sm.add_constant(X_train)

    sm_lr = sm.GLM(y_train,X_train_sm,family= sm.families.Binomial())

    sm_lr = sm_lr.fit()

    p_values = pd.DataFrame(columns=['Features','p-value'])
    p_values['Features'] = X_train.columns
    p_values_cal = list(round(sm_lr.pvalues,4))
    p_values_cal.pop(0)
    p_values['p-value'] = p_values_cal

    return p_values

