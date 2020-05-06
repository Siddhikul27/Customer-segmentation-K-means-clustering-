
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import datetime as dt 
 

data= pd.read_csv('C:/Users/User/Desktop/Bicycle_new.csv')
#print(data.head())

#print(data.isnull().sum(axis=0))

data=data[pd.notnull(data['standard_cost'])]
data['transaction_date'] = pd.to_datetime(data['transaction_date'])
#print(data.isnull().sum(axis=0))
info = data.info()
print(info)

Latest_date = dt.datetime(2017,12,30)

RFM_score = data.groupby('customer_id').agg({'transaction_date':lambda x: (Latest_date - x.max()).days, 'customer_id':lambda x: len(x), 'standard_cost':lambda x: x.sum()})
RFM_score['transaction_date']=RFM_score['transaction_date'].astype(int)

RFM_score.rename(columns={'transaction_date':'Recency','customer_id':'Frequency','standard_cost':'Monetary'}, inplace=True)
index=RFM_score.reset_index().head()
#print(index)
#Recencey_des=RFM_score.Recency.describe()
#print(Recencey_des)
Monetary_des=RFM_score.Monetary.describe()
print(Monetary_des)

quantiles=RFM_score.quantile(q=[0.25,0.5,0.75])
quantiles=quantiles.to_dict()
print(quantiles)

def RScoring(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
def FnMScoring(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1

RFM_score['R'] = RFM_score['Recency'].apply(RScoring, args=('Recency',quantiles,))
RFM_score['F'] = RFM_score['Frequency'].apply(FnMScoring, args=('Frequency',quantiles,))
RFM_score['M'] = RFM_score['Monetary'].apply(FnMScoring, args=('Monetary',quantiles,))

print(RFM_score.head())

RFM_score['RFMGroup'] = RFM_score.R.map(str) + RFM_score.F.map(str) + RFM_score.M.map(str)
RFM_score['RFMScore'] = RFM_score[['R', 'F', 'M']].sum(axis = 1)
print(RFM_score.head())

Loyalty_Level = ['Platinum', 'Gold', 'Silver', 'Bronze']
Score_cuts = pd.qcut(RFM_score.RFMScore, q = 4, labels = Loyalty_Level)
RFM_score['RFM_Loyalty_Level'] = Score_cuts.values
new2=RFM_score.reset_index()
#print(new2)

df=pd.DataFrame(new2, columns=['customer_id','Recency','Frequency','Monetary', 'RFMGroup','RFMScore','RFM_Loyalty_Level'])
df.to_excel (r'C:/Users/User/Desktop/Output1.xlsx', index = False, header=True)

#new3=RFM_score[RFM_score['RFMGroup']=='111'].sort_values('Monetary', ascending=False).reset_index().head(1000)
#print(new3)