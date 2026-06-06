import pandas as pd
train=pd.read_csv('train.csv')
d49=train[train['day']==49]['demand'].values
var=d49.var()
rmse=0.047675
r2=1-(rmse**2)/var
print(f'Var d49={var:.8f}')
print(f'R2={r2:.6f}')
print(f'score={max(0,100*r2):.2f}')
