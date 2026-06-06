import pandas as pd
BASE='d:/gridlock'
final = pd.read_csv(f'{BASE}/submission_final_92.csv')['demand'].values
ens60 = pd.read_csv(f'{BASE}/submission_adv_60_92_40_ens.csv')['demand'].values
ids = pd.read_csv(f'{BASE}/submission_final_92.csv')['Index'].values
for w in [0.8,0.85,0.9]:
    preds = (w * final + (1-w) * ens60).clip(0,1)
    name=f'{BASE}/submission_heavy_{int(w*100)}_{int((1-w)*100)}.csv'
    pd.DataFrame({'Index':ids,'demand':preds}).to_csv(name,index=False)
    print('Saved',name)
