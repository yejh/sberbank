import pandas as pd

tr = pd.read_csv('train.csv', index_col='id')
te = pd.read_csv('test.csv', index_col='id')

print tr.describe()

fx = pd.read_excel('BAD_ADDRESS_FIX.xlsx').drop_duplicates('id').set_index('id')

tr.update(fx)
te.update(fx)

print tr.describe()
