import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
macro_data = pd.read_csv('macro.csv')

train_sub = train_data[train_data.timestamp < '2015-01-01']
train_sub = train_sub[train_sub.product_type == "Investment"]

ind_1m = train_sub[train_sub.price_doc <= 1000000].index
ind_2m = train_sub[train_sub.price_doc == 2000000].index
ind_3m = train_sub[train_sub.price_doc == 3000000].index

train_index = set(train_data.index.copy())

for ind, gap in zip([ind_1m, ind_2m, ind_3m], [10, 3, 2]):
    ind_set = set(ind)
    ind_set_cut = ind.difference(set(ind[::gap]))

    train_index = train_index.difference(ind_set_cut)

train_data = train_data.loc[train_index]

# Split off columns that will be needed later.
train_ids = train_data['id'].values
test_ids = test_data['id'].values
train_prices = train_data['price_doc'].values
train_lprices = np.log1p(train_prices)

train_data.drop(['id', 'price_doc'], axis=1, inplace=True)
test_data.drop(['id'], axis=1, inplace=True)


good_macro_features = ['timestamp', 'balance_trade', 'balance_trade_growth', 'eurrub',
                       'average_provision_of_build_contract', 'micex_rgbi_tr',
                       'micex_cbi_tr', 'deposits_rate', 'mortgage_value',
                       'mortgage_rate', 'income_per_cap', 'rent_price_4.room_bus',
                       'museum_visitis_per_100_cap', 'apartment_build']
good_macro_data = pd.DataFrame(macro_data, columns=good_macro_features)

# Merge good features from macro.csv to training/test data.
n_train = len(train_data.index)
all_tt_data = pd.concat([train_data, test_data])
all_data = pd.merge(all_tt_data, good_macro_data, on='timestamp', how='left')

all_data.loc[all_data.state == 33] = 3
all_data.loc[all_data.build_year == 20052009] = 2007
all_data = all_data[all_data.sub_area != 3]

print('Extracting features from timestamps. . .')


years = pd.to_datetime(all_data.timestamp, errors='coerce').dt.year
months = pd.to_datetime(all_data.timestamp, errors='coerce').dt.month
dows = pd.to_datetime(all_data.timestamp, errors='coerce').dt.dayofweek
woys = pd.to_datetime(all_data.timestamp, errors='coerce').dt.weekofyear
doys = pd.to_datetime(all_data.timestamp, errors='coerce').dt.dayofyear

month_year = (months + years * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
week_year = (woys + years * 100)
week_year_cnt_map = week_year.value_counts().to_dict()

all_data['year'] = years
all_data['month'] = months

all_data['month_year_count'] = month_year.map(month_year_cnt_map)
all_data['week_year_count'] = week_year.map(week_year_cnt_map)

# Drop timestamps.
all_data.drop(['timestamp'], axis=1, inplace=True)

# II. Property-Specific Features
all_data['max_floor'] = all_data['max_floor'].replace(to_replace=0, value=np.nan)
all_data['rel_floor'] = all_data['floor'] / all_data['max_floor'].astype(float)
all_data['rel_kitch_sq'] = all_data['kitch_sq'] / all_data['full_sq'].astype(float)
all_data['rel_life_sq'] = all_data['life_sq'] / all_data['full_sq'].astype(float)
# Corrects for property with zero full_sq.
all_data['rel_life_sq'] = all_data['rel_life_sq'].replace(to_replace=np.inf, value=np.nan)
# Does not account for living room, but reasonable enough.
all_data['avg_room_sq'] = all_data['life_sq'] / all_data['num_room'].astype(float)
# Corrects for studios (zero rooms listed).
all_data['avg_room_sq'] = all_data['avg_room_sq'].replace(to_replace=np.inf, value=np.nan)

# Replace garbage values in build_year with NaNs, then find average build year
# in each sub_area.
all_data['build_year'] = all_data['build_year'].replace(
    to_replace=[0, 1, 2, 3, 20, 71, 215, 4965], value=np.nan)
mean_by_districts = pd.DataFrame(columns=['district', 'avg_build_year'])
sub_areas_unique = all_data['sub_area'].unique()
for sa in sub_areas_unique:
    temp = all_data.loc[all_data['sub_area'] == sa]
    mean_build_year = temp['build_year'].mean()
    new_df = pd.DataFrame([[sa, mean_build_year]], columns=['district', 'avg_build_year'])
    mean_by_districts = mean_by_districts.append(new_df, ignore_index=True)

mbd_dis_list = mean_by_districts['district'].tolist()
mbd_dis_full = all_data['sub_area'].tolist()
mbd_aby_np = np.array(mean_by_districts['avg_build_year'])
mbd_aby_full = np.zeros(len(all_data.index))

# (Could find a better way to do this.)
for i in range(len(all_data.index)):
    district = mbd_dis_full[i]
    mbd_aby_full[i] = mbd_aby_np[mbd_dis_list.index(district)]

all_data['avg_build_year'] = mbd_aby_full
all_data['rel_build_year'] = all_data['build_year'] - all_data['avg_build_year']

# III. Categorical Features, Treating NaNs

df_numeric = all_data.select_dtypes(exclude=['object'])
df_obj = all_data.select_dtypes(include=['object']).copy()
ecology_dict = {'no data': np.nan, 'poor': 1, 'satisfactory': 2, 'good': 3,
                'excellent': 4}


def one_hot_encode(x, n_classes):
    return np.eye(n_classes)[x]


for c in df_obj:
    factorized = pd.factorize(df_obj[c])
    f_values = factorized[0]
    f_labels = list(factorized[1])
    n_classes = len(f_labels)

    if (n_classes == 2 or n_classes == 3) and c != 'product_type':
        df_obj[c] = factorized[0]
    elif c == 'ecology':
        df_obj[c] = df_obj[c].map(ecology_dict)
    else:
        one_hot_features = one_hot_encode(f_values, n_classes)
        oh_features_df = pd.DataFrame(one_hot_features, columns=f_labels)
        df_obj = df_obj.drop(c, axis=1)
        df_obj = pd.concat([df_obj, oh_features_df], axis=1)

# for c in df_obj:
#    df_obj[c] = pd.factorize(df_obj[c])[0]

all_values = pd.concat([df_numeric, df_obj], axis=1)

full_feature_names = list(all_values)


x_train_all = all_values.values
y_train_all = train_lprices.ravel()      # Log(price)

x_test = x_train_all[n_train:]
x_test_df = pd.DataFrame(x_test, columns=full_feature_names)

lprices_df = pd.Series(train_lprices, name='log_price')

xgb_params = {
    'eta': 0.03,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train_all, y_train_all, feature_names=full_feature_names)
dtest = xgb.DMatrix(x_test, feature_names=full_feature_names)

cv_output = xgb.cv(xgb_params, 
    dtrain, 
    num_boost_round=2000, 
    early_stopping_rounds=20, 
    verbose_eval=50, 
    show_stdv=False)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
plt.show()


num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)

fig, ax = plt.subplots(1, 1, figsize=(8, 13))
xgb.plot_importance(model, height=0.5, ax=ax)
plt.show()

y_predict = model.predict(dtest)
y_predict = np.expm1(y_predict)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
print(output.head())

output.to_csv('xgbSub_2_full_features_treatment.csv', index=False)
