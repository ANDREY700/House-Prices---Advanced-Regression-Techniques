

# пробуем модель по домам
# 2025-01-23




# начнем со всех библиотек

import pandas as pd
import numpy as np
import sklearn
import sklearn as sk

# Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
sklearn.set_config(transform_output="pandas")

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, OrdinalEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, KFold

# for model learning
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score

#models
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from catboost import CatBoostRegressor
import xgboost as xgb

# Metrics
from sklearn.metrics import accuracy_score



# загрузка очищенных данных
# path = '/home/andrey/Documents/Working/31 W6D4 Оценка регрессоров/'
path = '/home/andrey/Documents/HOUSING/'
# массив Х для обучения
df = pd.read_csv(path + 'x_train_4.csv', sep='\t')
# массив Y для обучения
y0  = pd.read_csv(path + 'y_train_4.csv', sep='\t')
y0['y_log'] = np.log(y0['SalePrice'])
y0 = y0.drop('SalePrice', axis=1)
df.head()
y0.head()




# df.columns.to_list()

# x = 'BsmtFinType1'
# df.loc[:, x].head()
# df.loc[:, x].nunique() 

ordinal_code_list = [ ]
one_hot_code_list = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Alley',
                     'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                     'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                     'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
                     'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 
                     'MiscFeature', 'SaleType', 'SaleCondition']
scaling_code_list = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 
                     'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 
                     'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                     'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
                     '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

# перекодируем данные ----------------------------------------------------
# кодировщик для категорийных данных
my_coder = ColumnTransformer(
    transformers = [
        # ('ordinal_code', OrdinalEncoder(), ordinal_code_list),
        ('one_hot_code', OneHotEncoder(sparse_output=False), one_hot_code_list),
        ('scaling_code', StandardScaler(), scaling_code_list)
    ],
    verbose_feature_names_out=False,
    remainder='passthrough'
)

# df.loc[df['MSSubClass']=='MSSubClass', :].head()


# текстируем или готовим данные для загрузки
# true = submission
DECISION = False

# делим данные ------------------------------------------------------
X0 = my_coder.fit_transform(df)
# X0.head()
# X0.shape

# разделим данные на обучение и тест
if not DECISION:
    X0_train, X0_valid, y0_train, y0_valid = train_test_split(X0, y0, test_size=0.15, random_state=299)
else:
    X0_train = X0.copy()
    y0_train = y0

# X0_train.head() # смотрим глазками
# y0_train.head() # смотрим глазками

# X0_train.shape


# пара таблиц для отслеживания промежуточных результатов
middle_answer = np.zeros((len(X0_train), 20))
#middle_answer[1:10, :] # check
middle_metrics = pd.DataFrame([], columns=['#', 'name', 'mse'])


# middle_table = middle_metrics
# y_valid = y0_train
# y_pred = middle_answer[:, 0] 
# mod_num=1
# mod_name='LinearRegression'
# threshold = 0.5

# функция для добавления промежуточных результатов в таблицу
def add_metrics_to_table(middle_table, y_valid, y_pred, mod_num:int=0, 
                         mod_name:str=''): #, threshold:float=0.55

    mse=metrics.mean_squared_error(y_valid, y_pred) 
    middle_table = pd.concat([middle_table, 
                              pd.DataFrame({"#": [mod_num], 
                                            "name": [mod_name],
                                            "mse": [mse]
                                            })
                              ], ignore_index=True)

    return middle_table

import sklearn.metrics as metrics
def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    # mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    # print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))

# собираем ансабль моделей и делаем Стекинг ------------------------


# STEP 1 ----------------------------------------------------------
# 0
# метод наименьших квадратов
model_01 = sk.linear_model.LinearRegression()
model_01.fit(X0_train, y0_train)
middle_answer[:, 0] = model_01.predict(X0_train)[:, 0]
middle_metrics = add_metrics_to_table(middle_metrics, 
                                      y0_train, middle_answer[:, 0] , 
                                      mod_num=1, mod_name='LinearRegression')


# 1
# Ридж-регрессия
model_02 = sk.linear_model.Ridge(alpha=0.5)
# alpha
# solver
model_02.fit(X0_train, y0_train)
middle_answer[:, 1] = model_02.predict(X0_train)
middle_metrics = add_metrics_to_table(middle_metrics, 
                                      y0_train, middle_answer[:, 1] , 
                                      mod_num=2, mod_name='Ridge')


# 2
# LASSO
model_03 = sk.linear_model.Lasso(alpha=0.1)
# alpha
model_03.fit(X0_train, y0_train)
middle_answer[:, 2] = model_03.predict(X0_train)
middle_metrics = add_metrics_to_table(middle_metrics, 
                                      y0_train, middle_answer[:, 2] , 
                                      mod_num=3, mod_name='Lasso')

# 3
# LARS
model_04 = sk.linear_model.Lars(n_nonzero_coefs = 5)
# n_nonzero_coefs
model_04.fit(X0_train, y0_train)
middle_answer[:, 3] = model_04.predict(X0_train)
middle_metrics = add_metrics_to_table(middle_metrics, 
                                      y0_train, middle_answer[:, 3] , 
                                      mod_num=4, mod_name='Lars')


# 4
# ElasticNet
model_05 = sk.linear_model.ElasticNet(alpha = 0.1, l1_ratio = 0.5)
# random_state
model_05.fit(X0_train, y0_train)
middle_answer[:, 4] = model_05.predict(X0_train)
middle_metrics = add_metrics_to_table(middle_metrics, 
                                      y0_train, middle_answer[:, 4] , 
                                      mod_num=5, mod_name='ElasticNet')

# 5
# LassoLars
model_06 = sk.linear_model.LassoLars(alpha=0.1)
# alpha
model_06.fit(X0_train, y0_train)
middle_answer[:, 5] = model_06.predict(X0_train)
middle_metrics = add_metrics_to_table(middle_metrics, 
                                      y0_train, middle_answer[:, 5] , 
                                      mod_num=6, mod_name='LassoLars')

# 6
# BayesianRidge
model_07 = sk.linear_model.BayesianRidge()
# 
model_07.fit(X0_train, y0_train)
middle_answer[:, 6] = model_07.predict(X0_train)
middle_metrics = add_metrics_to_table(middle_metrics, 
                                      y0_train, middle_answer[:, 6] , 
                                      mod_num=7, mod_name='BayesianRidge')

# 7
# ARDRegression
model_08 = sk.linear_model.ARDRegression()
# 
model_08.fit(X0_train, y0_train)
middle_answer[:, 7] = model_08.predict(X0_train)
middle_metrics = add_metrics_to_table(middle_metrics, 
                                      y0_train, middle_answer[:, 7] , 
                                      mod_num=8, mod_name='ARDRegression')


# 8
# TweedieRegressor
model_09 = sk.linear_model.TweedieRegressor(alpha=0.5, power=1, link='log')
# alpha
# power
# link
model_09.fit(X0_train, y0_train)
middle_answer[:, 8] = model_09.predict(X0_train)
middle_metrics = add_metrics_to_table(middle_metrics, 
                                      y0_train, middle_answer[:, 8] , 
                                      mod_num=9, mod_name='TweedieRegressor')

# 9
# CatBoostRegressor
model_10 = CatBoostRegressor(iterations=20, learning_rate=1, depth=10)
model_10.fit(X0_train, y0_train)
middle_answer[:, 9] = model_10.predict(X0_train)
middle_metrics = add_metrics_to_table(middle_metrics, 
                                      y0_train, middle_answer[:, 9] , 
                                      mod_num=10, mod_name='CatBoostRegressor')


# 10
# CatBoostRegressor
model_11 = xgb.XGBRegressor(verbosity=0) 
model_11.fit(X0_train, y0_train)
middle_answer[:, 10] = model_11.predict(X0_train)
middle_metrics = add_metrics_to_table(middle_metrics, 
                                      y0_train, middle_answer[:, 10] , 
                                      mod_num=11, mod_name='XGBRegressor')

# 9
# 
# model_10 = RandomForestClassifier(n_estimators=100, random_state=99)
# # n_estimators
# # random_state
# model_10.fit(X0_train, y0_train)
# middle_answer[:, 9] = model_10.predict(X0_train)[0]
# middle_metrics = add_metrics_to_table(middle_metrics, 
#                                       y0_train, middle_answer[:, 9] , 
#                                       mod_num=10, mod_name='RandomForestClassifier')




# STEP 2 ----------------------------------------------------------
step2_answer = np.zeros((len(X0_train), 20))


model_21 = sk.linear_model.LinearRegression()
model_21.fit(middle_answer, y0_train)
step2_answer[:, 0] = model_21.predict(middle_answer)[:, 0]
middle_metrics = add_metrics_to_table(middle_metrics, 
                                      y0_train, step2_answer[:, 0] , 
                                      mod_num=21, mod_name='STEP-2: LinearRegression')

# regression_results(y0_train, step2_answer[:, 0])

model_22 = sk.linear_model.Ridge(alpha=0.5)
model_22.fit(middle_answer, y0_train)
step2_answer[:, 1] = model_22.predict(middle_answer)
middle_metrics = add_metrics_to_table(middle_metrics, 
                                      y0_train, step2_answer[:, 1] , 
                                      mod_num=22, mod_name='STEP-2: Ridge')


model_23 = sk.linear_model.Lasso(alpha=0.1)
model_23.fit(middle_answer, y0_train)
step2_answer[:, 2] = model_23.predict(middle_answer)
middle_metrics = add_metrics_to_table(middle_metrics, 
                                      y0_train, step2_answer[:, 2] , 
                                      mod_num=23, mod_name='STEP-2: Lasso')


model_24 = sk.linear_model.Lars(n_nonzero_coefs = 1)
model_24.fit(middle_answer, y0_train)
step2_answer[:, 3] = model_24.predict(middle_answer)
middle_metrics = add_metrics_to_table(middle_metrics, 
                                      y0_train, step2_answer[:, 3] , 
                                      mod_num=24, mod_name='STEP-2: Lars')


model_25 = sk.linear_model.ElasticNet(alpha = 0.1, l1_ratio = 0.5)
model_25.fit(middle_answer, y0_train)
step2_answer[:, 4] = model_25.predict(middle_answer)
middle_metrics = add_metrics_to_table(middle_metrics, 
                                      y0_train, step2_answer[:, 4] , 
                                      mod_num=25, mod_name='STEP-2: ElasticNet')


model_26 = sk.linear_model.LassoLars(alpha=0.1)
model_26.fit(middle_answer, y0_train)
step2_answer[:, 5] = model_26.predict(middle_answer)
middle_metrics = add_metrics_to_table(middle_metrics, 
                                      y0_train, step2_answer[:, 5] , 
                                      mod_num=26, mod_name='STEP-2: LassoLars')

model_27 = sk.linear_model.BayesianRidge()
model_27.fit(middle_answer, y0_train)
step2_answer[:, 6] = model_27.predict(middle_answer)
middle_metrics = add_metrics_to_table(middle_metrics, 
                                      y0_train, step2_answer[:, 6] , 
                                      mod_num=27, mod_name='STEP-2: BayesianRidge')


model_28 = sk.linear_model.ARDRegression()
model_28.fit(middle_answer, y0_train)
step2_answer[:, 7] = model_28.predict(middle_answer)
middle_metrics = add_metrics_to_table(middle_metrics, 
                                      y0_train, step2_answer[:, 7] , 
                                      mod_num=28, mod_name='STEP-2: ARDRegression')

model_29 = sk.linear_model.TweedieRegressor(alpha=0.5, power=2, link='log')
model_29.fit(middle_answer, y0_train)
step2_answer[:, 8] = model_29.predict(middle_answer)
middle_metrics = add_metrics_to_table(middle_metrics, 
                                      y0_train, step2_answer[:, 8] , 
                                      mod_num=29, mod_name='STEP-2: TweedieRegressor')


# model_30 = CatBoostRegressor(iterations=2, learning_rate=1, depth=2)
model_30 = CatBoostRegressor(iterations=20, learning_rate=1, depth=10)
model_30.fit(middle_answer, y0_train)
step2_answer[:, 9] = model_30.predict(middle_answer)
middle_metrics = add_metrics_to_table(middle_metrics, 
                                      y0_train, step2_answer[:, 9] , 
                                      mod_num=30, mod_name='STEP-2: CatBoostRegressor')



model_31 = xgb.XGBRegressor(verbosity=0) 
model_31.fit(middle_answer, y0_train)
step2_answer[:, 10] = model_31.predict(middle_answer)
middle_metrics = add_metrics_to_table(middle_metrics, 
                                      y0_train, step2_answer[:, 10] , 
                                      mod_num=31, mod_name='STEP-2: XGBRegressor')

# model_30 = RandomForestClassifier(n_estimators=10, random_state=200)
# model_30.fit(middle_answer, y0_train)
# step2_answer[:, 9] = model_30.predict(middle_answer)[0]
# middle_metrics = add_metrics_to_table(middle_metrics, 
#                                       y0_train, step2_answer[:, 9] , 
#                                       mod_num=30, mod_name='STEP-2: RandomForestClassifier')


# testing portion
if not DECISION:
    # пара таблиц для отслеживания промежуточных результатов
    valid_answer_step2 = np.zeros((len(X0_valid), 20))

    valid_answer_step2[:, 0] = model_01.predict(X0_valid)[:, 0]
    middle_metrics = add_metrics_to_table(middle_metrics, y0_valid, valid_answer_step2[:, 0], mod_num=40, mod_name='STEP-1v: LinearRegression valid')
    valid_answer_step2[:, 1] = model_02.predict(X0_valid)
    middle_metrics = add_metrics_to_table(middle_metrics, y0_valid, valid_answer_step2[:, 1], mod_num=41, mod_name='STEP-1v: Ridge valid')
    valid_answer_step2[:, 2] = model_03.predict(X0_valid)
    middle_metrics = add_metrics_to_table(middle_metrics, y0_valid, valid_answer_step2[:, 2], mod_num=42, mod_name='STEP-1v: Lasso valid')
    valid_answer_step2[:, 3] = model_04.predict(X0_valid)
    middle_metrics = add_metrics_to_table(middle_metrics, y0_valid, valid_answer_step2[:, 3], mod_num=43, mod_name='STEP-1v: Lars valid')
    valid_answer_step2[:, 4] = model_05.predict(X0_valid)
    middle_metrics = add_metrics_to_table(middle_metrics, y0_valid, valid_answer_step2[:, 4], mod_num=44, mod_name='STEP-1v: ElasticNet valid')
    valid_answer_step2[:, 5] = model_06.predict(X0_valid)
    middle_metrics = add_metrics_to_table(middle_metrics, y0_valid, valid_answer_step2[:, 5], mod_num=45, mod_name='STEP-1v: LassoLars valid')
    valid_answer_step2[:, 6] = model_07.predict(X0_valid)
    middle_metrics = add_metrics_to_table(middle_metrics, y0_valid, valid_answer_step2[:, 6], mod_num=46, mod_name='STEP-1v: BayesianRidge valid')
    valid_answer_step2[:, 7] = model_08.predict(X0_valid)
    middle_metrics = add_metrics_to_table(middle_metrics, y0_valid, valid_answer_step2[:, 7], mod_num=47, mod_name='STEP-1v: ARDRegression valid')
    valid_answer_step2[:, 8] = model_09.predict(X0_valid)
    middle_metrics = add_metrics_to_table(middle_metrics, y0_valid, valid_answer_step2[:, 8], mod_num=48, mod_name='STEP-1v: TweedieRegressor valid')
    valid_answer_step2[:, 9] = model_10.predict(X0_valid)
    middle_metrics = add_metrics_to_table(middle_metrics, y0_valid, valid_answer_step2[:, 9], mod_num=49, mod_name='STEP-1v: CatBoostRegressor valid')
    valid_answer_step2[:, 10] = model_11.predict(X0_valid)
    middle_metrics = add_metrics_to_table(middle_metrics, y0_valid, valid_answer_step2[:, 10], mod_num=50, mod_name='STEP-1v: XGBRegressor valid')


    valid_answer_step2_final = np.zeros((len(X0_valid), 20))

    valid_answer_step2_final[:, 0] = model_21.predict(valid_answer_step2)[:, 0]
    middle_metrics = add_metrics_to_table(middle_metrics, y0_valid, valid_answer_step2_final[:, 0], mod_num=60, mod_name='STEP-2v: LinearRegression valid')

    valid_answer_step2_final[:, 1] = model_22.predict(valid_answer_step2)
    middle_metrics = add_metrics_to_table(middle_metrics, y0_valid, valid_answer_step2_final[:, 1], mod_num=61, mod_name='STEP-2v: Ridge valid')

    valid_answer_step2_final[:, 2] = model_23.predict(valid_answer_step2)
    middle_metrics = add_metrics_to_table(middle_metrics, y0_valid,  valid_answer_step2_final[:, 2], mod_num=62, mod_name='STEP-2v: Lasso valid')

    valid_answer_step2_final[:, 3] = model_24.predict(valid_answer_step2)
    middle_metrics = add_metrics_to_table(middle_metrics, y0_valid, valid_answer_step2_final[:, 3], mod_num=63, mod_name='STEP-2v: Lars valid')

    valid_answer_step2_final[:, 4] = model_25.predict(valid_answer_step2)
    middle_metrics = add_metrics_to_table(middle_metrics, y0_valid, valid_answer_step2_final[:, 4], mod_num=64, mod_name='STEP-2v: ElasticNet valid')

    valid_answer_step2_final[:, 5] = model_26.predict(valid_answer_step2)
    middle_metrics = add_metrics_to_table(middle_metrics, y0_valid, valid_answer_step2_final[:, 5], mod_num=65, mod_name='STEP-2v: LassoLars valid')

    valid_answer_step2_final[:, 6] = model_27.predict(valid_answer_step2)
    middle_metrics = add_metrics_to_table(middle_metrics,  y0_valid, valid_answer_step2_final[:, 6], mod_num=66, mod_name='STEP-2v: BayesianRidge valid')

    valid_answer_step2_final[:, 7] = model_28.predict(valid_answer_step2)
    middle_metrics = add_metrics_to_table(middle_metrics,  y0_valid, valid_answer_step2_final[:, 7], mod_num=67, mod_name='STEP-2v: ARDRegression valid')

    valid_answer_step2_final[:, 8] = model_29.predict(valid_answer_step2)
    middle_metrics = add_metrics_to_table(middle_metrics, y0_valid, valid_answer_step2_final[:, 8], mod_num=68, mod_name='STEP-2v: TweedieRegressor valid')

    valid_answer_step2_final[:, 9] = model_30.predict(valid_answer_step2)
    middle_metrics = add_metrics_to_table(middle_metrics, y0_valid, valid_answer_step2_final[:, 9], mod_num=69, mod_name='STEP-2v: CatBoostRegressor valid')
    valid_answer_step2_final[:, 10] = model_30.predict(valid_answer_step2)
    middle_metrics = add_metrics_to_table(middle_metrics, y0_valid, valid_answer_step2_final[:, 10], mod_num=70, mod_name='STEP-2v: XGBRegressor valid')

    valid_answer_step2_final = pd.DataFrame(valid_answer_step2_final)
    valid_answer_step2 = pd.DataFrame(valid_answer_step2)

middle_metrics



# middle_answer2 = pd.DataFrame(middle_answer)
# path2 = '/home/andrey/Documents/Working/31 W6D4 Оценка регрессоров/out/'
# middle_answer2.to_csv(path2 + 'middle_answer_03.csv', header=True, index=None, sep='\t', mode='a')


    # # ---------------------------------------------------------------------------------------------
    # # собираем тестовую сборку для Кэггле 

if DECISION:
    x_kaggle  = pd.read_csv(path + 'x_test_4.csv', sep='\t')
    x_kaggle.head()

    x_kaggle_Id = x_kaggle['Id']
    x_kaggle = x_kaggle.drop('Id', axis=1)

    X100 = my_coder.transform(x_kaggle)

    X100_step1 = np.zeros((len(X100), 20))
    X100_step1[:, 0] = model_01.predict(X100)[:, 0]
    X100_step1[:, 1] = model_02.predict(X100)
    X100_step1[:, 2] = model_03.predict(X100)
    X100_step1[:, 3] = model_04.predict(X100)
    X100_step1[:, 4] = model_05.predict(X100)
    X100_step1[:, 5] = model_06.predict(X100)
    X100_step1[:, 6] = model_07.predict(X100)
    X100_step1[:, 7] = model_08.predict(X100)
    X100_step1[:, 8] = model_09.predict(X100)
    X100_step1[:, 9] = model_10.predict(X100)
    X100_step1[:, 10] = model_11.predict(X100)

    X100_step2 = np.zeros((len(X100), 20))
    X100_step2[:, 0] = model_01.predict(X100_step1)[:, 0]

    answer = np.exp(X100_step2[:, 0])

    answer = pd.DataFrame({'Id': x_kaggle_Id, 'SalePrice' :  answer})
    len(answer)

    answer.to_csv(path + 'submission_03.csv', header=True, index=None, sep=',')


