#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:03:07 2020

@author: hercule
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
import random
# from xgboost import XGBRegressor
from sklearn.ensemble import IsolationForest, RandomForestRegressor
import pickle


# Read in the data.
df = pd.read_csv('craigslist_car.csv')

# Price is our target variable, there are 176 missing values in price variable. I would drop them our here.
df.dropna(subset=['price'], inplace = True)

#Because delivery available and cryptocurrency ok feaures are null value, drop them directly
#All used cars listed in craiglist have cleant title.
#VIN is the vehicle identification number, it doesn't have impact on price
df.drop(['delivery available','cryptocurrency ok','title status','VIN','post_time'],axis=1,inplace=True)



# Drop rows where all values are NA.
df.dropna(how = 'all', inplace=True)

# Drop columns where all values are NA.
df.dropna(axis = 1, how = 'all', inplace=True)


# Split data into a training set and a test set, using 80 vs. 20 split
train_raw, test_raw = train_test_split(df, test_size = 0.2, random_state = 2020)

train_raw.reset_index(inplace=True)
test_raw.reset_index(inplace=True)


# Data preprocessing and feature engineering
class Transformer:
    def fit(self, X, y = None):
        pass
    
    def transform(self, X, y = None):
        df = pd.DataFrame()
        df['price'] = X.price.map(self.price_to_num)
        df['year'] = X.title.map(self.get_vehicle_year)
        df['maker'] = X.title.map(self.get_vehicle_maker)
        df = pd.concat([df, X.loc[:,'condition':'fuel']], axis = 1)
        df['cylinders'] = df.cylinders.map(self.cylinder_to_num)
        df['odometer'] = X.odometer.map(self.odometer_to_num)
        df = pd.concat([df, X.loc[:,'paint color':'type']], axis = 1)
        
        # Transform all string typed columns into lower cases.
        df = df.applymap(lambda x: x.lower() if type(x) == str else x)
        
        return df
        
    
    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.transform(X)
    
#####################################################################    

    def price_to_num(self, x):
        if type(x) == str:
            x = float(x.strip('$'))
        return x
    
    def get_vehicle_year(self, x):
        if type(x) == str:
            x = x.split()[0]
        return int(x)
    
    def get_vehicle_maker(self, x):
        if type(x) == str:
            s = x.split()
            x = s[1].lower()
            if x.isnumeric() and len(s) > 2:
                x = s[2].lower()
        return x
    
    def cylinder_to_num(self, x):
        if type(x) == str and x != 'other':
            x = int(x.split()[0])
        else:
            x = None
        return x
    
    
    def odometer_to_num(self, x):
        if type(x) == str:
            x = float(x)
        return x


train_new = Transformer().fit_transform(train_raw)







class Encoder:
    '''
    This class will encode categorical vairables into appropriate forms.
    '''
    def fit(self, X, y = None):
        df = X.copy()
    
        # Get the frequency dict of car maker
        self.freq_maker = self.encode_maker_type(df.maker)
        # Get the frequency dict of car type
        self.freq_type = self.encode_maker_type(df.type)
    
    def transform(self, X, y = None):
        df = X.copy()
        
        # Get the indexes where fuel type is equal to 'electric'
        electric_idx = np.where(df.fuel == 'electric')[0]
        # Impute missing values in cylinders variable with 0 where the according fuel type is electric.
        df.cylinders.loc[electric_idx, ] = 0
        
        df.condition = df.condition.map(self.encode_condition)
        
        color_tmp = self.encode_color(df['paint color'])
        df = pd.concat([df, color_tmp], axis = 1)
        
        
        # Clean makers.
        correct_makers = {'impala':'chevrolet','mercedes':'mercedes-benz','mini':'mini-cooper','volkswagon':'volkswagen',
                  'alfa':'alfa-romeo','caddilac':'cadillac','mustang':'ford','kentworth':'kenworth','vw':'volkswagen',
                  'cadilac':'cadillac','savana':'gmc','chrystler':'chrysler','toyt':'toyota','jetta':'volkswagen',
                  'infinity':'infiniti','magnum':'dodge','volkwagen':'volkswagen','tahoe':'chevrolet',
                  'toyoya':'toyota','posche':'porsche','toyata':'toyota','chevelle':'chevrolet',
                  'camry':'toyota','f-350':'ford','535x':'bmw','gnc':'gmc','gto':'pontiac','528i':'bmw',
                  'sr5':'toyota','camero':'chevrolet','2007':'ford','corvette':'chevrolet',
                  'olds':'oldsmobile-cutlass','mighty':'mitsubishi','range':'land-rover','ve':'volvo',
                  'land':'land-rover','gt':'bmw'}
        df['maker'] = df.maker.replace(correct_makers)        

        
        # Add car maker frequency column to the dataframe
        df['maker'] = df.maker.map(lambda x: self.freq_maker.get(x, None))
        
         
        # Add car type frequency column to the dataframe
        df['type'] = df.type.map(lambda x: self.freq_type.get(x, None))
        
        df['size'] = df['size'].map(self.encode_size)
        
        dummy_drive = pd.get_dummies(df.drive, prefix='drive')
        dummy_fuel = pd.get_dummies(df.fuel, prefix='fuel')
        dummy_transmission = pd.get_dummies(df.transmission, prefix='transmission')
        df = pd.concat([df, dummy_drive, dummy_fuel, dummy_transmission], axis = 1)              
        
        # Drop the original columns.
        df.drop(['paint color', 'drive', 'fuel', 'transmission'], axis = 1, inplace = True)

        
        return df
    
    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.transform(X)
    
#########################################################################################

    def encode_condition(self, x):
        ords = {'salvage': 0, 'fair': 1, 'good': 2, 'like new': 3, 'new': 4, 'excellent': 5}
        if type(x) == str:
            x = ords[x]
        return x
    
    
    def encode_color(self, S):
        rgb = {'silver': [192, 192, 192], 'red': [255, 0, 0], 'black': [0, 0, 0], 'brown': [165, 42, 42], 'grey': [128, 128, 128], 
      'white': [255, 255, 255], 'blue': [0, 0, 255], 'green': [0, 128, 0], 'orange': [255, 165, 0], 'yellow': [255, 255, 0], 
      'purple': [128, 0, 128], 'custom': [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]}
        
        color_rgb = []
        for c in S:
            if type(c) == str:
                color_rgb.append(rgb[c])
            else:
                color_rgb.append([np.nan] * 3)

        color_df = pd.DataFrame(color_rgb, columns=['color_r', 'color_g', 'color_b'])
        return color_df    
     
    
    def encode_maker_type(self, X, y=None):
        '''
        Transform car maker into their frequency in the dataset.
        '''
        freq_dict = {}
        for item in X:
            if item in freq_dict:
                freq_dict[item] += 1
            else:
                freq_dict[item] = 1
        return freq_dict
    
    
    def encode_size(self, x):
        ords = {'full-size': 4, 'mid-size': 3, 'compact': 2, 'sub-compact': 1}
        if type(x) == str:
            x = ords[x]
        return x



p = Encoder()
train = p.fit_transform(train_new)

with open('encoder.pkl', 'wb') as e:
    pickle.dump(p, e)


test_new = Transformer().fit_transform(test_raw)
test = p.transform(test_new)


# Get the target variable and predictor variables from training set and test set respectively.
X_train_miss = train.drop('price', axis=1)
y_train = train.price.values

X_test_miss = test.drop('price', axis=1)
y_test = test.price.values


# Missing data imputation
# Use IterativeImputer to impute the rest of missing values in the data.
imputer = IterativeImputer(n_nearest_features = 3, random_state = 2020)
imputer.fit(X_train_miss)

X_train = imputer.transform(X_train_miss)
X_test = imputer.transform(X_test_miss)


# Build Random Forest Model
rf = RandomForestRegressor(random_state=2020)
rf.fit(X_train, y_train)

y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

# print('train MSE: {0:.2e}'.format(mean_squared_error(y_train, y_train_pred)))
# print('train R2: {0:.3f}'.format(r2_score(y_train, y_train_pred)), '\n')

# print('test MSE: {0:.2e}'.format(mean_squared_error(y_test, y_test_pred)))
# print('test R2: {0:.3f}'.format(r2_score(y_test, y_test_pred)))























