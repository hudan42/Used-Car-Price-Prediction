#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 21:33:10 2020

@author: hercule
"""

from flask import Flask, jsonify, request
import json
import numpy as np
import pickle
import pandas as pd
from car_model_building import Encoder


with open("car_model.pkl", "rb") as f:
    model = pickle.load(f)

if __name__=='__main__':
    with open('encoder.pkl', 'rb') as t:
        p_encoder = pickle.load(t)
        
        
app = Flask(__name__)


@app.route('/')
def index():
    return \
    """
    <!DOCTYPE html>
    <html>
    <body>
    
    <h2>Predict Prices of Used Cars in San Francisco Bay Area</h2>
    
    <form method="POST" action="/result">
    
      Car Model (Year):<br>
      <input type="number" name="year" value=2000>
      <br><br>
      
      Car Model (Maker):<br>
      <input type="text" name="maker" value='audi'>
      <br><br>
      
      Car Condition:<br>
      <input type="radio" name="condition" value="salvage"> Salvage<br>
      <input type="radio" name="condition" value="fair"> Fair<br>
      <input type="radio" name="condition" value="good"> Good<br>
      <input type="radio" name="condition" value="like new"> Like New<br>
      <input type="radio" name="condition" value="new"> New<br>
      <input type="radio" name="condition" value="excellent"> Excellent
      <br><br>
      
      Number of Cylinders:<br>
      <input type="number" name="cylinder" value=4>
      <br><br>
      
      Drive:<br>
      <input type="radio" name="drive" value="fwd"> FWD<br>
      <input type="radio" name="drive" value="rwd"> RWD<br>
      <input type="radio" name="drive" value="4wd"> 4WD
      <br><br>
      
      Fuel:<br>
      <input type="radio" name="fuel" value="gas"> Gas<br>
      <input type="radio" name="fuel" value="diesel"> Diesel<br>
      <input type="radio" name="fuel" value="hybrid"> Hybrid<br>
      <input type="radio" name="fuel" value="electric"> Electric<br>
      <input type="radio" name="fuel" value="other"> Other
      <br><br>
      
      Odometer:<br>
      <input type="number" name="odometer" value=1000>
      <br><br>
      
      Color:<br>
      <select name='color'>
          <option value="silver">silver</option>
          <option value="red">red</option>
          <option value="black">black</option>
          <option value="brown">brown</option>
          <option value="brown">brown</option>
          <option value="grey">grey</option>
          <option value="white">white</option>
          <option value="blue">blue</option>
          <option value="green">green</option>
          <option value="orange">orange</option>
          <option value="yellow">yellow</option>
          <option value="purple">purple</option>
      </select>
      <br><br>

      
      Car Size:<br>
      <input type="radio" name="size" value="full-size"> Full Size<br>
      <input type="radio" name="size" value="mid-size"> Middle Size<br>
      <input type="radio" name="size" value="compact"> Compact<br>
      <input type="radio" name="size" value="sub-compact"> Sub Compact
      <br><br>
      
      
      Transmission:<br>
      <input type="radio" name="trans" value="automatic"> Automatic<br>
      <input type="radio" name="trans" value="manual"> Manual<br>
      <input type="radio" name="trans" value="other"> Other
      <br><br>
      
      
      Car Type:<br>
      <select name='type'>
          <option value="truck">truck</option>
          <option value="sedan">sedan</option>
          <option value="suv">suv</option>
          <option value="convertible">convertible</option>
          <option value="hatchback">hatchback</option>
          <option value="coupe">coupe</option>
          <option value="wagon">wagon</option>
          <option value="offroad">offroad</option>
          <option value="mini-van">mini van</option>
          <option value="bus">bus</option>
          <option value="other">other</option>
      </select>
      <br><br>
      
      <input type="submit" value="Submit">
    </form> 
    </body>
    </html>
    """


@app.route('/result', methods=['POST', 'GET'])
def result():
    year = request.form["year"]
    maker = request.form["maker"]
    condition = request.form["condition"]
    cylinder = request.form['cylinder']
    drive = request.form['drive']
    odometer = request.form['odometer']
    color = request.form['color']
    fuel = request.form['fuel']
    size = request.form['size']
    transmission = request.form['trans']
    types = request.form['type']
    
    features = ['year', 'maker', 'condition', 'cylinders', 'drive','odometer', 'paint color', 
                'fuel', 'size', 'transmission', 'type']
    tmp = pd.DataFrame([[year, maker, condition, cylinder, drive, odometer, color, 
                         fuel, size, transmission, types]], 
                       index = [0], columns = features)
    
    
    fakedf = pd.DataFrame([
         [1991, 'audi','good', 4,'fwd',1000,'red','gas', 'compact', 'manual', 'truck'], 
         [1991, 'audi','good', 4,'rwd',1000,'red','other', 'compact', 'automatic', 'truck'], 
         [1991, 'audi','good', 4,'4wd',1000,'red','diesel', 'compact', 'other', 'truck'], 
         [1991, 'audi','good', 4,'4wd',1000,'red','hybrid', 'compact', 'manual', 'truck'], 
         [1991, 'audi','good', 4,'4wd',1000,'red','electric', 'compact', 'manual', 'truck']], 
        columns=features) 
    
    X_tmp = pd.concat([tmp, fakedf]).reset_index(drop=True)
    
    X = p_encoder.transform(X_tmp).loc[0,:].values.reshape(1,-1)
    
    pred = model.predict(X)
    return \
    """
    <!DOCTYPE html>
    <html>
    <body>
    
    The Estimated Price for this Car is: <br><br>
    {0}<br><br>
    
    <form action="/">
      <input type="submit" value="Recalculate">
    </form> 
    </body>
    </html>
    """.format(pred)


@app.route('/scoring', methods=['POST'])
def get_keywords():
    year = request.form["year"]
    maker = request.form["maker"]
    condition = request.form["condition"]
    cylinder = request.form['cylinder']
    drive = request.form['drive']
    odometer = request.form['odometer']
    color = request.form['color']
    fuel = request.form['fuel']
    size = request.form['size']
    transmission = request.form['trans']
    types = request.form['type']
    
    features = ['year', 'maker', 'condition', 'cylinders', 'drive','odometer', 'paint color', 
                'fuel', 'size', 'transmission', 'type']
    tmp = pd.DataFrame([[year, maker, condition, cylinder, drive, odometer, color, 
                         fuel, size, transmission, types]], 
                       index = [0], columns = features)
    
    
    fakedf = pd.DataFrame([
         [1991, 'audi','good', 4,'fwd',1000,'red','gas', 'compact', 'manual', 'truck'], 
         [1991, 'audi','good', 4,'rwd',1000,'red','other', 'compact', 'automatic', 'truck'], 
         [1991, 'audi','good', 4,'4wd',1000,'red','diesel', 'compact', 'other', 'truck'], 
         [1991, 'audi','good', 4,'4wd',1000,'red','hybrid', 'compact', 'manual', 'truck'], 
         [1991, 'audi','good', 4,'4wd',1000,'red','electric', 'compact', 'manual', 'truck']], 
        columns=features) 
    
    X_tmp = pd.concat([tmp, fakedf]).reset_index(drop=True)
    
    X = p_encoder.transform(X_tmp).loc[0,:].values.reshape(1,-1)
        
    results = {"car_price":model.predict(X)}
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    