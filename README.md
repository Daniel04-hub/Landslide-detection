# Landslide-detection
A web-based Landslide Detection System built using Flask and Machine Learning (Random Forest). The app predicts the risk level of landslides based on environmental factors such as rainfall, slope angle, and moisture content. It provides a user-friendly dashboard, graphical confidence visualization. 


# Landslide Detection Web App

This is a Flask-based web application that predicts the risk level of landslides using a trained Machine Learning model (Random Forest).  
The system also provides user authentication, visualization of prediction confidence, and SMS alerts using the Twilio API.

## Features
- Predict landslide risk based on rainfall, slope, and moisture data  
- User registration and login with SQLite database  
- Graph visualization of model confidence  
- SMS alert system for high-risk areas  

## Tech Stack
- Python, Flask  
- scikit-learn, pandas, numpy  
- SQLite, Matplotlib  
- Twilio API for SMS notifications  

## How to Run
1. Clone the repository  
2. Create and activate a virtual environment  
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
