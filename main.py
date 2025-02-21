from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Form

from src.logging import logger
from src.prediction.predict import Predictor

from typing import Optional
import pandas as pd
import time
app = FastAPI()

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "Home Page"})

@app.get("/about")
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request, "title": "About Us"})

model = Predictor()

@app.post("/predict", response_class=HTMLResponse)
async def predict_price(
    request: Request,
    brand: str = Form(...),
    material: str = Form(...),
    size: str = Form(...),
    compartments: int = Form(...),
    laptop_compartment: str = Form(...),
    waterproof: str = Form(...),
    style: str = Form(...),
    color: str = Form(...),
    weight_capacity: str = Form("")
):
    try:
        weight = float(weight_capacity) if weight_capacity else 18.00
    except ValueError:
        logger.logging.info(f"Weight({weight_capacity}) can not be converted into float type!")
        weight = 18.00
    data = {
            'Brand' : brand,
            'Material' : material,
            'Size' : size,
            'Compartments' : compartments,
            'Laptop Compartment' : laptop_compartment,
            'Waterproof' : waterproof,
            'Style' : style,
            'Color' : color,
            'Weight Capacity (kg)' : weight,
    }
    prediction = {}
    try:
        prediction = model.predict(data)
    except Exception as e:
        logger.logging.info(f"Error has occured! {e}")
        prediction = {}

    # Render the result in an HTML template
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "brand": brand if brand!='None' else 'Not Specified',
            "material" : material if brand!='None' else 'Not Specified',
            "size": size if brand!='None' else 'Not Specified',
            "compartments": compartments,
            "laptop_compartment": laptop_compartment if brand!='None' else 'Not Specified',
            "waterproof": waterproof if brand!='None' else 'Not Specified',
            "style": style if brand!='None' else 'Not Specified',
            "color": color if brand!='None' else 'Not Specified',
            "weight_capacity": weight_capacity,
            "prediction": prediction.get('final prediction', 0.00),
            'adaboost_prediction' : prediction.get('adaboost', 0.00),
            'xgb_prediction' : prediction.get('xgb',0.00),            
            'lgbm_prediction' : prediction.get('lgbm',0.00),            
            'xgbrf_prediction' : prediction.get('xgbrf',0.00),            

        })