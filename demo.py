from src.prediction.predict import Predictor

model = Predictor()
data = {
    'Brand' : 'Puma',
    'Material' : 'Leather',
    'Size' : 'Medium',
    'Compartments' : 1,
    'Laptop Compartment' : 'Yes',
    'Waterproof' : 'Yes',
    'Style' : 'Tote',
    'Color' : 'Black',
    'Weight Capacity (kg)' : 22,
}
print(model.predict(data))