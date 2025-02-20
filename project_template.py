import os
from pathlib import Path
import logging
list_of_files = [
    'src/__init__.py',

    'src/logging/__init__.py',
    'src/logging/logger.py',

    'src/exception/__init__.py',
    'src/exception/exception.py',
    
    'src/components/__init__.py',

    'src/prediction/__init__.py',
    'src/prediction/predict.py',

    'templates/index.html',
    'templates/about.html',

    'static/styles.css',
    'static/script.js'
]

for path in list_of_files:
    filepath  = Path(path)
    filedir, filename = os.path.split(path)
    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,"w") as f:
            logging.info(f"Creating empty file: {filepath}")