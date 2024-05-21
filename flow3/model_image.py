import os
from roboflow import Roboflow
from dotenv import load_dotenv
load_dotenv(verbose=True)


rf = Roboflow(api_key=os.environ['ROBOFLOW_API_KEY'])
project = rf.workspace().project("car-lmzmc")
model = project.version('1').model

model.predict("test_image.jpg", confidence=40, overlap=30).save("prediction_image.jpg")

