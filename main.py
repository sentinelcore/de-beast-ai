from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()

@app.get("/")
def home():
    return {"message": "FastAPI is working!"}

handler = Mangum(app)