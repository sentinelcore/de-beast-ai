from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()

@app.get("/")
def home():
    print("✅ Root API was accessed!")  # Forces logs in Vercel
    return {"message": "FastAPI is running successfully on Vercel!"}

@app.get("/get_price")
def get_optimal_price():
    print("✅ Pricing API was accessed!")  # Logs API calls
    return {"Optimal Charging Price": "$90"}

# Required for Vercel Functions to detect FastAPI correctly
handler = Mangum(app)
