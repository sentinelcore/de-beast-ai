from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()

@app.get("/")
def home():
    return {"message": "FastAPI is running on Vercel!"}

@app.get("/get_price")
def get_optimal_price():
    return {"Optimal Charging Price": "$90"}

# Required for Vercel
handler = Mangum(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

