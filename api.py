from fastapi import FastAPI , Body
from utils import get_recommendations
from pydantic import BaseModel

app = FastAPI()

class Product(BaseModel):
    name:str

@app.post("/")
async def predict(product:Product):
    try:
        products = get_recommendations(product.name)
        return products
    except IndexError:
        return {"error":"Product not found"}
