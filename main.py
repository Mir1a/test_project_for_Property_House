from fastapi import FastAPI
from crud import router as nltk_router

app = FastAPI(
    title="Test Project for Property House"
)

app.include_router(nltk_router)
