from fastapi import FastAPI
from app.core.logging import setup_logging
from app.routes.health import router as health_router
from app.routes.recommend import router as recommend_router
from app.routes.pricing import router as pricing_router
from app.routes.inventory import router as inventory_router
from app.routes.chatbot import router as chatbot_router

setup_logging()

app = FastAPI(title="NextGen Retail AI API", version="1.0.0")
app.include_router(health_router)
app.include_router(recommend_router)
app.include_router(pricing_router)
app.include_router(inventory_router)
app.include_router(chatbot_router)
