import os
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..configs.loader import load_config
from ..models import build_model


API_TOKEN = os.environ.get("TRUTHGPT_API_TOKEN", "changeme")
CONFIG_PATH = os.environ.get(
    "TRUTHGPT_CONFIG",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs",
        "llm_default.yaml",
    ),
)


def _load_model():
    cfg = load_config(CONFIG_PATH, overrides=None)
    return build_model(cfg.model.family, cfg.dict())


app = FastAPI(title="TruthGPT Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None


@app.on_event("startup")
def startup() -> None:
    global MODEL
    MODEL = _load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/generate")
def generate(q: str, max_new_tokens: int = 64, temperature: float = 0.8, authorization: Optional[str] = Header(None)):
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")
    out = MODEL.infer({"text": q, "max_new_tokens": max_new_tokens, "temperature": temperature})
    return out






