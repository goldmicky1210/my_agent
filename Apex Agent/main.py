from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

from agent import handle_act

WAIT_ACTION = {"type": "WaitAction", "time_seconds": 1}

app = FastAPI(title="SN36 Apex Agent")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/act")
async def act(request: Request):
    try:
        body = await request.json()
    except Exception:
        logger.warning("Failed to parse request body")
        return {"actions": [WAIT_ACTION]}

    actions = await handle_act(
        task_id=body.get("task_id"),
        prompt=body.get("prompt"),
        url=body.get("url"),
        snapshot_html=body.get("snapshot_html"),
        screenshot=body.get("screenshot"),
        step_index=body.get("step_index"),
        web_project_id=body.get("web_project_id"),
        history=body.get("history"),
    )
    return {"actions": actions}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {type(exc).__name__}: {exc}")
    return JSONResponse(status_code=200, content={"actions": [WAIT_ACTION]})
