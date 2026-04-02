"""SN36 Apex Agent - FastAPI entry point."""
from __future__ import annotations
import logging
from typing import Any

from fastapi import FastAPI, Body

from agent import handle_act

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(title="SN36 Apex Agent", version="2.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/act")
async def act(payload: dict[str, Any] = Body(...)):
    actions = await handle_act(
        task_id=payload.get("task_id"),
        prompt=payload.get("prompt") or payload.get("task_prompt"),
        url=payload.get("url"),
        snapshot_html=payload.get("snapshot_html"),
        screenshot=payload.get("screenshot"),
        step_index=payload.get("step_index"),
        web_project_id=payload.get("web_project_id"),
        history=payload.get("history"),
        relevant_data=payload.get("relevant_data") if isinstance(payload.get("relevant_data"), dict) else None,
    )
    return {"actions": actions}


@app.post("/step")
async def step(payload: dict[str, Any] = Body(...)):
    return await act(payload)
