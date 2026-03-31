"""Data models used across the agent."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel


class Selector(BaseModel):
    type: str
    attribute: str | None = None
    value: str
    case_sensitive: bool = False


class Candidate(BaseModel):
    index: int
    tag: str
    text: str
    selector: Selector
    input_type: str | None = None
    name: str | None = None
    placeholder: str | None = None
    href: str | None = None
    role: str | None = None
    context: str = ""           # surrounding card/container text for disambiguation
    parent_form: str | None = None  # enclosing <form> id/name
    options: list[str] = []     # for <select> elements
    current_value: str = ""     # current value of input


class PageContext(BaseModel):
    url: str
    title: str = ""
    headings: list[str] = []


class PageIR(BaseModel):
    context: PageContext
    candidates: list[Candidate]
    raw_text: str = ""


class Constraint(BaseModel):
    field: str
    operator: str
    value: Any


class ActionRecord(BaseModel):
    action_type: str
    selector_value: str
    url: str
    step_index: int
    text: str = ""


class TaskState(BaseModel):
    task_id: str
    history: list[ActionRecord] = []
    filled_fields: set[str] = set()
    constraints: list[Constraint] = []
    task_type: str = "general"
    login_done: bool = False
    model_config = {"arbitrary_types_allowed": True}
