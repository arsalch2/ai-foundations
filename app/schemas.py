"""Request/response models. The OrderExtraction model doubles as the JSON schema the LLM must fill."""

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    temperature: float = Field(default=0.2, ge=0, le=2)


class ChatResponse(BaseModel):
    answer: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


# ---------- Structured extraction (preview of portfolio project 2) ----------

class OrderItem(BaseModel):
    product: str = Field(description="Product name or article number exactly as written in the email")
    quantity: int = Field(description="Ordered quantity as a whole number")
    unit: str | None = Field(description="Unit such as 'pcs', 'Stück', 'kg'; null if not stated")


class OrderExtraction(BaseModel):
    is_order: bool = Field(
        description="True if the email places a new order or changes quantities of an existing one. "
        "False for questions, cancellations and spam."
    )
    customer_company: str | None = Field(description="Company name of the sender; null if unknown")
    customer_email: str | None = Field(description="Sender email address; null if unknown")
    items: list[OrderItem] = Field(description="Ordered items; empty list if not an order")
    requested_delivery_date: str | None = Field(
        description="Requested delivery date as YYYY-MM-DD; null if not stated or ambiguous"
    )
    language: Literal["de", "en", "other"] = Field(description="Main language of the email")


class ExtractRequest(BaseModel):
    email_text: str = Field(min_length=1, max_length=20_000)


class ExtractResponse(BaseModel):
    result: OrderExtraction | None
    refusal: str | None = None


# ---------- Tool calling ----------

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)


class ToolCallLog(BaseModel):
    name: str
    arguments: dict
    result: dict


class AskResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCallLog]
