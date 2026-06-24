from pydantic import BaseModel

class TextInput(BaseModel):
    request_id: str | None = None
    sentence: str