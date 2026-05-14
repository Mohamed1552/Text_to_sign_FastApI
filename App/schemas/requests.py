from pydantic import BaseModel

class TextInput(BaseModel):
    sentence: str