from pydantic import BaseModel


class InputSchema(BaseModel):
    arxiv_url: str
