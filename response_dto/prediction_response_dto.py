from pydantic import BaseModel
from typing import List


class PredictionResponseDto(BaseModel):
    filename: str
    contenttype: str
    likely_class: str
