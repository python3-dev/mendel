"""Defintions of API models."""

from pydantic import BaseModel


class GeneticRequest(BaseModel):
    """Pydantic basemodel for API request."""

    ...


class GeneticResponse(BaseModel):
    """Pydantic basemodel for API response."""

    ...
