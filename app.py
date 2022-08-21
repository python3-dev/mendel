"""Execute genetic algorithm."""


import fastapi
import uvicorn

from models.api import GeneticRequest, GeneticResponse

app = fastapi.FastAPI(
    debug=True,
    title="Mendel Genetic Optimisation API",
    description="API for Genetic Optimisation",
    version="0.1.0",
)


@app.post("/process", response_class=GeneticResponse)
async def process(request: GeneticRequest):
    """View for /process endpoint"""
    return None


if __name__ == "__main__":
    uvicorn.run("app:app", port=800, reload=True, reload_excludes=["tests/"])
