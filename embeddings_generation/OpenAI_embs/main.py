from fastapi import FastAPI
import openai
from pydantic import BaseModel


app = FastAPI()
client = openai.Client()


class RequestModel(BaseModel):
    question: str
    engine: str


@app.get("/respond")
def root(request: RequestModel):
    response = client.embeddings.create(
        input=request.question,
        model=request.engine,
    )

    return response.data[0].embedding

