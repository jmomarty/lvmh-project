import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from recommendation_system import ThompsonSamplingRecSys


class RequestFeedback(BaseModel):
    feedback: bool


rs = ThompsonSamplingRecSys(config_fn='test_config.json')

app = FastAPI()


@app.get("/recommend")
def recommend():
    return {"recommendation": rs.recommend()}


@app.post("/feedback")
def feedback(feedback_body: RequestFeedback):
    rs.get_feedback(1 if feedback_body.feedback else 0)
    return {"msg": "Feedback received"}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
