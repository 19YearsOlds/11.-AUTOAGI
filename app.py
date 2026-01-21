from fastapi import FastAPI
from main import research_agent, generate_ai_model, task_executor, self_learning

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to AutoAGI"}

@app.get("/research")
def research():
    return {"AI Research": research_agent()}

@app.get("/generate_model")
def generate():
    return {"AI Model": generate_ai_model()}

@app.get("/execute_task")
def execute():
    return {"Task Execution": task_executor("Improve AI model")}

@app.get("/self_learning")
def learn():
    return {"Self-learning": self_learning()}