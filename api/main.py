from fastapi import FastAPI

# 1. Initialize the framework instance
app = FastAPI(title="Data Science API Server")

# 2. Define a GET route (Reading data)
@app.get("/")
def home():
    return {"status": "Active", "message": "Welcome to the Data Science Production Hub"}
