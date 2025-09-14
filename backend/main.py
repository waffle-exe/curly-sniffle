from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub import InferenceClient
from pydantic import BaseModel
import os
import json
import traceback
from typing import List, Optional
import requests
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------- CONFIGURATION ----------------
# The model 'Qwen/Qwen3-235B-A22B' is hosted by Fireworks AI.
# We must specify the provider to connect to the correct service.
# Ensure you have FIREWORKS_API_KEY in your .env file.
FIREWORKS_API_KEY = os.getenv("HF_TOKEN")
if not FIREWORKS_API_KEY:
    raise RuntimeError("Missing Fireworks API key. Please set FIREWORKS_API_KEY in .env")

# Set a long timeout for the InferenceClient to handle slow model responses.
# The `timeout` parameter here is for the underlying httpx client.
client = InferenceClient(
    provider="fireworks-ai",
    api_key=FIREWORKS_API_KEY,
    timeout=120.0 # Set a generous timeout, e.g., 120 seconds
)

USERS_FILE = "payments.json"
VERCEL_TOKEN = os.getenv("VERCEL_ACCESS_TOKEN")
VERCEL_TEAM_ID = os.getenv("VERCEL_TEAM_ID")
CHAT_HISTORY_PROJECT_NAME = '__chat_history__'


# --- FastAPI App Setup ---
app = FastAPI(
    title="Sitee AI Backend",
    description="API for managing users, projects, AI generation, and publishing.",
    version="2.6.1" # Version bump
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- PYDANTIC MODELS ----------------
class Project(BaseModel):
    name: str
    html: str
    timestamp: int
    published_url: Optional[str] = None
    react: Optional[str] = None

class User(BaseModel):
    id: str
    credits: int
    projects: List[Project] = []

class GenerationRequest(BaseModel):
    prompt: str
    user_id: str
    is_chat_mode: bool = False

class HtmlContent(BaseModel):
    html_content: str

# ---------------- DATABASE HELPERS ----------------
def read_users_db() -> List[User]:
    if not os.path.exists(USERS_FILE):
        return []
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [User(**user_data) for user_data in data]
    except (json.JSONDecodeError, IOError):
        return []

def write_users_db(users: List[User]):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump([user.model_dump() for user in users], f, indent=4)

# ---------------- API ENDPOINTS ----------------

@app.post("/users/{user_id}/projects/{timestamp}/publish")
async def publish_site(user_id: str, timestamp: int, content: HtmlContent):
    if not VERCEL_TOKEN:
        raise HTTPException(status_code=500, detail="Server is not configured for publishing. Missing VERCEL_ACCESS_TOKEN.")

    users = read_users_db()
    user = next((u for u in users if u.id == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    project = next((p for p in user.projects if p.timestamp == timestamp), None)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_name = f"sitee-{user_id.lower().replace('_', '-')}-{project.timestamp}"
    headers = {"Authorization": f"Bearer {VERCEL_TOKEN}"}
    if VERCEL_TEAM_ID:
        headers["x-vercel-team-id"] = VERCEL_TEAM_ID

    api_url = "https://api.vercel.com/v13/deployments"
    payload = {
        "name": project_name,
        "files": [{"file": "index.html", "data": content.html_content}],
        "projectSettings": {"framework": None}
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        deployment_url = f"https://{data['url']}"
        project.published_url = deployment_url
        write_users_db(users)
        return {"url": deployment_url}
    except requests.exceptions.RequestException as e:
        error_details = e.response.json() if e.response else {"error": {"message": str(e)}}
        raise HTTPException(status_code=502, detail=f"Publishing service error: {error_details.get('error', {}).get('message')}")

@app.post("/generate/")
async def generate_code(request: GenerationRequest):
    users = read_users_db()
    user = next((u for u in users if u.id == request.user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.credits <= 0:
        raise HTTPException(status_code=400, detail="Insufficient credits")
    
    user.credits -= 1
    write_users_db(users)

    try:
        model = "Qwen/Qwen3-235B-A22B" 
        if request.is_chat_mode:
            messages = [
                {"role": "system", "content": "You are Sitee, an AI assistant. Provide concise, direct answers."},
                {"role": "user", "content": request.prompt}
            ]
            completion = client.chat.completions.create(
                model=model, messages=messages, max_tokens=2048, temperature=0.5,
                timeout=120.0 # Add timeout for chat as well
            )
            chat_response = completion.choices[0].message.content
            return JSONResponse(content={"html": chat_response, "credits_remaining": user.credits})
        
        else:
            # --- Step 1: Generate HTML ---
            html_messages = [
                {"role": "system", "content": "You are an expert web developer specializing in creating single-file HTML websites using Tailwind CSS. Your output must be only the raw, runnable HTML code. The design must be professional and responsive. Use vanilla JavaScript inside a `<script>` tag. Include a footer that says 'Made with love by sitee'."},
                {"role": "user", "content": request.prompt}
            ]
            completion = client.chat.completions.create(
                model=model, messages=html_messages, max_tokens=8192, temperature=0.7,
                timeout=120.0 # Add a timeout to this call
            )
            generated_content = completion.choices[0].message.content
            
            html_start_tag = "<!DOCTYPE html>"
            html_start_index = generated_content.find(html_start_tag)
            if html_start_index != -1:
                generated_content = generated_content[html_start_index:]
            html_code = generated_content

            # --- Step 2: Generate React from HTML ---
            react_messages = [
                {"role": "system", "content": """You are an expert React developer. Convert the provided HTML into a single, self-contained React JSX file. The root component must be `App` and exported as default. Use functional components and hooks. Convert all HTML to JSX syntax (`className`, etc.). Convert `<style>` blocks into a component that injects styles into the document head using `useEffect`. Convert `<script>` logic into `useEffect` hooks and event handlers. The entire output MUST be ONLY the raw, runnable JSX code, with no explanations."""},
                {"role": "user", "content": html_code}
            ]
            react_completion = client.chat.completions.create(
                model=model, messages=react_messages, max_tokens=8192, temperature=0.4,
                timeout=120.0 # Add a timeout to this call
            )
            react_code = react_completion.choices[0].message.content

            return JSONResponse(content={
                "html": html_code, 
                "react": react_code,
                "credits_remaining": user.credits
            })

    except Exception as e:
        user.credits += 1
        write_users_db(users)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred during generation: {e}")

# === User and Project Management Endpoints ===

@app.get("/users/{user_id}", response_model=User)
def get_user_data(user_id: str):
    users = read_users_db()
    user = next((u for u in users if u.id == user_id), None)
    if user:
        return user
    raise HTTPException(status_code=404, detail="User not found")

@app.post("/users/{user_id}/projects", response_model=Project)
def save_project(user_id: str, project: Project):
    users = read_users_db()
    user = next((u for u in users if u.id == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if project.name == CHAT_HISTORY_PROJECT_NAME:
        existing_chat = next((p for p in user.projects if p.name == CHAT_HISTORY_PROJECT_NAME), None)
        if existing_chat:
            existing_chat.html = project.html
            existing_chat.timestamp = project.timestamp
            write_users_db(users)
            return existing_chat

    user.projects.append(project)
    write_users_db(users)
    return project

@app.put("/users/{user_id}/projects/{timestamp}", response_model=Project)
def update_project_code(user_id: str, timestamp: int, updated_project: Project = Body(...)):
    users = read_users_db()
    user = next((u for u in users if u.id == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    project_to_update = next((p for p in user.projects if p.timestamp == timestamp), None)
    if not project_to_update:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project_to_update.html = updated_project.html
    project_to_update.name = updated_project.name
    project_to_update.published_url = updated_project.published_url
    project_to_update.react = updated_project.react
    
    write_users_db(users)
    return project_to_update

@app.delete("/users/{user_id}/projects/{timestamp}", status_code=204)
def delete_project(user_id: str, timestamp: int):
    users = read_users_db()
    user = next((u for u in users if u.id == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    initial_project_count = len(user.projects)
    user.projects = [p for p in user.projects if p.timestamp != timestamp]
    if len(user.projects) == initial_project_count:
        raise HTTPException(status_code=404, detail="Project not found to delete")
    write_users_db(users)
    return {}

@app.delete("/users/{user_id}/projects")
def delete_all_projects(user_id: str):
    users = read_users_db()
    user = next((u for u in users if u.id == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.projects = [p for p in user.projects if p.name == CHAT_HISTORY_PROJECT_NAME]
    write_users_db(users)
    return {"message": "All projects for user have been deleted."}

@app.post("/create-user", response_model=User, status_code=201)
def create_user(user_data: dict):
    users = read_users_db()
    user_id = user_data.get("id")
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    if any(u.id == user_id for u in users):
        raise HTTPException(status_code=400, detail="User with this ID already exists")
    
    new_user = User(id=user_id, credits=user_data.get("credits", 10), projects=[])
    users.append(new_user)
    write_users_db(users)
    return new_user

@app.get("/all-users", response_model=List[User])
def get_all_users():
    return read_users_db()

# --- Main Entry Point ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
