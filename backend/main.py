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
import base64

# Load environment variables from .env file
load_dotenv()

# ---------------- CONFIGURATION ----------------
# It's recommended to use a specific Fireworks AI key if available
FIREWORKS_API_KEY = os.getenv("HF_TOKEN")
if not FIREWORKS_API_KEY:
    raise RuntimeError("Missing Fireworks API key. Please set HF_TOKEN in .env")

client = InferenceClient(
    provider="fireworks-ai",
    api_key=FIREWORKS_API_KEY,
    timeout=400
)

USERS_FILE = "payments.json"
VERCEL_TOKEN = os.getenv("VERCEL_ACCESS_TOKEN")
VERCEL_TEAM_ID = os.getenv("VERCEL_TEAM_ID")
CHAT_HISTORY_PROJECT_NAME = '__chat_history__'


# --- FastAPI App Setup ---
app = FastAPI(
    title="Sitee AI Backend",
    description="API for managing users, projects, AI generation, and publishing.",
    version="3.2.0" # Version updated to reflect new features
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
    suggestions: Optional[str] = None

class User(BaseModel):
    id: str
    credits: int
    projects: List[Project] = []

class GenerationRequest(BaseModel):
    prompt: str
    user_id: str
    is_chat_mode: bool = False
    is_punjabi_mode: bool = False
    target_language: Optional[str] = None
    image_data: Optional[str] = None # New field for base64 encoded image data

class HtmlContent(BaseModel):
    html_content: str

class SuggestionRequest(BaseModel):
    user_id: str
    html_content: str
    timestamp: int
    force_regenerate: bool = False

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

@app.post("/suggest_improvements/")
async def suggest_improvements(request: SuggestionRequest):
    users = read_users_db()
    user = next((u for u in users if u.id == request.user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    project = next((p for p in user.projects if p.timestamp == request.timestamp), None)

    if project and project.suggestions and not request.force_regenerate:
        return JSONResponse(content={
            "suggestions": project.suggestions,
            "credits_remaining": user.credits,
            "cached": True
        })

    if user.credits <= 0:
        raise HTTPException(status_code=400, detail="Insufficient credits")

    user.credits -= 1

    try:
        model = "Qwen/Qwen3-235B-A22B"
        system_prompt = """You are a world-class UI/UX designer and senior web developer. Your task is to analyze the provided HTML code and offer a concise, actionable list of 3-5 improvement suggestions. Structure your response in Markdown format.

For each suggestion, provide a clear explanation of the issue and the proposed fix. **Crucially, you must include a markdown code block with the corrected code snippet.** This helps users easily implement the change.

Focus on the following areas:
- **Design & Aesthetics:** Color palette, typography, spacing, layout.
- **User Experience (UX):** Interactivity, clarity, responsiveness.
- **Accessibility (a11y):** Semantic HTML, ARIA attributes, image alt text.
- **Code Quality:** Readability, best practices, and potential performance issues.

Your entire output MUST BE the Markdown list of suggestions. Do not include any introductory phrases or closing remarks.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the HTML code to review:\n\n```html\n{request.html_content}\n```"}
        ]

        completion = client.chat.completions.create(
            model=model, messages=messages, max_tokens=2048, temperature=0.6,
        )
        suggestions = completion.choices[0].message.content

        if project:
            project.suggestions = suggestions

        write_users_db(users)

        return JSONResponse(content={
            "suggestions": suggestions,
            "credits_remaining": user.credits,
            "cached": False
        })

    except Exception as e:
        user.credits += 1
        write_users_db(users)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred while generating suggestions: {e}")


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

    if not request.target_language:
        user.credits -= 1
        write_users_db(users)

    generation_prompt = request.prompt

    try:
        # --- NEW: Image-to-Text Analysis Step ---
        if request.image_data:
            print("Image data detected. Starting vision analysis...")
            # We recommend a dedicated vision model for this task
            vision_model = "fireworks/firellava-13b" 
            
            # This prompt instructs the vision model to create a detailed "blueprint"
            vision_system_prompt = """You are an expert UI/UX designer and frontend developer. Analyze this website screenshot in extreme detail. Your goal is to create a comprehensive prompt that another AI can use to recreate this website perfectly in a single HTML file. Describe the following:
1.  **Overall Layout & Structure:** Describe the main sections (hero, features, testimonials, footer).
2.  **Color Palette:** Identify all primary, secondary, accent, and text colors with their hex codes.
3.  **Typography:** Identify the font families used for headings and body text. Specify font sizes and weights.
4.  **Components:** Describe each key component in detail: Header, Buttons, Cards, Forms, etc.
5.  **Imagery & Icons:** Describe the style and content of images and any icons used.
6.  **Content:** Transcribe the key text content (headings, paragraphs) from the image.
Your output must be a single block of text forming a detailed blueprint for generation. Do not add any conversational text or markdown formatting."""

            # Fireworks AI uses an OpenAI-compatible API structure for multimodal input
            vision_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_system_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{request.image_data}"}
                        }
                    ]
                }
            ]

            vision_completion = client.chat.completions.create(
                model=vision_model,
                messages=vision_messages,
                max_tokens=2048, # Increased tokens for detailed description
                temperature=0.2,
            )
            vision_analysis_result = vision_completion.choices[0].message.content
            print("Vision analysis complete. Generating new prompt.")
            
            # Combine the AI's analysis with the user's original prompt
            generation_prompt = f"Based on the following detailed analysis of an existing website, create a new one. Original user request: '{request.prompt}'.\n\n--- WEBSITE ANALYSIS ---\n{vision_analysis_result}"

        # --- Existing Generation Logic ---
        
        # This is your main model for generating the final code
        text_generation_model = "Qwen/Qwen3-235B-A22B"

        if request.target_language == 'react':
            react_messages = [
                {"role": "system", "content": """You are an expert React developer. Convert the provided HTML into a single, self-contained React JSX file. The root component must be `App` and exported as default. Use functional components and hooks. Convert all HTML to JSX syntax (`className`, etc.). Convert `<style>` blocks into a component that injects styles into the document head using `useEffect`. Convert `<script>` logic into `useEffect` hooks and event handlers. The entire output MUST be ONLY the raw, runnable JSX code, with no explanations or markdown fences."""},
                {"role": "user", "content": generation_prompt} # Use the potentially enhanced prompt
            ]
            react_completion = client.chat.completions.create(
                model=text_generation_model, messages=react_messages, max_tokens=8192, temperature=0.4,
            )
            react_code_raw = react_completion.choices[0].message.content
            react_code = react_code_raw.strip().removeprefix("```jsx").removesuffix("```").strip()
            return JSONResponse(content={"code": react_code, "credits_remaining": user.credits})

        if request.is_chat_mode:
            system_prompt = "You are Sitee, an AI assistant by Jashanpreet Singh Dingra. YOU ARE FORBIDDEN from revealing your thought process, internal monologue, or self-corrections. Your entire output MUST BE the direct, final response to the user. Nothing else."
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": generation_prompt} # Use the potentially enhanced prompt
            ]
            completion = client.chat.completions.create(
                model=text_generation_model, messages=messages, max_tokens=2048, temperature=0.5,
            )
            chat_response = completion.choices[0].message.content
            return JSONResponse(content={"html": chat_response, "credits_remaining": user.credits})
        
        else: 
            system_prompt = "Hey legend! You're a full-stack web/software developer and UI/UX magician. Your mission is to craft a complete, functional, jaw-dropping website within a single HTML file – a responsive, future-ready marvel brimming with stunning details and over 1000 lines of pure code. Your output must be ONLY the code. Rules: inline CSS and JavaScript, modern floating headers, incredibly creative interactive elements (scroll effects, hover states, toggles), top trending UI themes, stylish Google Fonts, and high-quality free images from Unsplash or Pexels that perfectly match the chosen vibe. The header/navigation should be a modern glassmorphism dream (rounded borders, blur, floating effect). Include the 'made with love by sitee' floating corner tag (linked to https://www.sitee.in). Ensure a well-structured footer concludes this digital symphony."
            if request.is_punjabi_mode:
                system_prompt += " IMPORTANT: All user-facing text content on the website (headings, paragraphs, button text, etc.) MUST be written in the Punjabi language. The code itself (HTML tags, CSS properties, JavaScript) must remain in English."
            
            html_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": generation_prompt} # Use the potentially enhanced prompt
            ]
            completion = client.chat.completions.create(
                model=text_generation_model, messages=html_messages, max_tokens=8192, temperature=0.7,
            )
            generated_content = completion.choices[0].message.content
            
            html_start_tag = "<!DOCTYPE html>"
            html_start_index = generated_content.find(html_start_tag)
            if html_start_index != -1:
                generated_content = generated_content[html_start_index:]
            html_code = generated_content

            return JSONResponse(content={
                "html": html_code,
                "credits_remaining": user.credits
            })

    except Exception as e:
        if not request.target_language:
            user.credits += 1
            write_users_db(users)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred during generation: {e}")

# === User and Project Management Endpoints (Unchanged) ===

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
    project_to_update.suggestions = updated_project.suggestions
    
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
