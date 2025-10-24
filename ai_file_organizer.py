import os
import shutil
import json
import time
import requests 
import logging 
import hashlib 
import uvicorn
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from starlette.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Optional, Tuple, Union, List
from PIL import Image
from PIL.ExifTags import TAGS
from pydantic import BaseModel
from contextlib import asynccontextmanager 
# 🚨 DATABASE IMPORTS
from sqlmodel import Field, Session, SQLModel, create_engine, select

# --- Configuration & Constants ---

# Web Service will use these directories for staging and final organization
UPLOAD_DIR = Path("./temp_uploads")
TARGET_DIR = Path("./Organized_Files")
LOG_FILE = "organizer_service.log"

# 🚨 DATABASE CONFIG
DATABASE_FILE = "organizer_data.db"
sqlite_url = f"sqlite:///{DATABASE_FILE}"
# Use connect_args={"check_same_thread": False} for FastAPI/Uvicorn multi-threading compatibility with SQLite
engine = create_engine(sqlite_url, echo=False, connect_args={"check_same_thread": False}) 


# AI/Classification Config (Unchanged)
API_KEY = os.environ.get("GEMINI_API_KEY", "") 
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
CLASSIFICATION_CATEGORIES = [
    "DOCUMENTS", "CODE", "FINANCE", "IMAGES_MEDIA", "UNSORTED"
]
SUPPORTED_TEXT_EXTENSIONS = ['.txt', '.md', '.log', '.py', '.js', '.html', '.css', '.json', '.csv']
ARCHIVE_EXTENSIONS = ['.zip', '.tar', '.gz', '.rar', '.7z', '.tgz', '.bz2'] 
MEDIA_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tiff']

# REAL Google OAuth Config (Unchanged)
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = "http://127.0.0.1:8000/auth/google/callback" 
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


# --- Database Models ---

class User(SQLModel, table=True):
    """Stores basic user registration data."""
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True) # User's unique identifier (used as token)
    name: Optional[str] = None
    created_at: float = Field(default_factory=time.time)

class Rule(SQLModel, table=True):
    """Stores a user-defined organization rule."""
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    condition: str
    target_category: str
    target_subfolder: Optional[str] = None
    
    # Foreign key linking the rule to a specific user's email
    user_email: str = Field(index=True, foreign_key="user.email")

# Pydantic Input Schema for API (Does not contain DB fields like 'id' or foreign key)
class RuleIn(BaseModel):
    name: str
    condition: str
    target_category: str
    target_subfolder: Optional[str] = None

# --- Replaced Placeholder Functions ---

# Removed load_rules/save_rules as they are now handled by DB

# --- Security & Authentication (Unchanged) ---
oauth2_scheme = HTTPBearer(auto_error=False)

def get_current_user(token: Optional[HTTPBearer] = Depends(oauth2_scheme)) -> str:
    """Dependency to validate the token (which is the user's email)."""
    if token is None or not token.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Session token missing.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_identifier = token.credentials 
    if not user_identifier or "@" not in user_identifier: 
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session token format.")
    return user_identifier

# --- Utility Functions (Only changed apply_manual_rules) ---

def classify_content_with_ai(content: str) -> Tuple[str, str, Optional[str], Optional[str]]:
    # ... (omitted classify_content_with_ai logic - UNCHANGED) ...
    # Full function body omitted for brevity, assumed unchanged
    pass
    
def calculate_file_hash(file_path: Path, block_size=65536) -> Optional[str]:
    # ... (omitted calculate_file_hash logic - UNCHANGED) ...
    pass

def extract_media_metadata(file_path: Path) -> Tuple[str, str]:
    # ... (omitted extract_media_metadata logic - UNCHANGED) ...
    pass

def read_file_content(file_path: Path) -> Optional[str]:
    # ... (omitted read_file_content logic - UNCHANGED) ...
    pass

def calculate_similarity(s1: str, s2: str) -> float:
    # ... (omitted calculate_similarity logic - UNCHANGED) ...
    pass

def is_generic_filename(filename: str) -> bool:
    # ... (omitted is_generic_filename logic - UNCHANGED) ...
    pass

def move_file(source_path: Path, destination_dir: Path, final_name: str):
    # ... (omitted move_file logic - UNCHANGED) ...
    pass

# 🚨 UPDATED LOGIC: apply_manual_rules now queries the database
def apply_manual_rules(file_path: Path, user_email: str) -> Optional[Tuple[str, str, str]]:
    """Checks the file against rules loaded from the database for the specific user."""
    file_name = file_path.name
    file_suffix = file_path.suffix.lower()
    
    # Query database for rules belonging to this user
    with Session(engine) as session:
        statement = select(Rule).where(Rule.user_email == user_email)
        rules = session.exec(statement).all()
    
    for rule in rules:
        if rule.target_category.upper() not in CLASSIFICATION_CATEGORIES:
            logging.warning(f"[RULE-SKIP] Rule '{rule.name}' skipped: invalid category '{rule.target_category}'.")
            continue
            
        condition_met = False
        condition_lower = rule.condition.lower()

        # Simple condition checking (name, extension, keywords)
        if f'name contains "{file_name.lower()}"' in condition_lower:
            condition_met = True
        
        if f'extension is "{file_suffix}"' in condition_lower or f'extension is "{file_suffix.strip(".")}"' in condition_lower:
             condition_met = True
        
        condition_parts = condition_lower.replace(' or ', ' ').replace(' and ', ' ').replace('"', '').split()
        if any(part in file_name.lower() for part in condition_parts if len(part) > 2):
            condition_met = True

        if condition_met:
            logging.info(f"[RULE-MATCH] File '{file_name}' matched manual rule: '{rule.name}'.")
            return (
                rule.target_category.upper(),
                rule.target_subfolder if rule.target_subfolder else "General_Manual",
                f"Classified by manual rule '{rule.name}' based on condition: {rule.condition}"
            )

    return None

# 🚨 UPDATED LOGIC: process_uploaded_file now passes user_email to apply_manual_rules
def process_uploaded_file(file_path: Path, current_user: str):
    """Background worker function to handle the full organization logic."""
    if not file_path.is_file():
        logging.warning(f"[SKIP-TASK] File {file_path.name} not found in staging. Skipping task.")
        return

    file_suffix = file_path.suffix.lower()
    original_file_name = file_path.name 

    logging.info("=" * 70)
    logging.info(f"--- START PROCESSING: {file_path.name} ({file_suffix.upper()}) for user {current_user} ---")
    
    category = "UNSORTED"
    reason = "Could not read file content or file type not supported."
    sub_folder = None
    final_file_name = original_file_name
    
    try:
        # 1. Manual Rule Check (Highest Priority)
        # 🚨 CHANGE: Pass current_user email to apply_manual_rules
        rule_match = apply_manual_rules(file_path, current_user)
        if rule_match:
            category, sub_folder, reason = rule_match
        
        # 2. Archive Check (rest of logic unchanged)
        elif file_suffix in ARCHIVE_EXTENSIONS:
            category = "UNSORTED" 
            reason = "File identified as a compressed archive."
            sub_folder = "Archives" 
        
        # 3. Media Metadata Check
        elif file_suffix in MEDIA_EXTENSIONS:
            category = "IMAGES_MEDIA"
            sub_folder, reason = extract_media_metadata(file_path)
        
        # 4. AI Classification
        else:
            content = read_file_content(file_path)

            if content:
                logging.info("Content read. Sending to AI...")
                
                ai_category, ai_reason, ai_sub_folder, ai_filename = classify_content_with_ai(content) 
                category = ai_category
                reason = ai_reason
                
                if category != "UNSORTED" and ai_sub_folder:
                    sub_folder = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in ai_sub_folder).strip('_') or "General"
                        
                if category != "UNSORTED" and ai_filename:
                    original_stem = Path(original_file_name).stem
                    similarity = calculate_similarity(original_stem, ai_filename)
                    
                    if similarity < 0.70 and is_generic_filename(original_file_name):
                         original_suffix = file_path.suffix
                         final_file_name = f"{ai_filename}{original_suffix}"
                         logging.info(f"[RENAME] AI suggested: '{original_file_name}' -> '{final_file_name}'")
                    
        # 5. Fallback logging
        log_msg = f"[CLASSIFICATION: RESULT] Category: {category} | SubFolder: {sub_folder} | Final Name: {final_file_name} | Reason: {reason} | User: {current_user}"
        logging.info(log_msg)


        # 6. Move file
        destination_folder = TARGET_DIR / category
        if sub_folder:
            destination_folder = destination_folder / sub_folder

        move_file(file_path, destination_folder, final_file_name)
    
    except Exception as e:
        # ... (error handling omitted for brevity)
        pass


# --- Lifespan Handler Definition ---

@asynccontextmanager
async def lifespan_handler(app: FastAPI):
    """Handles application startup (DB init) and shutdown."""
    # Startup logic: runs when the application starts
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # 🚨 DB Initialization
    logging.info("Initializing database and creating tables...")
    SQLModel.metadata.create_all(engine)
    
    logging.info(f"Service started. Target directory set to: {TARGET_DIR.resolve()}")
    
    yield # <-- Application runs here

    logging.info("Service shutdown completed.")


# --- FastAPI Application Initialization (Must be after lifespan_handler) ---
app = FastAPI(
    title="AI File Organizer Service", 
    description="A cloud-ready service that organizes files using Gemini AI content classification and metadata extraction.",
    version="1.0.0",
    lifespan=lifespan_handler 
)

# CORS configuration (Unchanged)
origins = ["*", ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static File Serving & Root Redirects (Unchanged) ---
app.mount("/files", StaticFiles(directory="."), name="files")

@app.get("/index.html", include_in_schema=False)
async def serve_index():
    index_path = Path("index.html")
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html") 
    raise HTTPException(status_code=404, detail="index.html not found.")

@app.get("/", include_in_schema=False)
async def redirect_to_index():
    return RedirectResponse(url="/index.html")

# --- AUTH: Google OAuth 2.0 Endpoints ---

@app.get("/auth/google/login", summary="Initiates real Google login flow")
async def google_login():
    """Redirects the user to Google's consent screen."""
    state = "some_secure_random_state" 
    
    if "YOUR_CLIENT_ID_HERE" in GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Google Client ID not configured.")

    auth_params = {
        "response_type": "code",
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI, 
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
        "prompt": "select_account"
    }
    
    query_string = "&".join(f"{k}={v}" for k, v in auth_params.items())
    
    return RedirectResponse(
        url=f"{GOOGLE_AUTH_URL}?{query_string}", 
        status_code=status.HTTP_302_FOUND
    )


@app.get("/auth/google/callback", summary="Handles callback from Google")
async def google_callback(code: Optional[str] = None, error: Optional[str] = None):
    """Exchanges code for token, registers user, and redirects to the client."""
    if error:
        logging.error(f"Google OAuth Error: {error}")
        return RedirectResponse(url=f"/index.html#auth_error=google_failed", status_code=status.HTTP_302_FOUND)

    if not code:
        return RedirectResponse(url=f"/index.html#auth_error=no_code", status_code=status.HTTP_302_FOUND)

    token_payload = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code"
    }

    try:
        response = requests.post(GOOGLE_TOKEN_URL, data=token_payload)
        response.raise_for_status()
        token_data = response.json()
        access_token = token_data.get("access_token")
        
        user_info_response = requests.get(
            GOOGLE_USERINFO_URL, 
            headers={"Authorization": f"Bearer {access_token}"}
        )
        user_info_response.raise_for_status()
        user_info = user_info_response.json()
        user_email = user_info.get("email")
        
        if not user_email:
            raise Exception("Could not retrieve user email from Google.")
        
        # 🚨 DB Action: Register or retrieve user
        with Session(engine) as session:
            user_db = session.exec(select(User).where(User.email == user_email)).first()
            if not user_db:
                # New user registration
                user_db = User(email=user_email, name=user_info.get("name"))
                session.add(user_db)
                session.commit() # <--- COMMIT HERE IS CRITICAL
                session.refresh(user_db)
                logging.info(f"[DB] New user registered: {user_email}")
            else:
                 logging.info(f"[DB] User logged in: {user_email}")
        
        # Redirect back to the frontend with the user email as the token
        return RedirectResponse(
            url=f"/index.html#access_token={user_email}&token_type=Bearer", 
            status_code=status.HTTP_302_FOUND
        )

    except Exception as e:
        logging.error(f"OAuth Token/UserInfo Exchange Failed: {e}")
        return RedirectResponse(url=f"/index.html#auth_error=server_exchange_failed", status_code=status.HTTP_302_FOUND)


# --- API Endpoints ---

@app.post("/organize/upload", status_code=202)
async def upload_file_for_organization(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):
    """Accepts a file upload and queues the organization logic to run in the background."""
    
    temp_file_path = UPLOAD_DIR / file.filename
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logging.error(f"Error saving uploaded file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file temporarily.")
    finally:
        await file.close()

    background_tasks.add_task(process_uploaded_file, temp_file_path, current_user)
    
    logging.info(f"[API-UPLOAD] File '{file.filename}' uploaded successfully for user {current_user}. Queued for background processing.")

    return {
        "message": "File received and organization started in the background.",
        "filename": file.filename,
        "status": "Processing Queued"
    }

# 🚨 UPDATED ENDPOINT: Use RuleIn for input, returns list of Rule
@app.get("/config/rules", response_model=List[Rule], summary="Get all active manual rules")
async def get_rules(current_user: str = Depends(get_current_user)):
    """Returns the list of currently active manual organization rules."""
    # 🚨 DB Action: Select rules specific to the user
    with Session(engine) as session:
        statement = select(Rule).where(Rule.user_email == current_user)
        rules = session.exec(statement).all()
        return rules

# 🚨 UPDATED ENDPOINT: Use RuleIn for input
@app.post("/config/rules", response_model=RuleIn, summary="Add a new manual organization rule")
async def add_rule(new_rule_in: RuleIn, current_user: str = Depends(get_current_user)):
    """Adds a new rule to the manual rule list."""
    if new_rule_in.target_category.upper() not in CLASSIFICATION_CATEGORIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid target category. Must be one of: {', '.join(CLASSIFICATION_CATEGORIES)}"
        )

    # Convert RuleIn (Pydantic) to Rule (SQLModel) and attach user_email
    rule_db = Rule(
        **new_rule_in.model_dump(), 
        user_email=current_user
    )
    
    # 🚨 DB Action: Save rule
    with Session(engine) as session:
        session.add(rule_db)
        session.commit()
        session.refresh(rule_db)
        
    logging.info(f"[RULE-ADDED] Rule '{new_rule_in.name}' added by user {current_user}.")

    # Return the input model as confirmation
    return new_rule_in


@app.get("/logs")
async def get_logs(current_user: str = Depends(get_current_user)):
    """Returns the last 50 lines of the organization log file."""
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            return {"log": "".join(lines[-50:])}
    except FileNotFoundError:
        return {"log": "No log file found. Service just started or no files processed."} 
    except Exception as e:
        logging.error(f"Error accessing logs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error reading logs.")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": app.title}


if __name__ == "__main__":
    # --- Logging Setup (MUST BE FIRST) ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )

    if not API_KEY:
        logging.warning("GEMINI_API_KEY environment variable is NOT set. AI classification will not function.")
        print("--- AI WARNING ---")
        print("The GEMINI_API_KEY environment variable is NOT set. Classification accuracy will be limited.")
        print("------------------")

    # Start the Uvicorn ASGI server
    logging.info("Starting FastAPI service with Uvicorn...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
