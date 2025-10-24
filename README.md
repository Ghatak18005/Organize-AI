# 🧠 AI-Powered File Organizer Service

### Short Description  
An **AI-powered, industrial-grade file organization service** built on **Python’s FastAPI** and an asynchronous queue system.  
It leverages the **Gemini API** for intelligent, content-based file classification and features a **mock Google OAuth2 flow**, **user-defined rule processing**, and **robust duplicate detection**.

---

## 🚀 Project Overview & Motivation

The **AI File Organizer Service** elevates file management from simple sorting to **intelligent, autonomous organization**.  
It solves the challenge of managing diverse digital assets (code, documents, media) by analyzing content and metadata in a **secure, non-blocking** way.  

The asynchronous architecture ensures scalability and responsiveness — ideal for use as a **backend utility in cloud environments** or as part of larger workflow automation pipelines.

---

## ⚙️ Core Features

| Feature | Technology Used | Benefit |
|----------|----------------|----------|
| **Intelligent Classification** | Gemini API (LLM) | Analyzes text content to assign files to semantic categories (e.g., `FINANCE`, `CODE`, `DOCUMENTS`) and suggests subfolders or descriptive renaming. |
| **Asynchronous Task Queue** | FastAPI `BackgroundTasks` | Simulates production queues (Celery/SQS). Uploads return `202 Accepted` immediately, avoiding timeouts while heavy processing runs in background threads. |
| **Rule Engine Pre-Classification** | Internal Rule JSON + Python Logic | Enables zero-latency sorting for known files. Custom user rules (e.g., extensions, keywords) are checked before invoking AI classification. |
| **Mock Google OAuth2** | FastAPI Redirects + Bearer Tokens | Demonstrates secure OAuth2 login flow. Users receive mock tokens for authenticated endpoints. |
| **Metadata Sorting** | Python Pillow (EXIF) | Extracts EXIF metadata (e.g., camera model, creation date) to classify image files without AI. |
| **Smart Conflict Resolution** | Python `hashlib` (SHA-256) | Detects and removes exact duplicates by hashing file content, conserving storage. |

## 🧰 Tech Stack

| **Layer** | **Technology** | **Description** |
|------------|----------------|-----------------|
| Backend Framework | FastAPI | High-performance async web framework for the backend. |
| Async Task Handling | FastAPI BackgroundTasks | Enables non-blocking file processing using background workers. |
| AI Integration | Gemini API | Provides intelligent, semantic file classification using LLMs. |
| Auth Layer | Mock Google OAuth2 | Demonstrates secure login and token handling flow. |
| File & Metadata Processing | Pillow, hashlib | Extracts metadata (EXIF) and performs duplicate detection via hashing. |
| Frontend | HTML, TailwindCSS, JavaScript | Simple, responsive dashboard for file management and logs. |

---

## 🧩 Local Setup & Running

### **Prerequisites**
- Python **3.8+**
- Gemini API Key (for AI classification)

### **1. Clone the Repository**
```bash
git clone https://github.com/<your-username>/ai-file-organizer-service.git
cd ai-file-organizer-service
2. Install Dependencies
bash
Copy code
pip install fastapi uvicorn[standard] python-multipart Pillow requests
3. Set the Gemini API Key
If the key is missing, all AI-based classification defaults to UNSORTED.

Windows (PowerShell):

powershell
Copy code
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
Linux/macOS (bash/zsh):

bash
Copy code
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
4. Run the Backend Service
bash
Copy code
python ai_file_organizer.py
Once started, you should see:

nginx
Copy code
Uvicorn running on http://127.0.0.1:8000
5. Open the Frontend
Simply open the index.html file in your browser to access the UI.

🔐 Authentication Flow (Mock Google OAuth2)

The API is protected by a Bearer Token mechanism.

Steps:

Click “Sign in with Google (Mock)” on the landing page.

You’ll be redirected to /auth/google/login, which immediately returns a mock access token (the user’s email).

The frontend saves this token in localStorage and attaches it as an Authorization: Bearer header for all future requests.

🧠 Core Functionality Walkthrough

A. File Upload & Asynchronous Processing
Upload: Drag-and-drop or select files from your system.

Queueing: API responds instantly with 202 Accepted.

Processing: File is classified in the background (via rules, metadata, or AI).

Monitoring: Live log entries show classification progress.

B. Live Processing Log
The frontend polls logs every 5 seconds for real-time updates.

Log Tag	Description
[API-UPLOAD]	File upload event received.
[CLASSIFICATION: AI RESULT]	File analyzed by Gemini LLM — shows final category and rename.
[CLASSIFICATION: METADATA]	File sorted using EXIF metadata (no AI needed).
[CLASSIFICATION: RULE MATCH]	User-defined rule matched; AI skipped.
[DUPLICATE SKIPPED]	Identical file detected and removed.
[SUCCESSFULLY MOVED]	File moved to final destination folder.

C. Rule Configuration Editor
Define persistent, manual rules that override AI classification.

Structure:
Each rule checks file properties (e.g., extension, name keywords).

Example:

json
Copy code
{
  "name": "Log Files",
  "conditions": [
    {"property": "extension", "equals": ".log"},
    {"property": "name_contains", "value": "debug"}
  ],
  "target": "LOGS"
}
Storage:
Rules are stored locally in rules.json and loaded at startup.

Priority:
Rules are executed before AI classification for zero-latency sorting.

**💡 Future Enhancements**
🗄 Real Database Integration — Migrate rules.json to a persistent DB (PostgreSQL, Firestore) for multi-device sync.

👥 Multi-User Segmentation — Add user-specific file paths (e.g., /data/<user_id>/CATEGORY/...).

↩ Undo Command — Implement /undo endpoint using a transaction log to reverse the last N file operations.

📈 WebSocket Logs — Replace polling with real-time WebSocket streaming for better scalability.

**🧾 License**
This project is licensed under the MIT License — feel free to use, modify, and build upon it.

**⭐ Contribute**
Pull requests are welcome!
Open an issue for bugs, enhancements, or suggestions.
