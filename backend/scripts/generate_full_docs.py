import os
from fpdf import FPDF
import datetime

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'DrugGuard Project Documentation', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 16)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 12, title, 0, 1, 'L', 1)
        self.ln(4)

    def section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
    
    def file_desc(self, filename, description):
        self.set_font('Arial', 'B', 10)
        self.cell(0, 6, f"{filename}", 0, 1)
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, description)
        self.ln(3)

pdf = PDF()
pdf.alias_nb_pages()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

# Title Page
pdf.set_font('Arial', 'B', 24)
pdf.cell(0, 60, '', 0, 1)
pdf.cell(0, 10, 'DrugGuard', 0, 1, 'C')
pdf.cell(0, 10, 'Comprehensive System Documentation', 0, 1, 'C')
pdf.set_font('Arial', '', 12)
pdf.cell(0, 20, f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d')}", 0, 1, 'C')
pdf.add_page()

# --- BACKEND ---
pdf.chapter_title("Backend Architecture (c:\\Drug\\backend)")

pdf.section_title("1. Root Directory")
pdf.file_desc(".env", "Stores sensitive environment variables like API keys (Gemini, FDA), Database URLs, and Secret Keys. Keeps configuration out of the codebase.")
pdf.file_desc(".env.example", "A template for environment variables. Developers copy this to .env and fill in their real keys.")
pdf.file_desc("requirements.txt", "Lists all Python library dependencies (FastAPI, SQLAlchemy, PyTorch, etc.) needed to run the backend.")
pdf.file_desc("run.bat / build_all.bat", "Windows batch scripts to automate the startup process of both the backend server and frontend application.")
pdf.file_desc("drug_interactions.db", "Approx 27GB read-only SQLite database containing the massive dataset of known drug-drug interactions.")
pdf.file_desc("druguard.db", "The application's main writeable database. Stores User accounts, Profiles, History, and Logs.")
pdf.file_desc("test_quick.py", "A lightweight smoke-test script to verify that the Database and API are reachable without running the full test suite.")

pdf.section_title("2. Core Application (app/)")
pdf.file_desc("main.py", "The entry point of the FastAPI application. Initializes the app, configures CORS/Security, and mounts all API routers.")
pdf.file_desc("config.py", "Loads settings from .env and exposes them as a strongly-typed Settings object for use throughout the app.")
pdf.file_desc("database.py", "Manages the SQLAlchemy database connection, session creation, and engine configuration.")
pdf.file_desc("models.py", "Defines global database tables using SQLAlchemy ORM (e.g., Users, Roles, AuditLogs).")
pdf.file_desc("schemas.py", "Defines Pydantic models for API data validation (Request/Response bodies) shared across modules.")

pdf.section_title("3. Service Layer (app/services/)")
pdf.file_desc("api_client.py", "Client for communicating with external medical APIs (OpenFDA, RxNav) to fetch real-time drug information.")
pdf.file_desc("auth.py", "Handles User Authentication: Password hashing (Bcrypt), JWT token generation, and user verification.")
pdf.file_desc("cache.py", "Implements caching logic to store results of expensive queries (like drug lookups) to improve performance.")
pdf.file_desc("interaction_service.py", "The core engine that queries the drug_interactions.db to find risks between a list of medications.")
pdf.file_desc("food_interactions.py", "Specialized service for identifying Drug-Food interactions (e.g., avoiding dairy with certain antibiotics).")
pdf.file_desc("ocr_service.py", "Handles Optical Character Recognition (OCR) to convert uploaded images of documents into text.")
pdf.file_desc("pdf_generator.py", "Generates downloadable PDF medical reports for patients based on their analysis results.")
pdf.file_desc("robust_fetcher.py", "A resilient data fetching utility with retry logic and error handling for unstable external APIs.")
pdf.file_desc("rate_limiter.py", "Middleware that prevents abuse by limiting the number of API requests a user can make in a given timeframe.")

pdf.section_title("4. Diabetic Module (app/diabetic/)")
pdf.file_desc("router.py", "Defines API endpoints specifically for diabetes management (e.g., /diabetic/analyze, /diabetic/predict).")
pdf.file_desc("service.py", "Orchestrates the diabetic analysis workflow: calls Rules Engine, ML Models, and LLM Explainer.")
pdf.file_desc("rules.py", "Contains hard-coded clinical rules and contraindications specific to diabetes care (e.g., Metformin & Kidney function).")
pdf.file_desc("lab_report_analyzer.py", "Uses Gemini Vision / OCR to parse uploaded blood work PDFs and extract structured data (HbA1c, Gluecose).")
pdf.file_desc("ml_predictor.py", "Interface for the Machine Learning model that predicts diabetic complication risks.")
pdf.file_desc("llm_drug_checker.py", "Uses LLM (Gemini) as a 'Second Opinion' to cross-check drug safety for diabetic patients.")
pdf.file_desc("llm_explainer.py", "Translates complex medical risks into simple, patient-friendly language using GenAI.")

pdf.section_title("5. Prescription Module (app/prescription/)")
pdf.file_desc("router.py", "API endpoints for prescription management: Uploading scripts and asking RAG-based questions.")
pdf.file_desc("service.py", "Manages the Prescription analysis pipeline: OCR -> Entity Extraction -> Interaction Check.")
pdf.file_desc("vision_ocr.py", "Specialized Computer Vision module optimized for reading handwritten doctor's prescriptions.")
pdf.file_desc("rag_service.py", "Retrieval-Augmented Generation service. Finds relevant medical docs to help the LLM answer user questions accurately.")
pdf.file_desc("knowledge_base.py", "Manages the interface to the Vector Database (ChromaDB) for semantic searching of medical texts.")

pdf.section_title("6. Machine Learning (app/ml/)")
pdf.file_desc("predictor.py", "Generic class for loading trained models and performing inference.")
pdf.file_desc("trainer.py", "Script for training new models from raw datasets.")
pdf.file_desc("feature_engineering.py", "Transforms raw patient data into numerical features suitable for ML models.")
pdf.file_desc("explainability_service.py", "Generates SHAP values or other metrics to explain WHY a model made a specific prediction.")

# --- FRONTEND ---
pdf.add_page()
pdf.chapter_title("Frontend Architecture (c:\\Drug\\frontend)")

pdf.section_title("1. Configuration & Root")
pdf.file_desc("package.json", "Defines the React project metadata, dependencies (React, Vite, Tailwind), and build scripts.")
pdf.file_desc("vite.config.js", "Configuration for the Vite build tool. Handles plugins, proxy settings for API, and build optimizations.")
pdf.file_desc("tailwind.config.js", "Configures the Tailwind CSS framework: custom colors, fonts, and responsive breakpoints.")
pdf.file_desc("eslint.config.js", "Rules for code quality and style checking (Linting) to ensure clean JavaScript/React code.")

pdf.section_title("2. Core Logic (src/)")
pdf.file_desc("main.jsx", "The JavaScript entry point. Mounts the React application into the DOM.")
pdf.file_desc("App.jsx", "The Root React Component. Sets up the Router (Navigation), Global Providers (Theme, Auth), and main Layout.")
pdf.file_desc("index.css", "Global CSS styles and Tailwind directives. Defines the base look and feel.")

pdf.section_title("3. Components (src/components/)")
pdf.file_desc("Navbar.jsx / Footer.jsx", "Shared navigation and footer components visible on all pages.")
pdf.file_desc("Hero.jsx", "The main landing page component with welcome message and call-to-action.")
pdf.file_desc("InteractionChecker.jsx", "CORE FEATURE: The UI form where users enter drugs to check for interactions. Displays alerts and risks.")
pdf.file_desc("PrescriptionRAG.jsx", "CORE FEATURE: Interface for the 'Chat with your Prescription' feature. Handles user questions and displays AI answers.")
pdf.file_desc("PatientPrescriptionScanner.jsx", "Handles the UI for uploading prescription images and displaying the extracted medication list.")
pdf.file_desc("DiabetesManager.jsx", "Detailed dashboard for diabetic patients. Shows trends, logs glucose, and displays ML risk predictions.")
pdf.file_desc("ModelDashboard.jsx", "Visualizes the Machine Learning model's confidence and specific risk factors (charts/graphs).")
pdf.file_desc("ExplainabilityView.jsx", "Displays the 'Why' behind a risk. Shows user-friendly explanations for medical alerts.")
pdf.file_desc("CameraCapture.jsx", "A utility component to access the device camera for taking photos of prescriptions.")
pdf.file_desc("VoiceInput.jsx", "Allows users to speak their symptoms or drug names instead of typing. Uses Speech-to-Text.")
pdf.file_desc("SideEffects.jsx", "A dedicated view for listing potential side effects of the user's current medication regimen.")
pdf.file_desc("MedicationSchedule.jsx", "A calendar/timeline view helping users track when to take their medicines.")
pdf.file_desc("DosageCalculator.jsx", "Helper tool to calculate correct dosages based on weight/age (if applicable).")

pdf.section_title("4. Services & Utilities")
pdf.file_desc("src/services/api.js", "Centralized Axios instance for making HTTP requests to the Backend API. Handles headers and base URLs.")
pdf.file_desc("src/context/ThemeContext.jsx", "React Context provider for managing Light/Dark mode preferences across the app.")
pdf.file_desc("src/utils/useDebouncedSearch.js", "Custom Hook to delay API calls while the user is typing, preventing server overload.")

pdf.output("c:\\Drug\\backend\\DrugGuard_System_Documentation.pdf")
print("PDF Generated Successfully at c:\\Drug\\backend\\DrugGuard_System_Documentation.pdf")
