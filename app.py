"""
Bureaucracy Breaker - Hackathon Backend
Transforms government PDF forms into conversational experiences.
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid
import PyPDF2
import io
import os
import json
import requests

# Initialize FastAPI app
app = FastAPI(
    title="Bureaucracy Breaker",
    description="Transform government PDF forms into conversational experiences",
    version="1.0.0"
)

# Global error handlers for demo stability
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with clean error messages"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors with helpful messages"""
    print(f"[WARN] Request validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"error": "Invalid request format. Please check your request body."}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle unexpected errors gracefully"""
    print(f"[ERROR] Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Something went wrong. Please try again."}
    )

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class HealthResponse(BaseModel):
    status: str

class StartSessionRequest(BaseModel):
    session_id: Optional[str] = None

class StartSessionResponse(BaseModel):
    session_id: str
    question: Optional[Dict[str, str]] = None
    message: Optional[str] = None

class NextQuestionRequest(BaseModel):
    session_id: str
    answer: Optional[str] = None

class NextQuestionResponse(BaseModel):
    session_id: Optional[str] = None
    question: Optional[Dict[str, str]] = None
    completed: Optional[bool] = None
    error: Optional[str] = None

class PDFUploadResponse(BaseModel):
    session_id: str
    total_fields: int

class PDFField(BaseModel):
    name: str
    type: str

class PDFFieldsResponse(BaseModel):
    fields: List[PDFField]

class GeneratePDFRequest(BaseModel):
    session_id: str

# Session data model
@dataclass
class Session:
    session_id: str
    form_fields: List[Dict]  # Changed from List[str] to List[Dict] to store field objects
    current_field_index: int
    answers: Dict[str, str]
    created_at: datetime
    original_pdf: Optional[bytes] = None  # Store original PDF bytes for generation
    answered_checkbox_groups: set = None  # Track answered checkbox groups to avoid repetition
    
    def __post_init__(self):
        """Initialize answered_checkbox_groups as empty set if None"""
        if self.answered_checkbox_groups is None:
            self.answered_checkbox_groups = set()

# Global session store (in-memory for hackathon simplicity)
sessions: Dict[str, Session] = {}

# PDF Processing Class
class PDFProcessor:
    """Handles PDF form processing operations."""
    
    @staticmethod
    def extract_fields(pdf_bytes: bytes) -> List[Dict]:
        """
        Extract field information from a PDF's AcroForm.
        
        Args:
            pdf_bytes: Raw PDF file content as bytes
            
        Returns:
            List of field dictionaries with name and type
        """
        try:
            # Create a BytesIO object from the PDF bytes
            pdf_stream = io.BytesIO(pdf_bytes)
            
            # Create PDF reader
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            
            # Use get_fields() method if available
            if hasattr(pdf_reader, 'get_fields'):
                fields_dict = pdf_reader.get_fields()
                if not fields_dict:
                    return []
                
                field_list = []
                for field_name, field_obj in fields_dict.items():
                    field_type = PDFProcessor._determine_field_type(field_obj)
                    field_info = {
                        "name": field_name,
                        "type": field_type
                    }
                    
                    # For checkbox fields, extract export value
                    if field_type == "checkbox":
                        export_value = PDFProcessor._extract_checkbox_export_value(field_obj)
                        field_info["export_value"] = export_value
                    
                    field_list.append(field_info)
                return field_list
            
            # Fallback to manual extraction if get_fields() not available
            return PDFProcessor._manual_field_extraction(pdf_reader)
            
        except Exception:
            # Return empty list for any parsing errors
            return []
    
    @staticmethod
    def _determine_field_type(field_obj) -> str:
        """
        Determine field type based on PDF field object.
        
        Args:
            field_obj: PyPDF2 field object
            
        Returns:
            Field type string
        """
        try:
            # Check if field_obj has /FT attribute
            if hasattr(field_obj, 'get') and '/FT' in field_obj:
                ft_value = field_obj['/FT']
                
                # Map PDF field types to our simplified types
                if ft_value == '/Tx':
                    return "text"
                elif ft_value == '/Btn':
                    return "checkbox"
                elif ft_value == '/Ch':
                    return "choice"
            
            return "unknown"
            
        except Exception:
            return "unknown"
    
    @staticmethod
    def _extract_checkbox_export_value(field_obj) -> str:
        """
        Extract the export value for a checkbox field.
        
        PDF checkboxes use export values (e.g., "Male", "Female") instead of generic "Yes".
        This method attempts to find the actual export value from the field object.
        
        Args:
            field_obj: PyPDF2 field object for a checkbox
            
        Returns:
            Export value string, defaults to "Yes" if not found
        """
        try:
            # Try to get the export value from various PDF field attributes
            
            # Method 1: Check /AS (Appearance State) - current value
            if hasattr(field_obj, 'get') and '/AS' in field_obj:
                as_value = field_obj['/AS']
                if as_value and as_value != '/Off':
                    return str(as_value).lstrip('/')
            
            # Method 2: Check /V (Value) - field value
            if hasattr(field_obj, 'get') and '/V' in field_obj:
                v_value = field_obj['/V']
                if v_value and v_value != '/Off':
                    return str(v_value).lstrip('/')
            
            # Method 3: Check /AP (Appearance) dictionary for state names
            if hasattr(field_obj, 'get') and '/AP' in field_obj:
                ap_dict = field_obj['/AP']
                if hasattr(ap_dict, 'get') and '/N' in ap_dict:
                    n_dict = ap_dict['/N']
                    if hasattr(n_dict, 'keys'):
                        # Get all appearance states except /Off
                        states = [str(key).lstrip('/') for key in n_dict.keys() 
                                if str(key) != '/Off']
                        if states:
                            return states[0]  # Use first non-Off state
            
            # Method 4: Check /Opt (Options) for choice-like checkboxes
            if hasattr(field_obj, 'get') and '/Opt' in field_obj:
                opt_value = field_obj['/Opt']
                if opt_value and len(opt_value) > 0:
                    return str(opt_value[0])
            
            # Default fallback
            return "Yes"
            
        except Exception:
            # If all extraction methods fail, use default
            return "Yes"
    
    @staticmethod
    def _manual_field_extraction(pdf_reader) -> List[Dict]:
        """
        Manual field extraction fallback method.
        
        Args:
            pdf_reader: PyPDF2.PdfReader object
            
        Returns:
            List of field dictionaries
        """
        try:
            # Check if the PDF has form fields
            if "/AcroForm" not in pdf_reader.trailer["/Root"]:
                return []
            
            # Get the AcroForm
            acro_form = pdf_reader.trailer["/Root"]["/AcroForm"]
            
            # Check if Fields exist in AcroForm
            if "/Fields" not in acro_form:
                return []
            
            # Extract field information
            fields = acro_form["/Fields"]
            field_list = []
            
            for field in fields:
                field_obj = field.get_object()
                if "/T" in field_obj:  # /T is the field name
                    field_name = field_obj["/T"]
                    if isinstance(field_name, str):
                        field_type = PDFProcessor._determine_field_type(field_obj)
                        field_info = {
                            "name": field_name,
                            "type": field_type
                        }
                        
                        # For checkbox fields, extract export value
                        if field_type == "checkbox":
                            export_value = PDFProcessor._extract_checkbox_export_value(field_obj)
                            field_info["export_value"] = export_value
                        
                        field_list.append(field_info)
            
            return field_list
            
        except Exception:
            return []
    
    @staticmethod
    def fill_pdf(pdf_bytes: bytes, answers: Dict[str, str]) -> bytes:
        """
        Fill PDF form fields with provided answers.
        
        Args:
            pdf_bytes: Raw PDF file content as bytes
            answers: Dictionary mapping field names to values
            
        Returns:
            Updated PDF as bytes
        """
        try:
            # Create BytesIO objects for input and output
            input_stream = io.BytesIO(pdf_bytes)
            output_stream = io.BytesIO()
            
            # Create PDF reader and writer
            pdf_reader = PyPDF2.PdfReader(input_stream)
            pdf_writer = PyPDF2.PdfWriter()
            
            # Copy all pages to writer
            for page in pdf_reader.pages:
                pdf_writer.add_page(page)
            
            # Update form fields if they exist
            if hasattr(pdf_writer, 'update_page_form_field_values'):
                # Try to update fields for each page
                for page_num in range(len(pdf_writer.pages)):
                    try:
                        pdf_writer.update_page_form_field_values(
                            pdf_writer.pages[page_num], 
                            answers
                        )
                    except Exception:
                        # Continue if updating this page fails
                        continue
            
            # Write the updated PDF to output stream
            pdf_writer.write(output_stream)
            
            # Return the PDF bytes
            output_stream.seek(0)
            return output_stream.read()
            
        except Exception:
            # Return original PDF if filling fails
            return pdf_bytes

# AI Question Generation Class
class AIConverter:
    """
    Handles AI-based question generation for form fields using OpenRouter.
    
    Uses AI to convert technical PDF field names into human-friendly questions.
    Includes comprehensive fallback logic to ensure the system never fails
    due to AI service issues - essential for demo reliability.
    """
    
    # OpenRouter configuration
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MODEL = "openai/gpt-3.5-turbo"
    
    @staticmethod
    def generate_question(field: Dict) -> Optional[Dict]:
        """
        Generate a human-friendly question for a form field using OpenRouter.
        
        Args:
            field: Dictionary with 'name' and 'type' keys
            
        Returns:
            Dictionary with 'question' and 'explanation' or fallback if failed
        """
        try:
            # Try to use OpenRouter API
            return AIConverter._call_openrouter(field)
        except Exception as e:
            print(f"[WARN] AI generation failed for field {field.get('name', 'unknown')}, using fallback")
            # Return fallback question
            return AIConverter._fallback_question(field)
    
    @staticmethod
    def _call_openrouter(field: Dict) -> Optional[Dict]:
        """
        Call OpenRouter API to generate question.
        
        Args:
            field: Dictionary with 'name' and 'type' keys
            
        Returns:
            Dictionary with 'question' and 'explanation' or None
        """
        field_name = field.get('name', 'unknown')
        
        # Check for API key
        api_key = os.getenv('OPENROUTER_API_KEY')
        
        if not api_key:
            print("[INFO] OpenRouter API key not found, using fallback questions")
            return AIConverter._fallback_question(field)
        
        print(f"[INFO] Generating AI question for field: {field_name}")
        
        try:
            # Create the prompt
            prompt = AIConverter._create_prompt(field)
            
            # Prepare the request
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": AIConverter.DEFAULT_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful government clerk assisting citizens with form completion. Convert technical form field names into clear, friendly questions using simple, non-legal language. Never mention internal field names. For checkbox fields, ask yes/no questions. For text fields, ask open-ended questions. Keep explanations brief and helpful."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": 200,
                "temperature": 0.3
            }
            
            # Make the API call
            response = requests.post(
                AIConverter.OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                # Parse the response
                response_data = response.json()
                
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    content = response_data['choices'][0]['message']['content'].strip()
                    
                    # Convert the response to our format
                    return AIConverter._parse_response(content, field)
                else:
                    print("[WARN] OpenRouter returned invalid response, using fallback")
                    return AIConverter._fallback_question(field)
            else:
                print(f"[WARN] OpenRouter API error {response.status_code}, using fallback")
                return AIConverter._fallback_question(field)
                
        except requests.exceptions.Timeout:
            print("[WARN] OpenRouter API timeout, using fallback")
            return AIConverter._fallback_question(field)
        except requests.exceptions.RequestException as e:
            print("[WARN] OpenRouter API request failed, using fallback")
            return AIConverter._fallback_question(field)
        except json.JSONDecodeError as e:
            print("[WARN] OpenRouter response parsing failed, using fallback")
            return AIConverter._fallback_question(field)
        except Exception as e:
            print("[WARN] Unexpected OpenRouter error, using fallback")
            return AIConverter._fallback_question(field)
    
    @staticmethod
    def _create_prompt(field: Dict) -> str:
        """
        Create prompt for OpenRouter based on field information.
        
        Args:
            field: Dictionary with 'name' and 'type' keys
            
        Returns:
            Prompt string for OpenRouter
        """
        field_name = field.get('name', 'unknown')
        field_type = field.get('type', 'unknown')
        
        prompt = f"Convert this form field into a friendly question:\n"
        prompt += f"Field name: {field_name}\n"
        prompt += f"Field type: {field_type}\n\n"
        
        if field_type == "checkbox":
            prompt += "This is a checkbox field - create a yes/no question.\n"
        elif field_type == "text":
            prompt += "This is a text field - create an open-ended question.\n"
        elif field_type == "choice":
            prompt += "This is a choice field - ask the user to select an option.\n"
        else:
            prompt += "Field type is unknown - create a general question.\n"
        
        prompt += "\nRemember: Don't mention the internal field name in your question.\n"
        prompt += "\nProvide both a question and a brief explanation of why this information is needed."
        
        return prompt
    
    @staticmethod
    def _parse_response(content: str, field: Dict) -> Dict:
        """
        Parse OpenRouter response into question and explanation.
        
        Args:
            content: Response content from OpenRouter
            field: Original field dictionary
            
        Returns:
            Dictionary with 'question' and 'explanation'
        """
        try:
            # Try to extract question and explanation from the response
            lines = content.split('\n')
            
            question = ""
            explanation = ""
            
            # Look for patterns in the response
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if it looks like a question
                if '?' in line and not question:
                    question = line
                # Check if it looks like an explanation
                elif line and not explanation and question:
                    explanation = line
            
            # If we couldn't parse it properly, use the first line as question
            if not question:
                question = lines[0].strip() if lines else "Please provide information for this field."
            
            if not explanation:
                explanation = "This information is required to complete the form."
            
            # Clean up the question and explanation
            question = question.replace('"', '').replace("Question:", "").replace("Q:", "").strip()
            explanation = explanation.replace('"', '').replace("Explanation:", "").replace("A:", "").strip()
            
            return {
                "question": question,
                "explanation": explanation
            }
            
        except Exception as e:
            print("[WARN] AI response parsing failed, using fallback")
            return AIConverter._fallback_question(field)
    
    @staticmethod
    def _fallback_question(field: Dict) -> Dict:
        """
        Generate fallback question when AI fails.
        
        Args:
            field: Dictionary with 'name' and 'type' keys
            
        Returns:
            Dictionary with fallback question and explanation
        """
        return {
            "question": "Please provide information for this field.",
            "explanation": "This information is required to complete the form."
        }

# Form Validation Class
class FormValidator:
    """Handles validation and formatting of form field answers."""
    
    @staticmethod
    def validate_and_format(field: Dict, answer: str) -> Dict:
        """
        Validate and format a user's answer for a form field.
        
        Args:
            field: Dictionary with 'name' and 'type' keys
            answer: User's answer as string
            
        Returns:
            Dictionary with 'is_valid', 'value', and 'error' keys
        """
        field_type = field.get('type', 'unknown')
        field_name = field.get('name', 'unknown')
        
        # Handle different field types
        if field_type == 'text':
            return FormValidator._validate_text(answer)
        elif field_type == 'checkbox':
            return FormValidator._validate_checkbox(answer)
        elif field_type == 'choice':
            return FormValidator._validate_choice(answer)
        elif field_type == 'unknown':
            return FormValidator._validate_unknown(answer)
        else:
            # Default to text validation for unrecognized types
            return FormValidator._validate_text(answer)
    
    @staticmethod
    def _validate_text(answer: str) -> Dict:
        """
        Validate text field answer.
        
        Args:
            answer: User's answer
            
        Returns:
            Validation result dictionary
        """
        if not answer or not answer.strip():
            return {
                "is_valid": False,
                "value": None,
                "error": "Please provide a response. This field cannot be empty."
            }
        
        # Valid text - return trimmed value
        return {
            "is_valid": True,
            "value": answer.strip(),
            "error": None
        }
    
    @staticmethod
    def _validate_checkbox(answer: str) -> Dict:
        """
        Validate checkbox field answer.
        
        Args:
            answer: User's answer
            
        Returns:
            Validation result dictionary
        """
        if not answer:
            return {
                "is_valid": False,
                "value": None,
                "error": "Please answer Yes or No."
            }
        
        # Normalize answer
        normalized = answer.strip().lower()
        
        # Check valid values for checked (true)
        if normalized in ['yes', 'y', 'true', '1']:
            return {
                "is_valid": True,
                "value": "Yes",  # PDF checkbox checked value
                "error": None
            }
        # Check valid values for unchecked (false)
        elif normalized in ['no', 'n', 'false', '0']:
            return {
                "is_valid": True,
                "value": "",  # PDF checkbox unchecked value (empty string)
                "error": None
            }
        else:
            return {
                "is_valid": False,
                "value": None,
                "error": "Please answer Yes or No."
            }
    
    @staticmethod
    def _validate_choice(answer: str) -> Dict:
        """
        Validate choice field answer.
        
        Args:
            answer: User's answer
            
        Returns:
            Validation result dictionary
        """
        if not answer or not answer.strip():
            return {
                "is_valid": False,
                "value": None,
                "error": "Please select an option."
            }
        
        # Valid choice - return trimmed value
        return {
            "is_valid": True,
            "value": answer.strip(),
            "error": None
        }
    
    @staticmethod
    def _validate_unknown(answer: str) -> Dict:
        """
        Validate unknown field type - always valid.
        
        Args:
            answer: User's answer
            
        Returns:
            Validation result dictionary
        """
        # Unknown fields are always valid
        return {
            "is_valid": True,
            "value": answer.strip() if answer else "",
            "error": None
        }

# Helper functions (keeping for backward compatibility)
def extract_pdf_fields(pdf_bytes: bytes) -> List[str]:
    """
    Legacy function - extract field names only.
    
    Args:
        pdf_bytes: Raw PDF file content as bytes
        
    Returns:
        List of field names, empty if no fields exist
    """
    fields = PDFProcessor.extract_fields(pdf_bytes)
    return [field["name"] for field in fields]

def create_session() -> Session:
    """Create a new session with unique ID."""
    session_id = str(uuid.uuid4())
    session = Session(
        session_id=session_id,
        form_fields=[],
        current_field_index=0,
        answers={},
        created_at=datetime.now(),
        original_pdf=None
    )
    sessions[session_id] = session
    print(f"[INFO] Created new session: {session_id}")
    return session

def get_session(session_id: str) -> Session:
    """Retrieve session by ID, raise HTTPException if not found."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return sessions[session_id]

def extract_checkbox_base_name(field_name: str) -> str:
    """
    Extract base name from checkbox field name by removing [index] pattern.
    
    Government forms often encode ONE multiple-choice question as multiple checkbox fields.
    Example: c1_1[0], c1_1[1], c1_1[2] represent one radio button group.
    This function extracts the base name 'c1_1' to group related checkboxes.
    
    Examples:
        c1_1[0] → c1_1
        c1_1[1] → c1_1
        checkbox_group[2] → checkbox_group
        regular_field → regular_field (no change)
    
    Args:
        field_name: The field name to process
        
    Returns:
        Base name without [index] suffix
    """
    import re
    # Remove [index] pattern from the end of field name
    base_name = re.sub(r'\[\d+\]$', '', field_name)
    return base_name

def prepare_checkbox_group_answers(session: Session) -> Dict[str, str]:
    """
    Prepare checkbox group answers for PDF filling using actual export values.
    
    For checkbox groups (radio buttons), we need to explicitly set all fields:
    - Selected field gets its export_value (e.g., "Male", "Female")
    - All other fields in the same group get "/Off"
    
    This ensures PDF radio button groups display correctly with proper export values.
    
    Args:
        session: Session with form fields and answers
        
    Returns:
        Complete answers dictionary with all checkbox fields set using export values
    """
    # Start with existing answers
    complete_answers = session.answers.copy()
    
    # Group checkbox fields by base name, keeping field objects for export values
    checkbox_groups = {}
    for field in session.form_fields:
        if field.get('type') == 'checkbox':
            field_name = field.get('name', '')
            base_name = extract_checkbox_base_name(field_name)
            
            if base_name not in checkbox_groups:
                checkbox_groups[base_name] = []
            checkbox_groups[base_name].append(field)  # Store full field object
    
    # For each checkbox group, ensure all fields are set with proper export values
    for base_name, fields in checkbox_groups.items():
        # Find which field (if any) was answered in this group
        selected_field = None
        for field in fields:
            field_name = field.get('name', '')
            # Check if this field was answered (user selected "Yes" for this option)
            if field_name in session.answers and session.answers[field_name] == "Yes":
                selected_field = field
                break
        
        # Set all fields in the group with appropriate values
        for field in fields:
            field_name = field.get('name', '')
            if field == selected_field:
                # Use the field's export value for the selected option
                export_value = field.get('export_value', 'Yes')
                complete_answers[field_name] = export_value
            else:
                # Use "/Off" for unselected options (PDF standard)
                complete_answers[field_name] = "/Off"
    
    return complete_answers

def get_next_valid_field(session: Session) -> Optional[Dict]:
    """
    Get the next valid field (skip unknown types and already answered checkbox groups).
    
    Args:
        session: Session object
        
    Returns:
        Next valid field dict or None if no more fields
    """
    while session.current_field_index < len(session.form_fields):
        field = session.form_fields[session.current_field_index]
        
        # Skip unknown field types
        if field.get('type') == 'unknown':
            session.current_field_index += 1
            continue
        
        # Handle checkbox groups to avoid repeated questions
        if field.get('type') == 'checkbox':
            field_name = field.get('name', '')
            base_name = extract_checkbox_base_name(field_name)
            
            # If this checkbox group has already been answered, skip it
            # This prevents asking the same question multiple times for government forms
            if base_name in session.answered_checkbox_groups:
                session.current_field_index += 1
                continue
            
        return field
    
    return None

# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/start-session", response_model=StartSessionResponse)
async def start_session(request: StartSessionRequest):
    """Begin form completion process with AI-generated questions."""
    
    # Handle two cases:
    # 1. No session_id provided → Create new empty session (original behavior)
    # 2. session_id provided → Use existing session from PDF upload
    
    if not request.session_id:
        # Original behavior: create a new empty session
        session = create_session()
        return {
            "session_id": session.session_id,
            "message": "Session created. Please upload a PDF to begin form completion."
        }
    
    # New behavior: use existing session with PDF fields
    session = get_session(request.session_id)
    
    # Check if there are any fields
    if not session.form_fields:
        return {
            "session_id": session.session_id,
            "message": "No form fields available"
        }
    
    # Reset to first field
    session.current_field_index = 0
    
    # Get the first valid field (skip unknown types)
    field = get_next_valid_field(session)
    
    if not field:
        return {
            "session_id": session.session_id,
            "message": "No form fields available"
        }
    
    # Generate AI question for the field
    question_data = AIConverter.generate_question(field)
    
    if not question_data:
        # Fallback if AI fails
        question_data = {
            "question": "Please provide information for this field.",
            "explanation": "This information is required to complete the form."
        }
    
    # Add field type to the response
    question_data["field_type"] = field.get("type", "unknown")
    
    return {
        "session_id": session.session_id,
        "question": {
            "text": question_data["question"],
            "explanation": question_data["explanation"],
            "field_type": question_data["field_type"]
        }
    }

@app.post("/next-question", response_model=NextQuestionResponse)
async def next_question(request: NextQuestionRequest):
    """Get next question in sequence with AI-generated questions and answer validation."""
    # Get the session
    session = get_session(request.session_id)
    
    # If an answer is provided, validate it for the current field
    if request.answer is not None:
        # Get the current field
        if session.current_field_index < len(session.form_fields):
            current_field = session.form_fields[session.current_field_index]
            
            # Validate the answer
            validation_result = FormValidator.validate_and_format(current_field, request.answer)
            
            if not validation_result["is_valid"]:
                # Return error without advancing to next field
                return {
                    "session_id": session.session_id,
                    "error": validation_result["error"]
                }
            
            # Store the valid answer
            field_name = current_field.get("name")
            if field_name:
                session.answers[field_name] = validation_result["value"]
                
                # Track answered checkbox groups to avoid repeated questions
                if current_field.get("type") == "checkbox":
                    base_name = extract_checkbox_base_name(field_name)
                    session.answered_checkbox_groups.add(base_name)
    
    # Move to next field
    session.current_field_index += 1
    
    # Get the next valid field (skip unknown types)
    field = get_next_valid_field(session)
    
    if not field:
        # No more fields - form is complete
        return {"completed": True}
    
    # Generate AI question for the field
    question_data = AIConverter.generate_question(field)
    
    if not question_data:
        # Fallback if AI fails
        question_data = {
            "question": "Please provide information for this field.",
            "explanation": "This information is required to complete the form."
        }
    
    # Add field type to the response
    question_data["field_type"] = field.get("type", "unknown")
    
    return {
        "session_id": session.session_id,
        "question": {
            "text": question_data["question"],
            "explanation": question_data["explanation"],
            "field_type": question_data["field_type"]
        }
    }

@app.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Accept PDF upload, extract fields, and create a session."""
    # Validate file is provided
    if not file:
        raise HTTPException(status_code=400, detail="File is required")
    
    # Validate file is a PDF
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Check content type
    if file.content_type and not file.content_type.startswith('application/pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Read file bytes
        pdf_bytes = await file.read()
        
        # Extract field information using PDFProcessor
        fields = PDFProcessor.extract_fields(pdf_bytes)
        
        # Create a new session with the extracted fields
        session = create_session()
        session.form_fields = fields
        session.current_field_index = 0
        session.original_pdf = pdf_bytes  # Store original PDF bytes
        
        return {
            "session_id": session.session_id,
            "total_fields": len(fields)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to process PDF: {str(e)}")

@app.post("/generate-pdf")
async def generate_pdf(request: GeneratePDFRequest):
    """Generate a completed PDF using collected answers and return it for download."""
    # Get the session (will raise 404 if not found)
    session = get_session(request.session_id)
    
    # Check if answers exist
    if not session.answers:
        raise HTTPException(status_code=400, detail="No answers available to generate PDF")
    
    # Check if original PDF exists
    if not session.original_pdf:
        raise HTTPException(status_code=400, detail="Original PDF not found in session")
    
    try:
        # Prepare complete answers including all checkbox group fields
        # This ensures radio button groups are filled correctly in PDFs
        complete_answers = prepare_checkbox_group_answers(session)
        
        # Generate filled PDF using PDFProcessor
        filled_pdf_bytes = PDFProcessor.fill_pdf(session.original_pdf, complete_answers)
        
        # Clean up session after successful PDF generation
        if request.session_id in sessions:
            del sessions[request.session_id]
            print(f"[INFO] Generated PDF and cleaned up session: {request.session_id}")
        
        # Return the PDF as a streaming response
        return StreamingResponse(
            io.BytesIO(filled_pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=completed_form.pdf"
            }
        )
        
    except Exception as e:
        print(f"[ERROR] PDF generation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Unable to generate PDF: {str(e)}")

@app.get("/debug/sessions")
async def debug_sessions():
    """Debug endpoint to list all active sessions."""
    session_info = []
    for session_id, session in sessions.items():
        session_info.append({
            "session_id": session_id,
            "field_count": len(session.form_fields),
            "answer_count": len(session.answers),
            "has_original_pdf": session.original_pdf is not None,
            "current_field_index": session.current_field_index,
            "answered_checkbox_groups": list(session.answered_checkbox_groups),
            "created_at": session.created_at.isoformat()
        })
    
    return {
        "total_sessions": len(sessions),
        "sessions": session_info
    }

@app.post("/debug/next-question")
async def debug_next_question(request: Request):
    """Debug endpoint to inspect next-question request format"""
    try:
        body = await request.body()
        print(f"[DEBUG] Raw request body: {body}")
        
        if body:
            import json
            try:
                parsed = json.loads(body)
                print(f"[DEBUG] Parsed JSON: {parsed}")
                return {"raw_body": body.decode(), "parsed_json": parsed}
            except json.JSONDecodeError as e:
                print(f"[DEBUG] JSON decode error: {e}")
                return {"raw_body": body.decode(), "json_error": str(e)}
        else:
            return {"error": "Empty request body"}
    except Exception as e:
        print(f"[DEBUG] Debug error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    print("[INFO] Starting Bureaucracy Breaker server on port 8004")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
