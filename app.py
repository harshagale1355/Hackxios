"""
Bureaucracy Breaker - Hackathon Backend
Transforms government PDF forms into conversational experiences.
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import uuid
import PyPDF2
import io
import os
import json
import requests
import re
import base64
import time
from PIL import Image
from dotenv import load_dotenv


# OpenRouter API Key (Mistral 7B)
load_dotenv()
OPENROUTER_API_KEY = os.getenv("My_api_key")


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
class StartSessionRequest(BaseModel):
    session_id: Optional[str] = None

class StartSessionResponse(BaseModel):
    session_id: str
    question: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

class NextQuestionRequest(BaseModel):
    session_id: str
    answer: Optional[str] = None

class NextQuestionResponse(BaseModel):
    session_id: Optional[str] = None
    question: Optional[Dict[str, Any]] = None
    completed: Optional[bool] = None
    error: Optional[str] = None

class PDFUploadResponse(BaseModel):
    session_id: str
    total_fields: int
    summary: Optional[str] = None

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    is_form_command: bool = False

class PDFField(BaseModel):
    name: str
    type: str

class PDFFieldsResponse(BaseModel):
    fields: List[PDFField]

class GeneratePDFRequest(BaseModel):
    session_id: str

class UploadImageRequest(BaseModel):
    session_id: str
    field_name: str

class UploadImageResponse(BaseModel):
    success: bool
    message: str
    field_name: str

# Session data model
@dataclass
class Session:
    session_id: str
    form_fields: List[Dict]
    current_field_index: int
    answers: Dict[str, str]
    created_at: datetime
    original_pdf: Optional[bytes] = None
    answered_checkbox_groups: set = None
    is_fillable_pdf: bool = True  # True for AcroForm, False for text-based PDF
    pdf_text_content: str = ""  # Extracted text for non-fillable PDFs
    pre_generated_questions: Optional[Dict[str, Dict]] = None  # AI-generated questions for all fields
    questions_generated: bool = False  # Flag to track if questions were pre-generated
    uploaded_images: Optional[Dict[str, bytes]] = None  # Store uploaded images for signature/photo fields
    document_summary: str = ""  # AI-generated summary of the document
    chat_mode: bool = False  # If True, user is chatting instead of filling form
    
    def __post_init__(self):
        if self.answered_checkbox_groups is None:
            self.answered_checkbox_groups = set()
        if self.pre_generated_questions is None:
            self.pre_generated_questions = {}
        if self.uploaded_images is None:
            self.uploaded_images = {}

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
            
            print(f"[DEBUG] PDF has {len(pdf_reader.pages)} pages")
            
            field_list = []
            
            # Method 1: Scan page annotations for Widget fields (most reliable)
            print("[DEBUG] Scanning page annotations for form fields...")
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    if '/Annots' in page:
                        annots = page['/Annots']
                        if annots:
                            for annot_ref in annots:
                                try:
                                    annot_obj = annot_ref.get_object() if hasattr(annot_ref, 'get_object') else annot_ref
                                    
                                    # Check if this is a form widget
                                    subtype = annot_obj.get('/Subtype', '')
                                    if str(subtype) == '/Widget':
                                        # Get field name
                                        field_name = annot_obj.get('/T', '')
                                        if hasattr(field_name, 'get_object'):
                                            field_name = field_name.get_object()
                                        field_name = str(field_name) if field_name else f'Field_{len(field_list)}'
                                        
                                        # Get field type
                                        field_type = PDFProcessor._determine_field_type(annot_obj)
                                        
                                        field_info = {
                                            "name": field_name,
                                            "type": field_type
                                        }
                                        
                                        if field_type == "checkbox":
                                            field_info["export_value"] = PDFProcessor._extract_checkbox_export_value(annot_obj)
                                        
                                        # Avoid duplicates
                                        if not any(f['name'] == field_name for f in field_list):
                                            field_list.append(field_info)
                                            print(f"[DEBUG] Found field: {field_name} ({field_type})")
                                except Exception as e:
                                    print(f"[DEBUG] Error processing annotation: {e}")
                                    continue
                except Exception as e:
                    print(f"[DEBUG] Error processing page {page_num}: {e}")
                    continue
            
            if field_list:
                print(f"[INFO] Found {len(field_list)} AcroForm fields")
                return field_list
            
            # Method 2: Try get_fields() as fallback
            print("[DEBUG] Trying get_fields() method...")
            try:
                if hasattr(pdf_reader, 'get_fields'):
                    fields_dict = pdf_reader.get_fields()
                    if fields_dict:
                        for field_name, field_obj in fields_dict.items():
                            field_type = PDFProcessor._determine_field_type(field_obj)
                            field_info = {
                                "name": str(field_name),
                                "type": field_type
                            }
                            if field_type == "checkbox":
                                field_info["export_value"] = PDFProcessor._extract_checkbox_export_value(field_obj)
                            field_list.append(field_info)
                            print(f"[DEBUG] Found field via get_fields: {field_name} ({field_type})")
            except Exception as e:
                print(f"[DEBUG] get_fields() failed: {e}")
            
            if field_list:
                print(f"[INFO] Found {len(field_list)} AcroForm fields via get_fields")
                return field_list
            
            print("[INFO] No AcroForm fields found in PDF")
            return []
            
        except Exception as e:
            print(f"[ERROR] Field extraction failed: {str(e)}")
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
            ft_value = None
            
            # Try different ways to get field type
            if hasattr(field_obj, 'get'):
                ft_value = field_obj.get('/FT')
            elif isinstance(field_obj, dict):
                ft_value = field_obj.get('/FT')
            
            # Also check the indirect object
            if not ft_value and hasattr(field_obj, 'get_object'):
                obj = field_obj.get_object()
                ft_value = obj.get('/FT') if hasattr(obj, 'get') else None
            
            if ft_value:
                ft_str = str(ft_value)
                # Map PDF field types to our simplified types
                if '/Tx' in ft_str or ft_str == '/Tx':
                    return "text"
                elif '/Btn' in ft_str or ft_str == '/Btn':
                    return "checkbox"
                elif '/Ch' in ft_str or ft_str == '/Ch':
                    return "choice"
            
            return "text"  # Default to text instead of unknown
            
        except Exception as e:
            print(f"[DEBUG] Field type detection error: {e}")
            return "text"
    
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
    def fill_pdf(pdf_bytes: bytes, answers: Dict[str, str], uploaded_images: Dict[str, bytes] = None) -> bytes:
        """
        Fill PDF form fields with provided answers and embed images where possible.
        
        Args:
            pdf_bytes: Raw PDF file content as bytes
            answers: Dictionary mapping field names to values
            uploaded_images: Dictionary of field names to image bytes
            
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
            
            # Note: PyPDF2 has limited support for embedding images in form fields
            # Images are stored but may need manual PDF editing software for full integration
            if uploaded_images:
                print(f"[INFO] {len(uploaded_images)} image(s) uploaded for this form")
                # Images are stored in session for future enhancement
            
            # Write the updated PDF to output stream
            pdf_writer.write(output_stream)
            
            # Return the PDF bytes
            output_stream.seek(0)
            return output_stream.read()
            
        except Exception:
            # Return original PDF if filling fails
            return pdf_bytes
    
    @staticmethod
    def extract_text(pdf_bytes: bytes) -> str:
        """
        Extract all text content from a PDF.
        
        Args:
            pdf_bytes: Raw PDF file content as bytes
            
        Returns:
            Extracted text as string
        """
        try:
            pdf_stream = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            
            text_content = []
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(f"--- Page {page_num + 1} ---\n{page_text}")
                except Exception as e:
                    print(f"[DEBUG] Error extracting text from page {page_num}: {e}")
                    continue
            
            full_text = "\n\n".join(text_content)
            print(f"[INFO] Extracted {len(full_text)} characters of text from PDF")
            return full_text
            
        except Exception as e:
            print(f"[ERROR] Text extraction failed: {e}")
            return ""
    
    @staticmethod
    def generate_text_pdf(original_pdf_bytes: bytes, answers: Dict[str, str]) -> bytes:
        """
        Generate a summary PDF with the collected answers for non-fillable PDFs.
        Creates a new page with all the answers appended to the original PDF.
        
        Args:
            original_pdf_bytes: Original PDF bytes
            answers: Dictionary of field names to answers
            
        Returns:
            PDF bytes with answers summary page
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import inch
            
            # Create a summary PDF page with answers
            summary_stream = io.BytesIO()
            c = canvas.Canvas(summary_stream, pagesize=letter)
            width, height = letter
            
            # Title
            c.setFont("Helvetica-Bold", 16)
            c.drawString(1*inch, height - 1*inch, "Form Completion Summary")
            
            c.setFont("Helvetica", 10)
            c.drawString(1*inch, height - 1.3*inch, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            
            # Draw answers
            y_position = height - 1.8*inch
            c.setFont("Helvetica", 11)
            
            for field_name, answer in answers.items():
                if y_position < 1*inch:
                    # New page if we run out of space
                    c.showPage()
                    y_position = height - 1*inch
                    c.setFont("Helvetica", 11)
                
                # Clean up field name for display
                display_name = field_name.replace('_', ' ').title()
                
                # Draw field name in bold
                c.setFont("Helvetica-Bold", 11)
                c.drawString(1*inch, y_position, f"{display_name}:")
                
                # Draw answer
                c.setFont("Helvetica", 11)
                # Handle long answers by wrapping
                answer_text = str(answer) if answer else "(No answer)"
                if len(answer_text) > 70:
                    answer_text = answer_text[:67] + "..."
                c.drawString(1*inch + 0.2*inch, y_position - 0.2*inch, answer_text)
                
                y_position -= 0.5*inch
            
            c.save()
            summary_stream.seek(0)
            
            # Merge original PDF with summary page
            original_stream = io.BytesIO(original_pdf_bytes)
            original_reader = PyPDF2.PdfReader(original_stream)
            summary_reader = PyPDF2.PdfReader(summary_stream)
            
            writer = PyPDF2.PdfWriter()
            
            # Add all original pages
            for page in original_reader.pages:
                writer.add_page(page)
            
            # Add summary page(s)
            for page in summary_reader.pages:
                writer.add_page(page)
            
            # Output
            output_stream = io.BytesIO()
            writer.write(output_stream)
            output_stream.seek(0)
            
            return output_stream.read()
            
        except ImportError:
            print("[WARN] reportlab not installed, returning original PDF")
            return original_pdf_bytes
        except Exception as e:
            print(f"[ERROR] Failed to generate text PDF: {e}")
            return original_pdf_bytes


# AI-based field extraction for non-fillable PDFs
class AIFieldExtractor:
    """
    Uses AI to analyze PDF text and identify form fields.
    For non-fillable PDFs where we can't detect AcroForm fields.
    """
    
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MODEL = "mistralai/mistral-7b-instruct"
    
    EXTRACTION_PROMPT = """Analyze this form/document text and identify ALL fields that need to be filled in.

This could be any type of form: government forms, invoices, quotes, applications, etc.

Look for:
- Labels followed by empty spaces or colons (Name:, Address:, Date:)
- Table headers that indicate data entry columns (Description, Quantity, Price, Total)
- Row labels in tables (Subtotal:, Tax:, Grand Total:)
- Any field that expects user input
- Checkboxes or Yes/No options
- Date fields, signature fields

For each field found, output in this EXACT format (one per line):
FIELD: [field_name] | TYPE: [text/number/date/checkbox] | LABEL: [what the form asks for]

Example for a Quote/Invoice form:
FIELD: company_name | TYPE: text | LABEL: Company Name
FIELD: address | TYPE: text | LABEL: Address
FIELD: contact_no | TYPE: text | LABEL: Contact Number
FIELD: email | TYPE: text | LABEL: Email
FIELD: date | TYPE: date | LABEL: Date
FIELD: client_name | TYPE: text | LABEL: Client Name
FIELD: item_1_description | TYPE: text | LABEL: Item 1 Description
FIELD: item_1_quantity | TYPE: number | LABEL: Item 1 Quantity
FIELD: item_1_unit_price | TYPE: number | LABEL: Item 1 Unit Price
FIELD: subtotal | TYPE: number | LABEL: Subtotal
FIELD: tax_percent | TYPE: number | LABEL: Tax Percentage

IMPORTANT:
- Identify ALL fields visible in the form
- For tables with multiple rows, create fields for at least 3 rows (item_1, item_2, item_3)
- Use snake_case for field names
- Be thorough - don't miss any fields
- Maximum 25 fields

FORM TEXT:
"""
    
    @staticmethod
    def extract_fields_from_text(pdf_text: str) -> List[Dict]:
        """
        Use AI to identify form fields from PDF text.
        
        Args:
            pdf_text: Extracted text from PDF
            
        Returns:
            List of field dictionaries
        """
        api_key = OPENROUTER_API_KEY or os.getenv('OPENROUTER_API_KEY')
        
        if not api_key:
            raise Exception("OpenRouter API key not configured")
        
        # Log the extracted text for debugging
        print(f"[DEBUG] Extracted PDF text preview: {pdf_text[:500]}...")
        
        # Truncate text if too long
        truncated_text = pdf_text[:4000] if len(pdf_text) > 4000 else pdf_text
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": AIFieldExtractor.DEFAULT_MODEL,
            "messages": [
                {"role": "user", "content": AIFieldExtractor.EXTRACTION_PROMPT + truncated_text}
            ],
            "max_tokens": 1500,
            "temperature": 0.2
        }
        
        # Retry logic for rate limiting
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    AIFieldExtractor.OPENROUTER_URL,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        content = response_data['choices'][0]['message']['content'].strip()
                        fields = AIFieldExtractor._parse_ai_fields(content)
                        if fields:
                            return fields
                            
                elif response.status_code == 429:
                    wait_time = (attempt + 1) * 2
                    print(f"[WARN] Rate limited, waiting {wait_time}s before retry...")
                    import time
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[ERROR] AI field extraction failed: {response.status_code}: {response.text[:200]}")
                    
            except requests.exceptions.Timeout:
                print(f"[WARN] OpenRouter timeout on attempt {attempt + 1}")
                continue
            except Exception as e:
                print(f"[ERROR] AI field extraction error: {e}")
        
        raise Exception("Failed to extract fields from PDF using AI")
    
    @staticmethod
    def _parse_ai_fields(ai_response: str) -> List[Dict]:
        """Parse AI response into field list."""
        fields = []
        
        for line in ai_response.split('\n'):
            line = line.strip()
            if not line or not line.startswith('FIELD:'):
                continue
            
            try:
                # Parse: FIELD: name | TYPE: text | LABEL: label
                parts = line.split('|')
                if len(parts) >= 2:
                    field_name = parts[0].replace('FIELD:', '').strip()
                    field_type = parts[1].replace('TYPE:', '').strip().lower()
                    label = parts[2].replace('LABEL:', '').strip() if len(parts) > 2 else field_name
                    
                    # Normalize field type
                    if field_type in ['date', 'signature']:
                        field_type = 'text'
                    elif field_type not in ['text', 'checkbox', 'choice']:
                        field_type = 'text'
                    
                    fields.append({
                        "name": field_name,
                        "type": field_type,
                        "label": label,
                        "ai_detected": True
                    })
            except Exception:
                continue
        
        print(f"[INFO] AI detected {len(fields)} fields from PDF text")
        return fields


# AI Question Generation Class
class AIConverter:
    """
    Handles AI-based question generation for form fields using OpenRouter (Mistral 7B).
    
    Dynamically generates questions based on field names and PDF context.
    """
    
    # OpenRouter configuration
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MODEL = "mistralai/mistral-7b-instruct"
    
    # System prompt for strict field-based question generation
    SYSTEM_PROMPT = """You are a professional form assistant helping users fill out government/tax forms. Your ONLY job is to convert PDF form field names into clear, professional questions.

CRITICAL RULES:
1. ONLY ask about the EXACT field provided - never invent or add questions
2. The field name tells you what information is needed - interpret it literally
3. Match question type to field type:
   - checkbox → Yes/No question
   - text → Open-ended question asking for the specific information
   - choice → Selection question
4. Be professional and helpful, like a government clerk assisting someone
5. Keep questions concise and clear
6. If the field name looks like a code (f1_1, c1_2, etc.), use the PDF context to understand what it's asking for
7. NEVER ask the same question twice - each field is unique

OUTPUT FORMAT (strict):
Question: [Your question here]
Help: [One sentence explaining what to enter]"""
    
    @staticmethod
    def generate_question(field: Dict, pdf_context: str = "") -> Optional[Dict]:
        """
        Generate a human-friendly question for a form field using AI.
        
        Uses OpenRouter API to generate contextual questions based on field name and PDF content.
        
        Args:
            field: Dictionary with 'name' and 'type' keys
            pdf_context: Optional PDF text for context
            
        Returns:
            Dictionary with 'question' and 'explanation'
        """
        field_name = field.get('name', 'unknown')
        
        # Always use AI to generate questions dynamically
        print(f"[INFO] Generating AI question for field: {field_name}")
        try:
            return AIConverter._call_openrouter(field, pdf_context)
        except Exception as e:
            print(f"[WARN] AI failed for field {field_name}: {e}")
            # Smart fallback based on field name patterns
            return AIConverter._generate_smart_fallback(field)
    
    @staticmethod
    def _call_openrouter(field: Dict, pdf_context: str = "") -> Optional[Dict]:
        """
        Call OpenRouter API (Mistral 7B) to generate question.
        
        Args:
            field: Dictionary with 'name' and 'type' keys
            pdf_context: Optional PDF text for context
            
        Returns:
            Dictionary with 'question' and 'explanation'
        """
        field_name = field.get('name', 'unknown')
        field_type = field.get('type', 'text')
        field_label = field.get('label', '')
        
        # Check for API key
        api_key = OPENROUTER_API_KEY or os.getenv('OPENROUTER_API_KEY')
        
        if not api_key:
            raise Exception("OpenRouter API key not configured")
        
        print(f"[INFO] Generating AI question for field: {field_name} (type: {field_type})")
        
        # Create the prompt
        prompt = AIConverter._create_prompt(field, pdf_context)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": AIConverter.DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": AIConverter.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 200,
            "temperature": 0.2
        }
        
        # Retry logic for rate limiting
        max_retries = 3
        last_status_code = None
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    AIConverter.OPENROUTER_URL,
                    headers=headers,
                    json=payload,
                    timeout=20
                )
                
                last_status_code = response.status_code
                print(f"[DEBUG] OpenRouter response status: {response.status_code}")
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        content = response_data['choices'][0]['message']['content'].strip()
                        print(f"[DEBUG] AI response: {content[:200]}...")
                        
                        return AIConverter._parse_response(content, field)
                    else:
                        print(f"[WARN] OpenRouter returned no choices: {response_data}")
                        
                elif response.status_code == 429 or response.status_code == 402:
                    if attempt == max_retries - 1:
                        # Last attempt failed, raise HTTPException
                        print(f"[ERROR] All retries exhausted with status {response.status_code}")
                        if response.status_code == 429:
                            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again in a moment.")
                        else:
                            raise HTTPException(status_code=402, detail="API credits exhausted. Please contact support.")
                    
                    wait_time = (attempt + 1) * 2
                    print(f"[WARN] Rate limited/No credits (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[ERROR] OpenRouter API error {response.status_code}: {response.text[:300]}")
                    
            except requests.exceptions.Timeout:
                print(f"[WARN] OpenRouter timeout on attempt {attempt + 1}")
                continue
            except HTTPException:
                # Re-raise HTTPException so it propagates to the frontend
                raise
            except Exception as e:
                print(f"[ERROR] OpenRouter error: {e}")
        
        # If all retries failed with other errors, check last status code
        if last_status_code == 429:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again in a moment.")
        elif last_status_code == 402:
            raise HTTPException(status_code=402, detail="API credits exhausted. Please contact support.")
        
        # Generic failure
        raise Exception(f"Failed to generate question for field: {field_name}")
    
    @staticmethod
    def _create_prompt(field: Dict, pdf_context: str = "") -> str:
        """
        Create prompt for OpenRouter based on field information.
        
        Args:
            field: Dictionary with 'name' and 'type' keys
            pdf_context: Optional PDF text for context
            
        Returns:
            Prompt string for Gemini
        """
        field_name = field.get('name', 'unknown')
        field_type = field.get('type', 'text')
        export_value = field.get('export_value', '')
        
        # Clean up field name for better AI understanding
        readable_name = field_name.replace('_', ' ').replace('-', ' ')
        
        # Check if field name is generic (just numbers)
        is_generic = re.match(r'^\d+$', field_name.strip())
        
        # Build a strict, structured prompt
        prompt = f"Convert this PDF form field into a professional question.\n\n"
        prompt += f"FIELD NAME: {field_name}\n"
        prompt += f"READABLE: {readable_name}\n"
        prompt += f"FIELD TYPE: {field_type}\n"
        
        if export_value and export_value not in ['Yes', 'On', '1']:
            prompt += f"CHECKBOX LABEL: {export_value}\n"
        
        # Add PDF context for generic field names
        if is_generic and pdf_context:
            prompt += f"\nFORM CONTEXT (use this to understand what field #{field_name} might be asking for):\n"
            prompt += pdf_context[:1000] + "\n"
        
        # Add type-specific instructions
        prompt += "\nINSTRUCTIONS:\n"
        
        if field_type == "checkbox":
            if export_value and export_value not in ['Yes', 'On', '1']:
                prompt += f"- This checkbox represents the option '{export_value}'\n"
                prompt += f"- Ask if the user wants to select '{export_value}' (Yes/No)\n"
            else:
                prompt += "- This is a checkbox field\n"
                prompt += "- Ask a Yes/No question based on the field name\n"
        elif field_type == "choice":
            prompt += "- This is a dropdown/selection field\n"
            prompt += "- Ask the user to select or specify their choice\n"
        else:  # text
            prompt += "- This is a text input field\n"
            prompt += "- Ask for the specific information indicated by the field name\n"
            if is_generic:
                prompt += "- The field name is just a number, so use the form context to determine what information is needed\n"
        
        prompt += "\nGenerate ONE question for this EXACT field only."
        
        return prompt
    
    @staticmethod
    def _parse_response(content: str, field: Dict) -> Dict:
        """
        Parse Gemini response into question and explanation.
        
        Args:
            content: Response content from Gemini
            field: Original field dictionary
            
        Returns:
            Dictionary with 'question' and 'explanation'
        """
        content = content.strip()
        question = ""
        explanation = ""
        
        # Try to parse "Question: ... Help: ..." format
        if 'Question:' in content:
            parts = content.split('Question:', 1)
            if len(parts) > 1:
                rest = parts[1]
                if 'Help:' in rest:
                    q_parts = rest.split('Help:', 1)
                    question = q_parts[0].strip()
                    explanation = q_parts[1].strip() if len(q_parts) > 1 else ""
                else:
                    question = rest.strip()
        
        # Try "Question? | Explanation" format
        if not question and '|' in content:
            parts = content.split('|', 1)
            question = parts[0].strip()
            explanation = parts[1].strip() if len(parts) > 1 else ""
        
        # Try splitting by newline
        if not question:
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            for line in lines:
                # Skip markdown formatting and labels
                clean_line = line.lstrip('*#-').strip()
                clean_line = clean_line.replace('Question:', '').replace('Help:', '').strip()
                
                if '?' in clean_line and not question:
                    question = clean_line
                elif clean_line and question and not explanation:
                    explanation = clean_line
            
            if not question and lines:
                question = lines[0].lstrip('*#-').strip()
        
        # Clean up
        question = question.replace('"', '').replace("Question:", "").replace("Q:", "").strip()
        explanation = explanation.replace('"', '').replace("Explanation:", "").replace("Help:", "").strip()
        
        # Remove any asterisks from markdown
        question = question.replace('**', '').replace('*', '')
        explanation = explanation.replace('**', '').replace('*', '')
        
        # If still no question, use the raw content
        if not question or len(question) < 3:
            question = content.split('\n')[0].strip() if content else f"Please provide {field.get('name', 'this field')}:"
        
        if not explanation:
            explanation = "Please provide this information accurately."
        
        return {
            "question": question,
            "explanation": explanation
        }
    
    @staticmethod
    def _generate_smart_fallback(field: Dict) -> Dict:
        """
        Generate a smart fallback question when AI fails.
        Uses field name patterns to create relevant questions.
        
        Args:
            field: Dictionary with 'name' and 'type' keys
            
        Returns:
            Dictionary with 'question' and 'explanation'
        """
        field_name = field.get('name', 'unknown')
        field_type = field.get('type', 'text')
        
        # Clean up field name for display
        readable_name = field_name.replace('_', ' ').replace('-', ' ')
        
        # Remove common prefixes like f1_, c1_, etc.
        cleaned = re.sub(r'^[fc]\d+_', '', readable_name)
        cleaned = re.sub(r'\[\d+\]', '', cleaned)  # Remove array indices
        cleaned = cleaned.strip()
        
        # Common field name patterns and their questions
        patterns = {
            r'name|nombre': ('What is your name?', 'Enter your full legal name.'),
            r'first.*name': ('What is your first name?', 'Enter your first name as it appears on official documents.'),
            r'last.*name|surname': ('What is your last name?', 'Enter your last name as it appears on official documents.'),
            r'address|direccion': ('What is your address?', 'Enter your street address.'),
            r'city|ciudad': ('What is your city?', 'Enter your city name.'),
            r'state|estado': ('What is your state?', 'Enter your state abbreviation.'),
            r'zip|postal': ('What is your ZIP code?', 'Enter your 5-digit ZIP code.'),
            r'ssn|social.*security': ('What is your Social Security Number?', 'Enter your 9-digit SSN.'),
            r'ein|employer.*id': ('What is your Employer Identification Number?', 'Enter your EIN.'),
            r'phone|tel': ('What is your phone number?', 'Enter your phone number.'),
            r'email|correo': ('What is your email address?', 'Enter your email address.'),
            r'date|fecha': ('What is the date?', 'Enter the date in MM/DD/YYYY format.'),
            r'signature|sign|firma': ('Please upload your signature', 'Upload an image of your signature (PNG, JPG).'),
            r'photo|picture|imagen|image': ('Please upload a photo', 'Upload your photo (PNG, JPG).'),
            r'business|company|empresa': ('What is your business name?', 'Enter your business or company name.'),
        }
        
        # Check for signature/photo fields first - these should be image upload type
        if re.search(r'signature|sign|firma|photo|picture|imagen|image', field_name.lower()) or \
           re.search(r'signature|sign|firma|photo|picture|imagen|image', cleaned.lower()):
            return {
                "question": "Please upload your signature or photo",
                "explanation": "Upload an image file (PNG, JPG, JPEG)",
                "field_type": "image"
            }
        
        # Check for pattern matches
        for pattern, (question, explanation) in patterns.items():
            if re.search(pattern, field_name.lower()) or re.search(pattern, cleaned.lower()):
                return {"question": question, "explanation": explanation}
        
        # Default fallback based on field type
        if field_type == 'checkbox':
            return {
                "question": f"Do you want to select '{cleaned or readable_name}'?",
                "explanation": "Answer Yes or No."
            }
        elif field_type == 'choice':
            return {
                "question": f"Please select your choice for '{cleaned or readable_name}':",
                "explanation": "Choose the option that applies to you."
            }
        else:
            return {
                "question": f"Please enter {cleaned or readable_name}:",
                "explanation": "Provide this information as accurately as possible."
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


def generate_document_summary(pdf_text: str) -> str:
    """
    Generate an AI summary of the PDF document.
    
    Args:
        pdf_text: Extracted text from the PDF
        
    Returns:
        Summary string describing the document
    """
    api_key = OPENROUTER_API_KEY or os.getenv('OPENROUTER_API_KEY')
    
    if not api_key:
        return "Unable to generate summary - API key not configured."
    
    if not pdf_text or len(pdf_text.strip()) < 50:
        return "This appears to be a fillable form document. I'll help you complete each field."
    
    # Truncate text if too long
    text_for_summary = pdf_text[:4000] if len(pdf_text) > 4000 else pdf_text
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "mistralai/mistral-7b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful document assistant. Provide a brief, friendly summary of documents. Be concise (2-3 sentences). Mention the document type and its main purpose."
                },
                {
                    "role": "user",
                    "content": f"Please summarize this document briefly:\n\n{text_for_summary}"
                }
            ],
            "max_tokens": 200,
            "temperature": 0.3
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                summary = data['choices'][0]['message']['content'].strip()
                print(f"[INFO] 📄 Generated document summary: {summary[:100]}...")
                return summary
        
        return "This is a government form document. I'll help you fill it out step by step."
        
    except Exception as e:
        print(f"[WARN] Failed to generate summary: {e}")
        return "This is a government form document. I'll guide you through completing it."


def chat_with_ai(user_message: str, pdf_context: str = "", chat_history: list = None) -> str:
    """
    Have a conversation with the AI about the document or general questions.
    
    Args:
        user_message: The user's message
        pdf_context: Context from the PDF document
        chat_history: Previous messages in the conversation
        
    Returns:
        AI response string
    """
    api_key = OPENROUTER_API_KEY or os.getenv('OPENROUTER_API_KEY')
    
    if not api_key:
        return "I'm sorry, I can't respond right now. API key not configured."
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = """You are a helpful, friendly assistant for Bureaucracy Breaker - an app that helps people fill out government forms.

You can:
1. Answer questions about the document being filled
2. Explain what certain form fields mean
3. Provide general guidance on government forms
4. Have casual conversation

Keep responses concise and helpful. If the user wants to continue filling the form, tell them to type "continue" or "next".

Document context (if available):
""" + (pdf_context[:2000] if pdf_context else "No document loaded yet.")
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add chat history if available
        if chat_history:
            for msg in chat_history[-6:]:  # Keep last 6 messages for context
                messages.append(msg)
        
        messages.append({"role": "user", "content": user_message})
        
        payload = {
            "model": "mistralai/mistral-7b-instruct",
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=20
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                return data['choices'][0]['message']['content'].strip()
        elif response.status_code == 429:
            return "I'm being rate limited. Please wait a moment and try again."
        elif response.status_code == 402:
            return "API credits exhausted. Please contact support."
        
        return "I'm having trouble responding right now. Please try again."
        
    except Exception as e:
        print(f"[ERROR] Chat error: {e}")
        return "Sorry, I encountered an error. Please try again."


def batch_generate_questions(session: Session) -> None:
    """
    Batch generate ALL questions upfront by analyzing the entire PDF.
    This allows the AI to plan and create better, more contextual questions.
    
    Args:
        session: Session object with form_fields and pdf_text_content
    """
    print(f"[INFO] 🤖 AI is analyzing the PDF and planning questions for {len(session.form_fields)} fields...")
    
    successful = 0
    failed = 0
    
    for field in session.form_fields:
        field_name = field.get('name', '')
        field_type = field.get('type', 'unknown')
        
        # Skip unknown types
        if field_type == 'unknown':
            continue
        
        # Check if this is part of a checkbox group we've already processed
        if field_type == 'checkbox':
            base_name = extract_checkbox_base_name(field_name)
            # Check if we already generated a question for this group
            if any(base_name in key for key in session.pre_generated_questions.keys()):
                continue
        
        try:
            print(f"[INFO] Generating question for: {field_name} ({field_type})")
            question_data = AIConverter.generate_question(field, session.pdf_text_content)
            
            if question_data:
                session.pre_generated_questions[field_name] = {
                    "question": question_data.get("question"),
                    "explanation": question_data.get("explanation"),
                    "field_type": field_type
                }
                successful += 1
            else:
                # Use fallback
                session.pre_generated_questions[field_name] = {
                    "question": f"Please provide {field_name}:",
                    "explanation": "Enter the required information.",
                    "field_type": field_type
                }
                failed += 1
                
        except HTTPException:
            # Re-raise rate limit and credit errors to frontend
            raise
        except Exception as e:
            print(f"[WARN] Failed to generate question for {field_name}: {e}")
            # Use fallback
            session.pre_generated_questions[field_name] = {
                "question": f"Please provide {field_name}:",
                "explanation": "Enter the required information.",
                "field_type": field_type
            }
            failed += 1
    
    session.questions_generated = True
    print(f"[INFO] ✅ Question generation complete! {successful} generated, {failed} fallbacks")


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

def get_checkbox_group_fields(session: Session, base_name: str) -> List[Dict]:
    """
    Get all fields in a checkbox group.
    
    Args:
        session: Session object
        base_name: Base name of checkbox group (e.g., 'c1_1')
        
    Returns:
        List of all fields in the group
    """
    group_fields = []
    for field in session.form_fields:
        if field.get('type') == 'checkbox':
            field_base = extract_checkbox_base_name(field.get('name', ''))
            if field_base == base_name:
                group_fields.append(field)
    return group_fields

def is_radio_button_group(fields: List[Dict]) -> bool:
    """
    Determine if a checkbox group is actually a radio button group (mutually exclusive).
    Radio groups have export values like '1', '2', '3', etc. or are marked in mappings.
    
    Args:
        fields: List of checkbox fields in the group
        
    Returns:
        True if this is a radio button group
    """
    if len(fields) < 2:
        return False
    
    # Check if export values are sequential numbers (1, 2, 3, etc.) or letters
    # indicating a radio button group (mutually exclusive options)
    export_values = [f.get('export_value', '') for f in fields]
    if all(v.isdigit() for v in export_values if v):
        return True
    
    # Check if they're all single letters (A, B, C, etc.)
    if all(len(v) == 1 and v.isalpha() for v in export_values if v):
        return True
    
    return False

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
            
            # Check if this is a radio button group
            group_fields = get_checkbox_group_fields(session, base_name)
            if is_radio_button_group(group_fields):
                # Return a special radio group field with all options
                return {
                    'name': field_name,
                    'type': 'radio_group',
                    'base_name': base_name,
                    'options': group_fields
                }
            
        return field
    
    return None

# API endpoints
@app.get("/health")
async def health_check():
    """
    Health check endpoint with detailed status information.
    Returns API status, AI service availability, and system info.
    """
    import psutil
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "api_version": "1.0.0",
        "ai_service": "OpenRouter (Mistral 7B)",
        "active_sessions": len(sessions),
    }
    
    # Check API key availability
    api_key = OPENROUTER_API_KEY or os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        health_status["status"] = "degraded"
        health_status["warning"] = "No AI API key configured"
    
    # Check system resources
    try:
        memory = psutil.virtual_memory()
        health_status["system"] = {
            "memory_usage_percent": memory.percent,
            "cpu_count": psutil.cpu_count()
        }
    except:
        pass
    
    return health_status

@app.get("/check-api-key")
async def check_api_key():
    """Check if OpenRouter API key is valid and has credits."""
    api_key = OPENROUTER_API_KEY or os.getenv('OPENROUTER_API_KEY')
    
    if not api_key:
        return {
            "valid": False,
            "error": "No API key configured",
            "message": "OpenRouter API key is not set."
        }
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "mistralai/mistral-7b-instruct",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return {
                "valid": True,
                "message": "OpenRouter API key is valid and working!",
                "model": "mistralai/mistral-7b-instruct"
            }
        elif response.status_code == 401:
            return {
                "valid": False,
                "error": "Invalid API key",
                "message": "The API key is invalid or expired."
            }
        elif response.status_code == 402:
            return {
                "valid": False,
                "error": "No credits",
                "message": "API key has no credits remaining. Please add credits at https://openrouter.ai/settings/credits"
            }
        elif response.status_code == 429:
            return {
                "valid": False,
                "error": "Rate limited",
                "message": "Too many requests. Please wait and try again."
            }
        else:
            error_detail = response.json() if response.text else {}
            return {
                "valid": False,
                "error": f"API error {response.status_code}",
                "message": error_detail.get('error', {}).get('message', response.text[:200]),
                "status_code": response.status_code
            }
            
    except requests.exceptions.Timeout:
        return {
            "valid": False,
            "error": "Timeout",
            "message": "API request timed out."
        }
    except Exception as e:
        return {
            "valid": False,
            "error": "Connection error",
            "message": str(e)
        }

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
    
    # Use pre-generated question if available (planning-first approach)
    field_name = field.get("name", "")
    if session.questions_generated and field_name in session.pre_generated_questions:
        question_data = session.pre_generated_questions[field_name]
        print(f"[INFO] Using pre-generated question for {field_name}")
    else:
        # Fallback to on-the-fly generation if needed
        print(f"[WARN] No pre-generated question for {field_name}, generating now...")
        question_data = AIConverter.generate_question(field, session.pdf_text_content)
        
        if not question_data:
            question_data = {
                "question": "Please provide information for this field.",
                "explanation": "This information is required to complete the form.",
                "field_type": field.get("type", "unknown")
            }
    
    return {
        "session_id": session.session_id,
        "question": {
            "text": question_data["question"],
            "explanation": question_data["explanation"],
            "field_type": question_data.get("field_type", field.get("type", "unknown")),
            "field_name": field_name,
            "current": session.current_field_index + 1,
            "total": len(session.form_fields)
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
            field_name = current_field.get("name")
            
            # Special handling for image fields
            field_type = session.pre_generated_questions.get(field_name, {}).get("field_type", current_field.get("type"))
            
            if field_type == "image":
                # Handle special image field responses
                if request.answer == "IMAGE_UPLOADED":
                    # Image was already uploaded via /upload-image endpoint
                    # Just mark the field as complete and advance
                    if field_name and field_name in session.uploaded_images:
                        session.answers[field_name] = "[IMAGE_UPLOADED]"
                        print(f"[INFO] ✅ Image confirmed for field: {field_name}")
                    else:
                        return {
                            "session_id": session.session_id,
                            "error": "No image was uploaded for this field. Please upload an image first."
                        }
                elif request.answer == "SKIP":
                    # User chose to skip this optional image field
                    if field_name:
                        session.answers[field_name] = "[SKIPPED]"
                        print(f"[INFO] ⏭️ Image field skipped: {field_name}")
                else:
                    return {
                        "session_id": session.session_id,
                        "error": "For image fields, please upload an image or click skip."
                    }
            else:
                # Regular field validation
                validation_result = FormValidator.validate_and_format(current_field, request.answer)
                
                if not validation_result["is_valid"]:
                    # Return error without advancing to next field
                    return {
                        "session_id": session.session_id,
                        "error": validation_result["error"]
                    }
                
                # Store the valid answer
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
    
    # Use pre-generated question if available (planning-first approach)
    field_name = field.get("name", "")
    if session.questions_generated and field_name in session.pre_generated_questions:
        question_data = session.pre_generated_questions[field_name]
        print(f"[INFO] Using pre-generated question for {field_name}")
    else:
        # Fallback to on-the-fly generation if needed
        print(f"[WARN] No pre-generated question for {field_name}, generating now...")
        question_data = AIConverter.generate_question(field, session.pdf_text_content)
        
        if not question_data:
            question_data = {
                "question": "Please provide information for this field.",
                "explanation": "This information is required to complete the form.",
                "field_type": field.get("type", "unknown")
            }
    
    return {
        "session_id": session.session_id,
        "question": {
            "text": question_data["question"],
            "explanation": question_data.get("explanation", ""),
            "field_type": question_data.get("field_type", field.get("type", "unknown")),
            "field_name": field_name,
            "current": session.current_field_index + 1,
            "total": len(session.form_fields)
        }
    }

@app.post("/upload-image", response_model=UploadImageResponse)
async def upload_image(
    session_id: str,
    field_name: str,
    file: UploadFile = File(...)
):
    """
    Upload an image for a signature or photo field.
    Accepts common image formats (PNG, JPG, JPEG).
    """
    # Validate session
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_ext = file.filename.lower().split('.')[-1]
    if file_ext not in ['png', 'jpg', 'jpeg', 'gif', 'bmp']:
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload PNG, JPG, or JPEG image."
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Validate it's a real image by opening it
        try:
            img = Image.open(io.BytesIO(image_bytes))
            # Convert to RGB if needed (for JPEG compatibility)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Resize if too large (max 800x600 for PDF embedding)
            max_size = (800, 600)
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                print(f"[INFO] Resized image from {img.size} to fit {max_size}")
            
            # Save processed image to bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            processed_image_bytes = img_buffer.getvalue()
            
        except Exception as img_err:
            print(f"[ERROR] Invalid image file: {img_err}")
            raise HTTPException(status_code=400, detail="Invalid or corrupted image file")
        
        # Store image in session
        session.uploaded_images[field_name] = processed_image_bytes
        
        print(f"[INFO] Image uploaded for field {field_name}: {len(processed_image_bytes)} bytes")
        
        return {
            "success": True,
            "message": f"Image uploaded successfully for {field_name}",
            "field_name": field_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Image upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat with the AI assistant about the document or ask questions.
    Users can ask questions, get help, or have a conversation.
    Type "continue", "next", or "fill" to resume form filling.
    """
    session = get_session(request.session_id)
    user_message = request.message.strip().lower()
    
    # Check if user wants to continue filling the form
    continue_keywords = ['continue', 'next', 'fill', 'proceed', 'resume', 'start filling', 'lets go', "let's go"]
    if any(keyword in user_message for keyword in continue_keywords):
        session.chat_mode = False
        return {
            "response": "Great! Let's continue filling out your form. I'll ask you the next question.",
            "is_form_command": True
        }
    
    # Set chat mode
    session.chat_mode = True
    
    # Get AI response
    ai_response = chat_with_ai(
        request.message, 
        session.pdf_text_content,
        None  # Could add chat history tracking in future
    )
    
    return {
        "response": ai_response,
        "is_form_command": False
    }


@app.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Accept PDF upload, extract fields, and create a session."""
    if not file:
        raise HTTPException(status_code=400, detail="File is required")
    
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        pdf_bytes = await file.read()
        
        # First try to extract AcroForm fields (fillable PDF)
        fields = PDFProcessor.extract_fields(pdf_bytes)
        is_fillable = len(fields) > 0
        pdf_text = ""
        
        # Always extract text for context (helps AI generate better questions)
        pdf_text = PDFProcessor.extract_text(pdf_bytes)
        
        if not is_fillable:
            # Non-fillable PDF - use AI to identify fields from text
            print("[INFO] No AcroForm fields found, analyzing PDF text...")
            
            if pdf_text:
                # Use AI to identify form fields from text
                fields = AIFieldExtractor.extract_fields_from_text(pdf_text)
                print(f"[INFO] AI identified {len(fields)} fields from PDF text")
            else:
                print("[WARN] Could not extract text from PDF")
                # Provide generic fields as fallback
                fields = AIFieldExtractor._basic_field_extraction("")
        
        session = create_session()
        session.original_pdf = pdf_bytes
        session.form_fields = fields
        session.current_field_index = 0
        session.is_fillable_pdf = is_fillable
        session.pdf_text_content = pdf_text  # Store text for AI context
        
        pdf_type = "fillable (AcroForm)" if is_fillable else "non-fillable (text-based)"
        print(f"[INFO] Processed {pdf_type} PDF with {len(fields)} fields")
        
        # 📄 Generate document summary first
        print("[INFO] 📝 Generating document summary...")
        session.document_summary = generate_document_summary(pdf_text)
        
        # 🤖 PLANNING PHASE: Generate ALL questions upfront
        print("[INFO] 🎯 Starting AI planning phase - analyzing PDF and creating all questions...")
        batch_generate_questions(session)
        
        return {
            "session_id": session.session_id,
            "total_fields": len(fields),
            "summary": session.document_summary
        }
        
    except Exception as e:
        print(f"[ERROR] PDF upload failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Unable to process PDF: {str(e)}")

@app.post("/generate-pdf")
async def generate_pdf(request: GeneratePDFRequest):
    """Generate a completed PDF using collected answers and return it for download."""
    session = get_session(request.session_id)
    
    if not session.answers:
        raise HTTPException(status_code=400, detail="No answers available to generate PDF")
    
    if not session.original_pdf:
        raise HTTPException(status_code=400, detail="Original PDF not found in session")
    
    try:
        complete_answers = prepare_checkbox_group_answers(session)
        
        # Choose generation method based on PDF type
        if session.is_fillable_pdf:
            # Fillable PDF - update form fields directly and pass uploaded images
            filled_pdf_bytes = PDFProcessor.fill_pdf(
                session.original_pdf, 
                complete_answers, 
                session.uploaded_images
            )
            print(f"[INFO] Filled AcroForm PDF fields with {len(session.uploaded_images)} images")
        else:
            # Non-fillable PDF - generate summary page with answers
            filled_pdf_bytes = PDFProcessor.generate_text_pdf(session.original_pdf, complete_answers)
            print(f"[INFO] Generated summary page for non-fillable PDF")
        
        if request.session_id in sessions:
            del sessions[request.session_id]
            print(f"[INFO] Generated PDF and cleaned up session: {request.session_id}")
        
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
            "fields": session.form_fields,
            "answer_count": len(session.answers),
            "answers": session.answers,
            "has_original_pdf": session.original_pdf is not None,
            "current_field_index": session.current_field_index,
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
