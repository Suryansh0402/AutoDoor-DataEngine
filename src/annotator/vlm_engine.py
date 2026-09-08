"""VLM Engine for querying Google Gemini Vision models to generate annotations."""

import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

from .prompt_templates import (
    ImageAnnotationResponse,
    DOOR_DETECTION_SYSTEM_INSTRUCTION
)


class VLMEngine:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or os.getenv("DEFAULT_VLM_MODEL", "gemini-2.0-flash")
        self._client = None

        if self.api_key:
            self._init_client()
        else:
            print("⚠️ Note: GEMINI_API_KEY not set. Operating in offline/dry-run mode until key is provided.")

    def _init_client(self):
        """Initializes the official google-genai client."""
        try:
            from google import genai
            from google.genai import types
            self._client = genai.Client(api_key=self.api_key)
            self._types = types
            print(f"✨ Initialized Gemini Client using model: '{self.model_name}'")
        except ImportError:
            try:
                # Fallback to google.generativeai if older package installed
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=self.api_key)
                self._client = legacy_genai.GenerativeModel(
                    self.model_name,
                    system_instruction=DOOR_DETECTION_SYSTEM_INSTRUCTION
                )
                self._types = None
                print(f"✨ Initialized legacy google.generativeai client using model: '{self.model_name}'")
            except ImportError:
                raise ImportError(
                    "Please install the Google GenAI SDK: pip install google-genai"
                )

    def annotate_image(
        self,
        image_path: str,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ) -> ImageAnnotationResponse:
        """
        Sends an image to Gemini and receives structured door annotations.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")

        if not self._client:
            raise ValueError(
                "Gemini API key is required to run annotations. "
                "Set GEMINI_API_KEY in your .env file or pass it to VLMEngine(api_key='...')."
            )

        pil_image = Image.open(image_path)

        for attempt in range(max_retries):
            try:
                # When using modern google-genai SDK
                if self._types:
                    response = self._client.models.generate_content(
                        model=self.model_name,
                        contents=[pil_image, "Identify all doors in this image with bounding boxes and openness percentage."],
                        config=self._types.GenerateContentConfig(
                            system_instruction=DOOR_DETECTION_SYSTEM_INSTRUCTION,
                            response_mime_type="application/json",
                            response_schema=ImageAnnotationResponse,
                            temperature=0.1
                        )
                    )
                    raw_text = response.text
                else:
                    # Legacy SDK path
                    prompt = (
                        "Analyze this image according to your system instruction. "
                        "Return ONLY valid JSON matching this schema: "
                        "{\"doors\": [{\"box_2d\": [ymin, xmin, ymax, xmax], \"label\": \"open_door\"|\"closed_door\", "
                        "\"openness_percentage\": int, \"door_type\": str, \"confidence\": float, \"reasoning\": str}]}"
                    )
                    response = self._client.generate_content([prompt, pil_image])
                    raw_text = response.text

                # Parse and validate using Pydantic
                clean_json = raw_text.strip()
                if clean_json.startswith("```json"):
                    clean_json = clean_json.replace("```json", "", 1).rsplit("```", 1)[0].strip()
                elif clean_json.startswith("```"):
                    clean_json = clean_json.replace("```", "", 1).rsplit("```", 1)[0].strip()

                parsed_data = json.loads(clean_json)
                return ImageAnnotationResponse.model_validate(parsed_data)

            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1}/{max_retries} failed for '{image_path.name}': {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    print(f"❌ Failed to annotate '{image_path.name}' after {max_retries} attempts.")
                    return ImageAnnotationResponse(doors=[])
