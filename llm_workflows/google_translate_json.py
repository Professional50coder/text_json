
# Google Translate JSON Translator
# Install: pip install google-cloud-translate

from google.cloud import translate_v2 as translate
import json
from typing import Dict, Any
import os

null =1 

class GoogleTranslateJSONConverter:
    # Mapping from language names to ISO 639-1 codes
    LANGUAGE_CODE_MAP = {
        "hindi": "hi",
        "gujarati": "gu",
        "english": "en",
        "odia": "or",
        "marathi": "mr",
        "punjabi": "pa",
        "bengali": "bn",
        "tamil": "ta",
        "telugu": "te",
        "malayalam": "ml",
        "kannada": "kn",
        "urdu": "ur",
        "french": "fr",
        "german": "de",
        "japanese": "ja",
        "korean": "ko",
        "chinese": "zh",
        "arabic": "ar",
        "russian": "ru",
        # Add more as needed
    }

    def __init__(self, project_id: str = None):
        """Initialize Google Translate client"""
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\swati\OneDrive\Desktop\text_json\gen-lang-client-0858700453-3ffbc840e488[1].json"
        self.translate_client = translate.Client()

    def detect_language(self, text: str) -> str:
        """Detect language using Google Translate API"""
        try:
            result = self.translate_client.detect_language(text)
            return result['language']
        except Exception as e:
            print(f"Language detection error: {e}")
            return 'en'  # Default to English

    def translate_text(self, text: str, target_language: str, source_language: str = None) -> str:
        """Translate text using Google Translate API"""
        # Convert language name to code if needed
        lang_code = self.LANGUAGE_CODE_MAP.get(target_language.lower(), target_language)
        try:
            result = self.translate_client.translate(
                text, 
                target_language=lang_code,
                source_language=source_language
            )
            return result['translatedText']
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original if translation fails

    def translate_json_content(self, json_data: Dict, target_language: str) -> Dict:
        """Translate all string values in JSON to target language"""

        def translate_recursive(obj):
            if isinstance(obj, dict):
                return {key: translate_recursive(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [translate_recursive(item) for item in obj]
            elif isinstance(obj, str) and len(obj.strip()) > 0:
                # Only translate non-empty strings
                return self.translate_text(obj, target_language)
            else:
                return obj

        # Detect source language from first text found
        first_text = self._extract_first_text(json_data)
        detected_lang = self.detect_language(first_text) if first_text else 'en'

        print(f"🔍 Detected language: {detected_lang}")
        print(f"🎯 Translating to: {target_language}")

        if detected_lang == target_language:
            print("✅ Same language - no translation needed")
            return json_data

        # Translate the entire JSON structure
        translated_data = translate_recursive(json_data)
        return translated_data

    def _extract_first_text(self, obj) -> str:
        """Helper to find first string in JSON for language detection"""
        if isinstance(obj, dict):
            for value in obj.values():
                result = self._extract_first_text(value)
                if result:
                    return result
        elif isinstance(obj, list):
            for item in obj:
                result = self._extract_first_text(item)
                if result:
                    return result
        elif isinstance(obj, str) and len(obj.strip()) > 0:
            return obj
        return ""

# Usage Example:
if __name__ == "__main__":
    # Sample JSON data
    sample_data ={
  "title": "SATYABHAMA FABRICATION",
  "tagline": null,
  "vision": "To become a good business.",
  "mission": "Providing best metal work to people and high quality welding service.",
  "language": "English and Odia",
  "stage": "Early Stage - Setup, Sustenance, and Expansion (first year)",
  "summary": null,
  "problem_and_customer": "The implicit problem is the customer's need for high-quality, reliable, and affordable metal fabrication products and services. Target customers include: Construction companies, Restaurants, Retail stores and shops, Home builders and contractors. Customers seek: Quality of work, Affordable price and value, Quality and reliability. They will buy due to: High-quality material & workmanship, Competitive pricing.",
  "solution_and_features": "The business provides metal fabrication services, including products like grills, gates, ladders, boxes, and tables. Key differentiators (USP) are: Warranty and guarantee, Reliable delivery, Free delivery around 10km.",
  "market_and_competitors": "The business states 'No competitors around Ekme' (a specific locality). Differentiation from competitors is based on: High quality material & workmanship, Competitive pricing, Expert technician, Problem-solving approach, Attention to customer problems, Continuous improvement.",
  "channels_and_revenue": "Channels for reaching customers include: Advertisement (posters), Facebook, Instagram, Word of mouth, Phone, and References. Expected total value of sales per month: Rs. 43,900 (Sustenance Phase, 2-7 months) and Rs. 50,200 (Expansion Phase, 8-12 months).",
  "operations_and_team": "The business is supported by friends/family. Required infrastructure includes: A shop, Power supply, Machinery, Phone, Raw Materials, and Water supply. Raw materials needed are: Steel, Stainless steel, Aluminum, Copper, and Brass, sourced from metal distributors (Mare Mangala), hardware stores (Sambit hardware), and wholesale metal markets. Transport is required for raw materials and finished products. The business plans to employ 2 people. Regular overhead expenses are anticipated. A shop/office will be set up in Kusiapay (nearest market) due to its optimal location for raw material procurement and product shipping.",
  "traction_and_funding": "Total business setup cost (first month): Rs. 410,000 (including rental deposit, first month rental, business registration, office/shop setup and interiors, furniture and equipment, signage boards, insurance, and machinery). Loan requirement for setup phase: Rs. 210,000. Estimated fixed costs for sustenance phase (2-7 months): Rs. 28,900 per month. Desired profit for sustenance phase: Rs. 15,000 per month. Loan requirement for sustenance phase: Rs. 173,400. Expected profit at the end of sustenance phase: Rs. 90,000. Estimated fixed costs for expansion phase (8-12 months): Rs. 33,200 per month. Desired profit for expansion phase: Rs. 17,000 per month. Loan requirement for expansion phase: Rs. 166,000. Promotion budget: Rs. 500.",
  "risks_and_mitigation": null,
  "social_and_environmental_impact": null
}


    # Initialize translator
    translator = GoogleTranslateJSONConverter()

    # Translate to Hindi
    translated_data = translator.translate_json_content(sample_data, 'hi')
    print(json.dumps(translated_data, indent=2, ensure_ascii=False))
