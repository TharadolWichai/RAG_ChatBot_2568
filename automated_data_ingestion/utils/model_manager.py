"""
Model Manager - จัดการ LLM models จาก KKU IntelSphere หรือ OpenAI-compatible API
"""
import os
import requests
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import streamlit as st


class ModelManager:
    """จัดการ LLM models สำหรับ dashboard"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize Model Manager
        
        Args:
            api_key: OpenAI API key (ถ้าไม่ระบุจะอ่านจาก env)
            base_url: Base URL สำหรับ API (ถ้าไม่ระบุจะอ่านจาก env)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://gen.ai.kku.ac.th/api/v1")
        
        # Default models (fallback ถ้า API ไม่ตอบ)
        self.default_models = [
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "claude-sonnet-4.5",
            "gpt-5-mini",
            "gpt-5",
        ]
    
    def fetch_available_models(self, timeout: int = 10) -> Optional[List[str]]:
        """
        ดึงรายการโมเดลจาก API
        
        Args:
            timeout: Timeout ในวินาที (default: 10)
            
        Returns:
            List of model names หรือ None ถ้าไม่สามารถดึงได้
        """
        if not self.api_key:
            print("⚠️ No API key found")
            return None
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=timeout
            )
            
            if response.status_code == 200:
                models_data = response.json()
                return self._parse_models_response(models_data)
            else:
                print(f"⚠️ API returned status {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print("⏱️ Request timeout")
            return None
        except Exception as e:
            print(f"❌ Error fetching models: {e}")
            return None
    
    def _parse_models_response(self, data: Any) -> List[str]:
        """
        Parse models response จาก API
        
        Args:
            data: Response data จาก API
            
        Returns:
            List of model names
        """
        models = []
        
        # OpenAI format: {"data": [{"id": "model-name"}, ...]}
        if isinstance(data, dict) and "data" in data:
            for model in data["data"]:
                model_id = model.get("id", "")
                if model_id:
                    models.append(model_id)
        
        # Simple list: ["model1", "model2", ...]
        elif isinstance(data, list):
            for model in data:
                if isinstance(model, str):
                    models.append(model)
                elif isinstance(model, dict):
                    model_id = model.get("id", model.get("name", ""))
                    if model_id:
                        models.append(model_id)
        
        # Dict with models key
        elif isinstance(data, dict) and "models" in data:
            models_list = data["models"]
            for model in models_list:
                if isinstance(model, str):
                    models.append(model)
                elif isinstance(model, dict):
                    model_id = model.get("id", model.get("name", ""))
                    if model_id:
                        models.append(model_id)
        
        return models
    
    def get_models_with_fallback(self, use_cache: bool = True, cache_duration: int = 300) -> List[str]:
        """
        ดึงรายการโมเดลพร้อม fallback
        
        Args:
            use_cache: ใช้ cache หรือไม่ (default: True)
            cache_duration: ระยะเวลา cache ในวินาที (default: 300 = 5 นาที)
            
        Returns:
            List of model names
        """
        # Check cache (ใช้ streamlit session state)
        if use_cache and hasattr(st, 'session_state'):
            if 'models_cache' in st.session_state:
                cache_data = st.session_state['models_cache']
                cache_time = cache_data.get('timestamp')
                
                # Check if cache is still valid
                if cache_time:
                    if datetime.now() - cache_time < timedelta(seconds=cache_duration):
                        cached_models = cache_data.get('models', [])
                        if cached_models:
                            print(f"✅ Using cached models ({len(cached_models)} models)")
                            return cached_models
        
        # Try to fetch from API
        print("📡 Fetching models from API...")
        models = self.fetch_available_models()
        
        if models and len(models) > 0:
            print(f"✅ Fetched {len(models)} models from API")
            
            # Save to cache
            if hasattr(st, 'session_state'):
                st.session_state['models_cache'] = {
                    'models': models,
                    'timestamp': datetime.now()
                }
            
            return models
        else:
            print(f"⚠️ Could not fetch models, using defaults ({len(self.default_models)} models)")
            return self.default_models
    
    def get_model_recommendations(self) -> Dict[str, Dict[str, str]]:
        """
        ข้อมูลแนะนำสำหรับแต่ละโมเดล
        
        Returns:
            Dict mapping model name to recommendation info
        """
        return {
            "gemini-2.5-flash-lite": {
                "category": "Gemini (Fast & Cheap)",
                "speed": "⚡⚡⚡",
                "quality": "⭐⭐⭐",
                "description": "เร็วและประหยัด เหมาะสำหรับงานทั่วไป"
            },
            "gemini-2.5-flash": {
                "category": "Gemini (Balanced)",
                "speed": "⚡⚡",
                "quality": "⭐⭐⭐⭐",
                "description": "สมดุลระหว่างความเร็วและคุณภาพ"
            },
            "gemini-2.5-pro": {
                "category": "Gemini (High Quality)",
                "speed": "⚡",
                "quality": "⭐⭐⭐⭐⭐",
                "description": "คุณภาพสูง เหมาะสำหรับงานซับซ้อน"
            },
            "claude-sonnet-4.6": {
                "category": "Claude (Latest)",
                "speed": "⚡⚡",
                "quality": "⭐⭐⭐⭐⭐",
                "description": "ใหม่สุด เหมาะสำหรับการเขียนและวิเคราะห์"
            },
            "claude-sonnet-4.5": {
                "category": "Claude (Stable)",
                "speed": "⚡⚡",
                "quality": "⭐⭐⭐⭐⭐",
                "description": "เสถียร คุณภาพสูง"
            },
            "gpt-5.2": {
                "category": "GPT (Latest)",
                "speed": "⚡⚡",
                "quality": "⭐⭐⭐⭐⭐",
                "description": "GPT รุ่นใหม่ คุณภาพสูง"
            },
            "gpt-5-mini": {
                "category": "GPT (Fast)",
                "speed": "⚡⚡⚡",
                "quality": "⭐⭐⭐⭐",
                "description": "เร็วและประหยัด"
            },
            "deepseek-v3.2": {
                "category": "DeepSeek (Code)",
                "speed": "⚡⚡⚡",
                "quality": "⭐⭐⭐⭐",
                "description": "เหมาะสำหรับงาน Code และ Technical"
            }
        }
    
    def get_model_info(self, model_name: str) -> str:
        """
        ดึงข้อมูลโมเดลสำหรับแสดง
        
        Args:
            model_name: ชื่อโมเดล
            
        Returns:
            String แสดงข้อมูลโมเดล
        """
        recommendations = self.get_model_recommendations()
        
        if model_name in recommendations:
            info = recommendations[model_name]
            return f"{model_name} | {info['speed']} Speed | {info['quality']} Quality"
        else:
            return model_name
    
    def categorize_models(self, models: List[str]) -> Dict[str, List[str]]:
        """
        จัดกลุ่มโมเดลตามประเภท
        
        Args:
            models: รายการโมเดลทั้งหมด
            
        Returns:
            Dict mapping category to list of models
        """
        categorized = {
            "🔥 แนะนำ": [],
            "💎 Gemini": [],
            "🤖 Claude": [],
            "🚀 GPT": [],
            "🦙 Llama": [],
            "⚡ DeepSeek": [],
            "🔧 Mistral/Codestral": [],
            "🌟 อื่นๆ": []
        }
        
        # Recommended models
        recommended = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "claude-sonnet-4.5", "gpt-5-mini"]
        
        for model in models:
            model_lower = model.lower()
            
            # Check recommended
            if model in recommended:
                categorized["🔥 แนะนำ"].append(model)
            
            # Categorize
            if "gemini" in model_lower:
                categorized["💎 Gemini"].append(model)
            elif "claude" in model_lower:
                categorized["🤖 Claude"].append(model)
            elif "gpt" in model_lower:
                categorized["🚀 GPT"].append(model)
            elif "llama" in model_lower:
                categorized["🦙 Llama"].append(model)
            elif "deepseek" in model_lower:
                categorized["⚡ DeepSeek"].append(model)
            elif "mistral" in model_lower or "codestral" in model_lower or "devstral" in model_lower:
                categorized["🔧 Mistral/Codestral"].append(model)
            else:
                categorized["🌟 อื่นๆ"].append(model)
        
        # Remove empty categories
        return {k: v for k, v in categorized.items() if v}


# Singleton instance
_model_manager_instance = None


def get_model_manager() -> ModelManager:
    """Get singleton instance of ModelManager"""
    global _model_manager_instance
    if _model_manager_instance is None:
        _model_manager_instance = ModelManager()
    return _model_manager_instance
