"""
LLM Service for interacting with Ollama
"""
import json
import logging
import time
import asyncio
from typing import Optional, List, Dict
from ollama import Client

from .prompts import SYSTEM_PROMPT, build_prediction_prompt, FALLBACK_RESPONSE
from .schemas import DrugPrediction

logger = logging.getLogger(__name__)


class LLMService:
    """Service for interacting with Ollama LLM"""
    
    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.client: Optional[Client] = None
        self._available_models: List[str] = []
        self._connection_cache: Optional[Dict] = None
        self._cache_timestamp: float = 0
        self._cache_ttl: float = 30.0  # Cache connection status for 30 seconds
    
    @property
    def available_models(self) -> List[str]:
        """Get list of available models"""
        return self._available_models.copy()
    
    async def initialize(self) -> bool:
        """Initialize Ollama connection and check available models"""
        try:
            self.client = Client(host="http://127.0.0.1:11434")
            
            # Check connection and get available models
            models_response = self.client.list()
            
            # Parse the response - Ollama returns a ListResponse object with a models attribute
            models_list = []
            if hasattr(models_response, 'models'):
                models_list = models_response.models
            elif isinstance(models_response, dict) and 'models' in models_response:
                models_list = models_response['models']
            
            self._available_models = []
            for model_info in models_list:
                # Ollama Model objects have a 'model' attribute (not 'name')
                model_name = None
                if hasattr(model_info, 'model'):
                    model_name = model_info.model
                elif hasattr(model_info, 'name'):
                    model_name = model_info.name
                elif isinstance(model_info, dict):
                    model_name = model_info.get('model') or model_info.get('name')
                
                if not model_name:
                    continue
                
                # Extract base model name (remove :latest, :tag, etc.)
                base_name = model_name.split(':')[0]
                if base_name not in self._available_models:
                    self._available_models.append(base_name)
            
            # Check if default model is available
            if self.model not in self._available_models and self._available_models:
                self.model = self._available_models[0]
                logger.info(f"Default model not found, using: {self.model}")
            
            if not self._available_models:
                logger.warning("No models found in Ollama")
                return False
            
            logger.info(f"Ollama connected. Available models: {self._available_models}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
    
    async def check_connection_async(self) -> bool:
        """
        Async version of connection check (non-blocking)
        """
        current_time = time.time()
        
        # Return cached result if still valid
        if (self._connection_cache is not None and 
            current_time - self._cache_timestamp < self._cache_ttl):
            return self._connection_cache.get('connected', False)
        
        # Check connection in executor to avoid blocking
        try:
            if not self.client:
                self.client = Client(host="http://127.0.0.1:11434")
            
            # Run blocking call in executor with timeout
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(None, self.client.list)
            await asyncio.wait_for(future, timeout=2.0)
            
            # Cache the result
            self._connection_cache = {'connected': True}
            self._cache_timestamp = current_time
            return True
            
        except (asyncio.TimeoutError, Exception) as e:
            logger.debug(f"Connection check failed: {e}")
            self._connection_cache = {'connected': False}
            self._cache_timestamp = current_time
            return False
    
    def check_connection(self) -> bool:
        """
        Synchronous version - returns cached result or False quickly
        Use check_connection_async() for actual checks
        """
        current_time = time.time()
        
        # Return cached result if still valid
        if (self._connection_cache is not None and 
            current_time - self._cache_timestamp < self._cache_ttl):
            return self._connection_cache.get('connected', False)
        
        # If no cache, return False (don't block)
        # The async version should be called periodically to update cache
        return False
    
    async def predict_interaction(
        self,
        drug1: str,
        drug2: str,
        twosides_data: Optional[Dict] = None
    ) -> DrugPrediction:
        """
        Predict drug interaction using LLM
        """
        if not self.client:
            logger.error("Ollama client not initialized")
            return DrugPrediction(**FALLBACK_RESPONSE)
        
        try:
            # Build prompt
            prompt = build_prediction_prompt(drug1, drug2, twosides_data)
            
            # Call Ollama in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    system=SYSTEM_PROMPT,
                    options={
                        "temperature": 0.3,  # Lower temperature for more consistent results
                        "top_p": 0.9,
                    }
                )
            )
            
            # Extract response text
            if hasattr(response, 'response'):
                response_text = response.response
            elif isinstance(response, dict) and 'response' in response:
                response_text = response['response']
            else:
                response_text = str(response)
            
            # Parse JSON from response
            # Try to extract JSON if wrapped in markdown or other text
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            try:
                prediction_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to find JSON object in the text
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    prediction_data = json.loads(response_text[start_idx:end_idx+1])
                else:
                    raise
            
            # Validate and create prediction
            return DrugPrediction(
                has_interaction=prediction_data.get("has_interaction", False),
                severity=prediction_data.get("severity", "unknown"),
                confidence=float(prediction_data.get("confidence", 0.0)),
                explanation=prediction_data.get("explanation", "No explanation provided"),
                mechanism=prediction_data.get("mechanism", "Unknown"),
                recommendations=prediction_data.get("recommendations", []),
                reasoning=prediction_data.get("reasoning", "No reasoning provided")
            )
            
        except Exception as e:
            logger.error(f"Error predicting interaction: {e}", exc_info=True)
            return DrugPrediction(**FALLBACK_RESPONSE)


# Global service instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> Optional[LLMService]:
    """Get the global LLM service instance"""
    return _llm_service


async def initialize_llm_service(model: str = "llama3.2") -> LLMService:
    """Initialize and return the global LLM service"""
    global _llm_service
    _llm_service = LLMService(model=model)
    await _llm_service.initialize()
    return _llm_service

