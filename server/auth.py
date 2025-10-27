"""
Authentication middleware for API key validation.
"""
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from typing import List
import os


# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKeyAuth:
    """API Key authentication handler."""
    
    def __init__(self):
        """Initialize with API keys from environment."""
        api_keys_str = os.getenv("API_KEYS", "")
        if not api_keys_str:
            raise ValueError(
                "API_KEYS environment variable is not set. "
                "Please set it in your .env file."
            )
        
        # Split by comma and strip whitespace
        self.api_keys: List[str] = [
            key.strip() for key in api_keys_str.split(",") if key.strip()
        ]
        
        if not self.api_keys:
            raise ValueError("No valid API keys found in API_KEYS environment variable")
        
        print(f"Loaded {len(self.api_keys)} API key(s)")
    
    def verify_api_key(self, api_key: str = Security(api_key_header)) -> str:
        """
        Verify the provided API key.
        
        Args:
            api_key: API key from request header
            
        Returns:
            The validated API key
            
        Raises:
            HTTPException: If API key is invalid or missing
        """
        if api_key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API Key is missing. Please provide X-API-Key header.",
            )
        
        if api_key not in self.api_keys:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API Key",
            )
        
        return api_key


# Global instance
auth_handler = None


def get_auth_handler() -> APIKeyAuth:
    """Get or create the global auth handler instance."""
    global auth_handler
    if auth_handler is None:
        auth_handler = APIKeyAuth()
    return auth_handler

