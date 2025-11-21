from fastapi import APIRouter, HTTPException, status, Depends
from server.inference import get_model_manager
from server.auth import get_auth_handler
from server.dto import ModelsListResponse, WindowSizesResponse
from typing import Optional
import models

router = APIRouter(
    prefix="/models",
    tags=["Models"],
    responses={404: {"description": "Not found"}},
)


@router.get(
    "/",
    response_model=ModelsListResponse,
    tags=["Models"],
    summary="List available models",
    dependencies=[Depends(get_auth_handler().verify_api_key)],
)
async def list_models(window_size: Optional[int] = None):
    """
    List all available models with their configurations.

    Returns detailed information about each model including:
    - Model type and window size
    - Dataset and task information
    - Class names
    - Loading status
    """
    try:
        models_info = get_model_manager().list_models(window_size=window_size)
        return ModelsListResponse(models=models_info, total_count=len(models_info))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}",
        )


@router.get(
    "/window-sizes",
    response_model=WindowSizesResponse,
    tags=["Models"],
    summary="List available window sizes",
    dependencies=[Depends(get_auth_handler().verify_api_key)],
)
async def get_window_sizes():
    """
    List all available window sizes across all models.
    
    Returns a list of unique window sizes supported by the available models.
    """
    try:
        window_sizes = get_model_manager().get_available_window_sizes()
        return WindowSizesResponse(window_sizes=window_sizes)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list window sizes: {str(e)}",
        )


@router.post(
    "/{model_name}/load",
    response_model=dict,
    tags=["Models"],
    summary="Load a specific model",
    dependencies=[Depends(get_auth_handler().verify_api_key)],
)
async def load_model(model_name: str):
    """
    Explicitly load a model into memory.

    Models are loaded lazily on first inference by default.
    Use this endpoint to pre-load models for faster first inference.
    """
    try:
        get_model_manager().load_model(model_name)
        return {
            "status": "success",
            "message": f"Model '{model_name}' loaded successfully",
            "model_name": model_name,
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}",
        )


@router.delete(
    "/{model_name}/unload",
    response_model=dict,
    tags=["Models"],
    summary="Unload a specific model",
    dependencies=[Depends(get_auth_handler().verify_api_key)],
)
async def unload_model(model_name: str):
    """
    Unload a model from memory.

    Use this to free up memory when a model is no longer needed.
    """
    try:
        get_model_manager().unload_model(model_name)
        return {
            "status": "success",
            "message": f"Model '{model_name}' unloaded successfully",
            "model_name": model_name,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unload model: {str(e)}",
        )
