import time
import io
import numpy as np
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends
from server.inference import get_model_manager
from server.auth import get_auth_handler
from server.dto import InferenceResponse, InferenceRequest, PredictionResult

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
    responses={404: {"description": "Not found"}},
)


@router.post(
    "/predict",
    response_model=InferenceResponse,
    tags=["Inference"],
    summary="Perform inference",
    dependencies=[Depends(get_auth_handler().verify_api_key)],
)
async def predict(request: InferenceRequest):
    """
    Perform inference on input signal data.

    Args:
        request: Inference request containing:
            - data: Input signal data with shape (batch_size, channels, signal_length)
            - model_name: Name of the model to use
            - return_probabilities: Whether to return class probabilities

    Returns:
        Inference results including predictions, class names, and confidence scores.

    Example:
        ```json
        {
            "data": [[[0.1, 0.2, ...]]],  # shape: (1, 1, 512)
            "model_name": "transformer_encoder_cwru_512",
            "return_probabilities": true
        }
        ```
    """
    start_time = time.time()
    model_manager = get_model_manager()
    try:
        # Convert data to numpy array
        data = np.array(request.data, dtype=np.float32)

        # Perform inference
        predictions, probabilities = model_manager.predict(
            model_name=request.model_name,
            data=data,
            return_probabilities=request.return_probabilities,
        )

        # Get class names
        class_names = model_manager.get_class_names(request.model_name)

        # Build response
        results = []
        for i, pred_class in enumerate(predictions):
            result = PredictionResult(
                predicted_class=int(pred_class),
                class_name=class_names[pred_class],
                confidence=(
                    float(probabilities[i, pred_class])
                    if probabilities is not None
                    else 0.0
                ),
                probabilities=(
                    probabilities[i].tolist() if probabilities is not None else None
                ),
            )
            results.append(result)

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        return InferenceResponse(
            model_name=request.model_name,
            predictions=results,
            processing_time_ms=processing_time,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )


@router.post(
    "/predict-csv",
    response_model=InferenceResponse,
    summary="Perform inference on uploaded CSV file",
    dependencies=[Depends(get_auth_handler().verify_api_key)],
)
async def predict_from_csv(
    file: UploadFile = File(
        ..., description="CSV file containing signal data (one channel)"
    ),
    model_name: str = Form(..., description="Model name to use for inference"),
    return_probabilities: bool = Form(False, description="Return class probabilities"),
):
    """
    Perform model inference on a CSV file containing one or more signal samples.

    Expected CSV format:
    ```
    row,ch1
    0.0,-0.06724814371257484
    1.0,-0.22822135728542914
    ...
    ```

    Example curl:
    ```
    curl -X POST "http://127.0.0.1:3000/csv/predict" \
        -H "X-API-Key: your_api_key" \
        -F "file=@sample1_Normal.csv" \
        -F "model_name=transformer_encoder_cwru_512"
    ```
    """

    model_manager = get_model_manager()

    start_time = time.time()

    try:
        # Read CSV into pandas DataFrame
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Validate column
        if "ch1" not in df.columns:
            raise ValueError("CSV file must contain a 'ch1' column.")

        # Extract signal
        signal = df["ch1"].values.astype(np.float32)

        # Get expected window size from model config
        config = model_manager.get_model_config(model_name)
        window_size = config.get("window_size", len(signal))

        # Pad or trim signal to expected length
        if len(signal) < window_size:
            pad_width = window_size - len(signal)
            signal = np.pad(signal, (0, pad_width), mode="constant")
        elif len(signal) > window_size:
            signal = signal[:window_size]

        # Reshape to (batch=1, channels=1, signal_length)
        input_data = np.expand_dims(np.expand_dims(signal, axis=0), axis=0)

        # Perform inference
        predictions, probabilities = model_manager.predict(
            model_name=model_name,
            data=input_data,
            return_probabilities=return_probabilities,
        )

        # Get class names
        class_names = model_manager.get_class_names(model_name)

        # Build prediction result
        pred_class = int(predictions[0])
        result = PredictionResult(
            predicted_class=pred_class,
            class_name=class_names[pred_class],
            confidence=(
                float(probabilities[0, pred_class])
                if probabilities is not None
                else 0.0
            ),
            probabilities=(
                probabilities[0].tolist() if probabilities is not None else None
            ),
        )

        processing_time = (time.time() - start_time) * 1000

        return InferenceResponse(
            model_name=model_name,
            predictions=[result],
            processing_time_ms=processing_time,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )
