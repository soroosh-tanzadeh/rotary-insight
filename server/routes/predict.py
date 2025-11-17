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
    file: UploadFile = File(..., description="CSV file containing signal data"),
    model_name: str = Form(...),
    return_probabilities: bool = Form(False),
):
    """
    Perform model inference on a CSV containing one or more signal channels.
    Supports multi-channel + automatic sliding-window segmentation.

    Expected CSV format example:
    row,ch1,ch2,ch3
    0.0,0.12,0.09,-0.03
    1.0,0.15,0.08,-0.02
    2.0,0.10,0.07,-0.01
    """

    model_manager = get_model_manager()
    start_time = time.time()

    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Detect channel columns
        channel_cols = [c for c in df.columns if c.lower().startswith("ch")]
        if not channel_cols:
            raise ValueError("CSV must contain at least one channel column (ch1, ch2, ...)")

        # Extract signals
        signals = df[channel_cols].values.astype(np.float32)  # (L, C)
        signal_length, num_channels = signals.shape

        # Model config
        config = model_manager.get_model_config(model_name)
        window_size = config.get("window_size", signal_length)
        expected_channels = config.get("num_channels", 1)

        if num_channels != expected_channels:
            raise ValueError(
                f"Model expects {expected_channels} channels, but got {num_channels}"
            )

        # SLIDING WINDOW
        if signal_length <= window_size:
            # pad if needed
            if signal_length < window_size:
                pad = window_size - signal_length
                signals = np.pad(signals, ((0, pad), (0, 0)), mode="constant")
            windows = [signals.T]  # shape: (C, window_size)
        else:
            # split into windows of equal size
            num_windows = signal_length // window_size
            windows = [
                signals[i * window_size : (i + 1) * window_size].T
                for i in range(num_windows)
            ]

        # shape: (num_windows, C, window_size)
        input_data = np.stack(windows, axis=0)

        # Inference
        predictions, probabilities = model_manager.predict(
            model_name=model_name,
            data=input_data,
            return_probabilities=return_probabilities,
        )

        class_names = model_manager.get_class_names(model_name)

        # AGGREGATION (multi-window result)
        if return_probabilities and probabilities is not None:
            avg_probs = probabilities.mean(axis=0)  # average over windows
            final_class = int(avg_probs.argmax())

            result = PredictionResult(
                predicted_class=final_class,
                class_name=class_names[final_class],
                confidence=float(avg_probs[final_class]),
                probabilities=avg_probs.tolist(),
            )

        else:
            # simple majority vote
            unique, counts = np.unique(predictions, return_counts=True)
            final_class = int(unique[counts.argmax()])

            result = PredictionResult(
                predicted_class=final_class,
                class_name=class_names[final_class],
                confidence=1.0,
                probabilities=None,
            )

        # return response
        processing_time = (time.time() - start_time) * 1000

        return InferenceResponse(
            model_name=model_name,
            predictions=[result],
            processing_time_ms=processing_time,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}",
        )
