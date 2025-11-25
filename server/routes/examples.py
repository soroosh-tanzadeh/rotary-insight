"""
Example files endpoints.
"""

import os
import re
from fastapi import APIRouter, HTTPException, status
from server.dto import ExamplesListResponse, ExampleFile, ExampleSignalResponse
import pandas as pd

# Directory containing CSV examples
EXAMPLES_DIR = os.getenv("EXAMPLES_DIR", "samples")

router = APIRouter(
    prefix="/examples",
    tags=["Examples"],
    responses={404: {"description": "Not found"}},
)


@router.get(
    "/",
    response_model=ExamplesListResponse,
    summary="List example CSV files for fault detection",
)
async def list_example_files():
    """
    List all available example CSV files for fault detection.
    The files are expected to follow the naming pattern:
    `sample{index}_{fault_name}.csv`
    """
    if not os.path.exists(EXAMPLES_DIR):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Examples directory not found: {EXAMPLES_DIR}",
        )

    files = [f for f in os.listdir(EXAMPLES_DIR) if f.endswith(".csv")]
    examples = []

    pattern = re.compile(r"sample(\d+)_(.+)\.csv", re.IGNORECASE)

    for f in files:
        match = pattern.match(f)
        if match:
            index = int(match.group(1))
            fault_name = match.group(2)
            examples.append(
                ExampleFile(
                    filename=f,
                    sample_index=index,
                    fault_name=fault_name,
                )
            )

    return ExamplesListResponse(examples=examples, total_count=len(examples))


@router.get(
    "/{filename}",
    response_model=ExampleSignalResponse,
    summary="Get signal data from a specific example file",
)
async def get_example_file(filename: str):
    """
    Get the signal data from a specific example CSV file.
    The file must exist in the examples directory.
    """
    file_path = os.path.join(EXAMPLES_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Example file not found: {filename}",
        )

    try:
        df = pd.read_csv(file_path)
        if "ch1" not in df.columns:
             raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Column 'ch1' not found in file: {filename}",
            )
        
        signal = df["ch1"].tolist()
        
        return ExampleSignalResponse(
            filename=filename,
            signal=signal,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading file: {str(e)}",
        )
