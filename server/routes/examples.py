"""
Example files endpoints.
"""

import os
import re
from fastapi import APIRouter, HTTPException, status
from server.dto import ExamplesListResponse, ExampleFile, ExampleSignalResponse
import pandas as pd

# Base directory containing CSV examples
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
    List all available CSV files from dataset subdirectories (CWRU, PU).
    Expected filename format:
        {dataset}_sample_window_{index}.csv
    Example:
        CWRU_sample_window_18.csv
    """
    if not os.path.exists(EXAMPLES_DIR):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Examples directory not found: {EXAMPLES_DIR}",
        )

    examples = []

    # Matches: CWRU_sample_window_18.csv  OR  pu_sample_window_10.csv
    pattern = re.compile(r"(.*)_sample_window_(\d+)\.csv", re.IGNORECASE)

    # Walk through all subdirectories inside samples/
    for root, _, files in os.walk(EXAMPLES_DIR):
        for f in files:
            if not f.endswith(".csv"):
                continue

            match = pattern.match(f)
            if match:
                dataset = match.group(1)               # CWRU or PU
                index = int(match.group(2))           # numeric index

                # full relative path (e.g. "CWRU/CWRU_sample_window_1.csv")
                relative_path = os.path.relpath(
                    os.path.join(root, f),
                    EXAMPLES_DIR
                )

                examples.append(
                    ExampleFile(
                        filename=relative_path,
                        sample_index=index,
                        fault_name=dataset,  # name dataset as fault_name (due to filename structure)
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
