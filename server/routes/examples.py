"""
Example files endpoints.
"""

import os
import json
from fastapi import APIRouter, HTTPException, status
from server.dto import (
    ExamplesListResponse,
    ExampleData,
    ExampleSignalResponse,
    ExampleFile,
)
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

    jsonFile = open(EXAMPLES_DIR + "/samples.json", "r")

    data = json.load(jsonFile)

    jsonFile.close()

    respData = ExampleData(CWRU=[], PU=[])

    for dataset in data:
        for file_info in data[dataset]:
            examplefile = ExampleFile(
                filename=file_info["filename"],
                sampling_rate=file_info["sampling_rate"],
                label=file_info["label"],
            )
            if dataset == "CWRU":
                respData.CWRU.append(examplefile)
            elif dataset == "PU":
                respData.PU.append(examplefile)
    return ExamplesListResponse(
        data=data, total_count=len(data["CWRU"]) + len(data["PU"])
    )


@router.get(
    path="/{folder}/{filename}",
    response_model=ExampleSignalResponse,
    summary="Get signal data from a specific example file",
)
async def get_example_file(folder: str, filename: str):
    """
    Get the signal data from a specific example CSV file.
    The file must exist in the examples directory.
    """
    print(folder, filename)
    file_path = os.path.join(EXAMPLES_DIR, folder, filename)

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
