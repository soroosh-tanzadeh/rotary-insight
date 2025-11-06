"""
Example files endpoints.
"""

import os
import re
from fastapi import APIRouter, HTTPException, status
from server.dto import ExamplesListResponse, ExampleFile

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
