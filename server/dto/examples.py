"""
Pydantic models for example files listing.
"""

from pydantic import BaseModel, Field
from typing import List


class ExampleFile(BaseModel):
    """Metadata about a sample example file."""

    filename: str = Field(..., description="Filename of the sample CSV file")
    sample_index: int = Field(..., description="Sample index extracted from the filename")
    fault_name: str = Field(..., description="Fault type extracted from the filename")


class ExamplesListResponse(BaseModel):
    """Response model for list of example files."""

    examples: List[ExampleFile] = Field(..., description="List of example CSV files")
    total_count: int = Field(..., description="Total number of example files found")


class ExampleSignalResponse(BaseModel):
    """Response model for example file signal data."""

    filename: str = Field(..., description="Filename of the example CSV file")
    signal: List[float] = Field(..., description="Signal data extracted from the file")

