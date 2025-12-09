"""
Pydantic models for example files listing.
"""

from pydantic import BaseModel, Field
from typing import List


class ExampleFile(BaseModel):
    """Metadata about a single example CSV file."""

    def __init__(
        self,
        filename: str = "",
        sampling_rate: int = 0,
        label: str = "",
    ):
        super().__init__(filename=filename, sampling_rate=sampling_rate, label=label)

    filename: str = Field(..., description="Relative path to the example CSV file")
    sampling_rate: int = Field(
        ..., description="Sampling rate associated with the example file"
    )
    label: str = Field(..., description="Label associated with the example file")


class ExampleData(BaseModel):
    """Metadata about a sample example file."""

    def __init__(self, CWRU: List[ExampleFile] = [], PU: List[ExampleFile] = []):
        super().__init__(CWRU=CWRU, PU=PU)

    CWRU: List[ExampleFile] = Field(..., description="List of CWRU example filenames")
    PU: List[ExampleFile] = Field(..., description="List of PU example filenames")


class ExamplesListResponse(BaseModel):
    """Response model for list of example files."""

    data: ExampleData = Field(..., description="List of example CSV files")
    total_count: int = Field(..., description="Total number of example files found")


class ExampleSignalResponse(BaseModel):
    """Response model for example file signal data."""

    filename: str = Field(..., description="Filename of the example CSV file")
    signal: List[float] = Field(..., description="Signal data extracted from the file")
