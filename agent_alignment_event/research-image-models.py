from pydantic import BaseModel, Field, HttpUrl, field_validator, ConfigDict
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

class ImageFormat(str, Enum):
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"
    RAW = "raw"
    HEIC = "heic"

class LocationMetadata(BaseModel):
    site_name: str
    coordinates: Optional[tuple[float, float]] = None
    country: str
    region: Optional[str] = None
    
class TechnicalMetadata(BaseModel):
    format: ImageFormat
    size_bytes: int = Field(gt=0)
    dimensions: tuple[int, int]
    dpi: Optional[int] = None
    color_space: Optional[str] = None
    
class ResearchContext(BaseModel):
    project_name: str
    study_area: str
    tags: List[str] = Field(default_factory=list)
    classification: Optional[str] = None
    notes: Optional[str] = None

class Researcher(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    institution: str
    email: str
    orcid: Optional[str] = None
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v
    
    @field_validator('orcid')
    @classmethod
    def validate_orcid(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.startswith('https://orcid.org/'):
            raise ValueError('ORCID must be a valid ORCID URL')
        return v

class ResearchImage(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
    )
    
    id: UUID = Field(default_factory=uuid4)
    filename: str
    filepath: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Metadata
    technical: TechnicalMetadata
    location: LocationMetadata
    research: ResearchContext
    
    # Researcher Information
    researcher: Researcher
    
    # Additional Fields
    storage_url: Optional[HttpUrl] = None
    backup_urls: List[HttpUrl] = Field(default_factory=list)
    custom_metadata: Dict = Field(default_factory=dict)
    
    def to_jsonl(self) -> str:
        """Convert the image metadata to JSONL format"""
        return self.model_dump_json(exclude_none=True)
    
    @field_validator('filepath')
    @classmethod
    def validate_filepath(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Filepath cannot be empty')
        return v
    
    def update_metadata(self, **kwargs):
        """Update the custom metadata dictionary"""
        self.custom_metadata.update(kwargs)
        self.updated_at = datetime.utcnow()

# Example usage
example_image = ResearchImage(
    filename="temple_wall_001.jpg",
    filepath="/data/egypt/edfu/2024/",
    technical=TechnicalMetadata(
        format=ImageFormat.JPEG,
        size_bytes=2048576,
        dimensions=(3000, 2000)
    ),
    location=LocationMetadata(
        site_name="Temple of Edfu",
        coordinates=(24.9776, 32.8729),
        country="Egypt",
        region="Aswan Governorate"
    ),
    research=ResearchContext(
        project_name="Egyptian Temple Documentation",
        study_area="Archaeological Photography",
        tags=["temple", "hieroglyphics", "wall-relief", "ptolemaic-period"],
        classification="architectural"
    ),
    researcher=Researcher(
        name="Dr. Jane Smith",
        institution="University of Archaeology",
        email="j.smith@arch.edu"
    )
)