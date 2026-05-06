"""Unified preprocessing pipeline (Step 1 through Step 10).

This module orchestrates all implemented preprocessing steps in one place and
produces the target feature schema for a single job-post row.

Pipeline order:
1. Text cleaning
2. Job ID generation
3. Date features
4. Title normalization
5. Description embedding
6. Location extraction
7. Remote detection
8. Job type extraction
9. Education extraction
10. Skills extraction
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, Optional

from Job_pipeline.preprocessing.clean_text import CleanTextModule
from Job_pipeline.preprocessing.date_features import DateFeaturesConfig, DateFeaturesModule
from Job_pipeline.preprocessing.description_embedding import DescriptionEmbeddingModule
from Job_pipeline.preprocessing.education_extraction import EducationExtractionModule
from Job_pipeline.preprocessing.job_id import JobIdConfig, JobIdModule
from Job_pipeline.preprocessing.job_type_extraction import JobTypeExtractionModule
from Job_pipeline.preprocessing.location_extraction import LocationExtractionModule
from Job_pipeline.preprocessing.remote_detection import RemoteDetectionModule
from Job_pipeline.preprocessing.skills_extraction import SkillsExtractionModule
from Job_pipeline.preprocessing.tech_job_validation import TechJobValidationModule
from Job_pipeline.preprocessing.title_normalization import TitleNormalizationModule


logger = logging.getLogger(__name__)


TARGET_FEATURES = [
    "year_month",
    "timestamp",
    "month",
    "holiday_flag",
    "job_id",
    "company_name",
    "job_title",
    "normalized_title",
    "DescriptionVec",
    "city",
    "region",
    "country",
    "is_remote",
    "job_type",
    "education_level",
    "skills",
]


@dataclass
class UnifiedPreprocessorConfig:
    """Configuration for unified preprocessing."""

    date_key: str = "created_at"
    title_key: str = "title"
    description_key: str = "description"
    company_key: str = "entity_name"
    source_key: str = "source"
    enable_gemini_fallback: bool = False


class UnifiedPreprocessor:
    """Run all preprocessing modules and emit the final feature contract."""

    def __init__(self, config: Optional[UnifiedPreprocessorConfig] = None):
        self.config = config or UnifiedPreprocessorConfig()

        gemini_callable = None
        if not self.config.enable_gemini_fallback:
            gemini_callable = lambda _prompt: None

        self.clean_text = CleanTextModule()
        self.tech_validator = TechJobValidationModule()
        self.job_id = JobIdModule(
            JobIdConfig(
                title_key=self.config.title_key,
                company_key=self.config.company_key,
                date_key=self.config.date_key,
                source_key=self.config.source_key,
            )
        )
        self.date_features = DateFeaturesModule(DateFeaturesConfig(date_key=self.config.date_key))
        self.title_normalizer = TitleNormalizationModule(gemini_callable=gemini_callable)
        self.description_embedding = DescriptionEmbeddingModule()
        self.location_extractor = LocationExtractionModule(gemini_callable=gemini_callable)
        self.remote_detector = RemoteDetectionModule(gemini_callable=gemini_callable)
        self.job_type_extractor = JobTypeExtractionModule(gemini_callable=gemini_callable)
        self.education_extractor = EducationExtractionModule(gemini_callable=gemini_callable)
        self.skills_extractor = SkillsExtractionModule(gemini_callable=gemini_callable)
        logger.info(
            "UnifiedPreprocessor initialized; gemini_fallback_enabled=%s",
            self.config.enable_gemini_fallback,
        )

    def _build_location_value(self, row: Dict[str, str]) -> str:
        city = (row.get("city") or "").strip()
        country = (row.get("country") or "").strip()
        if city and country:
            return f"{city}, {country}"
        return city or country

    def preprocess_row(
        self, row: Dict[str, str], source_name: Optional[str] = None
    ) -> Optional[Dict[str, object]]:
        """Preprocess one raw row and return target features only."""
        working = dict(row)
        logger.debug("preprocess_row start source_name=%s row_keys=%d", source_name, len(working))
        if source_name and not working.get(self.config.source_key):
            working[self.config.source_key] = source_name
        if not working.get(self.config.source_key):
            working[self.config.source_key] = "unknown"

        cleaned = self.clean_text.transform(
            title=working.get(self.config.title_key),
            description=working.get(self.config.description_key),
        )
        working.update(cleaned)

        tech_decision = self.tech_validator.classify(
            title=working.get("clean_title"),
            description=working.get("clean_description"),
        )
        if not tech_decision.get("is_tech", False):
            logger.info(
                "preprocess_row skipped_non_tech source=%s title=%s category=%s score=%s",
                working.get(self.config.source_key),
                working.get(self.config.title_key),
                tech_decision.get("matched_category"),
                tech_decision.get("confidence_score"),
            )
            return None

        date_out = self.date_features.transform(working)
        id_out = self.job_id.transform(working)
        title_out = self.title_normalizer.normalize(
            title=working.get("clean_title"),
            description=working.get("clean_description"),
        )
        desc_vec_out = self.description_embedding.transform(
            {"clean_description": working.get("clean_description", "")}
        )
        location_out = self.location_extractor.extract(
            description=working.get("clean_description"),
            location_value=self._build_location_value(working),
        )
        remote_out = self.remote_detector.detect(
            title=working.get("clean_title"),
            description=working.get("clean_description"),
        )
        job_type_out = self.job_type_extractor.extract(
            title=working.get("clean_title"),
            description=working.get("clean_description"),
        )
        education_out = self.education_extractor.extract(
            title=working.get("clean_title"),
            description=working.get("clean_description"),
        )
        skills_out = self.skills_extractor.extract(working.get("clean_description"))

        output = {
            "year_month": date_out.get("year_month"),
            "timestamp": date_out.get("timestamp"),
            "month": date_out.get("month"),
            "holiday_flag": date_out.get("holiday_flag"),
            "job_id": id_out.get("job_id"),
            "company_name": working.get(self.config.company_key, ""),
            "job_title": working.get(self.config.title_key, ""),
            "normalized_title": title_out.get("normalized_title"),
            "DescriptionVec": desc_vec_out.get("DescriptionVec"),
            "city": location_out.get("city", ""),
            "region": location_out.get("region", ""),
            "country": location_out.get("country", ""),
            "is_remote": remote_out.get("is_remote", False),
            "job_type": job_type_out.get("job_type", "not_specified"),
            "education_level": education_out.get("education_level", "Not specified"),
            "skills": skills_out.get("skills", []),
        }
        logger.info(
            "preprocess_row complete source=%s job_id=%s title=%s skills_count=%d",
            working.get(self.config.source_key),
            output.get("job_id"),
            output.get("normalized_title"),
            len(output.get("skills") or []),
        )
        return output


__all__ = ["UnifiedPreprocessorConfig", "UnifiedPreprocessor", "TARGET_FEATURES"]
