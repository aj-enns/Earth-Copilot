# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Cloud Configuration Module
Provides environment-specific endpoints and scopes for Azure Commercial and Government clouds.
Driven by the AZURE_CLOUD_ENVIRONMENT environment variable (set via Bicep/deploy).
"""

import os
from dataclasses import dataclass


@dataclass
class CloudConfig:
    """Configuration for Azure cloud endpoints and OAuth scopes."""

    # STAC API endpoints (Planetary Computer)
    stac_api_url: str
    stac_catalog_url: str

    # Azure Maps
    azure_maps_base_url: str
    azure_maps_scope: str

    # Azure Cognitive Services (OpenAI)
    cognitive_services_scope: str


# Azure Commercial (public cloud) configuration
_COMMERCIAL = CloudConfig(
    stac_api_url="https://planetarycomputer.microsoft.com/api/stac/v1",
    stac_catalog_url="https://planetarycomputer.microsoft.com/api/stac/v1",
    azure_maps_base_url="https://atlas.microsoft.com",
    azure_maps_scope="https://atlas.microsoft.com/.default",
    cognitive_services_scope="https://cognitiveservices.azure.com/.default",
)

# Azure Government configuration
_GOVERNMENT = CloudConfig(
    stac_api_url="https://planetarycomputer.microsoft.com/api/stac/v1",
    stac_catalog_url="https://planetarycomputer.microsoft.com/api/stac/v1",
    azure_maps_base_url="https://atlas.microsoft.us",
    azure_maps_scope="https://atlas.microsoft.us/.default",
    cognitive_services_scope="https://cognitiveservices.azure.us/.default",
)

_CONFIGS = {
    "Commercial": _COMMERCIAL,
    "Government": _GOVERNMENT,
    "AzureCloud": _COMMERCIAL,
    "AzureUSGovernment": _GOVERNMENT,
}

# Resolve configuration from environment variable
_env = os.environ.get("AZURE_CLOUD_ENVIRONMENT", "Commercial")
cloud_cfg: CloudConfig = _CONFIGS.get(_env, _COMMERCIAL)

# Allow STAC_API_URL env var to override the default
_stac_override = os.environ.get("STAC_API_URL")
if _stac_override:
    cloud_cfg = CloudConfig(
        stac_api_url=_stac_override,
        stac_catalog_url=_stac_override,
        azure_maps_base_url=cloud_cfg.azure_maps_base_url,
        azure_maps_scope=cloud_cfg.azure_maps_scope,
        cognitive_services_scope=cloud_cfg.cognitive_services_scope,
    )
