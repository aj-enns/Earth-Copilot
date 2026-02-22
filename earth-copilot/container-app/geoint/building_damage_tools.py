"""
GEOINT Building Damage Assessment Tools for Azure AI Agent Service

Standalone functions for building damage analysis, compatible with
Azure AI Agent Service FunctionTool.

Usage:
    from geoint.building_damage_tools import create_building_damage_functions
    functions = create_building_damage_functions()
    tool = AsyncFunctionTool(functions)
"""

import logging
import json
import asyncio
import concurrent.futures
import os
from typing import Dict, Any, Set, Callable

logger = logging.getLogger(__name__)


def _run_vision_analysis_sync(latitude: float, longitude: float, module_type: str,
                               radius_miles: float, user_query: str) -> Dict:
    """Run the async VisionAnalyzer in a dedicated thread with its own event loop.

    This avoids conflicts with the Agent SDK's running event loop.
    """
    from geoint.vision_analyzer import get_vision_analyzer
    vision_analyzer = get_vision_analyzer()

    def _run():
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                vision_analyzer.analyze_location_with_vision(
                    latitude=latitude,
                    longitude=longitude,
                    module_type=module_type,
                    radius_miles=radius_miles,
                    user_query=user_query,
                    additional_context=None,
                )
            )
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result(timeout=180)


def assess_building_damage(latitude: float, longitude: float, radius_miles: float = 5.0) -> str:
    """Assess building damage at a location using satellite imagery and GPT Vision analysis.
    Returns damage severity classification and structural integrity assessment.
    Use this when the user asks about building damage, structural damage, disaster impact, or damage assessment.

    :param latitude: Center latitude of the assessment area
    :param longitude: Center longitude of the assessment area
    :param radius_miles: Radius in miles for analysis area (default 5.0)
    :return: JSON string with damage assessment results including severity and visual analysis
    """
    try:
        vision_result = _run_vision_analysis_sync(
            latitude, longitude, "building_damage", radius_miles,
            "Assess building damage and structural integrity in this location",
        )
        return json.dumps({
            "location": {"latitude": latitude, "longitude": longitude},
            "radius_miles": radius_miles,
            "visual_assessment": vision_result.get("visual_analysis"),
            "features_identified": vision_result.get("features_identified", []),
            "imagery_metadata": vision_result.get("imagery_metadata", {}),
            "confidence": vision_result.get("confidence", 0.0),
            "methodology": "LLM Vision analysis of satellite imagery for structural damage indicators"
        })
    except Exception as e:
        logger.error(f"Building damage assessment failed: {e}")
        return json.dumps({
            "location": {"latitude": latitude, "longitude": longitude},
            "status": "error",
            "message": f"Unable to perform damage assessment: {str(e)}. Satellite imagery may not be available."
        })


def classify_damage_severity(latitude: float, longitude: float) -> str:
    """Classify damage severity at a location into standard categories:
    No Damage, Minor Damage, Major Damage, or Destroyed.
    Uses visual analysis of satellite imagery.

    :param latitude: Center latitude of the assessment area
    :param longitude: Center longitude of the assessment area
    :return: JSON string with severity classification and confidence level
    """
    try:
        vision_result = _run_vision_analysis_sync(
            latitude, longitude, "building_damage", 2.0,
            "Classify the damage severity at this location. Use one of: No Damage, Minor Damage, Major Damage, Destroyed. Look for collapsed structures, debris, burn scars, water damage.",
        )
        return json.dumps({
            "location": {"latitude": latitude, "longitude": longitude},
            "visual_assessment": vision_result.get("visual_analysis"),
            "features_identified": vision_result.get("features_identified", []),
            "confidence": vision_result.get("confidence", 0.0),
            "categories": ["No Damage", "Minor Damage", "Major Damage", "Destroyed"],
            "methodology": "LLM Vision classification of satellite imagery"
        })
    except Exception as e:
        logger.error(f"Damage severity classification failed: {e}")
        return json.dumps({
            "location": {"latitude": latitude, "longitude": longitude},
            "status": "error",
            "message": f"Unable to classify damage severity: {str(e)}"
        })


def create_building_damage_functions() -> Set[Callable]:
    """Return the set of building damage tool functions for AsyncFunctionTool registration."""
    return {
        assess_building_damage,
        classify_damage_severity,
    }
