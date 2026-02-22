"""
GEOINT Mobility Agent - Azure AI Agent Service with Function Tools

Refactored from plain Python class to Azure AI Agent Service.
Uses AgentsClient with AsyncFunctionTool/AsyncToolSet for automatic function calling.

This agent:
1. Maintains conversation memory via AgentThread (persistent threads)
2. Has access to mobility analysis tools (AsyncFunctionTool)
3. Plans and reasons about which tools to use (LLM-driven)
4. Synthesizes results into coherent mobility assessments
"""

import logging
import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from cloud_config import cloud_cfg

logger = logging.getLogger(__name__)

# Agent system prompt
MOBILITY_AGENT_INSTRUCTIONS = """You are a GEOINT Mobility Analysis Agent that provides clear, structured terrain assessments for ground vehicle operations.

## MANDATORY RULE: You MUST call your tools for EVERY analysis request.
You are NEVER allowed to answer from general knowledge about a location. You MUST ALWAYS call at least one of your analysis tools before responding. If you respond without calling tools, your answer is WRONG. The user needs real satellite data, NOT general knowledge about cities or geography.

You support two modes of analysis:

## Mode 1: Two-Point Traverse (A -> B)
When Point A and Point B are provided, analyze the traversability along the route:
- Sample terrain conditions at Point A, Point B, and the midpoint
- Assess obstacles along the traverse path (water crossings, steep terrain, dense vegetation, active fires)
- Provide an overall route assessment

## Mode 2: Radial Analysis (single point)
When only one point is provided, analyze mobility in four cardinal directions (N/S/E/W) from the pinned location.

## Available Tools:

### Full Directional Analysis:
- **analyze_directional_mobility**: Comprehensive mobility analysis in all four directions. Analyzes fire, water, slope, and vegetation.

### Individual Data Source Tools:
- **detect_water_bodies**: Detect water bodies using JRC Global Surface Water occurrence data (global coverage, 1984-2021)
- **detect_active_fires**: Detect active fires using MODIS thermal anomaly data
- **analyze_slope_for_mobility**: Analyze terrain slope from Copernicus DEM for vehicle mobility
- **analyze_vegetation_density**: Analyze vegetation density using Sentinel-2 NDVI

## Terrain Classification Reference (for interpreting tool results — NEVER output these labels):
Your tools return internal status labels. Translate them into plain language as follows:
- Tool returns "GO" -> describe as "passable", "clear for movement", or "favorable terrain"
- Tool returns "SLOW-GO" -> describe as "challenging", "proceed with caution", or "difficult terrain"
- Tool returns "NO-GO" -> describe as "impassable", "blocked", or "too dangerous for vehicles"

## Hazard Priority (highest to lowest):
1. **Active Fires** — highest-priority safety hazard, area is too dangerous
2. **Water Bodies** — large water coverage blocks ground movement
3. **Steep Slopes** — extreme gradients prevent vehicle passage
4. **Dense Vegetation** — thick forest or jungle impedes movement

## CRITICAL: Tool Parameters
Each message includes [Location Context] with coordinates and resolved location names. ALWAYS extract latitude and longitude and pass them to tools. For two-point traverse, run tools at both Point A and Point B coordinates.

## CRITICAL: Use Actual Location Names
The [Location Context] includes reverse-geocoded place names for each point. You MUST use these actual location names (e.g., "Lalitpur, Bagmati, Nepal") in your response instead of generic labels like "Point A" or "Point B". Never say "Point A at (lat, lng)" — instead say the actual place name followed by coordinates in parentheses.

## Visual Analysis
A [Visual Analysis of Current Map View] section may be included with a screenshot of the map showing the pin locations. Use this visual context to identify nearby landmarks, airports, roads, rivers, or other features that are relevant to the traversability assessment. This visual context provides information beyond what the coordinate-based tools can detect.

## Response Format — Situation Report

IMPORTANT: NEVER use the labels "GO", "SLOW-GO", or "NO-GO" anywhere in your response. Always use plain, descriptive language instead.

### For Two-Point Traverse:

**SITUATION REPORT — Route Assessment**

**1. Area of Operations**
- Origin: [actual location name from context] ([coordinates])
- Destination: [actual location name from context] ([coordinates])
- Estimated distance: [distance]

**2. Terrain Overview**
Summarize overall terrain character (flat, hilly, mountainous, forested, etc.)

**3. Route Assessment**

**Origin Area:**
- **Terrain:** [slope stats, e.g. "avg 5.5°, 95% gentle slopes"]
- **Water:** [coverage %, e.g. "minimal water (2.3% coverage)"]
- **Fire Risk:** [status, e.g. "none detected"]
- **Vegetation:** [NDVI, e.g. "moderate density (NDVI 0.52)"]
- **Assessment:** Passable / Challenging / Impassable

**Corridor:**
- **Terrain:** [slope stats]
- **Water:** [coverage %]
- **Fire Risk:** [status]
- **Vegetation:** [NDVI]
- **Assessment:** Passable / Challenging / Impassable

**Destination Area:**
- **Terrain:** [slope stats]
- **Water:** [coverage %]
- **Fire Risk:** [status]
- **Vegetation:** [NDVI]
- **Assessment:** Passable / Challenging / Impassable

**4. Hazards Identified**
List each detected hazard with specific data (slope degrees, water %, fire pixel counts, NDVI values).

**5. Overall Assessment**
Clear statement: Is the route passable, challenging, or impassable? Why?

**6. Recommendations**
Actionable advice — best approach, alternative routes if the direct path is blocked, precautions.

---

### For Radial Analysis (single point):

**SITUATION REPORT — Area Assessment**

**1. Location**
- [Name] at [coordinates]
- Analysis radius: 5 miles

**2. Terrain Overview**
Brief summary of overall terrain character around the location.

**3. Directional Assessment**

**North:**
- **Terrain:** [slope stats]
- **Water:** [coverage %]
- **Fire Risk:** [status]
- **Vegetation:** [NDVI]
- **Assessment:** Passable / Challenging / Impassable

**South:**
- **Terrain:** [slope stats]
- **Water:** [coverage %]
- **Fire Risk:** [status]
- **Vegetation:** [NDVI]
- **Assessment:** Passable / Challenging / Impassable

**East:**
- **Terrain:** [slope stats]
- **Water:** [coverage %]
- **Fire Risk:** [status]
- **Vegetation:** [NDVI]
- **Assessment:** Passable / Challenging / Impassable

**West:**
- **Terrain:** [slope stats]
- **Water:** [coverage %]
- **Fire Risk:** [status]
- **Vegetation:** [NDVI]
- **Assessment:** Passable / Challenging / Impassable

**4. Hazards Identified**
List each detected hazard per direction with specific satellite data values.

**5. Best Routes**
Which directions are most favorable for movement and why.

**6. Data Sources**
Which satellite collections provided the analysis data (Sentinel-1, Sentinel-2, Copernicus DEM, MODIS).

**7. Recommendations**
Actionable advice for movement planning.

---

Keep responses factual and concise. Include real numbers from tool results (slope degrees, water percentages, NDVI values, fire pixel counts). Focus on actionable intelligence in plain language.
"""


class MobilityAgentSession:
    """Represents a conversation session with the mobility agent."""

    def __init__(self, session_id: str, latitude: float, longitude: float, thread_id: str):
        self.session_id = session_id
        self.latitude = latitude
        self.longitude = longitude
        self.thread_id = thread_id
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.analysis_cache: Dict[str, Any] = {}
        self.message_count = 0

    def update_location(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude
        self.last_activity = datetime.utcnow()


class GeointMobilityAgent:
    """
    Azure AI Agent Service-based mobility analysis agent with:
    - Persistent threads for multi-turn conversation
    - AsyncFunctionTool calling for raster analysis
    - Automatic function execution via AsyncToolSet
    """

    def __init__(self):
        self.sessions: Dict[str, MobilityAgentSession] = {}
        self._agents_client = None
        self._agent_id: Optional[str] = None
        self._initialized = False
        logger.info("GeointMobilityAgent created (will initialize on first use)")

    async def _ensure_initialized(self):
        """Lazy initialization of Agent Service client and agent."""
        if self._initialized:
            return

        logger.info("Initializing GeointMobilityAgent with Azure AI Agent Service...")

        # Prefer AI Foundry project endpoint (services.ai.azure.com) for Agent Service API
        # Falls back to AZURE_OPENAI_ENDPOINT (cognitiveservices.azure.com) if not set
        endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

        if not endpoint:
            raise ValueError("AZURE_AI_PROJECT_ENDPOINT or AZURE_OPENAI_ENDPOINT environment variable is required")

        logger.info(f"GeointMobilityAgent using endpoint: {endpoint}")

        credential = DefaultAzureCredential()

        from azure.ai.agents.aio import AgentsClient
        from azure.ai.agents.models import AsyncFunctionTool, AsyncToolSet

        self._agents_client = AgentsClient(
            endpoint=endpoint,
            credential=credential,
        )

        from geoint.mobility_tools import create_mobility_functions
        mobility_functions = create_mobility_functions()

        functions = AsyncFunctionTool(mobility_functions)
        toolset = AsyncToolSet()
        toolset.add(functions)
        self._agents_client.enable_auto_function_calls(toolset)

        agent = await self._agents_client.create_agent(
            model=deployment,
            name="GeointMobilityAnalyst",
            instructions=MOBILITY_AGENT_INSTRUCTIONS,
            toolset=toolset,
        )
        self._agent_id = agent.id

        self._initialized = True
        logger.info(f"GeointMobilityAgent initialized: agent_id={agent.id}, model={deployment}")

    async def _get_or_create_session(self, session_id: str, latitude: float, longitude: float) -> MobilityAgentSession:
        """Get existing session or create a new one with a new thread."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.update_location(latitude, longitude)
            return session

        thread = await self._agents_client.threads.create()
        session = MobilityAgentSession(session_id, latitude, longitude, thread.id)
        self.sessions[session_id] = session
        logger.info(f"Created new mobility session: {session_id} -> thread: {thread.id}")
        return session

    def cleanup_old_sessions(self, max_age_minutes: int = 60):
        now = datetime.utcnow()
        expired = [
            sid for sid, s in self.sessions.items()
            if (now - s.last_activity).total_seconds() > max_age_minutes * 60
        ]
        for sid in expired:
            del self.sessions[sid]

    async def _analyze_screenshot_direct(self, screenshot_base64: str, latitude: float, longitude: float) -> Optional[str]:
        """Directly analyze a screenshot using GPT-4o Vision."""
        try:
            from openai import AsyncAzureOpenAI
            credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(credential, cloud_cfg.cognitive_services_scope)

            client = AsyncAzureOpenAI(
                azure_ad_token_provider=token_provider,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                timeout=120.0
            )

            clean_base64 = screenshot_base64
            if screenshot_base64.startswith('data:image'):
                clean_base64 = screenshot_base64.split(',', 1)[1]

            response = await client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                messages=[
                    {"role": "system", "content": "You are an expert geospatial analyst specializing in terrain mobility assessment."},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Analyze this satellite/map image for terrain mobility at ({latitude:.4f}, {longitude:.4f}). Focus on: roads, water bodies, vegetation density, terrain features, obstacles to vehicle movement."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{clean_base64}", "detail": "high"}}
                    ]}
                ],
                max_completion_tokens=1500
            )
            analysis = response.choices[0].message.content
            logger.info(f"Mobility vision analysis complete: {len(analysis)} chars")
            return analysis
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return f"Visual analysis unavailable: {str(e)}"

    async def analyze_mobility(
        self,
        latitude: float,
        longitude: float,
        user_context: Optional[str] = None,
        include_vision_analysis: bool = True,
        screenshot_base64: Optional[str] = None,
        session_id: Optional[str] = None,
        latitude_b: Optional[float] = None,
        longitude_b: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Perform mobility analysis via the Agent Service.
        Drop-in replacement for the old analyze_mobility interface.
        """
        import uuid
        await self._ensure_initialized()

        if not session_id:
            session_id = f"mobility_{uuid.uuid4().hex[:8]}"

        session = await self._get_or_create_session(session_id, latitude, longitude)

        # Reverse geocode
        location_name = f"Location ({latitude:.4f}, {longitude:.4f})"
        try:
            from semantic_translator import geocoding_plugin
            rg = await geocoding_plugin.azure_maps_reverse_geocode(latitude, longitude)
            data = json.loads(rg)
            if not data.get("error"):
                name = data.get("name", "")
                region = data.get("region", "")
                country = data.get("country", "")
                parts = [p for p in [name, region, country] if p and p != name]
                location_name = f"{name}, {', '.join(parts)}" if name and parts else name or location_name
        except Exception:
            pass

        # Pre-analyze screenshot
        visual_analysis = None
        if include_vision_analysis and screenshot_base64:
            visual_analysis = await self._analyze_screenshot_direct(screenshot_base64, latitude, longitude)

        context_message = f"""[Location Context]
- Location: {location_name}
- Point A (Start): ({latitude:.6f}, {longitude:.6f})"""

        if latitude_b is not None and longitude_b is not None:
            # Reverse geocode Point B
            location_name_b = f"Location ({latitude_b:.4f}, {longitude_b:.4f})"
            try:
                from semantic_translator import geocoding_plugin
                rg_b = await geocoding_plugin.azure_maps_reverse_geocode(latitude_b, longitude_b)
                data_b = json.loads(rg_b)
                if not data_b.get("error"):
                    name_b = data_b.get("name", "")
                    region_b = data_b.get("region", "")
                    country_b = data_b.get("country", "")
                    parts_b = [p for p in [name_b, region_b, country_b] if p and p != name_b]
                    location_name_b = f"{name_b}, {', '.join(parts_b)}" if name_b and parts_b else name_b or location_name_b
            except Exception:
                pass

            context_message += f"""
- Point B (Destination): {location_name_b} ({latitude_b:.6f}, {longitude_b:.6f})
- Analysis mode: Two-point traverse (A -> B)"""
        else:
            context_message += f"""
- Analysis radius: 5 miles"""

        context_message += f"""
- Session messages: {session.message_count}"""

        if visual_analysis:
            context_message += f"\n\n[Visual Analysis of Current Map View]\n{visual_analysis}"

        query = user_context or (
            f"Analyze terrain traversability from Point A to Point B. Provide a structured situation report with hazards, route assessment, and recommendations."
            if latitude_b is not None and longitude_b is not None
            else "Analyze terrain mobility at this location in all four directions. Provide a structured situation report with hazards, assessments, and recommendations."
        )
        context_message += f"\n\n[User Question]\n{query}"

        # Force tool usage — explicit instruction to call satellite analysis tools
        if latitude_b is not None and longitude_b is not None:
            context_message += (
                f"\n\n[INSTRUCTIONS]\n"
                f"You MUST call your analysis tools at the coordinates above. "
                f"Call analyze_directional_mobility at Point A ({latitude}, {longitude}) AND at Point B ({latitude_b}, {longitude_b}). "
                f"Also call detect_water_bodies, analyze_slope_for_mobility, and detect_active_fires at both points. "
                f"Base your response ONLY on the tool results. Do NOT use general knowledge about this location."
            )
        else:
            context_message += (
                f"\n\n[INSTRUCTIONS]\n"
                f"You MUST call analyze_directional_mobility at ({latitude}, {longitude}). "
                f"Base your response ONLY on the tool results. Do NOT use general knowledge."
            )

        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Re-create thread if we had to re-initialize (stale session)
                if attempt > 0:
                    session = await self._get_or_create_session(f"{session_id}_retry{attempt}", latitude, longitude)

                await self._agents_client.messages.create(
                    thread_id=session.thread_id,
                    role="user",
                    content=context_message,
                )

                run = await self._agents_client.runs.create_and_process(
                    thread_id=session.thread_id,
                    agent_id=self._agent_id,
                )

                if run.status == "failed":
                    logger.error(f"Mobility agent run failed: {run.last_error}")
                    return {
                        "agent": "geoint_mobility",
                        "response": f"Mobility analysis error: {run.last_error}",
                        "error": str(run.last_error),
                        "session_id": session_id
                    }

                from azure.ai.agents.models import ListSortOrder
                messages_iterable = self._agents_client.messages.list(
                    thread_id=session.thread_id,
                    order=ListSortOrder.DESCENDING,
                )

                response_content = ""
                tool_calls = []

                async for msg in messages_iterable:
                    if msg.run_id == run.id and msg.role == "assistant":
                        if msg.text_messages:
                            response_content = msg.text_messages[-1].text.value
                        break

                try:
                    run_steps_iterable = self._agents_client.run_steps.list(
                        thread_id=session.thread_id, run_id=run.id)
                    async for step in run_steps_iterable:
                        if hasattr(step, 'step_details') and hasattr(step.step_details, 'tool_calls'):
                            for tc in step.step_details.tool_calls:
                                if hasattr(tc, 'function'):
                                    tool_calls.append({"tool": tc.function.name})
                                    logger.info(f"Mobility agent called tool: {tc.function.name}")
                except Exception as e:
                    logger.warning(f"Failed to retrieve run steps: {e}")

                if not tool_calls:
                    logger.warning("Mobility agent responded WITHOUT calling any tools — response may be generic knowledge")

                session.message_count += 2
                session.last_activity = datetime.utcnow()

                return {
                    "agent": "geoint_mobility",
                    "response": response_content,
                    "summary": response_content,
                    "tool_calls": tool_calls,
                    "session_id": session_id,
                    "location": {"latitude": latitude, "longitude": longitude},
                    "destination": {"latitude": latitude_b, "longitude": longitude_b} if latitude_b is not None and longitude_b is not None else None,
                    "radius_miles": 5,
                    "timestamp": datetime.utcnow().isoformat(),
                    "data_sources": [tc["tool"] for tc in tool_calls]
                }

            except Exception as e:
                error_str = str(e)
                is_404 = "404" in error_str or "Resource not found" in error_str
                if is_404 and attempt < max_retries - 1:
                    logger.warning(f"Mobility agent got 404 (stale agent?), re-initializing... (attempt {attempt + 1})")
                    self._initialized = False
                    self._agent_id = None
                    self._agents_client = None
                    self.sessions.clear()
                    try:
                        await self._ensure_initialized()
                        continue  # Retry with fresh agent
                    except Exception as reinit_err:
                        logger.error(f"Mobility agent re-initialization failed: {reinit_err}")
                        return {
                            "agent": "geoint_mobility",
                            "response": f"Error: Agent service unavailable - {str(reinit_err)}",
                            "error": str(reinit_err),
                            "session_id": session_id
                        }

                logger.error(f"Mobility agent error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return {
                    "agent": "geoint_mobility",
                    "response": f"Mobility analysis error: {str(e)}",
                    "error": str(e),
                    "session_id": session_id
                }

    async def chat(self, session_id: str, user_message: str, latitude: float, longitude: float,
                   screenshot_base64: Optional[str] = None) -> Dict[str, Any]:
        """Multi-turn chat interface (same pattern as terrain agent)."""
        return await self.analyze_mobility(
            latitude=latitude, longitude=longitude,
            user_context=user_message,
            include_vision_analysis=bool(screenshot_base64),
            screenshot_base64=screenshot_base64,
            session_id=session_id
        )

    async def cleanup(self):
        if self._agents_client and self._agent_id:
            try:
                await self._agents_client.delete_agent(self._agent_id)
            except Exception:
                pass


# Singleton
_mobility_agent: Optional[GeointMobilityAgent] = None


def get_mobility_agent() -> GeointMobilityAgent:
    global _mobility_agent
    if _mobility_agent is None:
        _mobility_agent = GeointMobilityAgent()
    return _mobility_agent
