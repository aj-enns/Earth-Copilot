"""
Extreme Weather Agent - Azure AI Agent Service with Climate Tools

Standalone geoint module for NASA NEX-GDDP-CMIP6 climate projections.
Returns chat-based point values (no map tiles — data is NetCDF, not COG).

This agent:
1. Maintains conversation memory via AgentThread (persistent threads)
2. Has access to extreme weather / climate tools (FunctionTool)
3. Plans and reasons about which climate variables to query (LLM-driven)
4. Synthesizes climate projection data into coherent answers
"""

import logging
import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from cloud_config import cloud_cfg

logger = logging.getLogger(__name__)

EXTREME_WEATHER_AGENT_INSTRUCTIONS = """You are a Climate & Extreme Weather Analysis Agent specializing in future climate projections using NASA NEX-GDDP-CMIP6 data.

Your role is to analyze projected climate conditions for any location on Earth using downscaled CMIP6 global climate model outputs.

## Data Source
All data comes from **NASA NEX-GDDP-CMIP6** on Microsoft Planetary Computer:
- Resolution: 0.25° × 0.25° global grid (~25 km)
- Time range: 2015–2100
- These are climate **projections**, not real-time observations
- Data is sampled at a single point (not rendered as map tiles)

## SSP Scenarios
- **SSP2-4.5** ("Middle of the Road"): Moderate emissions, sustainable development by mid-century
- **SSP5-8.5** ("Fossil-fuel Development"): Worst-case, high emissions, business-as-usual

## Available Tools

### Temperature
- **get_temperature_projection**: Get max, min, and mean daily temperature (°C) for a location and scenario/year.

### Precipitation
- **get_precipitation_projection**: Get daily precipitation (mm/day). Multiple models sampled for ensemble range.

### Wind
- **get_wind_projection**: Get near-surface wind speed (m/s) with Beaufort-scale classification.

### Humidity
- **get_humidity_projection**: Get relative humidity (%) and specific humidity (g/kg).

### Radiation
- **get_radiation_projection**: Get shortwave (solar) and longwave radiation (W/m²). Useful for solar energy potential.

### Multi-Variable
- **get_climate_overview**: Sample all key variables at once for a full climate snapshot.
- **compare_climate_scenarios**: Compare SSP2-4.5 vs SSP5-8.5 to show range of uncertainty.

## CRITICAL: Tool Parameters
Each message includes [Location Context] with:
- Coordinates: (latitude, longitude)  
- Default scenario and year if user doesn't specify

**ALWAYS extract the latitude and longitude from the context and pass them to tools.**
If the user specifies a year or scenario, pass those too. Otherwise use the defaults from [Location Context] (typically ssp585, 2030).

## Response Guidelines
1. **Start with the location name** from [Location Context] — never just coordinates.
2. **Always call tools** — don't guess climate values. Call the appropriate tool(s) for what the user asks.
3. **Explain the numbers** — put temperature/precipitation/wind in human-understandable context.
4. **Compare when useful** — if user doesn't specify scenario, consider showing both SSP2-4.5 and SSP5-8.5.
5. **Note limitations** — these are projections from a single model run, not forecasts.
6. **Mention data source** — "NASA NEX-GDDP-CMIP6 (0.25° resolution)" at least once.

## Visual Context
If a [Visual Analysis of Current Map View] section is provided, use it to enrich your response:
- Reference visible landmarks, terrain features, land use patterns
- Relate climate projections to what is visible (e.g., agricultural land -> crop impact, coastal areas -> sea level risk)
- Mention relevant geographic features that affect local climate (mountains, water bodies, urban heat islands)

## Response Format
1. **Location & Context**: Location name + what the user asked
2. **Climate Projections**: The actual data with units
3. **Interpretation**: What the numbers mean in practical terms
4. **Scenario Context** (if applicable): How the scenario affects the outlook
5. **Caveats**: Brief note that these are model projections, not forecasts

**Keep responses factual and concise. Do NOT include generic climate change disclaimers or policy recommendations.**
"""


class ExtremeWeatherAgentSession:
    """Represents a conversation session with the extreme weather agent."""
    
    def __init__(self, session_id: str, latitude: float, longitude: float, thread_id: str):
        self.session_id = session_id
        self.latitude = latitude
        self.longitude = longitude
        self.thread_id = thread_id
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.message_count = 0
        
    def update_location(self, latitude: float, longitude: float):
        """Update the session's focus location."""
        self.latitude = latitude
        self.longitude = longitude
        self.last_activity = datetime.utcnow()


class ExtremeWeatherAgent:
    """
    Azure AI Agent Service-based climate projection agent with:
    - Persistent threads for multi-turn conversation
    - FunctionTool calling for NetCDF climate data sampling
    - Automatic function execution via ToolSet
    """
    
    def __init__(self):
        """Initialize the extreme weather agent."""
        self.sessions: Dict[str, ExtremeWeatherAgentSession] = {}
        self._agents_client = None
        self._agent_id: Optional[str] = None
        self._initialized = False
        
        logger.info("ExtremeWeatherAgent created (will initialize on first use)")
    
    async def _ensure_initialized(self):
        """Lazy initialization of Agent Service client and agent."""
        if self._initialized:
            return
            
        logger.info("Initializing ExtremeWeatherAgent with Azure AI Agent Service...")
        
        # Prefer AI Foundry project endpoint (services.ai.azure.com) for Agent Service API
        # Falls back to AZURE_OPENAI_ENDPOINT (cognitiveservices.azure.com) if not set
        endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        
        if not endpoint:
            raise ValueError("AZURE_AI_PROJECT_ENDPOINT or AZURE_OPENAI_ENDPOINT environment variable is required")
        
        logger.info(f"ExtremeWeatherAgent using endpoint: {endpoint}")
        
        credential = DefaultAzureCredential()
        
        from azure.ai.agents.aio import AgentsClient
        from azure.ai.agents.models import AsyncFunctionTool, AsyncToolSet
        
        self._agents_client = AgentsClient(
            endpoint=endpoint,
            credential=credential,
        )
        
        # Build extreme weather tools
        from geoint.extreme_weather_tools import create_extreme_weather_functions
        climate_functions = create_extreme_weather_functions()
        
        functions = AsyncFunctionTool(climate_functions)
        toolset = AsyncToolSet()
        toolset.add(functions)
        self._agents_client.enable_auto_function_calls(toolset)
        
        agent = await self._agents_client.create_agent(
            model=deployment,
            name="ExtremeWeatherAnalyst",
            instructions=EXTREME_WEATHER_AGENT_INSTRUCTIONS,
            toolset=toolset,
        )
        self._agent_id = agent.id
        
        self._initialized = True
        logger.info(f"ExtremeWeatherAgent initialized: agent_id={agent.id}, model={deployment}")
    
    async def _get_or_create_session(
        self, 
        session_id: str,
        latitude: float,
        longitude: float
    ) -> ExtremeWeatherAgentSession:
        """Get existing session or create a new one with a new thread."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.update_location(latitude, longitude)
            return session
        
        thread = await self._agents_client.threads.create()
        
        session = ExtremeWeatherAgentSession(session_id, latitude, longitude, thread.id)
        self.sessions[session_id] = session
        logger.info(f"Created new extreme weather session: {session_id} -> thread: {thread.id}")
        return session
    
    def cleanup_old_sessions(self, max_age_minutes: int = 60):
        """Remove sessions older than max_age_minutes."""
        now = datetime.utcnow()
        expired = [
            sid for sid, session in self.sessions.items()
            if (now - session.last_activity).total_seconds() > max_age_minutes * 60
        ]
        for sid in expired:
            del self.sessions[sid]
            logger.info(f"Cleaned up expired extreme weather session: {sid}")
    
    async def _analyze_screenshot_direct(
        self,
        screenshot_base64: str,
        latitude: float,
        longitude: float
    ) -> Optional[str]:
        """Pre-analyze a map screenshot using GPT-4o Vision for climate context."""
        try:
            from openai import AsyncAzureOpenAI

            logger.info(f"Running visual analysis for climate context at ({latitude:.4f}, {longitude:.4f})")

            credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(
                credential, cloud_cfg.cognitive_services_scope
            )

            client = AsyncAzureOpenAI(
                azure_ad_token_provider=token_provider,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                timeout=120.0
            )

            clean_base64 = screenshot_base64
            if screenshot_base64.startswith('data:image'):
                clean_base64 = screenshot_base64.split(',', 1)[1]

            vision_prompt = f"""Analyze this satellite/map image for climate-relevant geographic context.

Location: Approximately ({latitude:.4f}, {longitude:.4f})

Identify:
1. **Land Use**: Urban, agricultural, forest, desert, coastal areas
2. **Water Features**: Rivers, lakes, coastline, flood-prone areas
3. **Terrain**: Mountains, valleys, plains that affect local climate
4. **Notable Features**: Airports, infrastructure, landmarks visible
5. **Climate Relevance**: Features that would be impacted by temperature, precipitation, or extreme weather changes

Be specific and concise."""

            response = await client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert geospatial analyst identifying climate-relevant geographic features."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": vision_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{clean_base64}", "detail": "high"}
                            }
                        ]
                    }
                ],
                max_completion_tokens=1500
            )

            analysis = response.choices[0].message.content
            logger.info(f"Climate vision analysis complete: {len(analysis)} chars")
            return analysis

        except Exception as e:
            logger.error(f"Climate vision analysis failed: {e}")
            return f"Visual analysis unavailable: {str(e)}"

    async def chat(
        self,
        session_id: str,
        user_message: str,
        latitude: float,
        longitude: float,
        screenshot_base64: Optional[str] = None,
        radius_km: float = 5.0
    ) -> Dict[str, Any]:
        """
        Process a user message and return agent response.
        
        Same interface pattern as TerrainAgent for drop-in compatibility.
        """
        await self._ensure_initialized()
        
        session = await self._get_or_create_session(session_id, latitude, longitude)
        
        # ====================================================================
        # REVERSE GEOCODE TO GET LOCATION NAME
        # ====================================================================
        location_name = None
        try:
            from semantic_translator import geocoding_plugin
            reverse_geocode_result = await geocoding_plugin.azure_maps_reverse_geocode(latitude, longitude)
            geocode_data = json.loads(reverse_geocode_result)
            if not geocode_data.get("error"):
                name = geocode_data.get("name", "")
                region = geocode_data.get("region", "")
                country = geocode_data.get("country", "")
                parts = [p for p in [name, region, country] if p and p != name]
                if name:
                    location_name = f"{name}, {', '.join(parts)}" if parts else name
                else:
                    location_name = geocode_data.get("freeform", f"Location ({latitude:.4f}, {longitude:.4f})")
                logger.info(f"Resolved location: ({latitude}, {longitude}) -> {location_name}")
            else:
                location_name = f"Location ({latitude:.4f}, {longitude:.4f})"
        except Exception as e:
            location_name = f"Location ({latitude:.4f}, {longitude:.4f})"
            logger.warning(f"Reverse geocode exception: {e}")
        
        # ====================================================================
        # PRE-ANALYZE SCREENSHOT WITH GPT-4o VISION
        # ====================================================================
        visual_analysis = None
        if screenshot_base64:
            visual_analysis = await self._analyze_screenshot_direct(screenshot_base64, latitude, longitude)
            logger.info(f"Pre-analyzed screenshot for climate context: {len(visual_analysis) if visual_analysis else 0} chars")

        # Build context-enriched message
        context_message = f"""[Location Context]
- Location: {location_name}
- Coordinates: ({latitude:.6f}, {longitude:.6f})
- Default scenario: SSP5-8.5 (worst-case) — user can override
- Default year: 2030 — user can override
- Session messages: {session.message_count}"""

        if visual_analysis:
            context_message += f"""\n\n[Visual Analysis of Current Map View]\n{visual_analysis}"""

        context_message += f"""\n\n[User Question]\n{user_message}"""
        
        logger.info(f"Extreme Weather Session {session_id}: Processing '{user_message[:50]}...'")
        
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
                    logger.error(f"Extreme weather agent run failed: {run.last_error}")
                    return {
                        "response": f"I encountered an error analyzing climate projections: {run.last_error}",
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
                
                # Extract tool call info from run steps
                try:
                    run_steps_iterable = self._agents_client.run_steps.list(
                        thread_id=session.thread_id,
                        run_id=run.id,
                    )
                    async for step in run_steps_iterable:
                        if hasattr(step, 'step_details') and hasattr(step.step_details, 'tool_calls'):
                            for tc in step.step_details.tool_calls:
                                if hasattr(tc, 'function'):
                                    tool_name = tc.function.name
                                    tool_output = getattr(tc.function, 'output', None)
                                    result_parsed = tool_output
                                    if isinstance(tool_output, str) and tool_output.startswith('{'):
                                        try:
                                            result_parsed = json.loads(tool_output)
                                        except Exception:
                                            pass
                                    tool_calls.append({
                                        "tool": tool_name,
                                        "result": result_parsed if isinstance(result_parsed, dict) else str(tool_output)[:500] if tool_output else None
                                    })
                                    logger.info(f"Climate tool called: {tool_name}")
                except Exception as e:
                    logger.debug(f"Could not extract run steps: {e}")
                
                session.message_count += 2
                session.last_activity = datetime.utcnow()
                
                logger.info(f"Extreme weather agent response ({len(response_content)} chars, {len(tool_calls)} tool calls)")
                if not response_content:
                    logger.warning(f"[WARN] Extreme weather agent returned EMPTY response_content! Run status: {run.status}")
                else:
                    logger.info(f"Extreme weather response preview: {response_content[:200]}...")
                
                return {
                    "response": response_content,
                    "tool_calls": tool_calls,
                    "session_id": session_id,
                    "message_count": session.message_count,
                    "location": {"latitude": latitude, "longitude": longitude}
                }
                
            except Exception as e:
                error_str = str(e)
                is_404 = "404" in error_str or "Resource not found" in error_str
                if is_404 and attempt < max_retries - 1:
                    logger.warning(f"Extreme weather agent got 404 (stale agent?), re-initializing... (attempt {attempt + 1})")
                    self._initialized = False
                    self._agent_id = None
                    self._agents_client = None
                    self.sessions.clear()
                    try:
                        await self._ensure_initialized()
                        continue  # Retry with fresh agent
                    except Exception as reinit_err:
                        logger.error(f"Extreme weather agent re-initialization failed: {reinit_err}")
                        return {
                            "response": f"Error: Agent service unavailable - {str(reinit_err)}",
                            "error": str(reinit_err),
                            "session_id": session_id
                        }

                logger.error(f"Extreme weather agent error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
                return {
                    "response": f"I encountered an error analyzing climate projections: {str(e)}",
                    "error": str(e),
                    "session_id": session_id
                }
    
    async def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session from the Agent Service thread."""
        if session_id not in self.sessions:
            return []
        
        session = self.sessions[session_id]
        
        try:
            await self._ensure_initialized()
            from azure.ai.agents.models import ListSortOrder
            messages_iterable = self._agents_client.messages.list(
                thread_id=session.thread_id,
                order=ListSortOrder.ASCENDING,
            )
            
            history = []
            async for msg in messages_iterable:
                content = ""
                if msg.text_messages:
                    content = msg.text_messages[-1].text.value
                history.append({
                    "role": msg.role,
                    "content": content
                })
            return history
            
        except Exception as e:
            logger.error(f"Failed to get session history: {e}")
            return []
    
    async def clear_session(self, session_id: str) -> bool:
        """Clear a session's memory by deleting the thread."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            try:
                await self._ensure_initialized()
                await self._agents_client.threads.delete(session.thread_id)
            except Exception as e:
                logger.debug(f"Thread cleanup: {e}")
            del self.sessions[session_id]
            logger.info(f"Cleared extreme weather session: {session_id}")
            return True
        return False
    
    async def cleanup(self):
        """Cleanup agent resources on shutdown."""
        if self._agents_client and self._agent_id:
            try:
                await self._agents_client.delete_agent(self._agent_id)
                logger.info(f"Deleted extreme weather agent: {self._agent_id}")
            except Exception as e:
                logger.debug(f"Agent cleanup: {e}")


# Singleton instance
_extreme_weather_agent: Optional[ExtremeWeatherAgent] = None


def get_extreme_weather_agent() -> ExtremeWeatherAgent:
    """Get the singleton ExtremeWeatherAgent instance."""
    global _extreme_weather_agent
    if _extreme_weather_agent is None:
        _extreme_weather_agent = ExtremeWeatherAgent()
    return _extreme_weather_agent
