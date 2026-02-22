"""
Terrain Agent - Azure AI Agent Service with Function Tools

Refactored from Semantic Kernel ChatCompletionAgent to Azure AI Agent Service.
Uses AgentsClient with FunctionTool/ToolSet for automatic function calling.

This agent:
1. Maintains conversation memory via AgentThread (persistent threads)
2. Has access to terrain analysis tools (FunctionTool)
3. Plans and reasons about which tools to use (LLM-driven)
4. Synthesizes results into coherent answers
"""

import logging
import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from cloud_config import cloud_cfg

logger = logging.getLogger(__name__)

# Agent system prompt (unchanged - well-crafted for terrain analysis)
TERRAIN_AGENT_INSTRUCTIONS = """You are a Geospatial Intelligence (GEOINT) Terrain Analysis Agent specializing in site permitting and environmental suitability analysis.

Your role is to analyze terrain and answer questions about geographic locations using DEM (Digital Elevation Model) data, water occurrence data, land cover data, and visual analysis from satellite imagery.

## Available Tools:

### Terrain Analysis (DEM-based):
- **get_elevation_analysis**: Get elevation data (min, max, mean in meters) and terrain classification (flat/hilly/mountainous)
- **get_slope_analysis**: Analyze terrain steepness, traversability, and percentage of flat/moderate/steep areas
- **get_aspect_analysis**: Determine slope direction (N, S, E, W, etc.) and sun exposure
- **find_flat_areas**: Locate flat areas suitable for landing zones, construction, or camps

### Environmental & Permitting Analysis:
- **analyze_flood_risk**: Check historical flood occurrence (0-100%) using JRC Global Surface Water. Returns flood risk level (LOW/MODERATE/HIGH) and permitting recommendation.
- **analyze_water_proximity**: Calculate distance to nearest water body for setback requirements (e.g., 500m from wetlands). Returns whether setback is satisfied.
- **analyze_environmental_sensitivity**: Identify wetlands, forests, mangroves, and protected habitats using ESA WorldCover. Returns environmental constraints and permitting status.

## Permitting Use Case Workflows:

**Mining Site Permit**: Call get_slope_analysis -> analyze_flood_risk -> analyze_water_proximity -> analyze_environmental_sensitivity
**Nuclear Facility Siting**: Call get_elevation_analysis -> analyze_flood_risk -> get_slope_analysis
**Construction Permit**: Call get_slope_analysis -> find_flat_areas -> analyze_flood_risk
**Solar/Wind Farm**: Call get_aspect_analysis -> get_slope_analysis -> find_flat_areas

## Visual Analysis (Automatic)
A [Visual Analysis of Current Map View] section is automatically included in your context when a map screenshot is available.

## CRITICAL: Tool Parameters
Each message includes [Location Context] with:
- Coordinates: (latitude, longitude) - USE THESE VALUES when calling any terrain tool
- Analysis radius: X km - USE THIS as the radius_km parameter

**ALWAYS extract the latitude, longitude, and radius from the context and pass them to tools.**

## Guidelines:
1. **For permitting questions** - Call ALL relevant tools (slope, flood, water proximity, environmental)
2. **Always call DEM tools** for elevation, slope, and aspect - these provide accurate quantitative data
3. **Use Visual Analysis** for vegetation, water bodies, urban areas, roads, and land use patterns
4. **Combine both sources** - DEM data for terrain metrics + visual analysis for land cover
5. **Be specific** - Include actual numbers (elevations in meters, slope percentages, etc.)
6. **Summarize permitting status** - End with clear SUITABLE/CONDITIONAL/NOT SUITABLE recommendation

## Response Format:
1. **Terrain Overview**: ALWAYS start with the location name followed by terrain character summary
2. **Elevation & Topography**: min/max/mean elevation, terrain type from tools
3. **Slope & Traversability**: Steepness data, percentage flat/steep, traversability
4. **Environmental Assessment** (for permitting): Flood risk, water proximity, wetlands/forests
5. **Permitting Recommendation**: Clear SUITABLE / CONDITIONAL / NOT SUITABLE with reasons

**CRITICAL: Always use the Location name from [Location Context]. Never respond with just coordinates.**

**Keep responses factual and concise. Do NOT include:**
- "Actionable Insights" or recommendation sections beyond permitting status
- Summary paragraphs at the end restating what was said
- Generic suggestions unrelated to the user's question
"""


class TerrainAgentSession:
    """Represents a conversation session with the terrain agent.
    
    Maps a session_id to an Agent Service thread_id for persistent conversation.
    """
    
    def __init__(self, session_id: str, latitude: float, longitude: float, thread_id: str):
        self.session_id = session_id
        self.latitude = latitude
        self.longitude = longitude
        self.thread_id = thread_id  # Agent Service thread ID
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.analysis_cache: Dict[str, Any] = {}
        self.message_count = 0
        
    def update_location(self, latitude: float, longitude: float):
        """Update the session's focus location."""
        self.latitude = latitude
        self.longitude = longitude
        self.last_activity = datetime.utcnow()


class TerrainAgent:
    """
    Azure AI Agent Service-based terrain analysis agent with:
    - Persistent threads for multi-turn conversation
    - FunctionTool calling for raster analysis
    - Automatic function execution via ToolSet
    """
    
    def __init__(self):
        """Initialize the terrain agent."""
        self.sessions: Dict[str, TerrainAgentSession] = {}
        self._agents_client = None
        self._agent_id: Optional[str] = None
        self._initialized = False
        
        logger.info("TerrainAgent created (will initialize on first use)")
    
    async def _ensure_initialized(self):
        """Lazy initialization of Agent Service client and agent."""
        if self._initialized:
            return
            
        logger.info("Initializing TerrainAgent with Azure AI Agent Service...")
        
        # Prefer AI Foundry project endpoint (services.ai.azure.com) for Agent Service API
        # Falls back to AZURE_OPENAI_ENDPOINT (cognitiveservices.azure.com) if not set
        endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        
        if not endpoint:
            raise ValueError("AZURE_AI_PROJECT_ENDPOINT or AZURE_OPENAI_ENDPOINT environment variable is required")
        
        logger.info(f"TerrainAgent using endpoint: {endpoint}")
        
        # Use Managed Identity
        credential = DefaultAzureCredential()
        
        # Import Agent Service SDK
        from azure.ai.agents.aio import AgentsClient
        from azure.ai.agents.models import AsyncFunctionTool, AsyncToolSet
        
        # Create async AgentsClient
        self._agents_client = AgentsClient(
            endpoint=endpoint,
            credential=credential,
        )
        
        # Build terrain tools as standalone functions for FunctionTool
        from geoint.terrain_tools import create_terrain_functions
        terrain_functions = create_terrain_functions()
        
        # Create AsyncFunctionTool and AsyncToolSet with auto function calling
        functions = AsyncFunctionTool(terrain_functions)
        toolset = AsyncToolSet()
        toolset.add(functions)
        self._agents_client.enable_auto_function_calls(toolset)
        
        # Create the agent
        agent = await self._agents_client.create_agent(
            model=deployment,
            name="TerrainAnalyst",
            instructions=TERRAIN_AGENT_INSTRUCTIONS,
            toolset=toolset,
        )
        self._agent_id = agent.id
        
        self._initialized = True
        logger.info(f"TerrainAgent initialized: agent_id={agent.id}, model={deployment}")
    
    async def _get_or_create_session(
        self, 
        session_id: str,
        latitude: float,
        longitude: float
    ) -> TerrainAgentSession:
        """Get existing session or create a new one with a new thread."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.update_location(latitude, longitude)
            return session
        
        # Create a new Agent Service thread
        thread = await self._agents_client.threads.create()
        
        session = TerrainAgentSession(session_id, latitude, longitude, thread.id)
        self.sessions[session_id] = session
        logger.info(f"Created new session: {session_id} -> thread: {thread.id}")
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
            logger.info(f"Cleaned up expired session: {sid}")
    
    async def _analyze_screenshot_direct(
        self,
        screenshot_base64: str,
        latitude: float,
        longitude: float
    ) -> Optional[str]:
        """
        Directly analyze a screenshot using GPT-4o Vision.
        
        This runs BEFORE the agent invoke to ensure visual analysis
        is always available in the agent's context.
        Uses the standard OpenAI client (not Agent Service) for vision.
        """
        try:
            from openai import AsyncAzureOpenAI
            
            logger.info(f"Running direct vision analysis at ({latitude:.4f}, {longitude:.4f})")
            
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
            
            # Clean base64 if needed
            clean_base64 = screenshot_base64
            if screenshot_base64.startswith('data:image'):
                clean_base64 = screenshot_base64.split(',', 1)[1]
            
            vision_prompt = f"""Analyze this satellite/map image for terrain and geospatial intelligence.

Location: Approximately ({latitude:.4f}, {longitude:.4f})

Provide a comprehensive analysis covering:
1. **Land Use & Urban Development**: Urban vs rural, settlements, roads
2. **Vegetation & Land Cover**: Forest, grassland, agricultural areas
3. **Water Features**: Rivers, lakes, wetlands, flood-prone areas
4. **Terrain Features**: Hills, valleys, flat areas
5. **Notable Observations**: Distinctive landmarks or features

Be specific and quantitative where possible."""
            
            response = await client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert geospatial analyst specializing in terrain analysis and satellite imagery interpretation."
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
            logger.info(f"Vision analysis complete: {len(analysis)} chars")
            return analysis
            
        except Exception as e:
            logger.error(f"Direct vision analysis failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
        
        Same interface as the previous SK-based agent for drop-in compatibility.
        """
        await self._ensure_initialized()
        
        # Get or create session (creates Agent Service thread)
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
            session.analysis_cache["visual_analysis"] = visual_analysis
            logger.info(f"Pre-analyzed screenshot: {len(visual_analysis) if visual_analysis else 0} chars")
        
        # Build context-enriched message
        context_message = f"""[Location Context]
- Location: {location_name}
- Coordinates: ({latitude:.6f}, {longitude:.6f})
- Analysis radius: {radius_km} km
- Session messages: {session.message_count}"""
        
        if visual_analysis:
            context_message += f"""

[Visual Analysis of Current Map View]
{visual_analysis}"""
        
        context_message += f"""

[User Question]
{user_message}"""
        
        logger.info(f"Session {session_id}: Processing '{user_message[:50]}...'")
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Re-create thread if we had to re-initialize (stale session)
                if attempt > 0:
                    session = await self._get_or_create_session(f"{session_id}_retry{attempt}", latitude, longitude)

                # Add message to the Agent Service thread
                await self._agents_client.messages.create(
                    thread_id=session.thread_id,
                    role="user",
                    content=context_message,
                )
                
                # Create and process the run (auto-executes function tools via ToolSet)
                run = await self._agents_client.runs.create_and_process(
                    thread_id=session.thread_id,
                    agent_id=self._agent_id,
                )
                
                if run.status == "failed":
                    logger.error(f"Agent run failed: {run.last_error}")
                    return {
                        "response": f"I encountered an error analyzing this location: {run.last_error}",
                        "error": str(run.last_error),
                        "session_id": session_id
                    }
                
                # Get messages from the thread (newest first)
                from azure.ai.agents.models import ListSortOrder
                messages_iterable = self._agents_client.messages.list(
                    thread_id=session.thread_id,
                    order=ListSortOrder.DESCENDING,
                )
                
                # Extract the assistant's latest response
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
                                    logger.info(f"Tool called: {tool_name}")
                except Exception as e:
                    logger.debug(f"Could not extract run steps: {e}")
                
                session.message_count += 2  # user + assistant
                session.last_activity = datetime.utcnow()
                
                logger.info(f"Agent response ({len(response_content)} chars, {len(tool_calls)} tool calls)")
                
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
                    logger.warning(f"Terrain agent got 404 (stale agent?), re-initializing... (attempt {attempt + 1})")
                    self._initialized = False
                    self._agent_id = None
                    self._agents_client = None
                    self.sessions.clear()
                    try:
                        await self._ensure_initialized()
                        continue  # Retry with fresh agent
                    except Exception as reinit_err:
                        logger.error(f"Terrain agent re-initialization failed: {reinit_err}")
                        return {
                            "response": f"Error: Agent service unavailable - {str(reinit_err)}",
                            "error": str(reinit_err),
                            "session_id": session_id
                        }

                logger.error(f"Agent error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
                return {
                    "response": f"I encountered an error analyzing this location: {str(e)}",
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
            logger.info(f"Cleared session: {session_id}")
            return True
        return False
    
    async def cleanup(self):
        """Cleanup agent resources on shutdown."""
        if self._agents_client and self._agent_id:
            try:
                await self._agents_client.delete_agent(self._agent_id)
                logger.info(f"Deleted agent: {self._agent_id}")
            except Exception as e:
                logger.debug(f"Agent cleanup: {e}")


# Singleton instance
_terrain_agent: Optional[TerrainAgent] = None


def get_terrain_agent() -> TerrainAgent:
    """Get the singleton TerrainAgent instance."""
    global _terrain_agent
    if _terrain_agent is None:
        _terrain_agent = TerrainAgent()
    return _terrain_agent
