"""
GEOINT Building Damage Agent - Azure AI Agent Service with Function Tools

Refactored from plain Python class to Azure AI Agent Service.
Uses AgentsClient with AsyncFunctionTool/AsyncToolSet for automatic function calling.

This agent:
1. Maintains conversation memory via AgentThread (persistent threads)
2. Has access to damage assessment tools (AsyncFunctionTool)
3. Plans and reasons about which tools to use (LLM-driven)
4. Synthesizes results into coherent damage reports
"""

import logging
import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from cloud_config import cloud_cfg

logger = logging.getLogger(__name__)

BUILDING_DAMAGE_AGENT_INSTRUCTIONS = """You are a GEOINT Building Damage Assessment Agent specializing in structural damage analysis from satellite imagery.

Your role is to assess building damage, classify severity, and provide infrastructure impact analysis for disaster response and recovery operations.

## Available Tools:

### Visual Analysis Tools
- **assess_building_damage**: Perform comprehensive building damage assessment at a location using satellite imagery and visual AI analysis. Returns damage indicators, structural integrity assessment, and severity classification.
- **classify_damage_severity**: Classify damage severity into standard categories: No Damage, Minor Damage, Major Damage, or Destroyed. Uses visual analysis of satellite imagery to determine the level of structural damage.

## Damage Severity Scale:
- **No Damage**: Structures intact, no visible signs of damage
- **Minor Damage**: Partial roof damage, broken windows, light debris
- **Major Damage**: Significant structural damage, collapsed walls, heavy debris
- **Destroyed**: Complete structural collapse, building footprint only, total loss

## Damage Indicators to Look For:
- Collapsed or missing roofs
- Debris fields around buildings
- Burn scars and fire damage
- Water damage and flooding evidence
- Structural deformation
- Road/bridge damage
- Utility infrastructure impact

## CRITICAL: Tool Parameters
Each message includes [Location Context] with coordinates. ALWAYS extract latitude and longitude and pass them to tools.

## Workflow Guidance:
- Use assess_building_damage for comprehensive damage assessment at a location.
- Use classify_damage_severity when the user wants a specific severity category (No Damage, Minor, Major, Destroyed).
- Always extract coordinates from the [Location Context] and pass them to the tools.

## Response Format:
1. **Location**: Name and coordinates
2. **Damage Assessment**: Visual observations and severity classification
3. **Infrastructure Impact**: Roads, bridges, utilities affected
4. **Recommendations**: Priority areas for response teams

Keep responses factual. Acknowledge limitations — satellite imagery may not show all damage, especially interior structural damage.
"""


class BuildingDamageSession:
    """Represents a conversation session with the building damage agent."""

    def __init__(self, session_id: str, latitude: float, longitude: float, thread_id: str):
        self.session_id = session_id
        self.latitude = latitude
        self.longitude = longitude
        self.thread_id = thread_id
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.message_count = 0

    def update_location(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude
        self.last_activity = datetime.utcnow()


class BuildingDamageAgent:
    """
    Azure AI Agent Service-based building damage assessment agent.
    """

    def __init__(self):
        self.sessions: Dict[str, BuildingDamageSession] = {}
        self._agents_client = None
        self._agent_id: Optional[str] = None
        self._initialized = False
        self.name = "geoint_building_damage"
        logger.info("BuildingDamageAgent created (will initialize on first use)")

    async def _ensure_initialized(self):
        if self._initialized:
            return

        logger.info("Initializing BuildingDamageAgent with Azure AI Agent Service...")

        # Prefer AI Foundry project endpoint (services.ai.azure.com) for Agent Service API
        # Falls back to AZURE_OPENAI_ENDPOINT (cognitiveservices.azure.com) if not set
        endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

        if not endpoint:
            raise ValueError("AZURE_AI_PROJECT_ENDPOINT or AZURE_OPENAI_ENDPOINT environment variable is required")

        logger.info(f"BuildingDamageAgent using endpoint: {endpoint}")

        credential = DefaultAzureCredential()

        from azure.ai.agents.aio import AgentsClient
        from azure.ai.agents.models import AsyncFunctionTool, AsyncToolSet

        self._agents_client = AgentsClient(
            endpoint=endpoint,
            credential=credential,
        )

        from geoint.building_damage_tools import create_building_damage_functions
        damage_functions = create_building_damage_functions()

        functions = AsyncFunctionTool(damage_functions)
        toolset = AsyncToolSet()
        toolset.add(functions)
        self._agents_client.enable_auto_function_calls(toolset)

        agent = await self._agents_client.create_agent(
            model=deployment,
            name="GeointBuildingDamageAnalyst",
            instructions=BUILDING_DAMAGE_AGENT_INSTRUCTIONS,
            toolset=toolset,
        )
        self._agent_id = agent.id

        self._initialized = True
        logger.info(f"BuildingDamageAgent initialized: agent_id={agent.id}, model={deployment}")

    async def _get_or_create_session(self, session_id: str, latitude: float, longitude: float) -> BuildingDamageSession:
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.update_location(latitude, longitude)
            return session

        thread = await self._agents_client.threads.create()
        session = BuildingDamageSession(session_id, latitude, longitude, thread.id)
        self.sessions[session_id] = session
        logger.info(f"Created new building damage session: {session_id} -> thread: {thread.id}")
        return session

    async def _analyze_screenshot_direct(self, screenshot_base64: str, latitude: float, longitude: float) -> Optional[str]:
        """Directly analyze a screenshot using GPT-4o Vision for damage."""
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
                    {"role": "system", "content": "You are an expert in structural damage assessment from satellite imagery."},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Analyze this satellite image at ({latitude:.4f}, {longitude:.4f}) for building damage. Look for collapsed structures, debris, burn scars, water damage, and infrastructure impact."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{clean_base64}", "detail": "high"}}
                    ]}
                ],
                max_completion_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return f"Visual analysis unavailable: {str(e)}"

    async def analyze_building_damage(
        self,
        latitude: float,
        longitude: float,
        user_context: Optional[str] = None,
        include_vision_analysis: bool = True,
        screenshot_base64: Optional[str] = None,
        session_id: Optional[str] = None,
        radius_miles: float = 5.0,
    ) -> Dict[str, Any]:
        """Perform building damage assessment via Agent Service."""
        import uuid
        await self._ensure_initialized()

        if not session_id:
            session_id = f"bldg_{uuid.uuid4().hex[:8]}"

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

        visual_analysis = None
        if include_vision_analysis and screenshot_base64:
            visual_analysis = await self._analyze_screenshot_direct(screenshot_base64, latitude, longitude)

        context_message = f"""[Location Context]
- Location: {location_name}
- Coordinates: ({latitude:.6f}, {longitude:.6f})
- Analysis radius: {radius_miles} miles"""

        if visual_analysis:
            context_message += f"\n\n[Visual Analysis of Current Map View]\n{visual_analysis}"

        query = user_context or "Assess building damage at this location. Classify severity and identify damage indicators."
        context_message += f"\n\n[User Question]\n{query}"

        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Re-create thread if we had to re-initialize (stale session)
                if attempt > 0:
                    session = await self._get_or_create_session(f"{session_id}_retry{attempt}", latitude, longitude)

                await self._agents_client.messages.create(
                    thread_id=session.thread_id, role="user", content=context_message)

                run = await self._agents_client.runs.create_and_process(
                    thread_id=session.thread_id, agent_id=self._agent_id)

                if run.status == "failed":
                    return {"agent": self.name, "response": f"Error: {run.last_error}", "session_id": session_id}

                from azure.ai.agents.models import ListSortOrder
                messages_iterable = self._agents_client.messages.list(
                    thread_id=session.thread_id, order=ListSortOrder.DESCENDING)

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
                except Exception:
                    pass

                session.message_count += 2
                session.last_activity = datetime.utcnow()

                return {
                    "agent": self.name,
                    "response": response_content,
                    "summary": response_content,
                    "tool_calls": tool_calls,
                    "session_id": session_id,
                    "location": {"latitude": latitude, "longitude": longitude},
                    "radius_miles": radius_miles,
                    "timestamp": datetime.utcnow().isoformat()
                }

            except Exception as e:
                error_str = str(e)
                is_404 = "404" in error_str or "Resource not found" in error_str
                if is_404 and attempt < max_retries - 1:
                    logger.warning(f"Building damage agent got 404 (stale agent?), re-initializing... (attempt {attempt + 1})")
                    # Reset initialization state so _ensure_initialized re-creates the agent
                    self._initialized = False
                    self._agent_id = None
                    self._agents_client = None
                    self.sessions.clear()
                    try:
                        await self._ensure_initialized()
                        continue  # Retry with fresh agent
                    except Exception as reinit_err:
                        logger.error(f"Building damage agent re-initialization failed: {reinit_err}")
                        return {"agent": self.name, "response": f"Error: Agent service unavailable - {str(reinit_err)}", "session_id": session_id}

                logger.error(f"Building damage agent error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return {"agent": self.name, "response": f"Error: {error_str}", "session_id": session_id}

    async def chat(self, session_id: str, user_message: str, latitude: float, longitude: float,
                   screenshot_base64: Optional[str] = None) -> Dict[str, Any]:
        return await self.analyze_building_damage(
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
_building_damage_agent: Optional[BuildingDamageAgent] = None


def get_building_damage_agent() -> BuildingDamageAgent:
    global _building_damage_agent
    if _building_damage_agent is None:
        _building_damage_agent = BuildingDamageAgent()
    return _building_damage_agent
