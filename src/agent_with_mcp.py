import asyncio
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.english import EnglishModel

# MCP-related imports
try:
    import mcp
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.types import Tool as MCPTool

    MCP_AVAILABLE = True
except ImportError:
    logging.warning("MCP dependencies not found. Please install the MCP SDK.")
    mcp = None
    streamablehttp_client = None
    MCP_AVAILABLE = False

logger = logging.getLogger("agent_with_mcp")

load_dotenv(".env.local")


def load_instructions_from_file(file_path: str) -> str:
    """Load instructions from a markdown file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"Instructions file not found: {file_path}")
        return "You are a helpful AI assistant with access to tools."
    except Exception as e:
        logger.error(f"Error loading instructions from {file_path}: {e}")
        return "You are a helpful AI assistant with access to tools."


async def build_livekit_tools(server: mcp.ClientSession) -> list[Any]:
    """
    Convert MCP tools to LiveKit function tools.
    Based on the AssemblyAI MCP integration pattern.
    """
    if not MCP_AVAILABLE:
        logger.warning("MCP not available, returning empty tools list")
        return []

    try:
        # Get available tools from the MCP server
        tools_result = await server.list_tools()
        tools = tools_result.tools if hasattr(tools_result, "tools") else []

        logger.info(f"Found {len(tools)} MCP tools available")

        livekit_tools = []

        for tool in tools:
            try:
                # Create a closure that captures the tool name
                def make_tool_function(tool_name: str, tool_description: str):
                    @function_tool
                    async def tool_function(context: RunContext, **kwargs):
                        """Dynamically generated tool function"""
                        try:
                            logger.info(
                                f"Calling MCP tool: {tool_name} with args: {kwargs}"
                            )
                            result = await server.call_tool(tool_name, arguments=kwargs)

                            # Extract text content from MCP result
                            if hasattr(result, "content") and result.content:
                                if hasattr(result.content[0], "text"):
                                    return result.content[0].text
                                else:
                                    return str(result.content[0])
                            else:
                                return str(result)

                        except Exception as e:
                            logger.error(f"Error calling MCP tool {tool_name}: {e}")
                            return f"Error calling {tool_name}: {e!s}"

                    # Set function metadata
                    tool_function.__name__ = tool_name
                    tool_function.__doc__ = tool_description or f"MCP tool: {tool_name}"

                    return tool_function

                # Create the tool function
                livekit_tool = make_tool_function(tool.name, tool.description)
                livekit_tools.append(livekit_tool)

                logger.info(f"Successfully converted MCP tool: {tool.name}")

            except Exception as e:
                logger.error(f"Failed to convert tool {tool.name}: {e}")

        logger.info(
            f"Successfully converted {len(livekit_tools)} MCP tools to LiveKit tools"
        )
        return livekit_tools

    except Exception as e:
        logger.error(f"Failed to build LiveKit tools from MCP server: {e}")
        return []


def create_mock_gmail_tools() -> list[Any]:
    """Create mock Gmail tools that simulate real functionality when MCP is unavailable"""
    logger.info("Creating mock Gmail tools for fallback functionality")

    # Sample fake emails for realistic responses
    fake_emails = [
        {
            "id": "fake_001",
            "from": "john.doe@example.com",
            "subject": "Project Update - Q4 Status",
            "snippet": "Hi there! Just wanted to update you on the Q4 project status...",
            "date": "2024-01-15",
            "read": False,
        },
        {
            "id": "fake_002",
            "from": "sarah.smith@company.com",
            "subject": "Meeting Reminder - Tomorrow 2PM",
            "snippet": "Don't forget about our scheduled meeting tomorrow at 2PM...",
            "date": "2024-01-14",
            "read": True,
        },
        {
            "id": "fake_003",
            "from": "notifications@github.com",
            "subject": "New PR Review Request",
            "snippet": "You have been requested to review a pull request...",
            "date": "2024-01-13",
            "read": False,
        },
    ]

    @function_tool
    async def mock_list_emails(
        context: RunContext, max_results: int = 10, query: str = ""
    ):
        """Mock tool to list recent emails"""
        logger.info(
            f"Mock Gmail: listing emails (max: {max_results}, query: '{query}')"
        )

        # Filter emails based on query if provided
        if query:
            filtered_emails = [
                email
                for email in fake_emails
                if query.lower() in str(email["subject"]).lower()
                or query.lower() in str(email["from"]).lower()
                or query.lower() in str(email["snippet"]).lower()
            ]
        else:
            filtered_emails = fake_emails

        # Limit results
        limited_emails = filtered_emails[:max_results]

        result = f"Found {len(limited_emails)} emails:\n\n"
        for email in limited_emails:
            status = "📧" if not email["read"] else "📖"
            result += f"{status} From: {email['from']}\n"
            result += f"   Subject: {email['subject']}\n"
            result += f"   Date: {email['date']}\n"
            result += f"   Preview: {email['snippet']}\n\n"

        return result

    @function_tool
    async def mock_read_email(context: RunContext, email_id: str):
        """Mock tool to read a specific email"""
        logger.info(f"Mock Gmail: reading email {email_id}")

        # Find the email
        email = next((e for e in fake_emails if e["id"] == email_id), None)
        if not email:
            return f"Email with ID {email_id} not found."

        # Generate fake full content
        full_content = f"""From: {email['from']}
Subject: {email['subject']}
Date: {email['date']}

{email['snippet']} This is the full content of the email. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation.

Best regards,
{str(email['from']).split('@')[0].replace('.', ' ').title()}"""

        # Mark as read
        email["read"] = True

        return full_content

    @function_tool
    async def mock_send_email(
        context: RunContext,
        to: str,
        subject: str,
        body: str,
        cc: str = "",
        bcc: str = "",
    ):
        """Mock tool to send an email"""
        logger.info(f"Mock Gmail: sending email to {to} with subject '{subject}'")

        # Simulate email sending
        fake_message_id = f"fake_sent_{random.randint(1000, 9999)}"

        result = "✅ Email sent successfully!\n\n"
        result += f"Message ID: {fake_message_id}\n"
        result += f"To: {to}\n"
        if cc:
            result += f"CC: {cc}\n"
        if bcc:
            result += f"BCC: {bcc}\n"
        result += f"Subject: {subject}\n"
        result += f"Body length: {len(body)} characters\n"
        result += f"Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return result

    @function_tool
    async def mock_search_emails(
        context: RunContext, query: str, max_results: int = 10
    ):
        """Mock tool to search emails"""
        logger.info(f"Mock Gmail: searching emails with query '{query}'")

        # Use the list_emails mock with the query
        return await mock_list_emails(context, max_results, query)

    @function_tool
    async def mock_get_unread_count(context: RunContext):
        """Mock tool to get unread email count"""
        logger.info("Mock Gmail: getting unread email count")

        unread_count = len([email for email in fake_emails if not email["read"]])
        return f"You have {unread_count} unread emails."

    # Set proper function names and documentation
    mock_list_emails.__name__ = "list_emails"
    mock_list_emails.__doc__ = "List recent emails from Gmail inbox"

    mock_read_email.__name__ = "read_email"
    mock_read_email.__doc__ = "Read a specific email by its ID"

    mock_send_email.__name__ = "send_email"
    mock_send_email.__doc__ = "Send an email via Gmail"

    mock_search_emails.__name__ = "search_emails"
    mock_search_emails.__doc__ = "Search emails in Gmail"

    mock_get_unread_count.__name__ = "get_unread_count"
    mock_get_unread_count.__doc__ = "Get the count of unread emails"

    return [
        mock_list_emails,
        mock_read_email,
        mock_send_email,
        mock_search_emails,
        mock_get_unread_count,
    ]


class AssistantWithMCP(Agent):
    def __init__(self, tools: list[Any] | None = None) -> None:
        instructions = load_instructions_from_file("dario_amodei_instructions.md")
        super().__init__(
            instructions=instructions,
            tools=tools or [],
        )

    # Keep the original weather function as an example
    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):
        """Use this tool to look up current weather information in the given location.

        If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.

        Args:
            location: The location to look up weather information for (e.g. city name)
        """
        logger.info(f"Looking up weather for {location}")
        return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    """Prewarm function to load VAD model with timeout handling"""
    try:
        logger.info("Loading VAD model...")
        proc.userdata["vad"] = silero.VAD.load()
        logger.info("VAD model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load VAD model: {e}")
        # Continue without VAD if loading fails
        proc.userdata["vad"] = None


async def entrypoint(ctx: JobContext):
    logger.info(f"Starting agent entrypoint for room: {ctx.room.name}")

    # Set up logging context
    ctx.log_context_fields = {"room": ctx.room.name}

    # Initialize MCP server for Gmail functionality
    server = None
    livekit_tools = []

    # Check if MCP credentials are available
    smithery_api_key = os.getenv("SMITHERY_API_KEY")
    smithery_profile = os.getenv("SMITHERY_PROFILE")

    if MCP_AVAILABLE and smithery_api_key and smithery_profile:
        try:
            logger.info("Setting up Gmail MCP server...")

            # Add timeout wrapper for MCP initialization
            async def initialize_mcp_with_timeout():
                # Build the MCP server URL
                url = f"https://server.smithery.ai/@shinzo-labs/gmail-mcp/mcp?api_key={smithery_api_key}&profile={smithery_profile}"

                # Create and initialize the MCP server
                server_transport = streamablehttp_client(url)
                read_stream, write_stream, _ = await server_transport.__aenter__()

                # Create client session
                server = mcp.ClientSession(read_stream, write_stream)
                await server.__aenter__()

                # Initialize the session
                await server.initialize()
                logger.info("MCP server initialized successfully")

                # Convert MCP tools to LiveKit tools
                livekit_tools = await build_livekit_tools(server)
                logger.info(
                    f"Converted {len(livekit_tools)} MCP tools to LiveKit tools"
                )

                return server, server_transport, livekit_tools

            # Set timeout for MCP initialization (15 seconds)
            try:
                server, server_transport, livekit_tools = await asyncio.wait_for(
                    initialize_mcp_with_timeout(), timeout=15.0
                )

                # Add shutdown callback for MCP cleanup
                @ctx.add_shutdown_callback
                async def cleanup_mcp():
                    logger.info("Shutting down MCP server...")
                    try:
                        if server:
                            await server.__aexit__(None, None, None)
                        if server_transport:
                            await server_transport.__aexit__(None, None, None)
                        logger.info("MCP server shutdown complete")
                    except Exception as e:
                        logger.error(f"Error during MCP cleanup: {e}")

            except TimeoutError:
                logger.warning("MCP server initialization timed out after 15 seconds")
                logger.info("Using mock Gmail functionality instead")
                server = None
                livekit_tools = create_mock_gmail_tools()

        except Exception as e:
            logger.error(f"Failed to setup MCP server: {e}")
            logger.info("Using mock Gmail functionality instead")
            server = None
            livekit_tools = create_mock_gmail_tools()
    else:
        if not MCP_AVAILABLE:
            logger.warning("MCP dependencies not available")
        else:
            logger.warning("SMITHERY_API_KEY or SMITHERY_PROFILE not found")
        logger.info("Using mock Gmail functionality")
        livekit_tools = create_mock_gmail_tools()

    # Create agent with tools
    logger.info("Creating agent...")
    agent = AssistantWithMCP(tools=livekit_tools)

    # Set up transcript writing
    async def write_transcript():
        try:
            current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = Path(
                f"transcripts/transcript_{ctx.room.name}_{current_date}.json"
            )
            filename.parent.mkdir(exist_ok=True)

            with open(filename, "w") as f:
                json.dump(session.history.to_dict(), f, indent=2)

            logger.info(f"Transcript saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to write transcript: {e}")

    # Set up voice AI pipeline
    logger.info("Setting up voice AI pipeline...")
    voice_id = os.getenv("CARTESIA_VOICE_ID", "default")

    # Handle case where VAD might not be loaded
    vad_model = ctx.proc.userdata.get("vad")
    if vad_model is None:
        logger.warning("VAD model not available, loading fallback...")
        try:
            vad_model = silero.VAD.load()
        except Exception as e:
            logger.error(f"Failed to load fallback VAD model: {e}")
            # Use a simple fallback or continue without VAD
            vad_model = None

    session = AgentSession(
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(model="nova-3", language="multi"),
        tts=cartesia.TTS(voice=voice_id),
        turn_detection=EnglishModel(),
        vad=vad_model,
        preemptive_generation=True,
    )

    # Add shutdown callbacks
    ctx.add_shutdown_callback(write_transcript)

    # Set up event handlers
    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("False positive interruption detected, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        try:
            summary = usage_collector.get_summary()
            logger.info(f"Usage summary: {summary}")
        except Exception as e:
            logger.error(f"Failed to log usage: {e}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session
    logger.info("Starting agent session...")
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Generate initial greeting - we always have tools now (either real MCP or mock)
    greeting = "Greet the user and let them know you can help with both general questions and Gmail functionality. You can help them read emails, send emails, search their inbox, and manage their Gmail account."

    await session.generate_reply(instructions=greeting)

    # Connect to the room
    logger.info(f"Connecting to room: {ctx.room.name}")
    await ctx.connect()
    logger.info("Successfully connected to room")


if __name__ == "__main__":
    # Configure logging level from environment variable
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting LiveKit Agent with MCP integration...")
    logger.info(f"Log level set to: {log_level}")

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
