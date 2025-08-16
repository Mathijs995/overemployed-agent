# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

This is a LiveKit Agents Python starter project for building voice AI
applications. The codebase provides a complete voice AI assistant implementation
using LiveKit's agent framework with OpenAI, Cartesia, and Deepgram
integrations.

## Development Commands

### Setup and Installation

- `uv sync` - Install dependencies and set up virtual environment
- `uv run src/agent.py download-files` - Download required models (Silero VAD,
  turn detector)

### Running the Agent

- `uv run src/agent.py console` - Run agent directly in terminal for testing
- `uv run src/agent.py dev` - Run agent for frontend/telephony development
- `uv run src/agent.py start` - Run agent in production mode

### Testing and Quality

- `uv run pytest` - Run the complete evaluation test suite
- `uv run ruff check` - Run linting checks
- `uv run ruff format` - Format code

### Task Commands (using taskfile.yaml)

- `task install` - Bootstrap application for local development
- `task dev` - Interactive development mode

## Architecture

### Core Components

**Agent Structure (`src/agent.py`)**:

- `Assistant` class: Main agent implementation inheriting from `Agent`
- `entrypoint()`: Main application entry point setting up the voice pipeline
- `prewarm()`: Model preloading function for performance optimization

**Voice Pipeline Configuration**:

- **LLM**: OpenAI GPT-4o-mini for conversation processing
- **STT**: Deepgram Nova-3 with multilingual support
- **TTS**: Cartesia voice synthesis
- **VAD**: Silero Voice Activity Detection
- **Turn Detection**: Multilingual model for conversation flow

**Function Tools**: The agent supports custom function tools decorated with
`@function_tool`. Currently includes:

- `lookup_weather()`: Example weather lookup tool

### Key Features

- Preemptive generation for low-latency responses
- False interruption detection and recovery
- Metrics collection and usage tracking
- LiveKit Cloud noise cancellation integration
- Comprehensive evaluation framework

### Environment Configuration

The application uses `.env.local` for configuration. Required variables:

- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`

### Testing Framework

Uses LiveKit's evaluation framework with pytest-asyncio:

- Agent behavior evaluation with LLM judges
- Function tool testing with mocking capabilities
- Error handling and edge case validation
- Response quality assessment

### Project Structure

```
src/
├── __init__.py
└── agent.py          # Main agent implementation
tests/
└── test_agent.py     # Comprehensive evaluation suite
```

## Development Notes

- Uses `uv` for dependency management and Python environment
- Ruff configured for linting with line length 88, Python 3.9+ target
- Docker-ready with model pre-downloading in build stage
- Production deployment supported via LiveKit Cloud or self-hosted
