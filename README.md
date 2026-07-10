# serry-voiceassistant

A **fully local** AI voice agent (cascade pipeline: STT → LLM → TTS), running
entirely on-device on Apple Silicon via MLX.

Built on [Pipecat](https://github.com/pipecat-ai/pipecat) — the open-source
framework for real-time voice and multimodal AI agents — as its pipeline
foundation.

## Configuration

- **Bot Type**: Web
- **Transport(s)**: SmallWebRTC, Daily (WebRTC)
- **Pipeline**: Cascade — all local, no cloud API keys
  - **STT**: Qwen3-ASR-1.7B (bf16) via [mlx-audio](https://github.com/Blaizzy/mlx-audio) — `services_local.py`
  - **LLM**: local **LM Studio** (OpenAI-compatible) at `http://localhost:1234/v1`, model `qwen3.5-122b-a10b`
  - **TTS**: Qwen3-TTS-12Hz-1.7B-Base (bf16) via mlx-audio — `services_local.py`

### Requirements

- macOS on **Apple Silicon** (arm64)
- **LM Studio** running with `qwen3.5-122b-a10b` loaded and its local server started (port 1234)
- First run downloads the Qwen3 ASR/TTS weights (~7 GB total) from Hugging Face

Model choices and voice are overridable via `.env` (`QWEN3_ASR_MODEL`,
`QWEN3_TTS_MODEL`, `QWEN3_TTS_VOICE`, `LMSTUDIO_*`).

## Setup

### Server

1. **Navigate to server directory**:

   ```bash
   cd server
   ```

2. **Install dependencies**:

   ```bash
   uv sync
   ```

3. **Configure environment variables**:

   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

4. **Run the bot**:

   ```bash
   uv run bot.py
   ```

   The runner serves every transport; the caller selects which one (a web/mobile
   client picks its transport when it connects; a telephony provider connects to
   `/ws`).

## Project Structure

```
serry-voiceassistant/
├── server/              # Python bot server
│   ├── bot.py           # Main bot implementation
│   ├── services_local.py # Local Qwen3-ASR & Qwen3-TTS Pipecat services (mlx-audio)
│   ├── pyproject.toml   # Python dependencies
│   ├── .env.example     # Environment variables template
│   ├── .env             # Your API keys (git-ignored)
│   ├── Dockerfile       # Container image for Pipecat Cloud
│   └── pcc-deploy.toml  # Pipecat Cloud deployment config
├── .gitignore           # Git ignore patterns
└── README.md            # This file
```

## Deploying to Pipecat Cloud

This project is configured for deployment to Pipecat Cloud. You can learn how to deploy to Pipecat Cloud in the [Pipecat Quickstart Guide](https://docs.pipecat.ai/getting-started/quickstart#step-2-deploy-to-production).

Refer to the [Pipecat Cloud Documentation](https://docs.pipecat.ai/deployment/pipecat-cloud/introduction) to learn more about configuring, deploying, and managing your agents in Pipecat Cloud.

## Learn More

- [Pipecat Documentation](https://docs.pipecat.ai/)
- [Pipecat GitHub](https://github.com/pipecat-ai/pipecat)
- [Pipecat Examples](https://github.com/pipecat-ai/pipecat-examples)
- [Discord Community](https://discord.gg/pipecat)