# Claudio - Voice I/O for Claude Code
# Cleanser Labs Audio

default:
    @just --list

setup:
    uv sync

test:
    uv run --extra dev pytest tests/ -v -m "not slow"

# Run all tests including slow integration tests
test-all:
    uv run --extra dev pytest tests/ -v

# Run only slow integration tests (TTS backends)
test-slow:
    uv run --extra dev pytest tests/ -v -m "slow"

# Run tests with coverage
test-cov:
    uv run --extra dev pytest tests/ -v --cov=src/claudio --cov-report=term-missing

lint:
    uv run ruff check src/ tests/

format:
    uv run ruff format src/ tests/

# --- Voice Commands ---

# Speak text directly (no server needed)
say text:
    uv run claudio say "{{text}}"

# Speak text directly with device
say-to text device:
    uv run claudio say "{{text}}" -d {{device}}

# Speak text via server
speak text:
    uv run claudio speak "{{text}}"

# --- Server Modes ---

# Start HTTP server
server:
    uv run claudio server

# Start HTTP server with device
server-dev device:
    uv run claudio server -d {{device}}

# Start MCP server (stdio)
mcp:
    uv run claudio --mcp

# Start API proxy (auto-TTS for Claude responses)
proxy port="9000":
    uv run claudio proxy -p {{port}}

# --- Listen Commands ---

# Start listening for voice
listen-start trigger="send":
    uv run claudio listen start -t "{{trigger}}"

# Stop listening
listen-stop:
    uv run claudio listen stop

# Get pending transcript
listen-get:
    uv run claudio listen get

# --- Audio Devices ---

# List audio devices
devices:
    uv run claudio devices

# --- Model Management ---

# Download ASR model
fetch-asr model="mlx-community/parakeet-tdt-0.6b-v3":
    HF_HUB_ENABLE_HF_TRANSFER=1 uv run python -c "from parakeet_mlx import from_pretrained; from_pretrained('{{model}}')"

# --- Testing ---

# Test MCP server initialization
test-mcp:
    @echo '{"jsonrpc":"2.0","method":"initialize","id":1}' | uv run claudio --mcp

# Test MCP speak
test-mcp-speak text="Hello from MCP":
    @printf '{"jsonrpc":"2.0","method":"initialize","id":1}\n{"jsonrpc":"2.0","method":"tools/call","params":{"name":"speak","arguments":{"text":"{{text}}"}},"id":2}\n' | uv run claudio --mcp

# Kill running servers
kill:
    pkill -f "claudio" || true

# --- Installation ---

# Install as global uv tool
install:
    uv tool install ".[soprano,asr]" --force --prerelease=allow

# Install as editable (no reinstall needed after code changes)
install-dev:
    uv tool install -e ".[soprano,asr]" --force --prerelease=allow

# Uninstall then reinstall
reinstall:
    uv tool uninstall claudio || true
    uv tool install ".[soprano,asr]" --force --prerelease=allow

# Uninstall global tool
uninstall:
    uv tool uninstall claudio
