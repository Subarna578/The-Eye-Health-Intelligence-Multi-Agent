# The Eye: MirrorLife Prevention Intelligence System

An advanced multi-agent AI system designed to monitor health and well-being trajectories in the Reply Mirror digital ecosystem. The Eye identifies early signals of health deviation and recommends proactive preventive support through cooperative intelligent agents.

## Project Overview

**The Eye** is a preventive health intelligence system built for the MirrorLife infrastructure—a comprehensive platform for population-level health optimization in the Reply Mirror metropolis (2087). The system continuously monitors longitudinal health datasets to detect subtle deviations from optimal behavioural and physiological trajectories.

### Core Capabilities

- **Dynamic Well-being Monitoring**: Tracks lifestyle behaviors, physical activity, sleep patterns, stress indicators, environmental factors, and social determinants
- **Early Signal Detection**: Identifies emerging vulnerabilities through adaptive pattern recognition across multimodal data streams
- **Adaptive Intelligence**: Maintains robustness against distributional shifts, behavioral drifts, and temporal evolution of risk patterns
- **Cooperative Multi-Agent Architecture**: Orchestrates specialized agents for different aspects of health analysis and recommendation
- **Proactive Prevention**: Recommends personalized preventive interventions before health deterioration occurs

### Classification Output

For each individual-time instance, the system classifies:
- **0**: Continuation of standard monitoring (well-being trajectory within acceptable optimization range)
- **1**: Recommendation for activation of personalized preventive support

## Technical Stack

- **Language**: Python 3.12+
- **LLM Framework**: LangChain
- **Model**: Meta LLaMA 3 70B Instruct (via OpenRouter API)
- **API Integration**: OpenRouter, Langfuse (tracing & observability)
- **Data Processing**: Pandas
- **Utilities**: Python-dotenv, ULID

## Installation

### Prerequisites
- Python 3.12 or higher
- OpenRouter API key
- (Optional) Langfuse credentials for trace observability

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "AI Agent Creation with LangChain"
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1  # On Windows
   # source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   MODEL_NAME=meta-llama/llama-3-70b-instruct
   
   # Optional: Langfuse integration
   LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
   LANGFUSE_SECRET_KEY=your_langfuse_secret_key
   LANGFUSE_HOST=https://challenges.reply.com/langfuse
   
   # Optional: Logging
   LOG_LEVEL=INFO
   ```

## Project Structure

```
.
├── main.py                    # Core entry point for the system
├── multi_agent.py            # Multi-agent orchestration and coordination
├── resource_management.py    # Resource allocation and management utilities
├── trace_viewer.py           # Langfuse trace visualization utilities
├── AGENT.md                  # Detailed project specification and challenge description
├── personas.md               # User personas and behavioral profiles
├── pyproject.toml            # Project metadata and dependencies
├── requirements.txt          # Python dependencies
│
├── users.json                # Citizen profiles and metadata
├── locations.json            # Geographic locations and context
├── status.csv                # Well-being status records
│
└── public_level_*/           # Challenge datasets (multiple difficulty levels)
    ├── users.json
    ├── locations.json
    ├── personas.md
    └── status.csv
```

## Usage

### Running the System

```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Run the main agent system
python main.py

# Or run multi-agent system
python multi_agent.py
```

### Command-line Arguments

```bash
python main.py --help
```

### Example: Processing User Data

The system processes individual citizen records and generates preventive recommendations:

```python
from main import client, parse_model_response
import json

# Load user data
with open('users.json') as f:
    users = json.load(f)

# System analyzes well-being trajectories and generates recommendations
```

## Data Formats

### users.json
Contains citizen profiles with demographic information, behavioral indicators, and longitudinal health signals.

### status.csv
Tracks well-being status over time, including:
- Timestamp
- Citizen ID
- Behavioral indicators (activity, sleep, stress)
- Environmental factors
- Classification (0 = monitor, 1 = preventive intervention)

### locations.json
Geographic and environmental context for health signal interpretation.

### personas.md
Behavioral profiles and user archetypes for personalization.

## Key Features

✨ **Multi-Agent Architecture**
- Specialized agents for different health domains
- Cooperative decision-making framework
- Adaptive role assignment based on data context

🧠 **Advanced Pattern Recognition**
- Temporal trajectory analysis
- Non-linear signal interaction detection
- Distributional shift adaptation

🎯 **Preventive Intelligence**
- Early intervention recommendations
- Personalized preventive strategies
- Evidence-based decision support

📊 **Observable & Traceable**
- Langfuse integration for full execution traces
- Detailed logging for transparency
- Audit trail for preventive recommendations

## Development

### Project Dependencies

All dependencies are specified in `pyproject.toml`:

```
langchain>=1.2.15
langchain-openai>=1.1.12
langfuse>=3.0.0,<4.0.0
pandas>=3.0.2
python-dotenv>=1.2.2
ulid-py>=1.1.0
```

### Running Tests

```bash
# Currently no automated tests (to be implemented)
python -m pytest tests/
```

## Submission & Evaluation

The system is designed for multi-level challenge evaluation:
- **Level 1**: Basic well-being classification with complete feature set
- **Level 2**: Advanced classification with partial or noisy data
- **Level 3**: Temporal adaptation with distributional shifts

Each level includes training and evaluation datasets. Best submission per level is recorded in `submission.txt`.

## Observability & Debugging

### Langfuse Traces

Monitor agent execution and decision paths via Langfuse:

```bash
python trace_viewer.py
```

### Logging

Adjust log level in `.env`:
```env
LOG_LEVEL=DEBUG  # For detailed execution logs
```

## Challenge Context

This project is part of **The Eye initiative** within the Reply Mirror ecosystem. The challenge involves designing adaptive intelligence capable of:

1. ✓ Identifying early health signals
2. ✓ Adapting to behavioral and environmental changes
3. ✓ Maintaining performance across temporal distribution shifts
4. ✓ Scaling to large population-level datasets

## License

[Specify license here]

## Contact & Support

For questions or issues, refer to `AGENT.md` for detailed challenge specifications or contact the Reply team.

---

**Last Updated**: April 10, 2026  
**Status**: Active Development (Multi-Agent Architecture)
