# L3Agent (Layered Learning Logic Agent)

## Overview
L3Agent is a multi steps video analysis agent that progressively refines its understanding of a video to answer user questions. It leverages a combination of full-video context, focused segments, and key-image detail to generate concise, accurate answers backed by visual evidence.

Key features:
- **Layered Analysis**: Begins with full-video overview, drills down to important segments, and finally zooms in on a single key image.
- **Adaptive Zoom**: Automatically decides when to zoom into parts of video or single frames for deeper inspection.
- **Synthesized Answers**: Combines insights from all layers to deliver a final, comprehensive answer.
- **Extensible**: Available for different vision models, segments selection strategies, or hint prompts generation based on different video sources.
- **Resource Management**: Handles video download, caching, frame extraction, and cleanup.

## Development Tools
- **Language & Runtime**: Python 3.10+ (recommended)
- **Package Management**: pip (virtualenv/venv)
- **Version Control**: Git

## APIs & External Services
- **OpenAI API**: Vision + Chat endpoints (GPT-4O-mini, GPT-4, etc.) for image/video understanding and language synthesis
- **YouTube Data API**: Fetch video metadata and descriptions

## Assets (Models & Datasets)
- **Pre-trained Models**:
  - OpenAI's GPT-4O-mini (vision-enabled) for multimodal Q&A
  - Other models (e.g. DeepSeek) for fallback or high-detail analysis
- **Datasets & Videos**:
  - Local cached video files under `videos/` directory
  - Input queries downloaded from Hugging Face

## Libraries & Dependencies
- **Core**:
  - `langchain`
  - `opencv`
  - `openai`

- **A full list of dependencies are listed in requirements.txt.**

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/yiwen101/AIchem1st.git
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Provide your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```
5. Download or copy your target YouTube video(s) into `videos/`:
   ```bash
   youtube-dl -o "videos/%(id)s.%(ext)s" <VIDEO_URL>
   ```
6. Run the agent in your Python script or interactive session:
   ```python
   from app.L3Agent import L3Agent
   from app.model.structs import ParquetFileRow

   agent = L3Agent(model="gpt-4o-mini", display=False, high_detail=True)
   row = ParquetFileRow(
       qid="001",
       video_id="dQw4w9WgXcQ",
       question="What object is the person holding at 00:30?",
       ... # other required fields
   )
   answer = agent.get_answer(row)
   print("Answer:", answer)
   ```