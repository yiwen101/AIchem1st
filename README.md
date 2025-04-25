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
- **Language & Runtime**: Python 3.11
- **Package Management**: uv
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
  - Input questions downloaded from competition Hugging Face

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/yiwen101/AIchem1st.git
   ```
2. Install dependencies and activate the environment:
   ```bash
   uv sync
   source .venv/bin/activate
   # or in Windows
   .venv\Scripts\activate
   ```
3. Provide your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```
4. Prepare the question parquet
4. Unzip the YouTube videos into `videos/` folder, name the videos as `<video_id>.mp4`:
   ```bash
   # E.g. https://www.youtube.com/shorts/4XSX2-o8WxM
   # -> save as ->
   # videos/4XSX2-o8WxM.mp4
   ```
5. Run the agent (single-question):
   ```bash
   python main.py run-l3-single <questions>.parquet <question_id>
   # E.g.
   python main.py run-l3-single test-00000-of-00001.parquet 0008-0
   ```
   Answer will be found in the console output.
6. Run the agent (all questions):
   ```bash
   python main.py run-l3 <questions>.parquet
   # E.g.
   python main.py run-l3 test-00000-of-00001.parquet
   ```
   Answers will be found in the `results/` folder.
