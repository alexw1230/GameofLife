# DevFest2026 RPG Detection & Quest System

This project is a real-time RPG-themed webcam application for DevFest2026. It combines YOLOv8 person detection, RPG stat assignment, quest tracking, and interactive AI-powered character chat.

## Core Features

- **YOLOv8 Person Detection**: Detects people in the webcam feed using Ultralytics YOLOv8.
- **RPG Stat Assignment**: Each detected person is assigned HP and Mana based on their size and clothing color brightness.
- **Job Assignment**: Characters are given RPG jobs (Tank, Mage, Healer, etc.) based on HP/Mana, with fallback jobs for non-matching archetypes.
- **Quest System**:
  - Main and side quests are generated and tracked.
  - Quest completion is verified via image upload and AI validation.
  - EXP and leveling system for completing quests.
- **Interactive Character Cards**:
  - Click on a detected person to view their RPG card.
  - Card displays unique title, job, description, and stats.
  - Chat button opens an AI-powered roleplay chat window.
- **AI Chat System**:
  - Chat with any character using OpenAI API (Meta-Llama/Gemma models).
  - Custom user input, AI responses, scrollable chat history, and instructions.
- **Loading Screen**: Fullscreen loading image shown at startup.
- **UI Overlays**: Success/failure overlays, quest log, health/mana bars, EXP bar, and boss detection.

## Installation

1. Clone or navigate to the project directory
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Note: First-time setup may take a few minutes as it downloads the YOLOv8 model (~100MB)


## Installation

1. Clone the repository and navigate to the project directory.
2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
3. Place YOLOv8 model file at `models/yolov8n.pt` (auto-download on first run).
4. Add your OpenAI API key to `key.env` or set `OPENAI_API_KEY` environment variable.

## Usage


## Quest System
- Main quest is generated and tracked, with cooldown and regeneration logic.
- Side quests are randomly assigned from a pool.
- Completing quests awards EXP and can trigger level-ups.
- Quest completion is validated by uploading a screenshot and using AI to check evidence.

## RPG Stats & Job Assignment
- HP is calculated from bounding box size and reference distance.
- Mana is calculated from clothing color brightness.
- Jobs are assigned based on HP/Mana requirements:
  - Tank, Warrior, Warlock, Mage, Healer, Muggle (requirement-based)
  - Commoner, Blacksmith, Noble, Baker, Farmer (random fallback)
- Boss detection highlights the largest character each frame.

## AI Chat System
- Each character card has a chat button.
- Chat window allows roleplay conversation with the AI, using character stats and description.
- Chat history is scrollable, and instructions are shown in the UI.
- Color-coded bounding boxes and labels


## Configuration
- Edit `config.yaml` for parameters like smoothing alpha, reference distance, HP scale, quest timeouts, etc.
- Place your OpenAI API key in `key.env` or set as environment variable.

## Requirements
- Python 3.8+
- Webcam
- Ultralytics YOLOv8 model file (`models/yolov8n.pt`)
- OpenAI API key
- Dependencies: `ultralytics`, `opencv-python`, `numpy`, `PyYAML`, `torch`, `tkinter`

## Troubleshooting
- If webcam is not detected, ensure it is connected and not used by other apps.
- If HP/Mana values seem off, adjust `hp_scale` and `ref_distance` in `config.yaml`.
- For slow performance, use a smaller YOLO model or reduce frame resolution.
- If chat or quest validation fails, check your OpenAI API key and internet connection.

## Project Intent
This project demonstrates:
- Real-time computer vision for RPG character detection and stat assignment.
- Interactive quest and leveling system.
- AI-powered roleplay chat for immersive character interaction.
- Modular, extensible code for future RPG features and festival demos.
