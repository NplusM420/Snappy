Snappy:
Snappy is a powerful Discord bot that enables users to generate images, upscale images, enhance image quality, remove backgrounds, and create videos from images using state-of-the-art AI models such as Stable Diffusion, SDXL, and RealVis. Admins can easily configure API keys and manage required roles directly from Discord.

Table of Contents

Features:
Commands

/image
/i2vgen
/enhance
/upscale
/removebg
/settings
/help


Installation:

Prerequisites
Setup
Running the Bot Locally
Adding Snappy to Discord Servers
Contributing
License

Features:

Image Generation: Generate images from text prompts using models like Stable Diffusion, SDXL, and RealVis.
Image Upscaling: Enhance the resolution of images.
Image Enhancement: Improve image quality.
Background Removal: Remove backgrounds from images.
Image to Video: Generate videos from images.
Settings Management: Admins can configure API keys and required roles directly from Discord.

Commands:
/image
Generates an image using the selected model.
Parameters:

prompt: Description of the image to generate.
width: Width of the image (default: 768).
height: Height of the image (default: 768).
steps: Number of inference steps (default: 25).
seed: Seed for reproducibility (optional).

Example:
/image prompt: "A beautiful sunset", width: 1024, height: 768, steps: 50
/i2vgen
Generates a video based on an input image and a prompt.
Parameters:

image_url: URL of the input image.
prompt: Description of the desired video.
width, height: Dimensions of the video (default: 1024).
num_steps: Number of inference steps (default: 25).
seed: Seed for reproducibility (optional).

Example:
/i2vgen image_url: "http://example.com/image.png", prompt: "Turn this into a cinematic video"
/enhance
Enhances an image using CodeFormer.
Parameters:

image_url: URL of the image to enhance.
fidelity: Strength of enhancement (0.1-0.9, default: 0.5).

Example:
/enhance image_url: "http://example.com/image.png", fidelity: 0.7
/upscale
Upscales an image using Magic Image Refiner.
Parameters:

image_url: URL of the image to upscale.
prompt: Description of the desired upscaling.

Example:
/upscale image_url: "http://example.com/image.png", prompt: "Upscale to 4k"
/removebg
Removes the background from an image.
Parameters:

image_url: URL of the image.

Example:
/removebg image_url: "http://example.com/image.png"
/settings
Updates the bot settings (admin only).
Parameters:

api_key: Your new Replicate API key.
required_role: The new role required to use image generation commands.
remove_role: The role to remove from the required roles.

Example:
/settings api_key: "new-api-key", required_role: "NewRole", remove_role: "OldRole"
/help

Shows help information for using the image generation commands.
Installation/Prerequisites

Python 3.10
Git
A Discord Developer Application
Replicate API key

Setup:

Clone the repository:
git clone https://github.com/NplusM420/snappy.git
cd snappy

Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required dependencies:

pip install -r requirements.txt

Set up environment variables:
Create a .env file in the project root directory and add the following:
DISCORD_BOT_TOKEN=your-discord-bot-token
REPLICATE_API_TOKEN=your-replicate-api-key
REQUIRED_ROLE=ImageGenerator


Running the Bot Locally
Run the bot:
python bot.py