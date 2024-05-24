from keep_alive import keep_alive

keep_alive()

import os
import discord
import replicate
from io import BytesIO
from dotenv import load_dotenv
import asyncio
import aiohttp
import traceback
import random
from discord import app_commands, Interaction, File, Embed, Colour, ButtonStyle, SelectOption

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
REQUIRED_ROLE = os.getenv("REQUIRED_ROLE", "ImageGenerator")

if REPLICATE_API_TOKEN:
    replicate.Client(api_token=REPLICATE_API_TOKEN)

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True

MODELS = {
    "sdxl": {
        "name": "stability-ai/sdxl",
        "version": "7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        "requires_prompt": True
    },
    "realvisxl": {
        "name": "adirik/realvisxl-v3.0-turbo",
        "version": "3dc73c805b11b4b01a60555e532fd3ab3f0e60d26f6584d9b8ba7e1b95858243",
        "requires_prompt": True
    },
    "sd": {
        "name": "stability-ai/stable-diffusion",
        "version": "ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
        "requires_prompt": True
    }
}

class MyBot(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        await self.tree.sync()

bot = MyBot(intents=intents)

@bot.event
async def on_error(event, *args, **kwargs):
    error_message = f"Error in {event}: {args[0]}\n"
    traceback_info = traceback.format_exc()
    error_message += traceback_info
    print(error_message)

async def fetch_media(url: str):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    content_type = resp.headers.get('content-type')
                    filename = "output.png" if 'image' in content_type else "output.mp4" if 'video' in content_type else "output"
                    return File(BytesIO(await resp.read()), filename=filename)
                else:
                    raise Exception(f"Failed to download content: {resp.status}")
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return None

def has_image_generator_role():
    async def predicate(interaction: Interaction):
        role = discord.utils.get(interaction.guild.roles, name=REQUIRED_ROLE)
        if role in interaction.user.roles:
            return True
        else:
            await interaction.response.send_message(f"You need the '{REQUIRED_ROLE}' role to use this command.", ephemeral=True)
            return False
    return app_commands.check(predicate)

async def perform_image_generation(interaction: Interaction, model: str, prompt: str = None, width: int = 768, height: int = 768, num_steps: int = 25, seed: int = None):
    print(f"Starting image generation for model: {model}")

    if not REPLICATE_API_TOKEN:
        print("Missing Replicate API token.")
        await interaction.followup.send("Please have an admin add a Replicate API key using the /settings command.", ephemeral=True)
        return

    model_data = MODELS.get(model.lower())
    if not model_data:
        print(f"Invalid model selected: {model}")
        await interaction.followup.send("Invalid model selected.", ephemeral=True)
        return

    if model_data["requires_prompt"] and not prompt:
        print(f"Missing prompt for model: {model}")
        await interaction.followup.send(f"Please provide a prompt for {model}.", ephemeral=True)
        return

    if seed is None:
        seed = random.randint(0, 4294967295)

    input_data = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": num_steps,
        "seed": seed
    }

    if model.lower() == "sdxl":
        input_data["apply_watermark"] = False
    elif model.lower() == "realvisxl":
        input_data["refine"] = "no_refiner"
        input_data["scheduler"] = "DPM++_SDE_Karras"
        input_data["num_outputs"] = 1
        input_data["guidance_scale"] = 2
        input_data["apply_watermark"] = False
        input_data["high_noise_frac"] = 0.8
        input_data["negative_prompt"] = "(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth"
        input_data["prompt_strength"] = 0.8

    print(f"Input data for {model}: {input_data}")
    model_to_run = f"{model_data['name']}:{model_data['version']}"
    print(f"Model to run: {model_to_run}")

    try:
        await interaction.response.defer()
        output = await asyncio.to_thread(replicate.run, model_to_run, input=input_data)
        print(f"Output from replicate: {output}")
        image_url = output[0]
        image_file = await fetch_media(image_url)

        if image_file:
            embed = Embed(title="Image Generated!", color=Colour.red())
            embed.add_field(name="Model", value=model, inline=False)
            embed.add_field(name="Prompt", value=prompt, inline=False)
            embed.add_field(name="Width", value=str(width), inline=False)
            embed.add_field(name="Height", value=str(height), inline=False)
            embed.add_field(name="Steps", value=str(num_steps), inline=False)
            embed.add_field(name="Seed", value=str(seed), inline=False)
            embed.set_image(url=image_url)

            view = discord.ui.View()
            view.add_item(discord.ui.Button(style=ButtonStyle.secondary, label="Upscale ⬆️", custom_id="upscale"))
            view.add_item(discord.ui.Button(style=ButtonStyle.danger, label="Enhance ✨", custom_id="enhance"))
            view.add_item(discord.ui.Button(style=ButtonStyle.secondary, label="Remove BG ✂️", custom_id="removebg"))
            view.add_item(discord.ui.Button(style=ButtonStyle.danger, label="Img2Vid 🎥", custom_id="img2vid"))
            view.add_item(discord.ui.Button(style=ButtonStyle.link, label="Open Link 🔗", url=image_url))

            await interaction.followup.send(embed=embed, view=view)
        else:
            print("Failed to download generated image.")
            await interaction.followup.send(f"Error: Failed to download generated image.", ephemeral=True)
    except Exception as e:
        print(f"Error generating image: {e}")
        await interaction.followup.send(f"Error generating image: {e}", ephemeral=True)

async def fetch_media(url: str):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    content_type = resp.headers.get('content-type')
                    filename = "output.png" if 'image' in content_type else "output.mp4" if 'video' in content_type else "output"
                    return File(BytesIO(await resp.read()), filename=filename)
                else:
                    raise Exception(f"Failed to download content: {resp.status}")
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return None


class ModelSelect(discord.ui.Select):
    def __init__(self, data):
        self.data = data
        options = [
            discord.SelectOption(label="SD", description="Stable Diffusion", value="sd"),
            discord.SelectOption(label="SDXL", description="Stable Diffusion XL", value="sdxl"),
            discord.SelectOption(label="RealVisXL", description="RealVis XL", value="realvisxl")
        ]
        super().__init__(placeholder="Choose a model...", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        selected_model = self.values[0]
        print(f"Selected model: {selected_model}")
        await perform_image_generation(interaction, selected_model, self.data['prompt'], self.data['width'], self.data['height'], num_steps=self.data['steps'], seed=self.data['seed'])



@bot.tree.command(name="image", description="Generate an image")
@has_image_generator_role()
@app_commands.guild_only()
@app_commands.describe(
    prompt="Describe the image you want to generate",
    width="Width of the image (default: 768)",
    height="Height of the image (default: 768)",
    steps="Number of inference steps (default: 25)",
    seed="Seed value for reproducibility (leave empty for random)"
)
async def image_command(interaction: Interaction, prompt: str, width: int = 768, height: int = 768, steps: int = 25, seed: int = None):
    data = {
        'prompt': prompt,
        'width': width,
        'height': height,
        'steps': steps,
        'seed': seed
    }
    view = discord.ui.View()
    view.add_item(ModelSelect(data))
    await interaction.response.send_message("Choose a model:", view=view, ephemeral=True)

@bot.event
async def on_interaction(interaction: discord.Interaction):
    if interaction.type == discord.InteractionType.component:
        custom_id = interaction.data['custom_id']

        if custom_id == "upscale":
            await interaction.response.send_message("Please wait one moment while I upscale the image...")
            original_message = interaction.message
            content_url = original_message.embeds[0].image.url if original_message.embeds else None
            await perform_upscale(interaction, image_url=content_url, prompt="Upscale to 4k")

        elif custom_id == "enhance":
            await interaction.response.send_message("Please wait one moment while I enhance the image...")
            original_message = interaction.message
            content_url = original_message.embeds[0].image.url if original_message.embeds else None
            await perform_enhance(interaction, image_url=content_url, fidelity=0.5)

        elif custom_id == "removebg":
            await interaction.response.send_message("Please wait one moment while I remove the background...")
            original_message = interaction.message
            content_url = original_message.embeds[0].image.url if original_message.embeds else None
            await perform_removebg(interaction, image_url=content_url)

        elif custom_id == "img2vid":
            await interaction.response.send_message("Please wait one moment while I create the video...")
            original_message = interaction.message
            content_url = original_message.embeds[0].image.url if original_message.embeds else None

            try:
                width = original_message.width
                height = original_message.height
                num_steps = original_message.num_steps
                seed = original_message.seed
            except AttributeError:
                width = 768
                height = 768
                num_steps = 25
                seed = None

            await perform_i2vgen(interaction, image_url=content_url, prompt="make this into a video", width=width, height=height, num_steps=num_steps, seed=seed)

async def perform_i2vgen(interaction: Interaction, image_url: str, prompt: str, width: int, height: int, num_steps: int, seed: int):
    if seed is None:
        seed = random.randint(0, 4294967295)

    input_data = {
        "image": image_url,
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": num_steps,
        "apply_watermark": False,
        "seed": seed
    }
    model = "ali-vilab/i2vgen-xl:5821a338d00033abaaba89080a17eb8783d9a17ed710a6b4246a18e0900ccad4"

    try:
        output = await asyncio.to_thread(replicate.run, model, input=input_data)
        video_url = output

        video_file = await fetch_media(video_url)

        if video_file:
            embed = Embed(title="Video Generated!", color=Colour.red())
            embed.description = f"[Click here to view the video]({video_url})"
            await interaction.followup.send(embed=embed, file=video_file)
        else:
            await interaction.followup.send(f"Error: Failed to download generated video.")
    except Exception as e:
        await interaction.followup.send(f"Error generating video: {e}")

async def perform_enhance(interaction: Interaction, image_url: str, fidelity: float):
    input_data = {
        "image": image_url,
        "codeformer_fidelity": fidelity
    }
    model = "sczhou/codeformer:7de2ea26c616d5bf2245ad0d5e24f0ff9a6204578a5c876db53142edd9d2cd56"
    try:
        output = await asyncio.to_thread(replicate.run, model, input=input_data)
        image_file = await fetch_media(output)
        if image_file:
            embed = Embed(title="Enhanced Image!", color=Colour.red())
            embed.set_image(url=output)

            view = discord.ui.View()
            view.add_item(discord.ui.Button(style=ButtonStyle.secondary, label="Upscale ⬆️", custom_id="upscale"))
            view.add_item(discord.ui.Button(style=ButtonStyle.danger, label="Enhance ✨", custom_id="enhance"))
            view.add_item(discord.ui.Button(style=ButtonStyle.secondary, label="Remove BG ✂️", custom_id="removebg"))
            view.add_item(discord.ui.Button(style=ButtonStyle.danger, label="Img2Vid 🎥", custom_id="img2vid"))
            view.add_item(discord.ui.Button(style=ButtonStyle.link, label="Open Link 🔗", url=output))

            sent_message = await interaction.followup.send(embed=embed, view=view)

            sent_message.image_url = image_url
            sent_message.fidelity = fidelity
        else:
            await interaction.followup.send(f"Error: Failed to download generated image.")
    except Exception as e:
        await interaction.followup.send(f"Error enhancing image: {e}")

async def perform_upscale(interaction: Interaction, image_url: str, prompt: str):
    input_data = {
        "image": image_url,
        "prompt": prompt
    }
    model = "batouresearch/magic-image-refiner:507ddf6f977a7e30e46c0daefd30de7d563c72322f9e4cf7cbac52ef0f667b13"

    try:
        output = await asyncio.to_thread(replicate.run, model, input=input_data)
        image_file = await fetch_media(output[0])
        if image_file:
            embed = Embed(title="Upscaled Image!", color=Colour.red())
            embed.set_image(url=output[0])

            view = discord.ui.View()
            view.add_item(discord.ui.Button(style=ButtonStyle.secondary, label="Upscale ⬆️", custom_id="upscale"))
            view.add_item(discord.ui.Button(style=ButtonStyle.danger, label="Enhance ✨", custom_id="enhance"))
            view.add_item(discord.ui.Button(style=ButtonStyle.secondary, label="Remove BG ✂️", custom_id="removebg"))
            view.add_item(discord.ui.Button(style=ButtonStyle.danger, label="Img2Vid 🎥", custom_id="img2vid"))
            view.add_item(discord.ui.Button(style=ButtonStyle.link, label="Open Link 🔗", url=output[0]))

            sent_message = await interaction.followup.send(embed=embed, view=view)

            sent_message.image_url = image_url
            sent_message.prompt = prompt
        else:
            await interaction.followup.send(f"Error: Failed to download generated image.")
    except Exception as e:
        await interaction.followup.send(f"Error upscaling image: {e}")

async def perform_removebg(interaction: Interaction, image_url: str):
    input_data = {
        "image": image_url
    }
    model = "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003"
    try:
        output = await asyncio.to_thread(replicate.run, model, input=input_data)
        image_file = await fetch_media(output)
        if image_file:
            embed = Embed(title="Image With Background Removed!", color=Colour.red())
            embed.set_image(url=output)

            view = discord.ui.View()
            view.add_item(discord.ui.Button(style=ButtonStyle.secondary, label="Upscale ⬆️", custom_id="upscale"))
            view.add_item(discord.ui.Button(style=ButtonStyle.danger, label="Enhance ✨", custom_id="enhance"))
            view.add_item(discord.ui.Button(style=ButtonStyle.secondary, label="Remove BG ✂️", custom_id="removebg"))
            view.add_item(discord.ui.Button(style=ButtonStyle.danger, label="Img2Vid 🎥", custom_id="img2vid"))
            view.add_item(discord.ui.Button(style=ButtonStyle.link, label="Open Link 🔗", url=output))

            sent_message = await interaction.followup.send(embed=embed, view=view)

            sent_message.image_url = image_url
        else:
            await interaction.followup.send(f"Error: Failed to download generated image.")
    except Exception as e:
        await interaction.followup.send(f"Error removing background: {e}")

@bot.tree.command(name="i2vgen", description="Generate a video based on an input image and a prompt")
@has_image_generator_role()
@app_commands.guild_only()
@app_commands.describe(
    image_url="URL of the input image",
    prompt="Describe the video you want to generate",
    width="Width of the video in pixels",
    height="Height of the video in pixels",
    num_steps="Number of inference steps (higher is slower but potentially better quality)",
    seed="Seed value for reproducibility (leave empty for random seed)"
)
async def i2vgen(interaction: Interaction, image_url: str, prompt: str, width: int = 1024, height: int = 1024, num_steps: int = 25, seed: int = None):
    await interaction.response.defer()
    await perform_i2vgen(interaction, image_url, prompt, width, height, num_steps, seed)

@bot.tree.command(name="enhance", description="Enhance an image using CodeFormer")
@has_image_generator_role()
@app_commands.guild_only()
@app_commands.describe(
    image_url="URL of the image to enhance",
    fidelity="Fidelity of the enhancement (0.1 - 0.9, higher is stronger)"
)
async def enhance(interaction: Interaction, image_url: str, fidelity: float = 0.1):
    await interaction.response.defer()
    await perform_enhance(interaction, image_url, fidelity)

@bot.tree.command(name="upscale", description="Upscale an image using Magic Image Refiner")
@has_image_generator_role()
@app_commands.guild_only()
@app_commands.describe(
    image_url="URL of the image to upscale",
    prompt="Describe how you want to upscale the image"
)
async def upscale(interaction: Interaction, image_url: str, prompt: str):
    await interaction.response.defer()
    await perform_upscale(interaction, image_url, prompt)

@bot.tree.command(name="removebg", description="Remove the background of an image")
@has_image_generator_role()
@app_commands.guild_only()
@app_commands.describe(image_url="URL of the image")
async def removebg(interaction: Interaction, image_url: str):
    await interaction.response.defer()
    await perform_removebg(interaction, image_url)

@bot.tree.command(name="settings", description="Update the bot settings")
@app_commands.guild_only()
@app_commands.describe(
    api_key="Your new Replicate API key (optional)",
    required_role="The new role required to use image generation commands (optional)",
    remove_role="The role to remove from the required roles"
)
@app_commands.checks.has_permissions(administrator=True)
async def settings_command(interaction: Interaction, api_key: str = None, required_role: str = None, remove_role: str = None):
    try:
        updated = False
        if api_key:
            update_replicate_api_token(api_key)
            updated = True

        if required_role:
            update_required_role(required_role)
            updated = True

        if remove_role:
            remove_required_role(remove_role)
            updated = True

        if updated:
            await interaction.response.send_message("Settings updated successfully.", ephemeral=True)
        else:
            await interaction.response.send_message("No changes made.", ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(f"Failed to update settings: {e}", ephemeral=True)

def update_replicate_api_token(api_key: str):
    global REPLICATE_API_TOKEN
    with open('.env', 'r') as file:
        lines = file.readlines()
    with open('.env', 'w') as file:
        for line in lines:
            if line.startswith('REPLICATE_API_TOKEN='):
                file.write(f'REPLICATE_API_TOKEN={api_key}\n')
            else:
                file.write(line)
    load_dotenv(override=True)
    REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
    replicate.Client(api_token=REPLICATE_API_TOKEN)

def update_required_role(required_role: str):
    global REQUIRED_ROLE
    with open('.env', 'r') as file:
        lines = file.readlines()
    with open('.env', 'w') as file:
        for line in lines:
            if line.startswith('REQUIRED_ROLE='):
                file.write(f'REQUIRED_ROLE={required_role}\n')
            else:
                file.write(line)
    load_dotenv(override=True)
    REQUIRED_ROLE = required_role

def remove_required_role(remove_role: str):
    global REQUIRED_ROLE
    with open('.env', 'r') as file:
        lines = file.readlines()
    with open('.env', 'w') as file:
        for line in lines:
            if line.startswith('REQUIRED_ROLE=') and remove_role in line:
                file.write('REQUIRED_ROLE=\n')
            else:
                file.write(line)
    load_dotenv(override=True)
    REQUIRED_ROLE = os.getenv("REQUIRED_ROLE", "ImageGenerator")

@bot.tree.command(name="help", description="Shows help information for using the image generation commands")
async def help_command(interaction: Interaction):
    embed = Embed(
        title="Help - Image Generation",
        description="Learn how to use the image generation commands!",
        color=Colour.green()
    )
    embed.add_field(
        name="/image",
        value="Generates an image using the selected model.\n"
              "**Parameters:**\n"
              "`prompt`: Description of the image to generate.\n"
              "`width`: Width of the image (default: 768).\n"
              "`height`: Height of the image (default: 768).\n"
              "`steps`: Number of inference steps (default: 25).\n"
              "`seed`: Seed for reproducibility (optional).\n"
              "**Example:** /image prompt: 'A beautiful sunset', width: 1024, height: 768, steps: 50",
        inline=False
    )
    embed.add_field(
        name="/i2vgen",
        value="Generates a video based on an input image and a prompt.\n"
              "**Parameters:**\n"
              "`image_url`: URL of the input image.\n"
              "`prompt`: Description of the desired video.\n"
              "`width`, `height`: Dimensions of the video (default: 1024).\n"
              "`num_steps`: Number of inference steps (default: 25).\n"
              "`seed`: Seed for reproducibility (optional).\n"
              "**Example:** /i2vgen image_url: 'http://example.com/image.png', prompt: 'Turn this into a cinematic video'",
        inline=False
    )
    embed.add_field(
        name="/enhance",
        value="Enhances an image using CodeFormer.\n"
              "**Parameters:**\n"
              "`image_url`: URL of the image to enhance.\n"
              "`fidelity`: Strength of enhancement (0.1-0.9, default: 0.5).\n"
              "**Example:** /enhance image_url: 'http://example.com/image.png', fidelity: 0.7",
        inline=False
    )
    embed.add_field(
        name="/upscale",
        value="Upscales an image using Magic Image Refiner.\n"
              "**Parameters:**\n"
              "`image_url`: URL of the image to upscale.\n"
              "`prompt`: Description of the desired upscaling.\n"
              "**Example:** /upscale image_url: 'http://example.com/image.png', prompt: 'Upscale to 4k'",
        inline=False
    )
    embed.add_field(
        name="/removebg",
        value="Removes the background from an image.\n"
              "**Parameters:**\n"
              "`image_url`: URL of the image.\n"
              "**Example:** /removebg image_url: 'http://example.com/image.png'",
        inline=False
    )
    embed.add_field(
        name="/settings",
        value="Updates the bot settings (admin only).\n"
              "**Parameters:**\n"
              "`api_key`: Your new Replicate API key.\n"
              "`required_role`: The new role required to use image generation commands.\n"
              "`remove_role`: The role to remove from the required roles.\n"
              "**Example:** /settings api_key: 'new-api-key', required_role: 'NewRole', remove_role: 'OldRole'",
        inline=False
    )

    await interaction.response.send_message(embed=embed)

bot.run(DISCORD_BOT_TOKEN)
