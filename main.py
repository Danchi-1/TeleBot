import os
import asyncio
import httpx
import torch
from PIL import Image
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

load_dotenv()

# Env
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Image captioning (local)
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) # type: ignore

# Mistral via OpenRouter (async)
async def chat_with_mistral(prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        return "OpenRouter API key missing. Set OPENROUTER_API_KEY in your .env."

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # (optional but nice) helps OpenRouter dashboards
        "HTTP-Referer": "https://github.com/your-repo-or-app",
        "X-Title": "Danc Telegram Bot",
    }
    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        return f"OpenRouter HTTP error {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"OpenRouter request failed: {e}"

# Local captioning (CPU/GPU). Keep sync, run off-thread when called.
def describe_image_sync(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)
    try:
        with torch.inference_mode():
            output_ids = model.generate(pixel_values=pixel_values, max_length=16, num_beams=4)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except TypeError as e:
        raise RuntimeError(f"model.generate failed: {e} — pixel_values type/shape: {type(pixel_values)}, {getattr(pixel_values, 'shape', None)}")
    return caption

# Handlers
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    user_message = update.message.text or ""
    if user_message.strip().lower() in ("who are you?", "who are you"):
        await update.message.reply_text("I am a Danc bot that can chat and describe images using AI!")
        return
    if user_message.strip().lower() == "/start":
        await update.message.reply_text("Hello! Send me a message or a photo, and I'll respond accordingly")
        return

    reply = await chat_with_mistral(user_message)
    await update.message.reply_text(reply)

async def bot_persona(Update: Update):
    while Update.message is not None:
        if Update.message.text == "Who are you?":
            await Update.message.reply_text("I am a Danc bot that can chat and describe images using AI!")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return

    try:
        # highest-res photo
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)

        # Proper Telegram download (async, no requests.get on file_path)
        image_path = "temp_image.jpg"
        await file.download_to_drive(image_path)

        # Run heavy captioning off the event loop
        caption = await asyncio.to_thread(describe_image_sync, image_path)

        # Send caption to Mistral for reasoning
        prompt = f"Describe and reason about this image based on the caption: {caption}"
        bot_reply = await chat_with_mistral(prompt)
        await update.message.reply_text(bot_reply)

    except Exception as e:
        await update.message.reply_text(f"Error processing image: {e}")
    finally:
        # try to clean up
        try:
            if os.path.exists("temp_image.jpg"):
                os.remove("temp_image.jpg")
        except Exception:
            pass

# ── Boot
def main():
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing. Set it in your .env")

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    print("Bot is running…")
    app.run_polling()

if __name__ == "__main__":
    main()