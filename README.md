# Danc Telegram Bot

A small Telegram bot that:
- captions images locally using a Vision+Language model (nlpconnect/vit-gpt2-image-captioning)
- can chat via an external LLM provider (e.g. Mistral through OpenRouter)

## Repository layout
- `main.py` — bot entrypoint and handlers (image captioning + chat)
- `.gitignore` — ignores environment files and test artifacts
- (other helper files may be present in the folder)

## Requirements
- Python 3.8+
- Recommended packages:
  - python-telegram-bot>=20
  - transformers
  - torch
  - pillow
  - python-dotenv
  - httpx (or requests) for OpenRouter calls

Example:
```bash
python3 -m venv venv
source venv/bin/activate
pip install python-telegram-bot transformers torch pillow python-dotenv httpx
```

You can also create a `requirements.txt` with the above and run:
```bash
pip install -r requirements.txt
```

## Environment
Create a `.env` file in the project root (this file is already listed in `.gitignore`) with:
```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

## Run
Start the bot (from project folder):
```bash
python main.py
```
The bot will connect to Telegram using `TELEGRAM_BOT_TOKEN`. For chat features, `OPENROUTER_API_KEY` must be set.

## Usage
- Send a photo to the bot to receive a caption generated locally.
- Send a text message to receive chat responses (requires OpenRouter key).
- Common "persona" triggers (e.g. "Who are you?") should be handled by the persona handler.

## Troubleshooting
- "Object of type 'Tensor' is not callable" during captioning:
  - Ensure the image processor is given a batch and you pass the tensor (not the BatchEncoding) to the model:
    - `inputs = processor(images=[image], return_tensors="pt")`
    - `pixel_values = inputs.pixel_values.to(device)`
    - `output_ids = model.generate(pixel_values=pixel_values, ...)`
    - `caption = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)`
- If the bot appears to "hang" on start, check for top-level `input()` calls or other blocking operations; the bot should run under the `if __name__ == "__main__":` guard.
- If using GPU, verify `torch.cuda.is_available()` and that `model.to(device)` is called.

## Notes
- Keep secrets out of version control (`.env` is in `.gitignore`).
- Review `main.py` for any temporary debug prints and remove them before production.
- Update handlers in `main.py` if you want separate routes for persona vs general messages.

If you want, I can also create a `requirements.txt` or update `main.py` to include the safe `describe_image_sync` implementation