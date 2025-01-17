import os
import logging
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TelegramScraper")

# Load environment variables
API_ID = os.getenv("API_ID")  # Replace with your Telegram API ID
API_HASH = os.getenv("API_HASH")  # Replace with your Telegram API Hash
PHONE = os.getenv("PHONE")  # Replace with your phone number

# Telegram client setup
client = TelegramClient('session', API_ID, API_HASH)

async def scrape_channel(channel_username, output_file):
    """Scrapes messages from a Telegram channel and saves them to a file."""
    async with client:
        logger.info(f"Fetching messages from channel: {channel_username}")
        messages = []
        async for message in client.iter_messages(channel_username):
            messages.append({
                "id": message.id,
                "text": message.message,
                "timestamp": message.date.isoformat(),
                "sender_id": message.sender_id
            })
        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            import json
            json.dump(messages, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved {len(messages)} messages to {output_file}")

if __name__ == "__main__":
    import asyncio
    CHANNELS = ["@example_channel1", "@example_channel2"]  # Add your channel usernames
    OUTPUT_DIR = "data/raw/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Login and scrape channels
    async def main():
        try:
            await client.start(PHONE)
            for channel in CHANNELS:
                output_file = os.path.join(OUTPUT_DIR, f"{channel.strip('@')}_messages.json")
                await scrape_channel(channel, output_file)
        except SessionPasswordNeededError:
            logger.error("Two-factor authentication is enabled. Please enter your password.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    asyncio.run(main())
