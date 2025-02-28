import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', '')

# Add this to your .env file:
# NEWS_API_KEY=your_api_key_here
# ALPHA_VANTAGE_KEY=your_api_key_here
