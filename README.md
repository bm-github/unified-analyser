# Unified Social Media Analyzer

A powerful Python tool that analyzes user activity across Twitter and Reddit platforms using AI-powered insights. The analyzer leverages both Google's Gemini and Anthropic's Claude APIs to provide comprehensive analysis of social media behavior and patterns.

## Features

- Cross-platform analysis of Twitter and Reddit accounts
- AI-powered insights using Gemini and Claude
- Intelligent caching system for API rate limit management
- Media file downloading and handling
- Interactive command-line interface with rich text formatting
- Comprehensive error handling and rate limiting
- Support for both single-platform and multi-platform analysis

## Prerequisites

- Python 3.8 or higher
- Twitter (X) Developer Account with Bearer Token
- Reddit Developer Account with API credentials
- Google AI (Gemini) API key
- OpenRouter API key for Claude access

## Installation

1. Clone the repository
2. Install required dependencies:
```bash
pip install praw tweepy httpx google-generativeai rich pillow
```

## Configuration

Create a `keys` directory in the parent folder with the following files:

- `key-openrouter.txt`: Your OpenRouter API key
- `key-gemini.txt`: Your Google Gemini API key
- `x-token.txt`: Your Twitter Bearer token
- `reddit-credentials.json`: Your Reddit API credentials in the format:
```json
{
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "user_agent": "your_user_agent"
}
```

## Usage

Run the analyzer:
```bash
python unified-social-analyzer.py
```

The interactive interface will guide you through:
1. Selecting platforms to analyze (Twitter, Reddit, or both)
2. Entering usernames
3. Asking questions about the user's activity

### Commands

During analysis:
- `exit`: End the session
- `refresh`: Update all platform data
- `help`: Show available commands

## Data Storage

The analyzer creates and manages the following directories:
- `social_cache/`: Cached social media data
- `chat_history/`: Analysis conversation history
- `social_cache/media/`: Downloaded media files

## Features in Detail

### Twitter Analysis
- Fetches recent tweets (excluding retweets and replies)
- Downloads and processes media attachments
- Analyzes engagement metrics
- 24-hour cache duration

### Reddit Analysis
- Retrieves recent submissions and comments
- Calculates subreddit activity statistics
- Analyzes karma scores and engagement
- 24-hour cache duration

### AI Analysis
- Uses Gemini for image-heavy Twitter analysis
- Uses Claude for text-based and cross-platform analysis
- Provides insights on:
  - User behavior patterns
  - Content preferences
  - Engagement trends
  - Community participation

## Error Handling

The analyzer includes comprehensive error handling for:
- API authentication failures
- Rate limiting
- Network issues
- Invalid usernames
- Missing credentials
- Media download failures

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

MIT License

## Acknowledgments

- PRAW (Python Reddit API Wrapper)
- Tweepy
- Google Generative AI
- Anthropic Claude
- Rich CLI library