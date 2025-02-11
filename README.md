# Unified Social Media Analyser

## Overview
The **Unified Social Media Analyser** is a tool designed to collect and analyze user activity across multiple social media platforms. It integrates with **Twitter**, **Reddit**, and **HackerNews** to fetch posts, comments, and media for in-depth analysis.

## Features
- **Multi-platform Analysis**: Collects data from Twitter, Reddit, and HackerNews.
- **Rate Limit Handling**: Detects and manages API rate limits gracefully.
- **Media Processing**: Downloads and analyzes images.
- **User Interaction**: Supports both command-line prompts and JSON input.
- **Data Caching**: Stores fetched data for efficient retrieval.
- **Comprehensive Reports**: Generates analytical reports based on user queries.

## Dependencies
The project requires the following Python libraries:
- `httpx`
- `tweepy`
- `praw`
- `rich`
- `PIL` (Pillow)
- `base64`
- `argparse`
- `hashlib`
- `json`
- `logging`
- `threading`
- `datetime`
- `functools`

Ensure you have these installed before running the program.

## Installation
Clone the repository and install the required dependencies:

```sh
pip install -r requirements.txt
```

## Configuration
Set up API credentials as environment variables:

```sh
export TWITTER_BEARER_TOKEN="your_twitter_bearer_token"
export REDDIT_CLIENT_ID="your_reddit_client_id"
export REDDIT_CLIENT_SECRET="your_reddit_client_secret"
export REDDIT_USER_AGENT="your_reddit_user_agent"
export OPENROUTER_API_KEY="your_openrouter_api_key"
```

## Usage
### Interactive Mode
Run the script interactively to analyze social media activity:

```sh
python unified-analyser.py
```

### JSON Input Mode
To use JSON input via stdin:

```sh
echo '{"platforms": {"twitter": ["user1"], "reddit": ["user2"]}, "query": "Analyze user activity", "format": "markdown"}' | python unified-analyser.py --stdin
```

### Output Format
The output can be saved in `markdown` or `json` format using the `--format` flag.

```sh
python unified-analyser.py --format json
```

## Error Handling
- Handles API rate limits with retry mechanisms.
- Logs errors to `analyser.log` for debugging.
- Provides user-friendly error messages for missing API keys or connectivity issues.

## License
This project is licensed under the MIT License.

## Contributing
Feel free to submit issues or pull requests to improve the project!
