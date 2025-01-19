import praw
import tweepy
import json
import os
import sys
from typing import Dict, List, Optional, Generator
from datetime import datetime, timedelta
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
import google.generativeai as genai
import PIL.Image
import httpx

class UnifiedSocialAnalyzer:
    def __init__(self):
        self.console = Console()
        self.cache_dir = Path("social_cache")
        self.history_dir = Path("chat_history")
        self.media_dir = self.cache_dir / 'media'
        self.http_client = httpx.Client(timeout=10.0)  # Reusable HTTP client
        
        # Create necessary directories
        for directory in [self.cache_dir, self.history_dir, self.media_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        self._init_credentials()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.http_client.close()
        
    def _init_credentials(self):
        """Initialize all API credentials"""
        try:
            # Initialize OpenRouter
            with open('../../keys/key-openrouter.txt', 'r') as f:
                self.openrouter_api_key = f.read().strip()
                if not self.openrouter_api_key:
                    raise ValueError("OpenRouter API key file is empty")
            
            # Initialize Gemini
            key_path = Path('../../keys/key-gemini.txt')
            with open(key_path) as f:
                gemini_api_key = f.read().strip()
                if not gemini_api_key:
                    raise ValueError("Gemini API key file is empty")
            os.environ['GOOGLE_API_KEY'] = gemini_api_key
            
            # Configure Gemini
            genai.configure(api_key=gemini_api_key)
            genai.configure(transport="rest")
            self.gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest',
                                                     generation_config={
                                                         "max_output_tokens": 2048,
                                                         "temperature": 0.7,
                                                         "top_p": 0.8,
                                                         "top_k": 40
                                                     })
            
            # Initialize OpenRouter client
            self.openrouter_client = httpx.Client(
                base_url="https://openrouter.ai/api/v1",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "HTTP-Referer": "http://localhost",
                    "Content-Type": "application/json"
                }
            )
            
            # Initialize Reddit
            with open('../../keys/reddit-credentials.json', 'r') as f:
                reddit_creds = json.load(f)
                self.reddit = praw.Reddit(
                    client_id=reddit_creds['client_id'],
                    client_secret=reddit_creds['client_secret'],
                    user_agent=reddit_creds['user_agent']
                )
            
            # Initialize Twitter
            with open('../../keys/x-token.txt') as f:
                self.twitter_client = tweepy.Client(bearer_token=f.read().strip())
                
        except FileNotFoundError as e:
            raise Exception(f"Missing credentials file: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON in credentials file: {str(e)}")
        except Exception as e:
            raise Exception(f"Credentials initialization failed: {str(e)}")

    def get_cache_path(self, platform: str, username: str) -> Path:
        """Get the path for the user's cache file."""
        return self.cache_dir / f"{platform}_{username}.json"

    def get_history_path(self, platform: str, username: str) -> Path:
        """Get the path for the user's chat history file."""
        return self.history_dir / f"{platform}_{username}_history.json"

    def load_cached_data(self, platform: str, username: str) -> Optional[Dict]:
        """Load cached data from file."""
        cache_path = self.get_cache_path(platform, username)
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)
        return None

    def save_to_cache(self, platform: str, username: str, data: Dict):
        """Save data to cache file."""
        cache_path = self.get_cache_path(platform, username)
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _download_media(self, media_url: str, identifier: str) -> Optional[Path]:
        """Download media file and return path."""
        try:
            response = self.http_client.get(media_url)
            response.raise_for_status()
            
            # Determine file extension from content type
            content_type = response.headers.get('content-type', '')
            ext = {
                'image/jpeg': '.jpg',
                'image/png': '.png',
                'image/gif': '.gif'
            }.get(content_type.lower(), '.jpg')
            
            media_path = self.media_dir / f"{identifier}{ext}"
            media_path.write_bytes(response.content)
            return media_path
            
        except httpx.HTTPError as e:
            self.console.print(f"[yellow]Warning: Failed to download media: HTTP error {e.response.status_code}[/yellow]")
        except Exception as e:
            self.console.print(f"[yellow]Warning: Failed to download media: {str(e)}[/yellow]")
        return None

    def fetch_twitter_data(self, username: str, force_refresh: bool = False) -> Optional[Dict]:
        """Fetch Twitter user data with caching."""
        if not force_refresh:
            cached_data = self.load_cached_data('twitter', username)
            if cached_data:
                cache_time = datetime.fromisoformat(cached_data['timestamp'])
                if datetime.now() - cache_time < timedelta(hours=24):
                    self.console.print("[green]Using cached Twitter data[/green]")
                    return cached_data

        try:
            user = self.twitter_client.get_user(username=username)
            if not user.data:
                return None

            tweets = self.twitter_client.get_users_tweets(
                id=user.data.id,
                max_results=10,
                tweet_fields=['created_at', 'public_metrics', 'attachments'],
                media_fields=['url', 'preview_image_url'],
                expansions=['attachments.media_keys'],
                exclude=['retweets', 'replies']
            )

            if tweets.data:
                media_lookup = {m.media_key: m for m in (tweets.includes.get('media', []) or [])}
                processed_tweets = []

                for t in tweets.data:
                    tweet_data = {
                        'id': t.id,
                        'text': t.text,
                        'created_at': t.created_at.isoformat(),
                        'metrics': t.public_metrics,
                        'media': []
                    }

                    if hasattr(t, 'attachments') and t.attachments:
                        media_keys = t.attachments.get('media_keys', [])
                        for key in media_keys:
                            media = media_lookup.get(key)
                            if media:
                                media_url = media.url or media.preview_image_url
                                if media_url:
                                    media_path = self._download_media(media_url, f"twitter_{t.id}")
                                    if media_path:
                                        tweet_data['media'].append(str(media_path))

                    processed_tweets.append(tweet_data)

                data = {
                    'timestamp': datetime.now().isoformat(),
                    'tweets': processed_tweets
                }
                self.save_to_cache('twitter', username, data)
                return data

        except Exception as e:
            self.console.print(f"[red]Error fetching Twitter data: {str(e)}[/red]")
            return None

    def fetch_reddit_data(self, username: str, force_refresh: bool = False) -> Optional[Dict]:
        """Fetch Reddit user data with caching."""
        if not force_refresh:
            cached_data = self.load_cached_data('reddit', username)
            if cached_data:
                cache_age = datetime.now() - datetime.fromisoformat(cached_data['fetch_time'])
                if cache_age.days < 1:
                    self.console.print("[green]Using cached Reddit data[/green]")
                    return cached_data

        try:
            redditor = self.reddit.redditor(username)
            _ = redditor.created_utc  # Verify user exists
            
            submissions = list(self._fetch_submissions(redditor))
            comments = list(self._fetch_comments(redditor))
            
            subreddit_activity = {}
            for item in submissions + comments:
                subreddit = item['subreddit']
                subreddit_activity[subreddit] = subreddit_activity.get(subreddit, 0) + 1
            
            data = {
                'username': username,
                'fetch_time': datetime.now().isoformat(),
                'submissions': submissions,
                'comments': comments,
                'statistics': {
                    'total_submissions': len(submissions),
                    'total_comments': len(comments),
                    'top_subreddits': dict(sorted(
                        subreddit_activity.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5])
                }
            }
            
            self.save_to_cache('reddit', username, data)
            return data
            
        except Exception as e:
            self.console.print(f"[red]Error fetching Reddit data: {str(e)}[/red]")
            return None

    def _fetch_submissions(self, redditor: praw.models.Redditor) -> Generator[Dict, None, None]:
        """Fetch Reddit submissions."""
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Fetching submissions...", total=None)
            for submission in redditor.submissions.new(limit=100):
                progress.update(task, advance=1)
                yield {
                    'title': submission.title,
                    'selftext': submission.selftext[:1000],
                    'subreddit': submission.subreddit.display_name,
                    'created_utc': submission.created_utc,
                    'score': submission.score,
                    'num_comments': submission.num_comments
                }

    def _fetch_comments(self, redditor: praw.models.Redditor) -> Generator[Dict, None, None]:
        """Fetch Reddit comments."""
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Fetching comments...", total=None)
            for comment in redditor.comments.new(limit=100):
                progress.update(task, advance=1)
                yield {
                    'body': comment.body[:1000],
                    'subreddit': comment.subreddit.display_name,
                    'score': comment.score,
                    'created_utc': comment.created_utc
                }

    def analyze_data(self, platforms: Dict[str, str], question: str) -> str:
        """Analyze data from specified platforms using appropriate LLM."""
        combined_data = []
        
        for platform, username in platforms.items():
            if platform == 'twitter':
                data = self.fetch_twitter_data(username)
                if data:
                    combined_data.append(self._format_twitter_data(username, data))
            elif platform == 'reddit':
                data = self.fetch_reddit_data(username)
                if data:
                    combined_data.append(self._format_reddit_data(data))
        
        if not combined_data:
            return "Unable to fetch data from any platform."
        
        formatted_prompt = f"Analyze the following social media activity to answer: {question}\n\n"
        formatted_prompt += "\n\n".join(combined_data)
        
        try:
            # Use Gemini for Twitter-only analysis (handles images better)
            if len(platforms) == 1 and 'twitter' in platforms:
                response = self.gemini_model.generate_content(formatted_prompt)
                return response.text
            # Use OpenRouter (Claude) for Reddit-only or combined analysis
            else:
                response = self.openrouter_client.post(
                    "/chat/completions",
                    json={
                        "model": "anthropic/claude-3-sonnet",
                        "messages": [
                            {
                                "role": "system",
                                "content": "Analyze social media activity focusing on key patterns in behavior, interests, and communication style. Provide concise, data-driven insights based only on the provided data."
                            },
                            {
                                "role": "user",
                                "content": formatted_prompt
                            }
                        ]
                    }
                )
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
                
        except Exception as e:
            self.console.print(f"[red]Error during analysis: {str(e)}[/red]")
            return f"Error analyzing data: {str(e)}"

    def _format_twitter_data(self, username: str, data: Dict) -> str:
        """Format Twitter data for analysis."""
        content = [f"Twitter Data for @{username}:"]
        
        for tweet in data['tweets']:
            content.append(f"\nTweet: {tweet['text']}")
            metrics = tweet['metrics']
            content.append(f"Engagement: {metrics['like_count']} likes, "
                         f"{metrics['reply_count']} replies, "
                         f"{metrics['retweet_count']} retweets")
            
        return "\n".join(content)

    def _format_reddit_data(self, data: Dict) -> str:
        """Format Reddit data for analysis."""
        content = [f"Reddit Data for u/{data['username']}:"]
        
        stats = data['statistics']
        content.append(f"\nTotal Activity:")
        content.append(f"- Submissions: {stats['total_submissions']}")
        content.append(f"- Comments: {stats['total_comments']}")
        
        content.append("\nTop Subreddits:")
        for sub, count in stats['top_subreddits'].items():
            content.append(f"- r/{sub}: {count} posts/comments")
        
        # Add top submissions and comments
        for submission in sorted(data['submissions'], key=lambda x: x['score'], reverse=True)[:5]:
            content.append(f"\nSubmission in r/{submission['subreddit']}:")
            content.append(f"Title: {submission['title']}")
            if submission['selftext']:
                content.append(f"Content: {submission['selftext'][:200]}...")
            content.append(f"Score: {submission['score']}")
            for comment in sorted(data['comments'], key=lambda x: x['score'], reverse=True)[:5]:
             content.append(f"\nComment in r/{comment['subreddit']}:")
            content.append(f"Content: {comment['body'][:200]}...")
            content.append(f"Score: {comment['score']}")
        
        return "\n".join(content)

    def interactive_session(self, platforms: Dict[str, str]):
        """Run interactive analysis session."""
        platform_str = " and ".join(f"{p}:@{u}" for p, u in platforms.items())
        
        self.console.print(Panel.fit(
            f"[bold blue]Social Media Analysis Session[/bold blue]\n"
            f"Analyzing: {platform_str}\n"
            "Commands: 'exit', 'refresh', 'help'",
            title="Social Media Analyzer",
            border_style="blue"
        ))
        
        while True:
            question = self.console.input("\n[bold cyan]What would you like to know about these accounts?[/bold cyan] ")
            
            match question.lower():
                case 'exit':
                    break
                case 'refresh':
                    refresh_success = True
                    for platform, username in platforms.items():
                        if platform == 'twitter':
                            if not self.fetch_twitter_data(username, force_refresh=True):
                                refresh_success = False
                        elif platform == 'reddit':
                            if not self.fetch_reddit_data(username, force_refresh=True):
                                refresh_success = False
                    
                    if refresh_success:
                        self.console.print("[green]All data refreshed![/green]")
                    else:
                        self.console.print("[red]Some data failed to refresh[/red]")
                    continue
                case 'help':
                    self.console.print(Panel(
                        "Commands:\n"
                        "exit - End session\n"
                        "refresh - Update all platform data\n"
                        "help - Show this message\n\n"
                        "Ask any question about the user's activity, interests, or behavior across platforms!",
                        title="Help",
                        border_style="blue"
                    ))
                    continue
                    
            try:
                self.console.print("[yellow]Analyzing...[/yellow]")
                analysis = self.analyze_data(platforms, question)
                self.console.print(Panel(Markdown(analysis), border_style="green"))
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")

def main():
    console = Console()
    
    console.print(Panel.fit(
        "[bold blue]Unified Social Media Analyzer[/bold blue]\n"
        "This tool analyzes user activity across Twitter and Reddit platforms.",
        border_style="blue"
    ))
    
    while True:
        console.print("\n[bold cyan]Available platforms:[/bold cyan]")
        console.print("1. Twitter")
        console.print("2. Reddit")
        console.print("3. Both platforms")
        console.print("4. Exit")
        
        choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4"])
        
        if choice == "4":
            break
            
        platforms = {}
        try:
            # Use context manager to ensure proper resource cleanup
            with UnifiedSocialAnalyzer() as analyzer:
                if choice in ["1", "3"]:
                    twitter_user = Prompt.ask("Enter Twitter username (without @)")
                    if analyzer.fetch_twitter_data(twitter_user):
                        platforms['twitter'] = twitter_user
                    else:
                        console.print("[red]Failed to fetch Twitter data[/red]")
                        if choice == "1":
                            continue
                
                if choice in ["2", "3"]:
                    reddit_user = Prompt.ask("Enter Reddit username (without u/)")
                    if analyzer.fetch_reddit_data(reddit_user):
                        platforms['reddit'] = reddit_user
                    else:
                        console.print("[red]Failed to fetch Reddit data[/red]")
                        if choice == "2":
                            continue
                
                if platforms:
                    analyzer.interactive_session(platforms)
                else:
                    console.print("[red]Failed to fetch data from any platform[/red]")
                    
        except KeyboardInterrupt:
            if Confirm.ask("\nDo you want to exit the program?"):
                break
            continue
        except Exception as e:
            console.print(f"[red]An error occurred: {str(e)}[/red]")
            if Confirm.ask("\nWould you like to try again?"):
                continue
            break
    
    console.print("\nThanks for using the Social Media Analysis tool!")

if __name__ == "__main__":
    main()