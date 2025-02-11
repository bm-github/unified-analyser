import os
import sys
import json
import time
import hashlib
import logging
import argparse
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union
from functools import lru_cache
import httpx
import tweepy
import praw
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TaskID
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
import base64
from urllib.parse import quote_plus
from PIL import Image
import io

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analyser.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SocialAnalyser')

class RateLimitExceededError(Exception):
    pass

class SocialAnalyser:
    def __init__(self):
        self.console = Console()
        self.base_dir = Path("data")
        self._setup_directories()
        
        self.progress = Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            transient=True,
            console=self.console,
            refresh_per_second=10
        )
        self.current_task: Optional[TaskID] = None
        self._analysis_response: Optional[httpx.Response] = None
        self._analysis_exception: Optional[Exception] = None

    def _setup_directories(self):
        for dir_name in ['cache', 'media', 'outputs']:
            (self.base_dir / dir_name).mkdir(parents=True, exist_ok=True)

    @property
    def openrouter(self):
        if not hasattr(self, '_openrouter'):
            try:
                self._openrouter = httpx.Client(
                    base_url="https://openrouter.ai/api/v1",
                    headers={
                        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                        "HTTP-Referer": "http://localhost:3000",
                        "X-Title": "Social Media Analyser",
                        "Content-Type": "application/json"
                    },
                    timeout=30.0
                )
            except KeyError as e:
                raise RuntimeError(f"Missing OpenRouter API key: {e}")
        return self._openrouter

    @property
    def reddit(self):
        if not hasattr(self, '_reddit'):
            try:
                self._reddit = praw.Reddit(
                    client_id=os.environ['REDDIT_CLIENT_ID'],
                    client_secret=os.environ['REDDIT_CLIENT_SECRET'],
                    user_agent=os.environ['REDDIT_USER_AGENT']
                )
            except KeyError as e:
                raise RuntimeError(f"Missing Reddit credentials: {e}")
        return self._reddit

    @property
    def twitter(self):
        if not hasattr(self, '_twitter'):
            try:
                self._twitter = tweepy.Client(
                    bearer_token=os.environ['TWITTER_BEARER_TOKEN']
                )
            except KeyError as e:
                raise RuntimeError(f"Missing Twitter credentials: {e}")
        return self._twitter

    def _handle_rate_limit(self, platform: str):
        """Strict rate limit handler that raises an exception"""
        error_message = f"{platform} API rate limit exceeded"
        self.console.print(Panel(
            f"[bold red]Rate Limit Blocked: {platform}[/bold red]\n"
            "API requests are currently unavailable.",
            title="🚫 Rate Limit",
            border_style="red"
        ))
        raise RateLimitExceededError(error_message)

    def _get_media_path(self, url: str, platform: str, username: str) -> Path:
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.base_dir / 'media' / f"{platform}_{username}_{url_hash}.jpg"

    def _download_media(self, url: str, platform: str = None, username: str = None, headers: Optional[dict] = None) -> Optional[bytes]:
        if not url or not isinstance(url, str) or not url.startswith('http'):
            self.console.print(f"[red]Invalid URL: {url}[/red]")
            return None
        
        media_path = self._get_media_path(url, platform, username)
        
        if media_path.exists():
            return media_path.read_bytes()
        
        valid_types = [
            'image/jpeg', 'image/png', 'image/gif', 'image/webp',
            'image/svg+xml', 'video/mp4', 'video/quicktime',
            'application/octet-stream', 'binary'
        ]

        for attempt in range(3):
            try:
                logger.info(f"Downloading media from {url}")
                resp = httpx.get(url, headers=headers, timeout=10, follow_redirects=True)
                resp.raise_for_status()
                
                content_type = resp.headers.get('content-type', '').lower()
                
                if not any(ct in content_type for ct in valid_types):
                    logger.warning(f"Unsupported content type: {content_type}")
                    return None
                    
                media_data = resp.content
                media_path.write_bytes(media_data)
                return media_data
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get('Retry-After', '60'))
                    reset_time = datetime.now() + timedelta(seconds=retry_after)
                    reset_time_str = reset_time.strftime("%H:%M:%S")
                    
                    self.console.print(Panel(
                        f"[bold red]Rate Limit Exceeded[/bold red]\n"
                        f"Platform media download is temporarily blocked.\n"
                        f"[yellow]Rate limit resets at: {reset_time_str}[/yellow]\n"
                        f"Wait time: {retry_after} seconds",
                        title="⏳ Rate Limit",
                        border_style="red"
                    ))
                    
                    time.sleep(retry_after)
                    continue
                
                logger.error(f"HTTP error {e.response.status_code}")
            except Exception as e:
                logger.error(f"Download error: {str(e)}")
            
            if attempt < 2:
                logger.info(f"Retrying download ({attempt+1}/3)")
                time.sleep(5)
        
        logger.error("Failed to download media after 3 attempts")
        return None

    def analyse(self, platforms: Dict[str, Union[str, List[str]]], query: str) -> str:
        collected_data = []
        media_analysis = []
        twitter_accounts = {}

        try:
            collect_task = self.progress.add_task("[cyan]Collecting data...", total=sum(len(v) if isinstance(v, list) else 1 for v in platforms.values()))
            
            for platform, usernames in platforms.items():
                if isinstance(usernames, str):
                    usernames = [usernames]
                
                for username in usernames:
                    if fetcher := getattr(self, f'fetch_{platform}', None):
                        try:
                            data = fetcher(username)
                        except RateLimitExceededError as e:
                            self.console.print(f"[red]{e}[/red]")
                            return f"Analysis aborted: {str(e)}"
                        
                        if data:
                            collected_data.append(self._format_text_data(platform, username, data))
                            
                            if platform == 'twitter':
                                media_analysis.extend(data.get('media_analysis', []))
                                twitter_accounts[username] = data
                            
                            self.progress.advance(collect_task)
            
            self.progress.remove_task(collect_task)

            if not collected_data:
                return "No data available for analysis"

            analysis_components = []
            
            if len(twitter_accounts) > 1:
                comparison = self._compare_twitter_accounts(twitter_accounts)
                analysis_components.append(f"## Twitter Account Comparison\n{comparison}")

            if media_analysis:
                analysis_components.append("## Media Analysis (Google Gemini)\n" + "\n".join(f"- {m}" for m in media_analysis))
            
            if collected_data:
                analysis_components.append("## Text Analysis\n" + "\n".join(collected_data))

            prompt = f"Analysis request: {query}\n\n" + "\n\n".join(analysis_components)

            model = os.getenv('ANALYSIS_MODEL', 'anthropic/claude-3.5-sonnet')
                        
            analysis_task = self.progress.add_task(f"[magenta]Final analysis...", total=None)
            try:
                api_thread = threading.Thread(
                    target=self._call_openrouter,
                    kwargs={
                        "json_data": {
                            "model": model,
                            "messages": [
                                {"role": "system", "content": "Integrate these analyses into a comprehensive report:"},
                                {"role": "user", "content": prompt}
                            ]
                        }
                    }
                )
                api_thread.start()
                
                while api_thread.is_alive():
                    api_thread.join(0.1)
                    self.progress.refresh()
                
                if self._analysis_exception:
                    raise self._analysis_exception
                    
                response = self._analysis_response
                response.raise_for_status()
            finally:
                self.progress.remove_task(analysis_task)
                self._analysis_response = None
                self._analysis_exception = None

            analysis = response.json()['choices'][0]['message']['content']
            return f"## Comprehensive Analysis Report\n\n{analysis}"

        except Exception as e:
            return f"Analysis failed: {str(e)}"

    def _analyse_image(self, image_data: bytes, context: str = "") -> Optional[str]:
        try:
            # Use PIL to detect image type
            with Image.open(io.BytesIO(image_data)) as img:
                format_lower = img.format.lower() if img.format else None
                
                if format_lower not in ['jpeg', 'png', 'webp']:
                    logger.warning(f"Unsupported image type: {format_lower}")
                    return None
                
                content_type = f"image/{format_lower}"
                base64_image = base64.b64encode(image_data).decode('utf-8')
                base64_string = f"data:{content_type};base64,{base64_image}"
            
            prompt = f"Analyse this image{' from ' + context if context else ''}. Describe the content and any notable elements."
            
            response = self.openrouter.post(
                "/chat/completions",
                json={
                    "model": "google/gemini-2.0-flash-lite-preview-02-05:free",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": base64_string}}
                        ]
                    }]
                }
            )
            response.raise_for_status()
            data = response.json()
            
            return data['choices'][0]['message']['content']
            
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            return None

    @lru_cache(maxsize=100)
    def _get_cache_path(self, platform: str, username: str) -> Path:
        return self.base_dir / 'cache' / f"{platform}_{username}.json"

    def _load_cache(self, platform: str, username: str) -> Optional[dict]:
        cache_path = self._get_cache_path(platform, username)
        if not cache_path.exists():
            return None

        try:
            data = json.loads(cache_path.read_text())
            cache_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now(timezone.utc) - cache_time < timedelta(hours=24):
                return data
        except Exception:
            cache_path.unlink(missing_ok=True)
        return None

    def _save_cache(self, platform: str, username: str, data: dict):
        data['timestamp'] = datetime.now(timezone.utc).isoformat()
        self._get_cache_path(platform, username).write_text(json.dumps(data, indent=2))

    def fetch_twitter(self, username: str, force=False) -> Optional[dict]:
        try:
            if not force and (cached := self._load_cache('twitter', username)):
                return cached

            try:
                user = self.twitter.get_user(username=username)
                tweets = self.twitter.get_users_tweets(
                    id=user.data.id,
                    max_results=5,
                    tweet_fields=['created_at', 'public_metrics', 'attachments'],
                    expansions=['attachments.media_keys'],
                    media_fields=['url', 'preview_image_url', 'type']
                )
            except tweepy.TooManyRequests:
                self._handle_rate_limit('Twitter')
                return None

            processed = []
            media_analysis = []
            
            for tweet in tweets.data or []:
                tweet_data = {
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at.isoformat(),
                    'metrics': tweet.public_metrics,
                    'media': []
                }

                if tweet.attachments and tweets.includes:
                    media_includes = tweets.includes.get('media', [])
                    for media_key in tweet.attachments.get('media_keys', []):
                        media = next((m for m in media_includes if m.media_key == media_key), None)
                        if media:
                            url = None
                            if media.type == 'photo':
                                url = getattr(media, 'url', None)
                            elif media.type in ['video', 'animated_gif']:
                                url = getattr(media, 'preview_image_url', None)

                            if url:
                                media_data = self._download_media(
                                    url=url,
                                    platform='twitter',
                                    username=username
                                )
                                if media_data and (analysis := self._analyse_image(media_data, f"Twitter user @{username}")):
                                    tweet_data['media'].append({
                                        'type': media.type,
                                        'analysis': analysis,
                                        'url': url
                                    })
                                    media_analysis.append(analysis)

                processed.append(tweet_data)

            data = {
                'tweets': processed,
                'media_analysis': media_analysis,
                'user_info': {
                    'id': user.data.id,
                    'name': user.data.name,
                    'username': user.data.username,
                    'created_at': user.data.created_at.isoformat() if user.data.created_at else None
                }
            }
            self._save_cache('twitter', username, data)
            return data

        except Exception as e:
            logger.error(f"Twitter fetch failed: {str(e)}")
            return None

    def fetch_reddit(self, username: str, force=False) -> Optional[dict]:
        try:
            try:
                redditor = self.reddit.redditor(username)
                submissions = list(redditor.submissions.new(limit=50))
            except praw.exceptions.PRAWException as e:
                if 'rate_limit' in str(e).lower():
                    self._handle_rate_limit('Reddit')
                raise

            redditor = self.reddit.redditor(username)
            submissions = []
            comments = []

            for submission in redditor.submissions.new(limit=50):
                submissions.append({
                    'title': submission.title,
                    'text': submission.selftext[:1000],
                    'subreddit': submission.subreddit.display_name,
                    'score': submission.score
                })

            for comment in redditor.comments.new(limit=50):
                comments.append({
                    'text': comment.body[:1000],
                    'subreddit': comment.subreddit.display_name,
                    'score': comment.score
                })

            subreddit_counts = {}
            for item in submissions + comments:
                subreddit_counts[item['subreddit']] = subreddit_counts.get(item['subreddit'], 0) + 1
            top_subreddits = dict(sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True)[:5])

            data = {
                'submissions': submissions,
                'comments': comments,
                'stats': {
                    'total_posts': len(submissions) + len(comments),
                    'top_subreddits': top_subreddits
                }
            }
            self._save_cache('reddit', username, data)
            return data

        except praw.exceptions.PRAWException as e:
            logger.error(f"Reddit fetch failed: {str(e)}")
            return None

    def fetch_hackernews(self, username: str, force=False) -> Optional[dict]:
        if not force and (cached := self._load_cache('hackernews', username)):
            return cached

        try:
            url = f"https://hn.algolia.com/api/v1/search?tags=author_{quote_plus(username)}&hitsPerPage=50"
            
            with httpx.Client() as client:
                response = client.get(url, timeout=15)
                response.raise_for_status()
                data = response.json()

            submissions = []
            for hit in data.get('hits', []):
                submissions.append({
                    'title': hit.get('title', '[No Title]'),
                    'url': hit.get('url', f"https://news.ycombinator.com/item?id={hit['objectID']}"),
                    'points': hit.get('points', 0),
                    'num_comments': hit.get('num_comments', 0),
                    'created_at': datetime.fromtimestamp(hit['created_at_i'], tz=timezone.utc).isoformat(),
                    'text': (hit.get('story_text') or hit.get('comment_text', ''))[:1000]
                })

            result = {
                'submissions': submissions,
                'stats': {
                    'total_submissions': len(submissions),
                    'average_points': sum(s['points'] for s in submissions)/len(submissions) if submissions else 0
                }
            }
            
            self._save_cache('hackernews', username, result)
            return result

        except Exception as e:
            logger.error(f"Hacker News fetch failed: {str(e)}")
            return None

    def _format_text_data(self, platform: str, username: str, data: dict) -> str:
        if platform == 'twitter':
            return f"Twitter data for @{username}:\n" + "\n".join(
                f"- Tweet: {t['text']}\n  Likes: {t['metrics']['like_count']}"
                for t in data.get('tweets', [])[:5]
            )
        
        if platform == 'reddit':
            submissions = "\n".join(
                f"- Post in r/{p['subreddit']}: {p['title']}\n  Score: {p['score']}"
                for p in data.get('submissions', [])[:3]
            )
            comments = "\n".join(
                f"- Comment in r/{c['subreddit']}: {c['text']}\n  Score: {c['score']}"
                for c in data.get('comments', [])[:3]
            )
            return f"Reddit data for u/{username}:\nSubmissions:\n{submissions}\nComments:\n{comments}"
        
        if platform == 'hackernews':
            return f"Hacker News data for {username}:\n" + "\n".join(
                f"- Submission: {s['title']}\n  Points: {s['points']}, Comments: {s['num_comments']}"
                for s in data.get('submissions', [])[:5]
            )

        return ""

    def _analyse_image_url(self, image_url: str) -> Optional[str]:
        try:
            image_data = self._download_media(url=image_url, headers=None)
            if not image_data:
                return None
                
            with Image.open(io.BytesIO(image_data)) as img:
                format_lower = img.format.lower() if img.format else None
                
                if format_lower in ['jpeg', 'png', 'webp']:
                    content_type = f'image/{format_lower}'
                else:
                    self.console.print(f"[yellow]Unsupported image type: {format_lower}[/yellow]")
                    return None
                
                base64_image = base64.b64encode(image_data).decode('utf-8')
                base64_string = f"data:{content_type};base64,{base64_image}"
            
            response = self.openrouter.post(
                "/chat/completions",
                json={
                    "model": "google/gemini-2.0-flash-lite-preview-02-05:free",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please analyse this image and describe what you see."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_string
                                }
                            }
                        ]
                    }]
                }
            )
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except Exception as e:
            self.console.print(f"[yellow]Media analysis failed: {str(e)}[/yellow]")
            return None

    def _validate_image(self, image_data: bytes) -> Optional[str]:
        """Validate image data and return its content type if valid."""
        try:
            with Image.open(io.BytesIO(image_data)) as img:
                format_lower = img.format.lower() if img.format else None
                if format_lower in ['jpeg', 'png', 'webp']:
                    return f'image/{format_lower}'
            return None
        except Exception:
            return None

    def _compare_twitter_accounts(self, accounts_data: Dict[str, dict]) -> str:
        comparison_prompt = "Compare these Twitter accounts and find commonalities:\n\n"
        
        for username, data in accounts_data.items():
            comparison_prompt += f"=== @{username} ===\n"
            comparison_prompt += "\n".join(
                f"- Tweet: {t['text']}" for t in data.get('tweets', [])[:10]
            )
            comparison_prompt += "\n\n"
        
        try:
            response = self.openrouter.post(
                "/chat/completions",
                json={
                    "model": "anthropic/claude-3.5-sonnet",
                    "messages": [
                        {"role": "system", "content": "Identify common themes, topics, and patterns between these accounts. Highlight similarities in content, tone, and engagement patterns."},
                        {"role": "user", "content": comparison_prompt}
                    ]
                }
            )
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Comparison failed: {str(e)}"

    def _call_openrouter(self, json_data: dict):
        try:
            self._analysis_response = self.openrouter.post("/chat/completions", json=json_data)
        except Exception as e:
            self._analysis_exception = e

    def _save_output(self, content: str, format_type: str = "markdown"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.base_dir / 'outputs'
        
        try:
            if format_type == "json":
                filename = output_dir / f"analysis_{timestamp}.json"
                data = {
                    "timestamp": timestamp,
                    "content": content,
                    "format": "json"
                }
                # Add encoding here
                filename.write_text(json.dumps(data, indent=2), encoding='utf-8')
            else:
                filename = output_dir / f"analysis_{timestamp}.md"
                # Add encoding here
                filename.write_text(content, encoding='utf-8')
            
            self.console.print(f"[green]Analysis saved to: {filename}[/green]")
        except Exception as e:
            self.console.print(f"[red]Failed to save output: {str(e)}[/red]")

    def run(self):
        self.console.print(Panel(
            "[bold blue]Unified Social Media Analyser[/bold blue]\n"
            "This tool analyses user activity across multiple platforms.",
            border_style="blue"
        ))
        
        while True:
            self.console.print("\n[bold cyan]Options:[/bold cyan]")
            self.console.print("1. Twitter Analysis\n2. Reddit Analysis\n3. HackerNews Analysis\n4. Custom Combination\n5. Exit")
        
            choice = Prompt.ask("Select").strip()
            if choice == "5":
                break
                
            try:
                platforms = {}
                
                if choice in ["1", "4"]:
                    twitter_users = Prompt.ask("Twitter usernames (comma-separated, without @)", default="").strip()
                    if twitter_users:
                        platforms['twitter'] = [u.strip() for u in twitter_users.split(',') if u.strip()]
                
                if choice in ["2", "4"]:
                    reddit_users = Prompt.ask("Reddit usernames (comma-separated, without u/)", default="").strip()
                    if reddit_users:
                        platforms['reddit'] = [u.strip() for u in reddit_users.split(',') if u.strip()]
                
                if choice in ["3", "4"]:
                    hn_users = Prompt.ask("HackerNews usernames (comma-separated)", default="").strip()
                    if hn_users:
                        platforms['hackernews'] = [u.strip() for u in hn_users.split(',') if u.strip()]
                
                if not platforms:
                    self.console.print("[yellow]No valid platforms selected[/yellow]")
                    continue

                self._run_analysis_loop(platforms)
            
            except KeyboardInterrupt:
                if Confirm.ask("\nExit program?"):
                    break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
                if not Confirm.ask("Try again?"):
                    break

    def process_stdin(self):
        try:
            input_data = json.load(sys.stdin)
            platforms = input_data.get("platforms", {})
            query = input_data.get("query", "")
            output_format = input_data.get("format", "markdown")
            
            if not platforms or not query:
                raise ValueError("Invalid input format")
                
            for platform, value in platforms.items():
                if platform in ['twitter', 'reddit', 'hackernews'] and isinstance(value, str):
                    platforms[platform] = [value]
                    
            if analysis := self.analyse(platforms, query):
                self._save_output(analysis, output_format)
                return
                
        except Exception as e:
            sys.stderr.write(f"Error: {str(e)}\n")
            sys.exit(1)

    def _run_analysis_loop(self, platforms: Dict[str, Union[str, List[str]]]):
        platform_labels = []
        if 'twitter' in platforms:
            platform_labels.append(f"Twitter: {', '.join(['@'+u for u in platforms['twitter']])}")
        if 'reddit' in platforms:
            platform_labels.append(f"Reddit: {', '.join(['u/'+u for u in platforms['reddit']])}")
        if 'hackernews' in platforms:
            platform_labels.append(f"HN: {', '.join(platforms['hackernews'])}")
        
        platform_info = " | ".join(platform_labels)
        
        self.console.print(Panel(
            f"Analyzing: {platform_info}\nCommands: exit, refresh, help",
            title="Analysis Session",
            border_style="cyan"
        ))

        while True:
            try:
                query = Prompt.ask("\nAnalysis query").strip()
                if not query:
                    continue
                
                if query.lower() == 'exit':
                    break
                if query.lower() == 'refresh':
                    with self.progress:
                        refresh_task = self.progress.add_task("[yellow]Refreshing data...", total=sum(len(v) if isinstance(v, list) else 1 for v in platforms.values()))
                        for platform, usernames in platforms.items():
                            if isinstance(usernames, str):
                                usernames = [usernames]
                            for username in usernames:
                                getattr(self, f'fetch_{platform}')(username, force=True)
                                self.progress.advance(refresh_task)
                        self.progress.remove_task(refresh_task)
                    self.console.print("[green]Data refreshed[/green]")
                    continue
                if query.lower() == 'help':
                    self.console.print(Panel(
                        "Available commands:\n"
                        "- exit: End current session\n"
                        "- refresh: Force fresh data fetch\n"
                        "- help: Show this help\n"
                        "- Any other text: Analysis query",
                        title="Help",
                        border_style="blue"
                    ))
                    continue

                with self.progress:
                    if analysis := self.analyse(platforms, query):
                        self.console.print(Panel(
                            Markdown(analysis),
                            border_style="green"
                        ))
                        if args.format:
                            self._save_output(analysis, args.format)

            except KeyboardInterrupt:
                if Confirm.ask("\nExit analysis session?"):
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Social Media Analyser")
    parser.add_argument('--stdin', action='store_true', help="Read input from stdin as JSON")
    parser.add_argument('--format', choices=['json', 'markdown'], default='markdown',help="Output format")
    args = parser.parse_args()

    analyser = SocialAnalyser()
    if args.stdin:
        analyser.process_stdin()
    else:
        analyser.run()