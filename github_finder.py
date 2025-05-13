import os
import requests
import difflib

class MinimalRepoFinder:
    """
    Finds the official GitHub repository for a given paper title.
    Uses the GitHub Search API and validates the found link.
    """
    def __init__(self, github_token=None):
        self.session = requests.Session()
        # Optionally use a GitHub token to increase rate limits
        # token = github_token or os.getenv("GITHUB_TOKEN")
        # if token:
        #     self.session.headers.update({'Authorization': f'token {token}'})
        self.search_url = "https://api.github.com/search/repositories"

    def find_repo(self, title: str) -> str:
        # Build the search query: look in repo name and description
        query = f"{title} in:name,description"
        params = {
            'q': query,
            'sort': 'stars',
            'order': 'desc',
            'per_page': 5
        }
        try:
            response = self.session.get(self.search_url, params=params, timeout=10)
            response.raise_for_status()
        except requests.RequestException:
            return None
        items = response.json().get('items', [])
        if not items:
            return None
        # Choose best match by fuzzy similarity between title and repo name
        best = None
        best_score = 0.0
        for repo in items:
            name = repo.get('name', '')
            score = difflib.SequenceMatcher(None, title.lower(), name.lower()).ratio()
            if score > best_score:
                best_score = score
                best = repo
        if best:
            url = best.get('html_url')
            if self._validate_url(url):
                return url
        return None

    def _validate_url(self, url: str) -> bool:
        try:
            resp = self.session.head(url, allow_redirects=True, timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
