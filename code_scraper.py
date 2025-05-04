import os
import requests
import json
from urllib.parse import urlparse
import openai

class CodeScraper:
    """
    Given a GitHub repository URL, fetches README, lists .py and .ipynb files,
    comments their code with step-by-step explanations, and returns both per-file comments and a summary.
    Uses unauthenticated GitHub API calls (no token required) and OpenAI v1.
    """
    def __init__(self, openai_api_key=None):
        self.session = requests.Session()
        key = openai_api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = key

    def _parse_repo(self, repo_url):
        parsed = urlparse(repo_url)
        parts = parsed.path.strip('/').split('/')
        if len(parts) < 2:
            return None, None
        return parts[0], parts[1]

    def _get_default_branch(self, owner, repo):
        api = f"https://api.github.com/repos/{owner}/{repo}"
        r = self.session.get(api); r.raise_for_status()
        return r.json().get('default_branch', 'main')

    def _get_tree(self, owner, repo, branch):
        api = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        r = self.session.get(api); r.raise_for_status()
        return r.json().get('tree', [])

    def list_files(self, repo_url):
        owner, repo = self._parse_repo(repo_url)
        if not owner:
            return [], [], None, None, None
        branch = self._get_default_branch(owner, repo)
        tree = self._get_tree(owner, repo, branch)
        readmes = [item['path'] for item in tree
                   if item['type']=='blob' and item['path'].lower().startswith('readme.')]
        code_files = [item['path'] for item in tree
                      if item['type']=='blob' and (item['path'].endswith('.py') or item['path'].endswith('.ipynb'))]
        return readmes, code_files, owner, repo, branch

    def _fetch(self, owner, repo, branch, path):
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
        r = self.session.get(url); r.raise_for_status()
        return r.text

    def extract_code_from_ipynb(self, content):
        data = json.loads(content)
        blocks = []
        for cell in data.get('cells', []):
            if cell.get('cell_type')=='code':
                blocks.append(''.join(cell.get('source', [])))
        return '\n'.join(blocks)

    def _chat(self, prompt, max_tokens=512):
        resp = openai.chat.completions.create(
            model='gpt-4',
            messages=[{'role':'user','content':prompt}],
            temperature=0.2,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()

    def comment_code(self, code):
        """
        Use OpenAI to generate a step-by-step explanation for code.
        Returns markdown with a fenced code block followed by a numbered list under 'Step-by-Step Explanation:'
        """
        prompt = (
            "Here is a Python code snippet. Do NOT modify the code."
            "\n\n"
            "1) First, output the code exactly, enclosed in a markdown ```python fenced code block."
            "\n"
            "2) Then, under a heading 'Step-by-Step Explanation:', provide a numbered list where each item explains a logical step or block of the code, referencing key function names or line ranges."
            "\n\n" + code
        )
        return self._chat(prompt, max_tokens=1024)

    def summarise_text(self, text, title="Summary"):
        prompt = (
            f"Please provide a concise summary for the following {title}:\n" + text
        )
        return self._chat(prompt, max_tokens=256)

    def scrape_and_comment(self, repo_url):
        readmes, files, owner, repo, branch = self.list_files(repo_url)
        results = {}
        # README summary
        if readmes:
            content = self._fetch(owner, repo, branch, readmes[0])
            results['README_summary'] = self.summarise_text(content, title='README')
        # Comment each code file
        for path in files:
            content = self._fetch(owner, repo, branch, path)
            code = self.extract_code_from_ipynb(content) if path.endswith('.ipynb') else content
            if not code.strip():
                continue
            results[path] = self.comment_code(code)
        # Overall comments summary
        all_comments = '\n'.join([v for k,v in results.items() if k!='README_summary'])
        results['comments_summary'] = self.summarise_text(all_comments, title='Code Comments')
        return results
