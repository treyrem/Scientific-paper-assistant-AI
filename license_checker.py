# license_checker.py
import os
import requests
import toml
import ast
from urllib.parse import urlparse

class LicenseDependencyChecker:
    def __init__(self):
        self.session = requests.Session()

    def _parse_repo(self, repo_url):
        parsed = urlparse(repo_url)
        parts = parsed.path.strip("/").split("/")
        return (parts[0], parts[1]) if len(parts) >= 2 else (None, None)

    def fetch_license(self, repo_url):
        owner, repo = self._parse_repo(repo_url)
        if not owner:
            return None
        api = f"https://api.github.com/repos/{owner}/{repo}/license"
        r = self.session.get(api); r.raise_for_status()
        data = r.json().get("license", {})
        return {
            "spdx_id": data.get("spdx_id"),
            "name":     data.get("name"),
            "url":      data.get("html_url")
        }

    def fetch_file(self, owner, repo, branch, path):
        raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
        r = self.session.get(raw); r.raise_for_status()
        return r.text

    def find_default_branch(self, owner, repo):
        api = f"https://api.github.com/repos/{owner}/{repo}"
        r = self.session.get(api); r.raise_for_status()
        return r.json().get("default_branch", "main")

    def list_tree(self, owner, repo, branch):
        api = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        r = self.session.get(api); r.raise_for_status()
        return [item["path"] for item in r.json().get("tree", [])]

    def extract_dependencies(self, repo_url):
        owner, repo = self._parse_repo(repo_url)
        if not owner:
            return []
        branch = self.find_default_branch(owner, repo)
        tree = self.list_tree(owner, repo, branch)

        # prefer pyproject.toml, else setup.py, else requirements.txt
        if "pyproject.toml" in tree:
            content = self.fetch_file(owner, repo, branch, "pyproject.toml")
            data = toml.loads(content)
            deps = data.get("project", {}).get("dependencies", [])
        elif "setup.py" in tree:
            content = self.fetch_file(owner, repo, branch, "setup.py")
            # naive AST parse of install_requires
            tree_ast = ast.parse(content)
            deps = []
            for node in ast.walk(tree_ast):
                if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "setup":
                    for kw in node.keywords:
                        if kw.arg == "install_requires":
                            deps = ast.literal_eval(kw.value)
                            break
        elif "requirements.txt" in tree:
            content = self.fetch_file(owner, repo, branch, "requirements.txt")
            deps = [line.strip() for line in content.splitlines() if line and not line.startswith("#")]
        else:
            deps = []

        # normalize and dedupe
        seen = set()
        clean = []
        for d in deps:
            pkg = d.split()[0]  # e.g. "numpy>=1.19" → "numpy>=1.19"
            if pkg not in seen:
                seen.add(pkg)
                clean.append(pkg)
        return clean
