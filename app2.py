import streamlit as st
from github_finder import MinimalRepoFinder
from code_scraper import CodeScraper
import tempfile
from extract_title import extract_paper_title
import os
import streamlit as st
from license_checker import LicenseDependencyChecker

st.set_page_config(page_title="Paper Explorer", page_icon="🔍")
st.title("Research Paper Explorer")

uploaded_file = st.file_uploader("Upload a PDF of a research paper", type=["pdf"])
if not uploaded_file:
    st.info("Please upload a PDF to begin.")
    st.stop()

tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
tmp.write(uploaded_file.read())
pdf_path = tmp.name
paper_title = extract_paper_title(pdf_path)
st.subheader("Extracted Title")
st.info(paper_title if paper_title else "Title not found")

repo_finder = MinimalRepoFinder()
with st.spinner("🔍 Searching GitHub for repository..."):
    repo_url = repo_finder.find_repo(paper_title)

if not repo_url:
    st.warning("No GitHub repository found for this paper.")
    os.unlink(pdf_path)
    st.stop()

# Display repo link
st.success("Repository found!")
st.markdown(f"**GitHub Repository:** [{repo_url}]({repo_url})")

checker = LicenseDependencyChecker()

# Scrape and comment code
scraper = CodeScraper()
with st.spinner("🛠️ Scraping and commenting code..."):
    results = scraper.scrape_and_comment(repo_url)

    # README summary
    if results.get('README_summary'):
        with st.expander("📄 README Summary", expanded=False):
            st.info(results['README_summary'])

    # Per-file step-by-step explanations
    for path, commented_md in results.items():
        if path in ('README_summary', 'comments_summary'):
            continue
        with st.expander(f"🗂️ {path}", expanded=False):
            st.markdown(commented_md)

    # Overall comments summary
    if results.get('comments_summary'):
        with st.expander("📝 Code summary", expanded=False):
            st.info(results['comments_summary'])

    #Check license
    st.subheader("📜 License")
    license_info = checker.fetch_license(repo_url)
    if license_info and license_info["spdx_id"]:
        badge_url = f"https://img.shields.io/github/license/{license_info['spdx_id']}/{license_info['spdx_id']}"
        # better: shields supports: https://img.shields.io/github/license/owner/repo
        badge_url = f"https://img.shields.io/github/license/{license_info['spdx_id']}"
        st.image(badge_url, width=150)
        st.info(f"{license_info['name']} — [View on GitHub]({license_info['url']})")
    else:
        st.warning("No license file detected")

    # 2) Dependency table
    st.subheader("📦 Dependencies")
    deps = checker.extract_dependencies(repo_url)
    if deps:
        # build a markdown table
        md = "| Package | PyPI Link |\n|---|---|\n"
        for pkg in deps:
            name = pkg.split(">=")[0].split("==")[0].split("<")[0]
            link = f"https://pypi.org/project/{name}/"
            md += f"| `{pkg}` | [PyPI]({link}) |\n"
        st.markdown(md, unsafe_allow_html=True)
    else:
        st.info("No dependencies detected")


# Clean up
os.unlink(pdf_path)

