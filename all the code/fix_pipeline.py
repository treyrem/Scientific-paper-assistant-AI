# fix_pipeline.py

import os
import sys
import torch
from pathlib import Path

# Make a backup of the original file
original_file = Path("pipeline_modules.py")
if original_file.exists():
    backup_file = Path("pipeline_modules.py.bak")
    if not backup_file.exists():  # Only backup if not already done
        original_file.rename(backup_file)
        print(f"Backed up {original_file} to {backup_file}")

# Prepare the fix
qa_generator_fix = """
    def extract_keywords(self, sections: Dict[str, PaperSection]) -> List[Dict]:
        \"\"\"Extract key concepts and provide additional information\"\"\"
        keywords_with_info = []

        # Combine important sections (limit text size to avoid CUDA errors)
        important_texts = []
        for section_type in ["abstract", "results", "conclusion"]:
            if section_type in sections:
                # Only take first 300 words from each section
                section_words = sections[section_type].content.split()[:300]
                important_texts.append(" ".join(section_words))

        all_text = " ".join(important_texts)

        # Use TF-IDF to find important terms
        sentences = nltk.sent_tokenize(all_text)
        if not sentences:
            return []

        vectorizer = TfidfVectorizer(
            max_features=20, stop_words="english", ngram_range=(1, 3)
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)

            feature_names = vectorizer.get_feature_names_out()
            top_indices = tfidf_matrix.sum(axis=0).A1.argsort()[-10:][::-1]

            # Force QA pipeline to CPU to avoid CUDA errors
            # FIXED: Use a proper PyTorch device object instead of integer
            device = torch.device("cpu")
            self.qa_pipeline.model = self.qa_pipeline.model.to(device)
            
            # Create a new pipeline with the correct device
            from transformers import pipeline
            self.qa_pipeline = pipeline(
                "question-answering", 
                model=self.qa_pipeline.model,
                device=device
            )

            # Get top keywords and their context
            for idx in top_indices:
                keyword = feature_names[idx]

                # Find sentences containing this keyword
                context_sentences = [
                    sent for sent in sentences if keyword.lower() in sent.lower()
                ]

                # Get additional information using QA
                if context_sentences:
                    # Limit context size to avoid token length issues
                    context = " ".join(context_sentences[:1])  # Use just 1 sentence

                    # Ask for definition/explanation
                    qa_result = self.qa_pipeline(
                        {"question": f"What is {keyword}?", "context": context}
                    )

                    keywords_with_info.append(
                        {
                            "keyword": keyword,
                            "context": context,
                            "definition": qa_result["answer"],
                            "confidence": qa_result["score"],
                        }
                    )
        except ValueError as e:
            self.logger.error(f"Error extracting keywords: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error in keyword extraction: {e}")
            return []

        return keywords_with_info
"""

# Also fix the summarize_section method in the Summarizer class
summarizer_fix = """
    def summarize_section(self, text: str, max_length: int = 150) -> str:
        \"\"\"Summarize a section of text\"\"\"
        # Check if text is too short for summarization
        words = text.split()
        if len(words) < 50:  # Too short to summarize
            return text  # Return original text

        # Adjust max_length for short texts
        actual_max_length = min(max_length, len(words) // 2)
        actual_min_length = min(20, actual_max_length - 10)

        try:
            # Split long texts into smaller chunks to avoid CUDA errors
            max_words = 400  # Reduced from 800 to avoid CUDA memory issues

            # FIXED: Use proper PyTorch device object
            device = torch.device("cpu")
            self.summarizer.model = self.summarizer.model.to(device)
            
            # Create a new pipeline with the correct device
            from transformers import pipeline
            self.summarizer = pipeline(
                "summarization",
                model=self.summarizer.model,
                tokenizer=self.tokenizer,
                device=device
            )

            if len(words) <= max_words:
                summaries = [
                    self.summarizer(
                        text, max_length=actual_max_length, min_length=actual_min_length
                    )[0]["summary_text"]
                ]
            else:
                chunks = [
                    " ".join(words[i : i + max_words])
                    for i in range(0, len(words), max_words // 2)
                ]
                summaries = []
                for chunk in chunks:
                    summary = self.summarizer(
                        chunk, max_length=actual_max_length // 2, min_length=10
                    )[0]["summary_text"]
                    summaries.append(summary)

            return " ".join(summaries)
        except Exception as e:
            self.logger.error(f"Error summarizing text: {e}")
            return text[:150] + "..."  # Return first 150 chars as fallback
"""

# Read the backup file
with open("pipeline_modules.py.bak", "r") as f:
    content = f.read()

# Replace the problematic methods with fixed versions
import re

qa_pattern = r"def extract_keywords\(self, sections:.*?return keywords_with_info"
content = re.sub(qa_pattern, qa_generator_fix.strip(), content, flags=re.DOTALL)

summarizer_pattern = r'def summarize_section\(self, text: str,.*?return text\[:150\] \+ "\.\.\."  # Return first 150 chars as fallback'
content = re.sub(summarizer_pattern, summarizer_fix.strip(), content, flags=re.DOTALL)

# Write the fixed content to the original file
with open("pipeline_modules.py", "w") as f:
    f.write(content)

print("Fixed pipeline_modules.py successfully!")
print("\nNow run your pipeline with:")
print(
    'python run_pipeline.py "C:\\LabGit\\Scientific-paper-assistant-AI\\papers\\BERT.pdf"'
)
print("\nFor better stability, avoid using GPU:")
print(
    'python run_pipeline.py "C:\\LabGit\\Scientific-paper-assistant-AI\\papers\\BERT.pdf" --use-publaynet'
)
