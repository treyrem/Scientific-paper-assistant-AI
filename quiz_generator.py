# quiz_generator.py
# Generates a quiz from a paper analysis JSON file using OpenAI API

import os
import re
import json
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# --- OpenAI Integration ---
try:
    from openai import OpenAI, OpenAIError
except ImportError:
    OpenAI = None
    OpenAIError = None
    logging.warning("OpenAI library not found. pip install openai")
# --- End OpenAI Integration ---

# --- .env File Loading ---
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
    logging.warning(
        "python-dotenv library not found. pip install python-dotenv. API key cannot be loaded from .env file."
    )
# --- End .env File Loading ---

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- OpenAI Quiz Generation Function ---


def generate_quiz_openai(
    analysis_data: Dict[str, Any],
    openai_client: Optional[OpenAI],
    openai_model: str,
    num_questions: int = 5,
    paper_title: Optional[str] = "the paper",
) -> Optional[List[Dict]]:
    """Uses OpenAI API to generate quiz questions from extracted summaries."""

    if not openai_client:
        logger.error("OpenAI client not available. Cannot generate quiz.")
        return None

    logger.info(
        f"Generating {num_questions} quiz questions for '{paper_title}' using OpenAI model: {openai_model}"
    )

    # --- Prepare Input Text from Extracted Summaries ---
    input_parts = []
    key_sections = ["abstract", "introduction", "methods", "results", "conclusion"]
    for sec in key_sections:
        summary_key = f"{sec}_summary"
        # Use the final synthesized summaries if available, otherwise fallback to extractive
        content_to_use = analysis_data.get(summary_key)
        if not content_to_use and sec in analysis_data.get("sections", {}):
            # Fallback to original section content if summary is missing
            content_to_use = analysis_data["sections"][sec].get("content")

        if content_to_use:
            # Add section name for context, use original content if summary missing
            input_parts.append(f"{sec.capitalize()} Content/Summary:\n{content_to_use}")

    if not input_parts:
        logger.error(
            "No extracted summaries or section content found in the analysis JSON to generate quiz from."
        )
        return None

    input_text = "\n\n---\n\n".join(input_parts)
    logger.debug(
        f"Input text for OpenAI prompt (first 500 chars):\n{input_text[:500]}..."
    )

    # --- Construct Prompt ---
    prompt = f"""Based on the following extracted content/summaries from a scientific paper titled "{paper_title}", generate exactly {num_questions} multiple-choice quiz questions to test understanding of the paper's key aspects (e.g., main topic, methods, findings, contributions).

For each question:
1. Start the question with the question number followed by a period (e.g., "1.").
2. Provide the question text immediately after the number and period.
3. On separate lines below the question, provide exactly 4 answer choices, each starting with A), B), C), or D) followed by the choice text.
4. On a separate line after the choices, clearly indicate the correct answer using the format "Correct Answer: [Letter]" (e.g., "Correct Answer: C").
5. Ensure questions and answers are directly supported by the provided text.

Extracted Content/Summaries:
---
{input_text}
---

Generate Quiz Questions:
"""

    # --- Call OpenAI API ---
    try:
        response = openai_client.chat.completions.create(
            model=openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant skilled at creating multiple-choice quizzes based on scientific text summaries, following specific formatting instructions.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=1500,  # Increased slightly
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        quiz_text = response.choices[0].message.content.strip()
        logger.info("OpenAI quiz generation successful.")
        logger.debug(f"Raw OpenAI response:\n{quiz_text}")

        # --- Parse the Response (Improved Parsing) ---
        quiz_data = []
        # Split into potential question blocks based on the number and period pattern
        question_blocks = re.split(r"\n\s*(?=\d+\.\s)", quiz_text)

        for block in question_blocks:
            block = block.strip()
            if not block:
                continue

            # ** Refined Question Regex: Capture text until the first choice marker **
            # Match "1. Question Text..." up to the line starting with A) or B) etc.
            question_match = re.match(
                r"^\d+\.\s*(.*?)(?=\n\s*[A-D]\))", block, re.DOTALL | re.IGNORECASE
            )
            if not question_match:
                logger.warning(f"Could not parse question text from block:\n{block}")
                continue
            question = question_match.group(1).strip()

            choices = {"A": None, "B": None, "C": None, "D": None}
            # ** Refined Choice Regex: Capture text until next choice, Correct Answer line, or end of block **
            # Use non-greedy match .*? and lookahead for multiple terminators
            choice_matches = re.findall(
                r"([A-D])\)\s*(.*?)(?=\n\s*[A-D]\)|\n\s*Correct Answer:|$)",
                block,
                re.DOTALL | re.IGNORECASE,
            )
            parsed_choices = 0
            for letter, text in choice_matches:
                letter_upper = letter.upper()
                if letter_upper in choices:
                    choices[letter_upper] = text.strip()
                    parsed_choices += 1

            # ** Refined Correct Answer Regex: More robust search **
            correct_answer_match = re.search(
                r"Correct Answer:\s*([A-D])", block, re.IGNORECASE
            )
            correct_answer = (
                correct_answer_match.group(1).upper() if correct_answer_match else None
            )

            # Validate parsing
            if (
                question
                and parsed_choices == 4
                and all(choices.values())
                and correct_answer
            ):
                quiz_data.append(
                    {
                        "question": question,
                        "choices": choices,
                        "correct_answer": correct_answer,
                    }
                )
            else:
                logger.warning(
                    f"Could not fully parse question block (Q:{question is not None}, Choices:{parsed_choices}/4, Ans:{correct_answer is not None}):\n{block}"
                )

        if not quiz_data:
            logger.error(
                "Failed to parse any valid questions from the OpenAI response."
            )
            return None

        # Limit to the requested number of questions if more were generated (unlikely with prompt change)
        return quiz_data[:num_questions]

    except OpenAIError as e:
        logger.error(f"OpenAI API error during quiz generation: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during OpenAI quiz generation: {e}")

    return None  # Return None if quiz generation fails


# --- Main Execution ---
def main():
    """Main function to load analysis and generate quiz."""
    parser = argparse.ArgumentParser(
        description="Generate a quiz from a paper analysis JSON file using OpenAI."
    )
    parser.add_argument(
        "analysis_json_path", help="Path to the input paper analysis JSON file"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file path for the quiz (default: alongside input with _quiz.json suffix)",
        default=None,
    )
    parser.add_argument(
        "--num-questions",
        "-n",
        type=int,
        default=5,
        help="Number of quiz questions to generate",
    )
    parser.add_argument(
        "--openai-model",
        help="OpenAI model to use for quiz generation",
        default="gpt-3.5-turbo",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)

    # --- Load API Key from .env file ---
    openai_api_key = None
    if load_dotenv:
        env_path = Path(
            r"C:\LabGit\Scientific-paper-assistant-AI\api_keys\OPEN_AI_KEY.env"
        ).resolve()
        if env_path.is_file():
            logger.info(f"Loading OpenAI API key from: {env_path}")
            load_dotenv(dotenv_path=env_path)
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                logger.warning(f"OPENAI_API_KEY not found in {env_path}.")
        else:
            logger.warning(f".env file not found at specified path: {env_path}")
    else:
        logger.warning("python-dotenv not installed. Cannot load key from .env file.")

    # Check prerequisites
    if not openai_api_key:
        logger.error(
            "OpenAI API key not found. Please set it in the .env file or environment variable. Exiting."
        )
        return
    if OpenAI is None:
        logger.error(
            "OpenAI library not installed. Please run 'pip install openai'. Exiting."
        )
        return
    if not os.path.exists(args.analysis_json_path):
        logger.error(f"Input analysis file not found: {args.analysis_json_path}")
        return

    # Set output path
    if not args.output:
        output_dir = os.path.dirname(args.analysis_json_path) or "."
        base_name = (
            os.path.splitext(os.path.basename(args.analysis_json_path))[0]
            .replace("_analysis_openai", "")
            .replace("_analysis_extractive", "")
        )
        args.output = os.path.join(output_dir, f"{base_name}_quiz.json")
    else:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

    # --- Load Analysis Data ---
    try:
        with open(args.analysis_json_path, "r", encoding="utf-8") as f:
            analysis_data = json.load(f)
        logger.info(f"Successfully loaded analysis data from {args.analysis_json_path}")
    except json.JSONDecodeError:
        logger.error(
            f"Error decoding JSON from {args.analysis_json_path}. Ensure it's a valid JSON file."
        )
        return
    except Exception as e:
        logger.error(f"Error loading analysis file {args.analysis_json_path}: {e}")
        return

    # --- Initialize OpenAI Client ---
    openai_client = None
    try:
        openai_client = OpenAI(api_key=openai_api_key)
        logger.info(f"OpenAI client initialized for model: {args.openai_model}")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return  # Cannot proceed without client

    # --- Generate Quiz ---
    paper_title = analysis_data.get("title", "the paper")
    quiz_questions = generate_quiz_openai(
        analysis_data=analysis_data,
        openai_client=openai_client,
        openai_model=args.openai_model,
        num_questions=args.num_questions,
        paper_title=paper_title,
    )

    # --- Save Quiz ---
    if quiz_questions:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(quiz_questions, f, indent=2)
            logger.info(f"Quiz successfully saved to: {args.output}")
            print(f"\nQuiz generated and saved to: {args.output}")
            # Optionally print quiz to console
            print("\n--- Generated Quiz ---")
            for i, q in enumerate(quiz_questions):
                print(f"\n{i+1}. {q['question']}")
                for choice, text in q["choices"].items():
                    print(f"   {choice}) {text}")
                print(f"   Correct Answer: {q['correct_answer']}")
            print("--------------------")

        except Exception as e:
            logger.error(f"Error saving quiz to file {args.output}: {e}")
    else:
        logger.error("Quiz generation failed.")
        print("\nQuiz generation failed. Check logs for details.")


if __name__ == "__main__":
    main()
