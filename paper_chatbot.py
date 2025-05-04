# paper_chatbot.py
# An AI chatbot to answer questions about a research paper using its analysis JSON.

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

# --- Helper Functions ---


def load_analysis(json_path: str) -> Optional[Dict]:
    """Loads the analysis data from a JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Successfully loaded analysis data from {json_path}")
        return data
    except FileNotFoundError:
        logger.error(f"Analysis file not found: {json_path}")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {json_path}. Ensure it's valid.")
    except Exception as e:
        logger.error(f"Error loading analysis file {json_path}: {e}")
    return None


def prepare_context_from_analysis(analysis_data: Dict, max_chars: int = 15000) -> str:
    """Extracts and concatenates section content to form context."""
    context_parts = []
    paper_title = analysis_data.get("title", "the paper")
    context_parts.append(f"Paper Title: {paper_title}")

    # Prioritize core sections for context
    core_sections = ["abstract", "introduction", "methods", "results", "conclusion"]
    sections_data = analysis_data.get("sections", {})

    for sec_type in core_sections:
        if sec_type in sections_data and sections_data[sec_type].get("content"):
            content = sections_data[sec_type]["content"]
            # Basic cleaning of content before adding
            content = re.sub(
                r"<PAGEBREAK NUM=\d+>", "", content
            )  # Remove page markers if present
            content = re.sub(r"\s+", " ", content).strip()  # Normalize whitespace
            context_parts.append(f"--- {sec_type.capitalize()} ---\n{content}")

    full_context = "\n\n".join(context_parts)

    # Truncate if context is too long (simple truncation)
    if len(full_context) > max_chars:
        logger.warning(f"Combined context exceeds {max_chars} characters. Truncating.")
        full_context = full_context[:max_chars] + "\n... [Context Truncated]"

    return full_context


def get_chatbot_response(
    client: OpenAI,
    model: str,
    conversation_history: List[Dict[str, str]],
    paper_context: str,
    user_question: str,
) -> Optional[str]:
    """Gets a response from the OpenAI chatbot."""

    system_prompt = f"""You are an AI assistant designed to answer questions about a specific scientific research paper based ONLY on the provided context below. Do not use any external knowledge or information outside this context. If the answer cannot be found in the context, state that clearly.

Provided Context from the Paper:
--- START CONTEXT ---
{paper_context}
--- END CONTEXT ---

Answer the user's questions based solely on the information within the START CONTEXT and END CONTEXT markers."""

    # Add system prompt and current user question to the history for the API call
    messages_for_api = (
        [{"role": "system", "content": system_prompt}]
        + conversation_history
        + [{"role": "user", "content": user_question}]
    )

    logger.debug(f"Sending {len(messages_for_api)} messages to OpenAI.")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages_for_api,
            temperature=0.3,  # Lower temperature for more factual, context-based answers
            max_tokens=500,  # Adjust as needed
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        assistant_response = response.choices[0].message.content.strip()
        logger.info("Received response from OpenAI.")
        return assistant_response
    except OpenAIError as e:
        logger.error(f"OpenAI API error during chat: {e}")
        return "Sorry, I encountered an error communicating with the AI model."
    except Exception as e:
        logger.error(f"Unexpected error during OpenAI chat: {e}")
        return "Sorry, an unexpected error occurred."


# --- Main Chat Loop ---
def run_chat(analysis_data: Dict, openai_client: OpenAI, openai_model: str):
    """Runs the main interactive chat loop."""

    paper_context = prepare_context_from_analysis(analysis_data)
    if not paper_context:
        print("Could not prepare context from analysis data. Exiting.")
        return

    paper_title = analysis_data.get("title", "the research paper")
    print(f"\nWelcome! Ask me questions about the paper: '{paper_title}'.")
    print("Type 'quit' or 'exit' to end the chat.")

    conversation_history: List[Dict[str, str]] = (
        []
    )  # Stores {"role": "user/assistant", "content": ...}

    while True:
        try:
            user_input = input("\nYour Question: ")
        except EOFError:  # Handle Ctrl+D
            print("\nExiting chat.")
            break

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if not user_input.strip():
            continue

        # Get response from OpenAI
        assistant_response = get_chatbot_response(
            client=openai_client,
            model=openai_model,
            conversation_history=conversation_history,
            paper_context=paper_context,
            user_question=user_input,
        )

        print(f"\nAssistant: {assistant_response}")

        # Add user query and assistant response to history
        if assistant_response:  # Only add if we got a response
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append(
                {"role": "assistant", "content": assistant_response}
            )

        # Optional: Limit history length to prevent exceeding token limits
        # e.g., keep only the last N turns (N*2 messages)
        max_history_messages = 10  # Keep last 5 turns
        if len(conversation_history) > max_history_messages:
            conversation_history = conversation_history[-max_history_messages:]


# --- Main Execution ---
def main():
    """Main function to load analysis and start chatbot."""
    parser = argparse.ArgumentParser(
        description="Chatbot to answer questions about a paper using its analysis JSON."
    )
    parser.add_argument(
        "analysis_json_path", help="Path to the input paper analysis JSON file"
    )
    parser.add_argument(
        "--openai-model",
        help="OpenAI model to use for the chatbot",
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

    # --- Load Analysis Data ---
    analysis_data = load_analysis(args.analysis_json_path)
    if not analysis_data:
        return

    # --- Initialize OpenAI Client ---
    openai_client = None
    try:
        openai_client = OpenAI(api_key=openai_api_key)
        logger.info(f"OpenAI client initialized for model: {args.openai_model}")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return  # Cannot proceed without client

    # --- Start Chat ---
    run_chat(analysis_data, openai_client, args.openai_model)


if __name__ == "__main__":
    main()
