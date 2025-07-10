"""
Utility functions for parsing financial data from text.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_value(text_snippet: str, keyword: str) -> Optional[float]:
    """
    Helper function to extract a numerical value associated with a keyword from text.
    Designed to handle various formats including billions/millions/trillions.

    Args:
        text_snippet (str): The text to search within.
        keyword (str): The keyword to look for (e.g., "Revenue", "Net Income").

    Returns:
        Optional[float]: The extracted numerical value, or None if not found.
    """
    # Look for the keyword followed by a number, potentially with currency symbols or commas
    # This regex is simplified. For real-world data, more robust parsing is needed.
    # It also tries to capture values like "87.5 billion" or "123.4 million"
    match = re.search(
        rf"{re.escape(keyword)}[^.\d]*?([\$€£]?\s*[\d,]+\.?\d*(?:\s*(?:billion|million|trillion))?)",
        text_snippet,
        re.IGNORECASE
    )
    if match:
        value_str = match.group(1).lower().replace('$', '').replace('€', '').replace('£', '').replace(',', '').strip()
        multiplier = 1.0
        if "trillion" in value_str:
            multiplier = 1_000_000_000_000
            value_str = value_str.replace('trillion', '')
        elif "billion" in value_str:
            multiplier = 1_000_000_000
            value_str = value_str.replace('billion', '')
        elif "million" in value_str:
            multiplier = 1_000_000
            value_str = value_str.replace('million', '')
        
        try:
            return float(value_str) * multiplier
        except ValueError:
            logger.warning(f"Could not convert '{value_str}' to float for keyword '{keyword}'.")
            return None
    return None