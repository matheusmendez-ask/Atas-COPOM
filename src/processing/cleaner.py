"""Text sanitization and noise removal for Copom publications."""

import html
import logging
import re

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class AtaCleaner:
    """Sanitizer for raw Copom documents, removing HTML boilerplate and normalizing text."""

    def __init__(self) -> None:
        # Regex patterns for cleaning
        self._consecutive_spaces = re.compile(r"[ \t]+")
        self._consecutive_newlines = re.compile(r"\n{3,}")
        self._page_numbers = re.compile(
            r"\b(?:Página|Pág\.?)\s+\d+\s+(?:de\s+\d+)?\b", re.IGNORECASE
        )
        self._unwanted_footnotes = re.compile(r"\[\d+\]|\(\d+\)")

    def clean_html(self, raw_html_or_text: str) -> str:
        """Strip HTML tags while preserving logical paragraph and section breaks.

        Args:
            raw_html_or_text: The raw HTML content from the BCB API.

        Returns:
            Sanitized plain text document.
        """
        if not raw_html_or_text or not raw_html_or_text.strip():
            return ""

        # Decode HTML entities (e.g. &atilde;, &nbsp;)
        unescaped = html.unescape(raw_html_or_text)

        # Parse with BeautifulSoup
        soup = BeautifulSoup(unescaped, "html.parser")

        # Remove irrelevant and noise tags
        for element in soup(
            ["script", "style", "nav", "footer", "header", "noscript", "svg", "button"]
        ):
            element.decompose()

        # Add linebreaks before block elements to maintain document structure
        for block_tag in soup.find_all(
            ["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"]
        ):
            block_tag.insert_before("\n")
            block_tag.insert_after("\n")

        text = soup.get_text(separator=" ")

        # Perform string-level cleaning and normalization
        cleaned_text = self.normalize_text(text)
        return cleaned_text

    def normalize_text(self, text: str) -> str:
        """Apply deterministic cleaning rules to produce high-quality text for chunking."""
        if not text:
            return ""

        # Replace non-breaking spaces and zero-width spaces
        text = (
            text.replace("\xa0", " ")
            .replace("\u200b", "")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )

        # Remove noise patterns like page numbering
        text = self._page_numbers.sub("", text)

        # Process line by line
        lines = [self._consecutive_spaces.sub(" ", line).strip() for line in text.split("\n")]

        # Filter empty lines while preserving structural paragraph breaks
        cleaned_lines: list[str] = []
        for line in lines:
            if line:
                cleaned_lines.append(line)
            elif cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")

        result = "\n".join(cleaned_lines)
        # Collapse any 3+ newlines into 2
        result = self._consecutive_newlines.sub("\n\n", result).strip()

        return result
