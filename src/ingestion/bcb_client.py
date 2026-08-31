"""Resilient HTTP client for Banco Central do Brasil (BCB) API.

Provides rate-limiting handling, exponential backoff with Tenacity,
and parsing for Copom minutes and official publications.
"""

import logging
from typing import Any

import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings

logger = logging.getLogger(__name__)


class BCBClientError(Exception):
    """Base exception for BCB API client errors."""

    pass


class BCBClient:
    """Client for interacting with the Banco Central do Brasil APIs."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        backoff_factor: float | None = None,
    ) -> None:
        self.base_url = (base_url or settings.BCB_API_BASE_URL).rstrip("/")
        self.odata_url = settings.BCB_ODATA_URL
        self.timeout = timeout or settings.BCB_REQUEST_TIMEOUT
        self.max_retries = max_retries or settings.BCB_MAX_RETRIES
        self.backoff_factor = backoff_factor or settings.BCB_BACKOFF_FACTOR

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
            }
        )

    def _build_retry_decorator(self):
        """Construct a dynamic Tenacity retry decorator based on client settings."""
        return retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=self.backoff_factor, min=1, max=30),
            retry=retry_if_exception_type(
                (
                    requests.RequestException,
                    requests.Timeout,
                    requests.ConnectionError,
                    BCBClientError,
                )
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )

    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute resilient HTTP GET request with retries and status code validation."""

        @self._build_retry_decorator()
        def _execute_request() -> dict[str, Any]:
            logger.debug(f"Executing GET request to {url} with params {params}")
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code == 429:
                    logger.warning(f"Rate limit hit (429) at {url}. Backing off...")
                    raise BCBClientError("Rate limit encountered from BCB API (HTTP 429)")
                if response.status_code >= 500:
                    logger.warning(f"Server error ({response.status_code}) at {url}. Retrying...")
                    raise BCBClientError(f"BCB Server error HTTP {response.status_code}")

                response.raise_for_status()
                return response.json()
            except requests.exceptions.JSONDecodeError as err:
                logger.error(f"Failed to parse JSON response from {url}: {err}")
                raise BCBClientError(f"Invalid JSON payload from BCB API: {err}") from err
            except requests.RequestException as err:
                logger.warning(f"Request failed: {err}. Retrying if attempts remain...")
                raise

        return _execute_request()

    def fetch_atas_list(self, quantidade: int = 15) -> list[dict[str, Any]]:
        """Fetch the catalog list of latest Copom meeting minutes.

        Args:
            quantidade: Number of recent meetings to retrieve.

        Returns:
            List of raw meeting catalog objects.
        """
        endpoint = f"{self.base_url}/copom/atas"
        params = {"quantidade": quantidade}
        data = self._get(endpoint, params=params)

        items = data.get("conteudo", [])
        logger.info(f"Retrieved catalog with {len(items)} Copom meetings from BCB API.")
        return items

    def fetch_ata_details(self, nro_reuniao: int) -> dict[str, Any] | None:
        """Fetch full details and text for a specific Copom meeting.

        Args:
            nro_reuniao: The meeting number (e.g. 280, 279, ...).

        Returns:
            Dictionary containing meeting details and full HTML text, or None if not found.
        """
        endpoint = f"{self.base_url}/copom/atas_detalhes"
        params = {"nro_reuniao": nro_reuniao}
        data = self._get(endpoint, params=params)

        conteudo = data.get("conteudo", [])
        if not conteudo:
            logger.warning(f"No details found for meeting #{nro_reuniao}")
            return None

        return conteudo[0]

    def fetch_odata_publications(self, top: int = 10, skip: int = 0) -> list[dict[str, Any]]:
        """Fetch publications from the BCB OData service if accessible.

        Args:
            top: Number of items to retrieve.
            skip: Offset for pagination.

        Returns:
            List of publications from the OData response.
        """
        params = {
            "$top": top,
            "$skip": skip,
            "$format": "json",
        }
        data = self._get(self.odata_url, params=params)
        return data.get("value", [])
