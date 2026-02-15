"""
PubMed E-utilities async client.

This module provides an async HTTP client for interacting with the NCBI
PubMed E-utilities API to search for and fetch biomedical literature.

Features:
- Async HTTP requests with httpx
- Rate limiting (respects NCBI limits)
- Retry logic with exponential backoff
- XML parsing for article metadata
- Configurable search queries

NCBI E-utilities documentation:
https://www.ncbi.nlm.nih.gov/books/NBK25499/

Requirements addressed:
- REQ-DM-01: Fetch publications from PubMed E-utilities API
- REQ-DM-02: Parse title, abstract, authors, MeSH terms
"""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import httpx
from lxml import etree
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

# PubMed E-utilities base URLs
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{EUTILS_BASE}/esearch.fcgi"
EFETCH_URL = f"{EUTILS_BASE}/efetch.fcgi"

# Rate limiting: NCBI allows 3 req/sec without API key, 10 with key
# We use conservative defaults to avoid being blocked
DEFAULT_RATE_LIMIT_DELAY = 0.34  # ~3 requests per second
API_KEY_RATE_LIMIT_DELAY = 0.11  # ~9 requests per second (with margin)

# Batch size for fetching articles (NCBI recommends max 200)
DEFAULT_BATCH_SIZE = 100

# Default search query for Ambroxol-Parkinson's research
DEFAULT_QUERY = (
    "(Ambroxol[Title/Abstract] OR GBA1[Title/Abstract] OR GCase[Title/Abstract] "
    "OR glucocerebrosidase[Title/Abstract]) "
    "AND (Parkinson[Title/Abstract] OR Parkinsonism[Title/Abstract] "
    "OR neurodegeneration[Title/Abstract] OR alpha-synuclein[Title/Abstract])"
)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Author:
    """Represents an article author."""

    last_name: str
    first_name: str = ""
    initials: str = ""
    affiliation: str = ""

    @property
    def full_name(self) -> str:
        """Return full name as 'First Last' or 'Last' if no first name."""
        if self.first_name:
            return f"{self.first_name} {self.last_name}"
        if self.initials:
            return f"{self.initials} {self.last_name}"
        return self.last_name

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary for JSON storage."""
        return {
            "name": self.full_name,
            "last_name": self.last_name,
            "first_name": self.first_name,
            "initials": self.initials,
            "affiliation": self.affiliation,
        }


@dataclass
class PubMedArticle:
    """
    Represents a parsed PubMed article.

    Contains all metadata extracted from PubMed XML response.
    """

    pmid: str
    title: str
    abstract: str | None = None
    authors: list[Author] = field(default_factory=list)
    journal: str = ""
    publication_date: date | None = None
    doi: str = ""
    mesh_terms: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    publication_types: list[str] = field(default_factory=list)

    @property
    def has_abstract(self) -> bool:
        """Check if article has an abstract."""
        return bool(self.abstract and self.abstract.strip())

    @property
    def author_names(self) -> list[str]:
        """Get list of author full names."""
        return [a.full_name for a in self.authors]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "authors": [a.to_dict() for a in self.authors],
            "journal": self.journal,
            "publication_date": self.publication_date,
            "doi": self.doi,
            "mesh_terms": self.mesh_terms,
            "keywords": self.keywords,
            "publication_types": self.publication_types,
        }


# =============================================================================
# Exceptions
# =============================================================================


class PubMedError(Exception):
    """Base exception for PubMed client errors."""

    pass


class PubMedRateLimitError(PubMedError):
    """Raised when rate limited by NCBI."""

    pass


class PubMedAPIError(PubMedError):
    """Raised when API returns an error response."""

    pass


class PubMedParseError(PubMedError):
    """Raised when XML parsing fails."""

    pass


# =============================================================================
# PubMed Client
# =============================================================================


class PubMedClient:
    """
    Async client for PubMed E-utilities API.

    Features:
    - Search for articles by query
    - Fetch article details by PMID
    - Automatic rate limiting
    - Retry with exponential backoff
    - XML parsing for metadata extraction

    Usage:
        async with PubMedClient() as client:
            # Search for articles
            pmids = await client.search("Ambroxol Parkinson", max_results=100)

            # Fetch article details
            articles = await client.fetch_articles(pmids)

            for article in articles:
                print(f"{article.pmid}: {article.title}")

    With API key (faster rate limit):
        client = PubMedClient(api_key="your_ncbi_api_key")
    """

    def __init__(
        self,
        api_key: str | None = None,
        rate_limit_delay: float | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout: float = 30.0,
    ):
        """
        Initialize PubMed client.

        Args:
            api_key: NCBI API key for higher rate limits (optional)
            rate_limit_delay: Delay between requests in seconds
            batch_size: Number of articles to fetch per request
            timeout: HTTP request timeout in seconds
        """
        self.api_key = api_key or getattr(settings, "ncbi_api_key", None)
        self.batch_size = batch_size
        self.timeout = timeout

        # Set rate limit delay based on API key presence
        if rate_limit_delay is not None:
            self.rate_limit_delay = rate_limit_delay
        elif self.api_key:
            self.rate_limit_delay = API_KEY_RATE_LIMIT_DELAY
        else:
            self.rate_limit_delay = DEFAULT_RATE_LIMIT_DELAY

        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0

    async def __aenter__(self) -> "PubMedClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get the HTTP client, raising if not initialized."""
        if self._client is None:
            raise RuntimeError(
                "PubMedClient must be used as async context manager: "
                "async with PubMedClient() as client: ..."
            )
        return self._client

    async def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = asyncio.get_event_loop().time()
        elapsed = current_time - self._last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()

    def _get_base_params(self) -> dict[str, str]:
        """Get base parameters for all API requests."""
        params = {"db": "pubmed", "retmode": "xml"}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, PubMedRateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _request(self, url: str, params: dict[str, Any]) -> str:
        """
        Make a rate-limited HTTP request with retry logic.

        Args:
            url: API endpoint URL
            params: Query parameters

        Returns:
            Response text (XML)

        Raises:
            PubMedRateLimitError: If rate limited (429)
            PubMedAPIError: If API returns error status
        """
        await self._rate_limit()

        logger.debug("PubMed API request", url=url, params=params)

        response = await self.client.get(url, params=params)

        if response.status_code == 429:
            logger.warning("PubMed rate limit hit, will retry")
            raise PubMedRateLimitError("Rate limit exceeded")

        if response.status_code != 200:
            logger.error(
                "PubMed API error",
                status_code=response.status_code,
                response=response.text[:500],
            )
            raise PubMedAPIError(
                f"API returned status {response.status_code}: {response.text[:200]}"
            )

        return response.text

    # =========================================================================
    # Search
    # =========================================================================

    async def search(
        self,
        query: str = DEFAULT_QUERY,
        max_results: int = 100,
        min_date: str | None = None,
        max_date: str | None = None,
    ) -> list[str]:
        """
        Search PubMed for articles matching a query.

        Args:
            query: PubMed search query (supports field tags like [Title/Abstract])
            max_results: Maximum number of PMIDs to return
            min_date: Minimum publication date (YYYY/MM/DD or YYYY)
            max_date: Maximum publication date (YYYY/MM/DD or YYYY)

        Returns:
            List of PMIDs matching the query

        Example:
            pmids = await client.search(
                "Ambroxol AND Parkinson",
                max_results=50,
                min_date="2020",
            )
        """
        params = self._get_base_params()
        params.update(
            {
                "term": query,
                "retmax": str(max_results),
                "usehistory": "n",
                "sort": "relevance",
            }
        )

        if min_date:
            params["mindate"] = min_date
            params["datetype"] = "pdat"  # Publication date
        if max_date:
            params["maxdate"] = max_date
            params["datetype"] = "pdat"

        logger.info("Searching PubMed", query=query[:100], max_results=max_results)

        xml_text = await self._request(ESEARCH_URL, params)

        # Parse XML response
        try:
            root = etree.fromstring(xml_text.encode())
        except etree.XMLSyntaxError as e:
            logger.error("Failed to parse search response", error=str(e))
            raise PubMedParseError(f"Failed to parse search XML: {e}") from e

        # Extract PMIDs
        pmids = [elem.text for elem in root.findall(".//Id") if elem.text]

        # Get total count for logging
        count_elem = root.find(".//Count")
        total_count = int(count_elem.text) if count_elem is not None and count_elem.text else 0

        logger.info(
            "Search completed",
            found=len(pmids),
            total_available=total_count,
            query=query[:50],
        )

        return pmids

    # =========================================================================
    # Fetch Articles
    # =========================================================================

    async def fetch_articles(
        self,
        pmids: list[str],
        progress_callback: callable | None = None,
    ) -> list[PubMedArticle]:
        """
        Fetch article details for a list of PMIDs.

        Args:
            pmids: List of PubMed IDs to fetch
            progress_callback: Optional callback(fetched, total) for progress

        Returns:
            List of PubMedArticle objects with full metadata

        Example:
            articles = await client.fetch_articles(["12345678", "23456789"])
            for article in articles:
                print(f"{article.pmid}: {article.title}")
        """
        if not pmids:
            return []

        articles = []
        total = len(pmids)

        # Process in batches
        for i in range(0, total, self.batch_size):
            batch_pmids = pmids[i : i + self.batch_size]

            logger.debug(
                "Fetching article batch",
                batch_start=i,
                batch_size=len(batch_pmids),
                total=total,
            )

            batch_articles = await self._fetch_batch(batch_pmids)
            articles.extend(batch_articles)

            if progress_callback:
                progress_callback(len(articles), total)

        logger.info("Fetched articles", count=len(articles), requested=total)

        return articles

    async def _fetch_batch(self, pmids: list[str]) -> list[PubMedArticle]:
        """Fetch a batch of articles by PMIDs."""
        params = self._get_base_params()
        params.update(
            {
                "id": ",".join(pmids),
                "rettype": "xml",
            }
        )

        xml_text = await self._request(EFETCH_URL, params)

        # Parse XML response
        try:
            root = etree.fromstring(xml_text.encode())
        except etree.XMLSyntaxError as e:
            logger.error("Failed to parse fetch response", error=str(e))
            raise PubMedParseError(f"Failed to parse fetch XML: {e}") from e

        # Parse each article
        articles = []
        for article_elem in root.findall(".//PubmedArticle"):
            try:
                article = self._parse_article(article_elem)
                articles.append(article)
            except Exception as e:
                # Log but continue processing other articles
                logger.warning("Failed to parse article", error=str(e))

        return articles

    def _parse_article(self, elem: etree._Element) -> PubMedArticle:
        """
        Parse a PubmedArticle XML element into a PubMedArticle object.

        Args:
            elem: lxml Element for <PubmedArticle>

        Returns:
            Parsed PubMedArticle object
        """
        # Get PMID
        pmid_elem = elem.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None and pmid_elem.text else ""

        if not pmid:
            raise PubMedParseError("Article missing PMID")

        # Get title
        title_elem = elem.find(".//ArticleTitle")
        title = self._get_text_content(title_elem) if title_elem is not None else ""

        # Get abstract
        abstract = self._parse_abstract(elem)

        # Get authors
        authors = self._parse_authors(elem)

        # Get journal
        journal_elem = elem.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None and journal_elem.text else ""

        # Get publication date
        pub_date = self._parse_publication_date(elem)

        # Get DOI
        doi = self._parse_doi(elem)

        # Get MeSH terms
        mesh_terms = self._parse_mesh_terms(elem)

        # Get keywords
        keywords = self._parse_keywords(elem)

        # Get publication types
        pub_types = self._parse_publication_types(elem)

        return PubMedArticle(
            pmid=pmid,
            title=title,
            abstract=abstract,
            authors=authors,
            journal=journal,
            publication_date=pub_date,
            doi=doi,
            mesh_terms=mesh_terms,
            keywords=keywords,
            publication_types=pub_types,
        )

    def _get_text_content(self, elem: etree._Element) -> str:
        """
        Get all text content from an element, including nested elements.

        This handles cases like:
        <ArticleTitle>Effects of <i>in vitro</i> treatment</ArticleTitle>
        """
        if elem is None:
            return ""
        # Get all text including tail text from children
        return "".join(elem.itertext()).strip()

    def _parse_abstract(self, elem: etree._Element) -> str | None:
        """Parse abstract, handling structured abstracts with sections."""
        abstract_elem = elem.find(".//Abstract")
        if abstract_elem is None:
            return None

        # Handle structured abstracts (multiple AbstractText elements)
        abstract_texts = abstract_elem.findall(".//AbstractText")
        if not abstract_texts:
            return None

        parts = []
        for text_elem in abstract_texts:
            label = text_elem.get("Label", "")
            text = self._get_text_content(text_elem)

            if label and text:
                parts.append(f"{label}: {text}")
            elif text:
                parts.append(text)

        return "\n\n".join(parts)

    def _parse_authors(self, elem: etree._Element) -> list[Author]:
        """Parse author list."""
        authors = []
        author_list = elem.find(".//AuthorList")
        if author_list is None:
            return authors

        for author_elem in author_list.findall(".//Author"):
            # Skip collective/group authors for simplicity
            if author_elem.get("ValidYN") == "N":
                continue

            last_name_elem = author_elem.find("LastName")
            if last_name_elem is None or not last_name_elem.text:
                continue

            first_name_elem = author_elem.find("ForeName")
            initials_elem = author_elem.find("Initials")
            affiliation_elem = author_elem.find(".//Affiliation")

            author = Author(
                last_name=last_name_elem.text,
                first_name=first_name_elem.text if first_name_elem is not None and first_name_elem.text else "",
                initials=initials_elem.text if initials_elem is not None and initials_elem.text else "",
                affiliation=affiliation_elem.text if affiliation_elem is not None and affiliation_elem.text else "",
            )
            authors.append(author)

        return authors

    def _parse_publication_date(self, elem: etree._Element) -> date | None:
        """Parse publication date, trying multiple date fields."""
        # Try ArticleDate first (electronic publication)
        article_date = elem.find(".//ArticleDate")
        if article_date is not None:
            parsed = self._parse_date_element(article_date)
            if parsed:
                return parsed

        # Try PubDate (journal publication)
        pub_date = elem.find(".//PubDate")
        if pub_date is not None:
            parsed = self._parse_date_element(pub_date)
            if parsed:
                return parsed

        return None

    def _parse_date_element(self, date_elem: etree._Element) -> date | None:
        """Parse a date element (PubDate or ArticleDate)."""
        year_elem = date_elem.find("Year")
        if year_elem is None or not year_elem.text:
            # Try MedlineDate format
            medline_date = date_elem.find("MedlineDate")
            if medline_date is not None and medline_date.text:
                # Extract year from MedlineDate like "2023 Jan-Feb"
                match = re.match(r"(\d{4})", medline_date.text)
                if match:
                    return date(int(match.group(1)), 1, 1)
            return None

        year = int(year_elem.text)

        month_elem = date_elem.find("Month")
        if month_elem is not None and month_elem.text:
            # Month can be number or name
            month_text = month_elem.text
            if month_text.isdigit():
                month = int(month_text)
            else:
                month = self._month_name_to_number(month_text)
        else:
            month = 1

        day_elem = date_elem.find("Day")
        day = int(day_elem.text) if day_elem is not None and day_elem.text else 1

        try:
            return date(year, month, day)
        except ValueError:
            # Invalid date, return just year
            return date(year, 1, 1)

    def _month_name_to_number(self, month_name: str) -> int:
        """Convert month name (Jan, Feb, etc.) to number."""
        months = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "may": 5, "jun": 6, "jul": 7, "aug": 8,
            "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        return months.get(month_name.lower()[:3], 1)

    def _parse_doi(self, elem: etree._Element) -> str:
        """Parse DOI from article IDs."""
        for id_elem in elem.findall(".//ArticleId"):
            if id_elem.get("IdType") == "doi" and id_elem.text:
                return id_elem.text
        return ""

    def _parse_mesh_terms(self, elem: etree._Element) -> list[str]:
        """Parse MeSH terms."""
        mesh_terms = []
        for mesh_elem in elem.findall(".//MeshHeading/DescriptorName"):
            if mesh_elem.text:
                mesh_terms.append(mesh_elem.text)
        return mesh_terms

    def _parse_keywords(self, elem: etree._Element) -> list[str]:
        """Parse keywords."""
        keywords = []
        for kw_elem in elem.findall(".//Keyword"):
            if kw_elem.text:
                keywords.append(kw_elem.text)
        return keywords

    def _parse_publication_types(self, elem: etree._Element) -> list[str]:
        """Parse publication types."""
        pub_types = []
        for pt_elem in elem.findall(".//PublicationType"):
            if pt_elem.text:
                pub_types.append(pt_elem.text)
        return pub_types

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def search_and_fetch(
        self,
        query: str = DEFAULT_QUERY,
        max_results: int = 100,
        min_date: str | None = None,
        max_date: str | None = None,
        progress_callback: callable | None = None,
    ) -> list[PubMedArticle]:
        """
        Search for articles and fetch their full details in one call.

        This is a convenience method that combines search() and fetch_articles().

        Args:
            query: PubMed search query
            max_results: Maximum number of articles to return
            min_date: Minimum publication date
            max_date: Maximum publication date
            progress_callback: Optional callback(fetched, total) for progress

        Returns:
            List of PubMedArticle objects

        Example:
            articles = await client.search_and_fetch(
                "Ambroxol GCase Parkinson",
                max_results=50,
            )
        """
        pmids = await self.search(
            query=query,
            max_results=max_results,
            min_date=min_date,
            max_date=max_date,
        )

        if not pmids:
            logger.warning("No articles found for query", query=query[:50])
            return []

        return await self.fetch_articles(pmids, progress_callback=progress_callback)


# =============================================================================
# Utility Functions
# =============================================================================


async def fetch_ambroxol_parkinson_articles(
    max_results: int = 100,
    api_key: str | None = None,
) -> list[PubMedArticle]:
    """
    Convenience function to fetch Ambroxol-Parkinson's research articles.

    Uses the default query targeting Ambroxol, GBA1, GCase, and Parkinson's
    disease research.

    Args:
        max_results: Maximum number of articles to fetch
        api_key: Optional NCBI API key for higher rate limits

    Returns:
        List of PubMedArticle objects

    Example:
        articles = await fetch_ambroxol_parkinson_articles(max_results=50)
        print(f"Found {len(articles)} articles")
    """
    async with PubMedClient(api_key=api_key) as client:
        return await client.search_and_fetch(
            query=DEFAULT_QUERY,
            max_results=max_results,
        )
