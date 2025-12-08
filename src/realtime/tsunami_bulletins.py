"""Tsunami bulletin fetchers for PTWC and NTWC RSS feeds.

Fetches authoritative tsunami warnings and advisories from:
- Pacific Tsunami Warning Center (PTWC)
- National Tsunami Warning Center (NTWC)
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

try:
    import httpx
except ImportError:
    httpx = None

try:
    import feedparser  # type: ignore
except ImportError:
    feedparser = None

LOGGER = logging.getLogger(__name__)

# NOAA Tsunami RSS feeds
PTWC_RSS_URL = "https://www.tsunami.gov/events/xml/PHEBxml.xml"  # Pacific
NTWC_RSS_URL = "https://www.tsunami.gov/events/xml/WCEBxml.xml"  # West Coast/Alaska


def fetch_tsunami_bulletins(
    sources: Optional[List[str]] = None,
) -> List[Dict]:
    """Fetch tsunami bulletins from NOAA warning centers.
    
    Args:
        sources: List of sources to query. Options: ["ptwc", "ntwc"]
                Default: both sources
    
    Returns:
        List of bulletin dictionaries containing:
            - title: Bulletin title
            - summary: Bulletin summary/description
            - link: URL to full bulletin
            - published: Publication timestamp
            - source: Source warning center (PTWC or NTWC)
            - severity: Parsed severity level (if available)
    """
    if feedparser is None:
        raise ImportError("feedparser is required. Install with: pip install feedparser")
    
    if sources is None:
        sources = ["ptwc", "ntwc"]
    
    bulletins = []
    
    for source in sources:
        if source.lower() == "ptwc":
            url = PTWC_RSS_URL
            center = "PTWC"
        elif source.lower() == "ntwc":
            url = NTWC_RSS_URL
            center = "NTWC"
        else:
            LOGGER.warning(f"Unknown tsunami bulletin source: {source}")
            continue
        
        try:
            feed = feedparser.parse(url)
            
            if feed.bozo:
                LOGGER.warning(f"Error parsing {center} RSS feed: {feed.bozo_exception}")
                continue
            
            for entry in feed.entries:
                bulletin = {
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "link": entry.get("link", ""),
                    "published": _parse_published_date(entry),
                    "source": center,
                    "severity": _parse_severity(entry.get("title", "")),
                }
                bulletins.append(bulletin)
            
            LOGGER.info(f"Fetched {len(feed.entries)} bulletins from {center}")
            
        except Exception as exc:
            LOGGER.error(f"Failed to fetch bulletins from {center}: {exc}")
            continue
    
    # Sort by published date, most recent first
    bulletins.sort(key=lambda x: x["published"], reverse=True)
    
    return bulletins


def fetch_ptwc_bulletins() -> List[Dict]:
    """Fetch bulletins from Pacific Tsunami Warning Center.
    
    Returns:
        List of bulletin dictionaries
    """
    return fetch_tsunami_bulletins(sources=["ptwc"])


def fetch_ntwc_bulletins() -> List[Dict]:
    """Fetch bulletins from National Tsunami Warning Center.
    
    Returns:
        List of bulletin dictionaries
    """
    return fetch_tsunami_bulletins(sources=["ntwc"])


def _parse_published_date(entry: Dict) -> datetime:
    """Parse published date from RSS entry."""
    if "published_parsed" in entry and entry.published_parsed:
        import time
        return datetime(*entry.published_parsed[:6])
    elif "published" in entry:
        try:
            # Try various date formats
            for fmt in ["%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%S%z"]:
                try:
                    return datetime.strptime(entry.published, fmt)
                except ValueError:
                    continue
        except:
            pass
    return datetime.utcnow()


def _parse_severity(title: str) -> str:
    """Parse severity level from bulletin title.
    
    Common severity levels:
    - WARNING: Immediate threat to life and property
    - ADVISORY: Strong currents possible, stay away from shore
    - WATCH: Monitoring situation, threat not yet determined
    - INFORMATION: General information, no threat
    """
    title_upper = title.upper()
    
    if "WARNING" in title_upper:
        return "WARNING"
    elif "ADVISORY" in title_upper:
        return "ADVISORY"
    elif "WATCH" in title_upper:
        return "WATCH"
    elif "INFORMATION" in title_upper or "CANCELLATION" in title_upper:
        return "INFORMATION"
    else:
        return "UNKNOWN"


def get_active_warnings(bulletins: List[Dict]) -> List[Dict]:
    """Filter for active warnings (not information/cancellation).
    
    Args:
        bulletins: List of bulletins from fetch_tsunami_bulletins
    
    Returns:
        List of active warnings/advisories
    """
    active = [
        b for b in bulletins 
        if b["severity"] in ["WARNING", "ADVISORY", "WATCH"]
    ]
    return active


def format_bulletin_summary(bulletin: Dict) -> str:
    """Format a bulletin as human-readable summary.
    
    Args:
        bulletin: Bulletin dictionary
    
    Returns:
        Formatted string summary
    """
    published_str = bulletin["published"].strftime("%Y-%m-%d %H:%M UTC")
    
    summary = f"""
[{bulletin['source']}] {bulletin['severity']}
Published: {published_str}
Title: {bulletin['title']}

{bulletin['summary'][:300]}...

Full bulletin: {bulletin['link']}
    """.strip()
    
    return summary
