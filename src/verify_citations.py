"""
Citation Verification Module

Verifies extracted citations against external databases:
- Semantic Scholar API for academic papers
- CrossRef API for DOIs
- Manual verification flags for legal/historical
"""

import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import argparse
from tqdm import tqdm


class HallucinationType(Enum):
    """Classification of citation truthfulness."""
    REAL = "real"                    # Citation exists and supports claim
    FABRICATED = "fabricated"        # Citation doesn't exist (invented)
    MISATTRIBUTED = "misattributed"  # Citation exists but wrong claim
    UNVERIFIED = "unverified"        # Could not verify


@dataclass
class VerificationResult:
    """Result of citation verification."""
    citation_raw: str
    label: HallucinationType
    confidence: float
    verification_source: str
    matched_title: Optional[str] = None
    matched_authors: Optional[List[str]] = None
    matched_year: Optional[int] = None
    notes: Optional[str] = None


class SemanticScholarVerifier:
    """Verify academic citations via Semantic Scholar API."""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["x-api-key"] = api_key
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Rate limiting
    
    def _rate_limit(self):
        """Ensure we don't exceed API rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def search_paper(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for papers matching query."""
        self._rate_limit()
        
        url = f"{self.BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,year,citationCount,abstract"
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
            else:
                return []
        except Exception as e:
            print(f"API error: {e}")
            return []
    
    def verify_citation(
        self,
        authors: Optional[List[str]],
        year: Optional[int],
        title_hint: Optional[str] = None
    ) -> VerificationResult:
        """Verify if a citation refers to a real paper."""
        
        # Build search query
        query_parts = []
        if authors:
            query_parts.append(authors[0])  # Use first author
        if year:
            query_parts.append(str(year))
        if title_hint:
            query_parts.append(title_hint[:50])  # First 50 chars of title
        
        if not query_parts:
            return VerificationResult(
                citation_raw="",
                label=HallucinationType.UNVERIFIED,
                confidence=0.0,
                verification_source="semantic_scholar",
                notes="No searchable information in citation"
            )
        
        query = " ".join(query_parts)
        results = self.search_paper(query)
        
        if not results:
            # No results found - likely fabricated
            return VerificationResult(
                citation_raw=query,
                label=HallucinationType.FABRICATED,
                confidence=0.7,  # Not 100% confident - could be API limitation
                verification_source="semantic_scholar",
                notes="No matching papers found"
            )
        
        # Check if any result matches
        for paper in results:
            paper_year = paper.get("year")
            paper_authors = [a.get("name", "") for a in paper.get("authors", [])]
            
            # Year match check
            year_match = (year is None) or (paper_year == year)
            
            # Author match check (fuzzy)
            author_match = False
            if authors:
                for author in authors:
                    author_last = author.split()[-1].lower() if author else ""
                    for paper_author in paper_authors:
                        if author_last in paper_author.lower():
                            author_match = True
                            break
            else:
                author_match = True  # No author specified
            
            if year_match and author_match:
                return VerificationResult(
                    citation_raw=query,
                    label=HallucinationType.REAL,
                    confidence=0.85,
                    verification_source="semantic_scholar",
                    matched_title=paper.get("title"),
                    matched_authors=paper_authors[:3],
                    matched_year=paper_year,
                    notes="Matched in Semantic Scholar"
                )
        
        # Found papers but no good match
        return VerificationResult(
            citation_raw=query,
            label=HallucinationType.FABRICATED,
            confidence=0.6,
            verification_source="semantic_scholar",
            notes=f"Found {len(results)} papers but no exact match"
        )


class CrossRefVerifier:
    """Verify citations via CrossRef API."""
    
    BASE_URL = "https://api.crossref.org/works"
    
    def __init__(self, email: Optional[str] = None):
        self.email = email
        self.headers = {}
        if email:
            self.headers["User-Agent"] = f"CitationVerifier/1.0 (mailto:{email})"
    
    def search_by_query(self, query: str, limit: int = 5) -> List[Dict]:
        """Search CrossRef for matching works."""
        params = {"query": query, "rows": limit}
        
        try:
            response = requests.get(self.BASE_URL, params=params, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("message", {}).get("items", [])
            return []
        except Exception:
            return []


class CitationVerifier:
    """Main verification orchestrator."""
    
    def __init__(
        self,
        semantic_scholar_key: Optional[str] = None,
        crossref_email: Optional[str] = None
    ):
        self.ss_verifier = SemanticScholarVerifier(semantic_scholar_key)
        self.cr_verifier = CrossRefVerifier(crossref_email)
    
    def verify_academic_citation(self, citation: Dict) -> VerificationResult:
        """Verify an academic citation."""
        authors = citation.get("extracted_authors", [])
        year = citation.get("extracted_year")
        title = citation.get("extracted_title")
        raw = citation.get("raw_text", "")
        
        result = self.ss_verifier.verify_citation(authors, year, title)
        result.citation_raw = raw
        return result
    
    def verify_legal_citation(self, citation: Dict) -> VerificationResult:
        """Mark legal citations for manual verification."""
        raw = citation.get("raw_text", "")
        return VerificationResult(
            citation_raw=raw,
            label=HallucinationType.UNVERIFIED,
            confidence=0.0,
            verification_source="manual_required",
            notes="Legal citations require manual verification"
        )
    
    def verify_historical_citation(self, citation: Dict) -> VerificationResult:
        """Historical citations - try Semantic Scholar for academic sources."""
        # Try as academic first
        result = self.verify_academic_citation(citation)
        if result.label == HallucinationType.REAL:
            return result
        
        # Otherwise mark for manual
        raw = citation.get("raw_text", "")
        return VerificationResult(
            citation_raw=raw,
            label=HallucinationType.UNVERIFIED,
            confidence=0.0,
            verification_source="manual_required",
            notes="Historical citation - manual verification recommended"
        )
    
    def verify_citation(self, citation: Dict, domain: str) -> VerificationResult:
        """Route to appropriate verifier based on domain."""
        if domain == "scientific":
            return self.verify_academic_citation(citation)
        elif domain == "legal":
            return self.verify_legal_citation(citation)
        elif domain == "historical":
            return self.verify_historical_citation(citation)
        else:
            return self.verify_academic_citation(citation)


def verify_essays(essays_path: str, output_path: str, api_key: Optional[str] = None):
    """Verify all citations in a generated essays file."""
    
    with open(essays_path, 'r') as f:
        essays = json.load(f)
    
    verifier = CitationVerifier(semantic_scholar_key=api_key)
    
    verified_essays = []
    total_citations = 0
    verified_counts = {t.value: 0 for t in HallucinationType}
    
    for essay in tqdm(essays, desc="Verifying citations"):
        domain = essay.get("domain", "scientific")
        verified_citations = []
        
        for citation in essay.get("citations", []):
            total_citations += 1
            result = verifier.verify_citation(citation, domain)
            
            verified_citations.append({
                **citation,
                "label": result.label.value,
                "verification_confidence": result.confidence,
                "verification_source": result.verification_source,
                "matched_title": result.matched_title,
                "matched_authors": result.matched_authors,
                "matched_year": result.matched_year,
                "verification_notes": result.notes,
            })
            
            verified_counts[result.label.value] += 1
        
        essay["citations"] = verified_citations
        verified_essays.append(essay)
    
    # Save verified essays
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(verified_essays, f, indent=2)
    
    # Print statistics
    print(f"\nVerification Statistics:")
    print(f"Total citations: {total_citations}")
    for label, count in verified_counts.items():
        pct = (count / total_citations * 100) if total_citations > 0 else 0
        print(f"  {label}: {count} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Verify citations in generated essays")
    parser.add_argument("--input", type=str, required=True, help="Path to essays JSON")
    parser.add_argument("--output", type=str, required=True, help="Output path for verified essays")
    parser.add_argument("--api-key", type=str, help="Semantic Scholar API key")
    args = parser.parse_args()
    
    verify_essays(args.input, args.output, args.api_key)


if __name__ == "__main__":
    main()
