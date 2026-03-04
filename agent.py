import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb
import sys
import os
from typing import List, Dict, Optional

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
except ImportError:
    sync_playwright = None
    PlaywrightTimeoutError = TimeoutError

class FreeWebsiteBot:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize with free embedding model"""
        self.model = SentenceTransformer(model_name)
        self.chroma_client = chromadb.PersistentClient(path="./free_website_knowledge")
        self.collection = None
        self.playwright_enabled = sync_playwright is not None
        self.playwright_failures = 0
        self.max_playwright_failures = self._read_int_env(
            "CRAWL_PLAYWRIGHT_MAX_FAILURES", 2
        )
        self.request_timeout_s = self._read_float_env("CRAWL_REQUEST_TIMEOUT_S", 10.0)
        self.playwright_goto_timeout_ms = self._read_int_env(
            "CRAWL_PLAYWRIGHT_GOTO_TIMEOUT_MS", 12000
        )
        self.playwright_networkidle_timeout_ms = self._read_int_env(
            "CRAWL_PLAYWRIGHT_NETWORKIDLE_TIMEOUT_MS", 2500
        )
        self.playwright_settle_wait_ms = self._read_int_env(
            "CRAWL_PLAYWRIGHT_SETTLE_WAIT_MS", 300
        )
        if os.getenv("CRAWL_USE_PLAYWRIGHT", "1").strip().lower() in {"0", "false", "no"}:
            self.playwright_enabled = False

        # Retry transient HTTP errors and use a browser-like user agent.
        retries = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "HEAD"]),
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session = requests.Session()
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                )
            }
        )

    def _read_int_env(self, name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default

    def _read_float_env(self, name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default

    def _candidate_urls(self, url: str) -> List[str]:
        """Try HTTPS first, then fallback to HTTP for misconfigured hosts."""
        if url.startswith("https://"):
            return [url, "http://" + url[len("https://") :]]
        return [url]

    def _clean_text(self, soup: BeautifulSoup) -> str:
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return " ".join(chunk for chunk in chunks if chunk)

    def _extract_page_data(self, html: str, source_url: str) -> Dict:
        soup = BeautifulSoup(html, "html.parser")
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        return {"url": source_url, "title": title, "content": self._clean_text(soup)}

    def _looks_like_js_placeholder(self, text: str) -> bool:
        lowered = text.lower()
        return (
            "requires javascript to work" in lowered
            or "please enable javascript" in lowered
            or "browser with javascript support" in lowered
        )

    def _crawl_page_with_playwright(self, url: str) -> Optional[Dict]:
        if (
            sync_playwright is None
            or not self.playwright_enabled
            or self.playwright_failures >= self.max_playwright_failures
        ):
            return None

        browser = None
        context = None
        try:
            with sync_playwright() as p:
                try:
                    browser = p.chromium.launch(headless=True)
                    context = browser.new_context(
                        user_agent=self.session.headers.get("User-Agent")
                    )
                    page = context.new_page()
                    page.goto(
                        url,
                        wait_until="domcontentloaded",
                        timeout=self.playwright_goto_timeout_ms,
                    )
                    try:
                        page.wait_for_load_state(
                            "networkidle",
                            timeout=self.playwright_networkidle_timeout_ms,
                        )
                    except PlaywrightTimeoutError:
                        pass
                    if self.playwright_settle_wait_ms > 0:
                        page.wait_for_timeout(self.playwright_settle_wait_ms)
                    html = page.content()
                finally:
                    if context is not None:
                        try:
                            context.close()
                        except Exception:
                            pass
                    if browser is not None:
                        try:
                            browser.close()
                        except Exception:
                            pass
            page_data = self._extract_page_data(html, url)
            if page_data["content"]:
                print(f"Fetched rendered content with Playwright for {url}")
            return page_data
        except PlaywrightTimeoutError:
            self.playwright_failures += 1
            print(
                f"Playwright timed out for {url} "
                f"({self.playwright_failures}/{self.max_playwright_failures})."
            )
            if self.playwright_failures >= self.max_playwright_failures:
                self.playwright_enabled = False
                print("Playwright disabled for this run after repeated failures.")
            return None
        except Exception as e:
            self.playwright_failures += 1
            print(
                f"Playwright failed for {url}: {e} "
                f"({self.playwright_failures}/{self.max_playwright_failures})."
            )
            if self.playwright_failures >= self.max_playwright_failures:
                self.playwright_enabled = False
                print("Playwright disabled for this run after repeated failures.")
            return None

    def crawl_page(self, url: str) -> Optional[Dict]:
        """Crawl a single webpage"""
        last_error = None
        for candidate in self._candidate_urls(url):
            try:
                response = self.session.get(candidate, timeout=self.request_timeout_s)
                response.raise_for_status()
                page_data = self._extract_page_data(response.text, candidate)
                if self._looks_like_js_placeholder(page_data["content"]):
                    rendered_page_data = self._crawl_page_with_playwright(candidate)
                    if rendered_page_data and rendered_page_data["content"]:
                        return rendered_page_data
                return page_data
            except Exception as e:
                last_error = e
                rendered_page_data = self._crawl_page_with_playwright(candidate)
                if rendered_page_data and rendered_page_data["content"]:
                    return rendered_page_data

        print(f"Error crawling {url}: {last_error}")
        return None
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []

        step = max(chunk_size - overlap, 1)
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + chunk_size]).strip()
            if chunk:
                chunks.append(chunk)
            
        return chunks
    
    def create_collection(self, collection_name: str = "website_content"):
        """Create or get ChromaDB collection"""
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def learn_from_websites(self, urls: List[str]) -> int:
        """Main function to learn from websites"""
        self.create_collection()
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        successful_pages = 0
        
        for idx, url in enumerate(urls):
            print(f"Processing {url}...")
            page_data = self.crawl_page(url)
            
            if page_data and page_data["content"]:
                chunks = self.chunk_text(page_data["content"])
                if not chunks:
                    continue

                successful_pages += 1
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id = f"doc_{idx}_chunk_{chunk_idx}"
                    
                    all_chunks.append(chunk)
                    all_metadatas.append({
                        "url": page_data["url"],
                        "title": page_data["title"],
                        "chunk_index": chunk_idx
                    })
                    all_ids.append(chunk_id)

        if not all_chunks:
            print("No content was collected from the provided URLs.")
            return 0

        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.model.encode(all_chunks).tolist()
        
        # Upsert avoids duplicate-id errors when re-running training.
        self.collection.upsert(
            embeddings=embeddings,
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )

        print(
            f"Successfully learned {len(all_chunks)} chunks "
            f"from {successful_pages} pages!"
        )
        return len(all_chunks)
        
    def query(self, question: str, n_results: int = 3) -> List[Dict]:
        """Query the learned content"""
        if self.collection is None:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}

        collection_size = self.collection.count()
        if collection_size == 0:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}

        # Generate embedding for question
        question_embedding = self.model.encode([question]).tolist()
        
        # Query collection
        results = self.collection.query(
            query_embeddings=question_embedding,
            n_results=min(n_results, collection_size)
        )
        
        return results
    
    def generate_response(self, question: str) -> str:
        """Generate a response based on retrieved content"""
        results = self.query(question)
        
        if not results['documents'][0]:
            return "I couldn't find relevant information about that on the website."
        
        # Combine retrieved chunks
        context = "\n\n".join(results['documents'][0])
        sources = set([m['url'] for m in results['metadatas'][0]])
        
        # Simple response generation (for better responses, integrate with free LLM like Llama)
        response = f"Based on the website content:\n\n{context}\n\n"
        response += f"Sources: {', '.join(sources)}"
        
        return response

# Usage
def main_free():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    bot = FreeWebsiteBot()
    
    # Your website URLs
    urls = [
        "https://joelloter.free.nf",
       # "https://joelloter.free.nf/about",
        # "https://joelloter.free.nf/services"
    ]
    
    # Train the bot
    try:
        learned_chunks = bot.learn_from_websites(urls)
    except KeyboardInterrupt:
        print("\nInterrupted while indexing websites. Exiting.")
        return
    if learned_chunks == 0:
        print(
            "No website content indexed. Update URL(s) or retry, then ask questions."
        )
    
    # Query
    while True:
        try:
            question = input("\nAsk: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if question.lower() == 'quit':
            break
            
        response = bot.generate_response(question)
        print(f"\n{response}")

if __name__ == "__main__":
    main_free()
