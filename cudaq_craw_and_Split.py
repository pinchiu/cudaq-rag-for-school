import os
import bs4
import requests
import concurrent.futures
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
BASE_URL = "https://nvidia.github.io/cuda-quantum/0.7.0/"
ENTRY_URL = f"{BASE_URL}using/quick_start.html"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
INPUT_DIR = "cuda_quantum_full_docs"
OUTPUT_DIR = os.path.join(INPUT_DIR, "splits")

def get_all_links(url):
    """Analyzes the sidebar to find all relevant documentation pages."""
    print(f"[*] Analyzing navigation structure: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Target the sidebar or main navigation specifically
        sidebar = soup.select_one(".bd-sidebar") or soup.select_one("nav")
        links = []
        
        if sidebar:
            for a in sidebar.find_all('a', href=True):
                full_url = requests.compat.urljoin(url, a['href']).split('#')[0]
                # Filter to stay within the documentation version scope
                if full_url.startswith(BASE_URL) and full_url.endswith(".html"):
                    if full_url not in links:
                        links.append(full_url)
        
        unique_links = sorted(list(set(links)))
        print(f"[+] Found {len(unique_links)} unique pages to crawl.")
        return unique_links
    except Exception as e:
        print(f"[!] Error fetching links: {e}")
        return []

def clean_doc_content(doc):
    """Removes common documentation artifacts like the Sphinx '¶' symbol."""
    doc.page_content = doc.page_content.replace("¶", "").strip()
    return doc

def scrape_docs():
    """Performs parallel downloading and cleaning of all documentation pages."""
    all_pages = get_all_links(ENTRY_URL)
    if not all_pages:
        return

    print(f"[*] Starting parallel download of {len(all_pages)} pages...")
    
    # Using a SoupStrainer to isolate the technical content
    bs4_strainer = bs4.SoupStrainer(attrs={"itemprop": "articleBody"})
    
    # Note: Modern WebBaseLoader supports multi-threading natively for paths
    loader = WebBaseLoader(
        web_paths=all_pages,
        bs_kwargs={"parse_only": bs4_strainer},
        header_template=HEADERS
    )
    # Set requests per second to stay polite to the server
    loader.requests_per_second = 5 
    
    try:
        docs = loader.load()
        print(f"[+] Successfully downloaded {len(docs)} documents.")

        if not os.path.exists(INPUT_DIR):
            os.makedirs(INPUT_DIR)

        for i, doc in enumerate(docs):
            # Generate a readable filename from the URL
            rel_path = all_pages[i].replace(BASE_URL, "").replace("/", "_").replace(".html", "")
            if not rel_path: rel_path = "index"
            
            filename = os.path.join(INPUT_DIR, f"{rel_path}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                # Clean and save
                f.write(clean_doc_content(doc).page_content)
            
        print(f"[+] Raw files saved to: {INPUT_DIR}")
        return docs
    except Exception as e:
        print(f"[!] Crawler error: {e}")
        return []

def process_and_split_documents():
    """Chunks the documents for the vector database with metadata preservation."""
    if not os.path.exists(INPUT_DIR):
        print(f"[!] Directory {INPUT_DIR} not found. Run crawler first.")
        return

    print(f"[*] Initializing Text Splitter (Chunk Size: 1000, Overlap: 200)")
    text_splitter = RecursiveCharacterTextSplitter(
        # Ordered by priority: logic blocks -> lines -> sentences -> words
        separators=["\n\n", "\n", "。 ", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    all_docs_content = []
    metadatas = []

    print(f"[*] Loading files from '{INPUT_DIR}'...")
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    
    for filename in files:
        filepath = os.path.join(INPUT_DIR, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            if content.strip():
                all_docs_content.append(content)
                # Preserve the source name in metadata for RAG citations
                metadatas.append({"source": filename.replace(".txt", "")})
        except Exception as e:
            print(f"[!] Error reading {filename}: {e}")

    print(f"[*] Performing semantic splitting on {len(all_docs_content)} files...")
    splits = text_splitter.create_documents(all_docs_content, metadatas=metadatas)
    print(f"[+] Generated {len(splits)} total chunks.")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Save chunks for the embedding script to pick up
    for i, split in enumerate(splits):
        source_name = split.metadata.get("source", "unknown")
        chunk_filename = f"{source_name}_chunk_{i}.txt"
        chunk_path = os.path.join(OUTPUT_DIR, chunk_filename)
        
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(split.page_content)

    print(f"[SUCCESS] All chunks saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    # check if we need to crawl or if data exists
    has_data = os.path.exists(INPUT_DIR) and any(f.endswith('.txt') for f in os.listdir(INPUT_DIR))
    
    if not has_data:
        print("[!] No local data found. Starting crawl pipeline...")
        scrape_docs()
    else:
        print(f"[*] Data detected in '{INPUT_DIR}'. Skipping crawl...")
    
    process_and_split_documents()

