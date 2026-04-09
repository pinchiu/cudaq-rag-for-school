import os
import pypdf
import json
import subprocess
import bs4
import requests
import concurrent.futures
import shutil
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

# Reference Repositories
ACADEMIC_REPO = "https://github.com/NVIDIA/cuda-q-academic.git"
ACADEMIC_DIR = os.path.join(INPUT_DIR, "cuda-q-academic")
MAIN_REPO = "https://github.com/NVIDIA/cuda-quantum.git"
MAIN_REPO_DIR = os.path.join(INPUT_DIR, "cuda-quantum")

# Technical Papers
ARXIV_PAPERS = [
    "https://arxiv.org/pdf/2302.04631.pdf" # CUDA-Q foundational paper
]

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

def download_academic_repo():
    print(f"[*] Checking academic repository...")
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        
    if not os.path.exists(ACADEMIC_DIR):
        print(f"[*] Cloning academic repository from {ACADEMIC_REPO}...")
        try:
            subprocess.run(["git", "clone", ACADEMIC_REPO, ACADEMIC_DIR], check=True)
            print("[+] Successfully cloned academic repository.")
        except subprocess.CalledProcessError as e:
            print(f"[!] Error cloning repository: {e}")
    else:
        print(f"[*] Academic repository already exists at {ACADEMIC_DIR}. Pulling latest changes...")
        try:
            subprocess.run(["git", "-C", ACADEMIC_DIR, "pull"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[!] Error pulling repository updates: {e}")

def download_main_repo():
    print(f"[*] Checking main CUDA-Quantum repository...")
    if not os.path.exists(MAIN_REPO_DIR):
        print(f"[*] Cloning main repository from {MAIN_REPO}...")
        try:
            subprocess.run(["git", "clone", "--depth", "1", MAIN_REPO, MAIN_REPO_DIR], check=True)
            print("[+] Successfully cloned main repository.")
        except subprocess.CalledProcessError as e:
            print(f"[!] Error cloning repository: {e}")
    else:
        print(f"[*] Main repository already exists. Pulling latest code...")
        try:
            subprocess.run(["git", "-C", MAIN_REPO_DIR, "pull"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[!] Error pulling repository updates: {e}")

def download_technical_papers():
    print(f"[*] Checking for technical whitepapers...")
    paper_dir = os.path.join(INPUT_DIR, "papers")
    if not os.path.exists(paper_dir):
        os.makedirs(paper_dir)
        
    for url in ARXIV_PAPERS:
        filename = url.split("/")[-1]
        filepath = os.path.join(paper_dir, filename)
        if not os.path.exists(filepath):
            print(f"[*] Downloading paper: {filename}")
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                with open(filepath, "wb") as f:
                    shutil.copyfileobj(response.raw, f)
            except Exception as e:
                print(f"[!] Error downloading paper {filename}: {e}")

def extract_pdf(filepath):
    """Extracts text from a PDF file using pypdf."""
    text = ""
    try:
        reader = pypdf.PdfReader(filepath)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"--- Page {i+1} ---\n{page_text}\n\n"
    except Exception as e:
        print(f"[!] Error reading PDF {filepath}: {e}")
    return text


def extract_notebook(filepath):
    text = ""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for cell in data.get("cells", []):
                source = cell.get("source", [])
                if isinstance(source, list):
                    text += "".join(source) + "\n\n"
                else:
                    text += source + "\n\n"
    except Exception as e:
        print(f"[!] Error reading notebook {filepath}: {e}")
    return text

def extract_markdown(filepath):
    text = ""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"[!] Error reading markdown {filepath}: {e}")
    return text

def process_academic_materials():
    if not os.path.exists(ACADEMIC_DIR):
        return

    print("[*] Extracting academic materials (Notebooks, MD, PDF)...")
    extracted_count = 0
    for root, dirs, files in os.walk(ACADEMIC_DIR):
        if '.git' in root: continue
            
        for file in files:
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, ACADEMIC_DIR)
            
            text = ""
            if file.endswith('.ipynb'):
                text = extract_notebook(filepath)
            elif file.endswith('.md'):
                text = extract_markdown(filepath)
            elif file.endswith('.pdf'):
                text = extract_pdf(filepath)
            
            if text:
                safe_name = "academic_" + rel_path.replace(os.sep, "_").replace(".", "_")
                out_path = os.path.join(INPUT_DIR, safe_name + ".txt")
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                extracted_count += 1
                
    print(f"[+] Successfully extracted {extracted_count} academic files.")

def process_main_repo_examples():
    if not os.path.exists(MAIN_REPO_DIR):
        return

    print("[*] Extracting main repo examples (Python, C++, Headers)...")
    # Target folders that contain high-quality examples
    target_folders = ["examples", os.path.join("python", "examples"), "snippets", "docs"]
    extracted_count = 0

    for folder in target_folders:
        folder_path = os.path.join(MAIN_REPO_DIR, folder)
        if not os.path.exists(folder_path): continue

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Include Python, C++, and markdown documentation from the repo
                if file.endswith(('.py', '.cpp', '.h', '.cuh', '.md')):
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, MAIN_REPO_DIR)
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()
                        
                        if text:
                            # Add a header to help the LLM know which file this is
                            text = f"Source File: {rel_path}\n---\n" + text
                            safe_name = "mainrepo_" + rel_path.replace(os.sep, "_").replace(".", "_")
                            out_path = os.path.join(INPUT_DIR, safe_name + ".txt")
                            with open(out_path, 'w', encoding='utf-8') as f:
                                f.write(text)
                            extracted_count += 1
                    except:
                        pass
    
    print(f"[+] Successfully extracted {extracted_count} code examples from the main repo.")

def process_technical_papers():
    paper_dir = os.path.join(INPUT_DIR, "papers")
    if not os.path.exists(paper_dir): return

    print("[*] Extracting text from technical whitepapers...")
    extracted_count = 0
    for file in os.listdir(paper_dir):
        if file.endswith('.pdf'):
            filepath = os.path.join(paper_dir, file)
            text = extract_pdf(filepath)
            if text:
                out_path = os.path.join(INPUT_DIR, f"whitepaper_{file.replace('.pdf', '')}.txt")
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                extracted_count += 1
    print(f"[+] Successfully extracted {extracted_count} whitepapers.")


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
    # 1. Scrape Web Documentation
    has_web_data = os.path.exists(INPUT_DIR) and any(f.endswith('.txt') and not f.startswith(('academic_', 'mainrepo_', 'whitepaper_')) for f in os.listdir(INPUT_DIR))
    if not has_web_data:
        print("[!] No local web data found. Starting web crawl pipeline...")
        scrape_docs()
    else:
        print(f"[*] Web data detected in '{INPUT_DIR}'. Skipping crawl...")
    
    # 2. Sync and Process Repositories
    download_academic_repo()
    process_academic_materials()
    
    download_main_repo()
    process_main_repo_examples()
    
    # 3. Handle Technical Whitepapers
    download_technical_papers()
    process_technical_papers()
    
    # 4. Final Splitting
    process_and_split_documents()

