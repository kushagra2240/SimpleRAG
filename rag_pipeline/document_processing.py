import os
from typing import List, Dict
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

def clean_text(text: str) -> str:
    """Cleans up text by removing extra whitespace, tabs, and newlines."""
    text = text.replace('\t', ' ').replace('\n', ' ')
    text = ' '.join(text.split())
    return text

def load_documents_from_files(file_paths: List[str]) -> List[Dict]:
    all_docs = []
    for file_path in file_paths:
        try:
            reader = PdfReader(file_path)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                cleaned_text = clean_text(text)
                metadata = {"page_number": page_num + 1, "document_name": os.path.basename(file_path)}
                all_docs.append({"content": cleaned_text, "metadata": metadata})
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return all_docs

def load_documents_from_folder(directory: str) -> List[Dict]:
    all_docs = []
    if not os.path.isdir(directory):
        print(f"Error: Directory {directory} not found")
        return []
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            try:
                reader = PdfReader(file_path)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    cleaned_text = clean_text(text)
                    metadata = {"page_number": page_num + 1, "document_name": filename}
                    all_docs.append({"content": cleaned_text, "metadata": metadata})
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    return all_docs

def split_documents(documents: List[Dict], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    all_splits = []
    for doc in documents:
        splits = text_splitter.create_documents([doc['content']], metadatas=[doc['metadata']])
        all_splits.extend(splits)
    return all_splits