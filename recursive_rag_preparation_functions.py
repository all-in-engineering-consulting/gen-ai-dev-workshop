from general_functions import faiss_index
from langchain_text_splitters import MarkdownHeaderTextSplitter
import os


def create_index_for_recursive_rag(directory_path, index_name):
    headers_to_split_on = [
        ("##", "Header 2"),
    ]

    all_chunks = []

    # # List to store the content of all markdown files

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".md"):
            file_path = os.path.join(directory_path, filename)

            # Read the contents of each markdown file
            with open(file_path, 'r', encoding='utf-8') as file:
                md_doc = file.read()

            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
            chunks = markdown_splitter.split_text(md_doc)
            for chunk in chunks:
                chunk.metadata["agreement"] = filename
                all_chunks.append(chunk)

            print("\n\n")
            for chunk in chunks:
                print(chunk,"\n\n")

    return faiss_index(chunks=all_chunks, index_name=index_name)