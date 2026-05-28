from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("4-rag/documents/immoveable.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 50,
    length_function = len,
    separators = ["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_documents(docs)

print(f"Original Document: 1 doc with {len(docs[0].page_content)} characters")
print(f"After split: {len(chunks)} chunks with {len(chunks[0].page_content)} characters")

for i, chunk in enumerate (chunks[:4]):
    print(f"--- Chunk {i} ---")
    print(f"Size: {len(chunk.page_content)} characteres")
    print(f"Content:\n{chunk.page_content}\n")

