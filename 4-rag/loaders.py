from langchain_community.document_loaders import TextLoader, CSVLoader

# --- TextLoader ---
loader_txt = TextLoader("4-rag/documents/immoveable.txt", encoding="utf-8")
docs_txt = loader_txt.load()

print("=== TXT LOADER ===")
print(f"Quantity documents: {len(docs_txt)}")
print(f"Size: {len(docs_txt[0].page_content)} caracteres")
print(f"Metadata: {docs_txt[0].metadata}")
print(f"\nFirst 300 characters:\n{docs_txt[0].page_content[:300]}")

print("\n" + "="*50 + "\n")

# --- CSVLoader ---
loader_csv = CSVLoader("4-rag/documents/immoveable.csv", encoding="utf-8")
docs_csv = loader_csv.load()

print("=== CSV LOADER ===")
print(f"Quantity documents: {len(docs_csv)}")
print(f"\First document (first line of CSV):")
print(f"Content: {docs_csv[0].page_content}")
print(f"Metadata: {docs_csv[0].metadata}")

print("\n=== STRUCTURE OF DOCUMENT OBJECT ===")
doc = docs_txt[0]
print(f"Type: {type(doc)}")
print(f"Attributes: page_content, metadata")
print(f"doc.page_content is: {type(doc.page_content)}")
print(f"doc.metadata is: {type(doc.metadata)}")