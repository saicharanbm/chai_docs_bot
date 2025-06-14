from dotenv import load_dotenv
load_dotenv()
from fetch_all_sub_links import urls_list
from langchain_community.document_loaders import WebBaseLoader
from helper import initialize_clients_for_indexing
from configuration import COLLECTION_NAME 
from langchain_qdrant import QdrantVectorStore

# initialize an embedding model instance
embedding_model, text_splitter, qdrant_host, qdrant_api_key , console = initialize_clients_for_indexing()


error_occurred = False
vector_store = None

try:
    for i, url in enumerate(urls_list):
        console.print(f"Processing URL {i+1}/{len(urls_list)}: {url}")
        
        loader = WebBaseLoader(url)
        doc = loader.load()
        
        if not doc:
            console.print(f"⚠️ No content loaded from {url}")
            continue
            
        split_docs = text_splitter.split_documents(documents=doc)
        
        if not split_docs:
            console.print(f"⚠️ No documents after splitting from {url}")
            continue
        
        if vector_store is None:
            vector_store = QdrantVectorStore.from_documents(
                url=qdrant_host, 
                api_key=qdrant_api_key,
                documents=split_docs,
                collection_name=COLLECTION_NAME,
                embedding=embedding_model
            )
            console.print(f"✅ Created vector store and added {len(split_docs)} documents from {url}")
        else:
            vector_store.add_documents(split_docs)
            console.print(f"✅ Added {len(split_docs)} documents from {url}")

except Exception as e:
    console.print(f"❌ Error processing {url}: {str(e)}")
    error_occurred = True

if not error_occurred:
    console.print("✅ [bold green] Stored all the embeddings to Qdrant vector DB successfully! [/bold green]")
