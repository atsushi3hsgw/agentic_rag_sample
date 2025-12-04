import os
import json
import time
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from typing import List, Dict, Any
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("load2vector.log"),
        logging.StreamHandler()
    ]
)

def initialize_vector_store(
    pinecone_api_key: str,
    openai_api_key: str,
    index_name: str,
    embedding_model: str,
    base_url: str = 'https://openrouter.ai/api/v1'
) -> PineconeVectorStore:
    """Initialize Pinecone vector store with automatic index creation"""
    try:
        import pinecone
        
        # Initialize OpenAI embeddings with OpenRouter configuration
        embeddings = OpenAIEmbeddings(
            base_url=base_url,
            model=embedding_model,
            openai_api_key=openai_api_key
        )
        
        # Check and create index if necessary
        pc = Pinecone(api_key=pinecone_api_key)
        if index_name not in pc.list_indexes().names():
            logging.info(f"Creating new Pinecone index: {index_name}")
            # Get embedding dimension using a test embedding
            test_embedding = embeddings.embed_query("dimension test")
            dimension = len(test_embedding)
            logging.info("Detected embedding dimension: %d", dimension)
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
        index = pc.Index(index_name)
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings
        )
        logging.info(f"Using Pinecone index: {index_name}")
        return vector_store
    except Exception as e:
        logging.error(f"Initialize Pinecone vector store: {str(e)}")
        raise

def process_documents(
    file_path: str, 
    chunk_size: int, 
    chunk_overlap: int
) -> List[Document]:
    """Process JSONL file into Document objects"""
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    doc_id = doc['id']
                    content = '\n\n'.join(doc['paragraphs'])
                    
                    texts = text_splitter.split_text(content)
                    for i, text in enumerate(texts):
                        documents.append(Document(
                            page_content=text,
                            metadata={
                                'id': doc_id,
                                'title': doc['title'],
                                'chunk_index': i
                            }
                        ))
                    logging.info(f"Processed document {doc_id} ({len(texts)} chunks)")
                
                except json.JSONDecodeError:
                    logging.error(f"Invalid JSON on line {line_num}")
                except KeyError as e:
                    logging.error(f"Missing required field {str(e)}")
        
        return documents
    except Exception as e:
        logging.error(f"Document processing failed: {str(e)}")
        raise

# Final implementation

def main():
    try:
        # Load environment variables
        load_dotenv()
        
        # Configuration
        required_env_vars = {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'PINECONE_API_KEY': os.getenv('PINECONE_API_KEY')
        }
        
        for var, value in required_env_vars.items():
            if not value:
                raise ValueError(f"Missing required environment variable: {var}")
        
        config = {
            'pinecone_index': os.getenv('PINECONE_INDEX_NAME', 'agentic-rag-index'),
            'base_url': os.getenv('BASE_URL', 'https://openrouter.ai/api/v1'),
            'embedding_model': os.getenv('EMBEDDING_MODEL', 'openai/text-embedding-3-small'),
            'chunk_size': int(os.getenv('CHUNK_SIZE', 2000)),
            'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', 300))
        }
        
        # Verify command line argument
        if len(sys.argv) != 2:
            print("Usage: python load2vector.py <jsonl_file_path>")
            sys.exit(1)
        
        jsonl_path = sys.argv[1]
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Input file not found: {jsonl_path}")
                
        logging.info("Starting document processing")
        
        # Process documents into LangChain format
        documents = process_documents(
            file_path=jsonl_path,
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap']
        )
        
        # Initialize vector store
        logging.info("Initializing Pinecone connection")
        vector_store = initialize_vector_store(
            pinecone_api_key=required_env_vars['PINECONE_API_KEY'],
            openai_api_key=required_env_vars['OPENAI_API_KEY'], 
            index_name=config['pinecone_index'],
            embedding_model=config['embedding_model']
        )
        
        # Add documents to vector store with auto-generated embeddings
        logging.info(f"Uploading {len(documents)} documents")
        vector_store.add_documents(
            documents=documents,
            ids=[f"{doc.metadata['id']}_{doc.metadata['chunk_index']}" for doc in documents]
        )
        
        logging.info("Operation completed successfully")
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
