import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_chroma import Chroma
from chromadb.config import Settings
import os
from dotenv import load_dotenv
import openai
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings as LlamaSettings
import logging
from typing import List, Dict, Any
import chardet

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArxivRAGSystem:
    def __init__(self, llm_option="openai", embedding_model="BAAI/bge-small-en-v1.5"):
        """
        Initialize the ArXiv RAG system
        
        Args:
            llm_option: "openai" or "local"
            embedding_model: Hugging Face embedding model name
        """
        # Initialize logger for this instance
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing ArXiv RAG System...")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.logger.info(f"Loaded embedding model: {embedding_model}")
        
        # Initialize LLM
        if llm_option == "openai":
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            self.llm = OpenAI(
                model="gpt-4o-mini",
                api_key=openai_api_key,
                temperature=0.7,
                max_tokens=1500
            )
            self.logger.info("Initialized OpenAI LLM")
        else:
            raise NotImplementedError("Local LLM not implemented yet")
        
        # Configure LlamaIndex settings
        embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        LlamaSettings.llm = self.llm
        LlamaSettings.embed_model = embed_model
        
        # Initialize storage
        self.papers_df = None
        self.index = None
        self.chroma_client = None
        self.collection = None
        
        self.logger.info("ArXiv RAG System initialized successfully")
    
    def detect_file_encoding(self, file_path):
        """Detect the encoding of a file"""
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            self.logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
            return encoding
    
    def load_arxiv_data(self, file_path, sample_size=1000):
        """
        Load ArXiv data from JSON file with proper encoding handling
        
        Args:
            file_path: Path to the ArXiv JSON file
            sample_size: Number of papers to load (None for all)
        
        Returns:
            DataFrame with loaded papers
        """
        self.logger.info(f"Loading ArXiv data from {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Detect encoding
        encoding = self.detect_file_encoding(file_path)
        if encoding is None:
            encoding = 'utf-8'  # Fallback
            self.logger.warning("Could not detect encoding, using UTF-8")
        
        papers = []
        
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                for i, line in enumerate(f):
                    if sample_size and i >= sample_size:
                        break
                    
                    try:
                        paper = json.loads(line.strip())
                        
                        # Basic validation
                        if not paper.get('title') or not paper.get('abstract'):
                            continue
                        
                        # Process authors field properly
                        authors_str = "Unknown"
                        if paper.get('authors_parsed'):
                            # authors_parsed is a list of [last_name, first_name] lists
                            author_names = []
                            for author in paper.get('authors_parsed', [])[:3]:  # Limit to first 3 authors
                                if isinstance(author, list) and len(author) >= 2:
                                    # Format: "First Last"
                                    author_names.append(f"{author[1]} {author[0]}")
                                elif isinstance(author, list) and len(author) == 1:
                                    author_names.append(str(author[0]))
                                else:
                                    author_names.append(str(author))
                            authors_str = ', '.join(author_names)
                        elif paper.get('authors'):
                            # Fallback to regular authors field
                            authors_str = paper.get('authors', 'Unknown')
                        
                        # Clean and prepare data
                        clean_paper = {
                            'arxiv_id': paper.get('id', '').replace('http://arxiv.org/abs/', ''),
                            'title': paper.get('title', '').replace('\n', ' ').strip(),
                            'abstract': paper.get('abstract', '').replace('\n', ' ').strip(),
                            'authors': authors_str,
                            'categories': paper.get('categories', ''),
                            'primary_category': paper.get('categories', '').split()[0] if paper.get('categories') else '',
                            'date': paper.get('update_date', paper.get('created', '')),
                            'doi': paper.get('doi', ''),
                            'journal_ref': paper.get('journal-ref', '')
                        }
                        
                        papers.append(clean_paper)
                        
                        if i % 1000 == 0:
                            self.logger.info(f"Processed {i} papers...")
                            
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Skipping malformed JSON at line {i}: {e}")
                        continue
                    except Exception as e:
                        self.logger.warning(f"Error processing paper at line {i}: {e}")
                        continue
        
        except UnicodeDecodeError as e:
            self.logger.error(f"Unicode decode error: {e}")
            # Try with different encodings
            for alt_encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    self.logger.info(f"Trying encoding: {alt_encoding}")
                    with open(file_path, 'r', encoding=alt_encoding, errors='ignore') as f:
                        # Process with alternative encoding...
                        # (same logic as above but with different encoding)
                        pass
                    break
                except:
                    continue
            else:
                raise Exception("Could not read file with any encoding")
        
        if not papers:
            raise ValueError("No valid papers found in the file")
        
        self.papers_df = pd.DataFrame(papers)
        self.logger.info(f"Loaded {len(self.papers_df)} papers successfully")
        
        return self.papers_df
    
    def load_existing_index(self):
        """Load existing vector database and papers data"""
        import os
        import pickle
        
        # Check if database directory exists
        if not os.path.exists('./chroma_db'):
            raise FileNotFoundError("No existing database found at ./chroma_db")
        
        # Initialize ChromaDB client and get collection
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection_name = "arxiv_papers"
        
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            self.logger.info("Found existing ChromaDB collection")
        except Exception as e:
            self.logger.error(f"Could not load collection: {e}")
            raise FileNotFoundError(f"Collection '{collection_name}' not found in existing database")
        
        # Create LlamaIndex vector store from existing collection
        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create VectorStoreIndex from existing storage
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
        
        # Try to load papers DataFrame from pickle file
        papers_file = './chroma_db/papers_df.pkl'
        if os.path.exists(papers_file):
            with open(papers_file, 'rb') as f:
                self.papers_df = pickle.load(f)
            self.logger.info(f"Loaded {len(self.papers_df)} papers from existing database")
        else:
            # Fallback: try to reconstruct from vector database metadata
            self.logger.warning("No papers DataFrame found, reconstructing from database...")
            try:
                # Get all documents from the collection
                results = self.collection.get()
                if results and results.get('documents'):
                    # Create a basic DataFrame from the database contents
                    import pandas as pd
                    
                    papers_data = []
                    documents = results.get('documents', [])
                    metadatas = results.get('metadatas', [])
                    
                    for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
                        papers_data.append({
                            'arxiv_id': metadata.get('arxiv_id', f'unknown_{i}'),
                            'title': metadata.get('title', 'Unknown Title'),
                            'abstract': doc,
                            'authors': metadata.get('authors', 'Unknown'),
                            'categories': metadata.get('categories', ''),
                            'primary_category': metadata.get('category', ''),  # Note: using 'category' not 'primary_category'
                            'date': metadata.get('date', ''),
                            'doi': metadata.get('doi', ''),
                            'journal_ref': metadata.get('journal_ref', '')
                        })
                    
                    self.papers_df = pd.DataFrame(papers_data)
                    self.logger.info(f"Reconstructed {len(self.papers_df)} papers from database metadata")
                else:
                    raise ValueError("Database exists but contains no documents")
            except Exception as e:
                self.logger.error(f"Failed to reconstruct papers data: {e}")
                raise
        
        self.logger.info("Successfully loaded existing index and papers data")

    def save_papers_dataframe(self):
        """Save papers DataFrame to pickle file for faster loading"""
        import pickle
        import os
        
        if self.papers_df is not None:
            os.makedirs('./chroma_db', exist_ok=True)
            papers_file = './chroma_db/papers_df.pkl'
            with open(papers_file, 'wb') as f:
                pickle.dump(self.papers_df, f)
            self.logger.info(f"Saved papers DataFrame with {len(self.papers_df)} papers")

    def build_index(self):
        """Build vector index from loaded papers"""
        if self.papers_df is None:
            raise ValueError("No papers loaded. Call load_arxiv_data() first.")
        
        self.logger.info("Building vector index...")
        
        # Create documents for LlamaIndex
        documents = []
        for _, paper in self.papers_df.iterrows():
            text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}\n\nAuthors: {paper['authors']}\n\nCategory: {paper['primary_category']}"
            
            doc = Document(
                text=text,
                metadata={
                    'arxiv_id': paper['arxiv_id'],
                    'title': paper['title'],
                    'authors': paper['authors'],
                    'category': paper['primary_category'],
                    'date': paper['date']
                }
            )
            documents.append(doc)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collection
        collection_name = "arxiv_papers"
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            self.logger.info("Using existing ChromaDB collection")
        except:
            self.collection = self.chroma_client.create_collection(collection_name)
            self.logger.info("Created new ChromaDB collection")
        
        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Build index
        self.index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context,
            show_progress=True
        )
        
        self.logger.info("Vector index built successfully")
        self.save_papers_dataframe()
        return self.index
    
    def search_papers(self, query, top_k=5):
        """
        Search for papers using semantic similarity
        
        Args:
            query: Search query string
            top_k: Number of results to return
        
        Returns:
            List of paper dictionaries with similarity scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Query the index
        query_engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="no_text"  # We just want the retrieved documents
        )
        
        response = query_engine.query(query)
        
        results = []
        for node in response.source_nodes:
            metadata = node.metadata
            
            # Find the full paper data
            paper_row = self.papers_df[
                self.papers_df['arxiv_id'] == metadata['arxiv_id']
            ].iloc[0]
            
            result = {
                'arxiv_id': paper_row['arxiv_id'],
                'title': paper_row['title'],
                'abstract': paper_row['abstract'],
                'authors': paper_row['authors'],
                'category': paper_row['primary_category'],
                'date': paper_row['date'],
                'score': node.score if hasattr(node, 'score') else 1.0
            }
            results.append(result)
        
        return results
    
    def get_recommendations(self, query, top_k=5):
        """
        Get AI-powered recommendations for a query
        
        Args:
            query: User query
            top_k: Number of papers to analyze
        
        Returns:
            AI-generated recommendations text
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Get relevant papers
        papers = self.search_papers(query, top_k)
        
        # Create context for LLM
        context = f"User Query: {query}\n\n"
        context += "Relevant Papers Found:\n\n"
        
        for i, paper in enumerate(papers, 1):
            context += f"{i}. **{paper['title']}** (arXiv:{paper['arxiv_id']})\n"
            context += f"   Authors: {paper['authors']}\n"
            context += f"   Category: {paper['category']}\n"
            context += f"   Abstract: {paper['abstract'][:300]}...\n\n"
        
        # Generate recommendations
        prompt = f"""
        Based on the user's query and the relevant papers found, provide intelligent recommendations and analysis.
        
        {context}
        
        Please provide:
        1. A brief analysis of what the user is looking for
        2. Key themes and trends in the retrieved papers
        3. Specific recommendations on which papers are most relevant and why
        4. Suggestions for related research directions or keywords to explore
        
        Keep your response concise but insightful, focusing on actionable recommendations.
        """
        
        try:
            response = self.llm.complete(prompt)
            return response.text
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return f"Error generating recommendations: {e}"

# Example usage
if __name__ == "__main__":
    # Initialize system
    rag = ArxivRAGSystem(llm_option="openai")
    
    # Load data
    df = rag.load_arxiv_data("arxiv-metadata-oai-snapshot.json", sample_size=1000)
    print(f"Loaded {len(df)} papers")
    
    # Build index
    rag.build_index()
    print("Index built successfully")
    
    # Test search
    results = rag.search_papers("transformer neural networks", top_k=3)
    for paper in results:
        print(f"- {paper['title']} (Score: {paper['score']:.3f})")
    
    # Test recommendations
    recommendations = rag.get_recommendations("transformer neural networks", top_k=3)
    print("\nAI Recommendations:")
    print(recommendations)