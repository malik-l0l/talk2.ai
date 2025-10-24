import os
import chromadb
from sentence_transformers import SentenceTransformer
import uuid

def setup_vector_database():
    """Setup vector database for personality data"""
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Initialize sentence transformer for embeddings
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create data directory if it doesn't exist
    os.makedirs('assets/data', exist_ok=True)
    
    # Process Elon Musk data
    elon_data_path = 'assets/data/elon_musk_data.txt'
    if os.path.exists(elon_data_path):
        print("Processing Elon Musk data...")
        process_personality_data(chroma_client, embedding_model, 'elonmusk', elon_data_path)
    else:
        print(f"Warning: {elon_data_path} not found. Please add the file.")
    
    # Process Gandhi data
    gandhi_data_path = 'assets/data/gandhi_data.txt'
    if os.path.exists(gandhi_data_path):
        print("Processing Gandhi data...")
        process_personality_data(chroma_client, embedding_model, 'gandhi', gandhi_data_path)
    else:
        print(f"Warning: {gandhi_data_path} not found. Please add the file.")
    
    print("Vector database setup complete!")

def process_personality_data(chroma_client, embedding_model, personality_name, file_path):
    """Process and store personality data in vector database"""
    
    try:
        # Get or create collection
        try:
            collection = chroma_client.get_collection(name=personality_name)
            print(f"Collection {personality_name} already exists. Deleting and recreating...")
            chroma_client.delete_collection(name=personality_name)
        except:
            pass
        
        collection = chroma_client.create_collection(name=personality_name)
        
        # Read and process the data file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Split content into chunks (you can adjust chunk size as needed)
        chunks = split_into_chunks(content, chunk_size=500, overlap=50)
        
        print(f"Created {len(chunks)} chunks for {personality_name}")
        
        # Create embeddings and store in ChromaDB
        documents = []
        embeddings = []
        ids = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Skip empty chunks
                # Create embedding
                embedding = embedding_model.encode([chunk])
                
                documents.append(chunk)
                embeddings.append(embedding[0].tolist())
                ids.append(f"{personality_name}_{i}")
                metadatas.append({
                    'personality': personality_name,
                    'chunk_id': i,
                    'source': file_path
                })
        
        # Add to collection
        if documents:
            collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Successfully stored {len(documents)} chunks for {personality_name}")
        else:
            print(f"Warning: No valid chunks found for {personality_name}")
            
    except Exception as e:
        print(f"Error processing {personality_name} data: {e}")

def split_into_chunks(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        
        # Break if we've reached the end
        if i + chunk_size >= len(words):
            break
    
    return chunks

if __name__ == "__main__":
    print("Setting up vector database for personality simulator...")
    setup_vector_database()
