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

def create_sample_data():
    """Create sample data files if they don't exist"""
    
    os.makedirs('assets/data', exist_ok=True)
    
    # Create sample Elon Musk data
    elon_sample = """
    Elon Musk is known for his ambitious goals and innovative thinking. He believes in the importance of making life multiplanetary and has founded SpaceX to achieve this goal. 
    
    Musk is passionate about sustainable energy and has led Tesla to become a leader in electric vehicles. He often speaks about the urgency of transitioning to sustainable transport and energy.
    
    He is known for his direct communication style on social media and his willingness to challenge conventional wisdom. Musk believes that first principles thinking is essential for innovation.
    
    His companies include Tesla, SpaceX, Neuralink, and The Boring Company. Each addresses different aspects of advancing human civilization.
    
    Musk is known for setting aggressive timelines and pushing teams to achieve seemingly impossible goals. He believes that constraints breed creativity.
    
    He has spoken about the risks of artificial intelligence and the importance of developing AI safely. Through Neuralink, he's working on brain-computer interfaces.
    
    Musk believes in the power of engineering and manufacturing excellence. He often emphasizes the importance of the machine that builds the machine.
    """
    
    # Create sample Gandhi data
    gandhi_sample = """
    Mahatma Gandhi believed in non-violence (ahimsa) as the fundamental principle for social change. He demonstrated that peaceful resistance could overcome oppression.
    
    Gandhi taught that truth (satya) and non-violence are inseparable. He believed that means are as important as ends in achieving justice.
    
    He emphasized the importance of self-reliance and simple living. Gandhi spun his own cloth as a symbol of independence from British manufacturing.
    
    Gandhi believed in the unity of all religions and the importance of tolerance. He saw truth in all faiths and advocated for religious harmony.
    
    His philosophy included the concept of satyagraha - holding firmly to truth through non-violent protest. This became a powerful tool for social change.
    
    Gandhi stressed the importance of serving others, especially the poorest and most marginalized in society. He believed in the dignity of all people.
    
    He taught that real independence comes from within - from self-discipline, self-control, and moral courage.
    """
    
    # Write sample files
    elon_path = 'assets/data/elon_musk_data.txt'
    if not os.path.exists(elon_path):
        with open(elon_path, 'w', encoding='utf-8') as f:
            f.write(elon_sample)
        print(f"Created sample file: {elon_path}")
    
    gandhi_path = 'assets/data/gandhi_data.txt'
    if not os.path.exists(gandhi_path):
        with open(gandhi_path, 'w', encoding='utf-8') as f:
            f.write(gandhi_sample)
        print(f"Created sample file: {gandhi_path}")

if __name__ == "__main__":
    print("Setting up vector database for personality simulator...")
    
    # # Create sample data files
    # create_sample_data()
    
    # Setup vector database
    setup_vector_database()
    
    print("\nTo use your own data:")
    print("1. Replace assets/data/elon_musk_data.txt with your Elon Musk data")
    print("2. Replace assets/data/gandhi_data.txt with your Gandhi data")
    print("3. Run this script again to update the vector database")