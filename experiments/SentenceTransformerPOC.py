from sentence_transformers import SentenceTransformer, util
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_name="all-MiniLM-L6-v2"):
    """Load the Sentence Transformer model with error handling."""
    try:
        model = SentenceTransformer(model_name)
        logger.info(f"Successfully loaded model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        raise

def encode_sentences(model, sentences):
    """Encode sentences with error handling and validation."""
    if not sentences:
        raise ValueError("Sentences list cannot be empty")
    
    try:
        # Remove duplicates to avoid unnecessary computation
        unique_sentences = list(dict.fromkeys(sentences))
        logger.info(f"Encoding {len(unique_sentences)} unique sentences")
        
        embeddings = model.encode(unique_sentences, convert_to_tensor=True)
        return embeddings, unique_sentences
    except Exception as e:
        logger.error(f"Failed to encode sentences: {str(e)}")
        raise

def calculate_similarities(embeddings, sentences):
    """Calculate cosine similarities between sentence pairs."""
    try:
        if len(embeddings) < 2:
            raise ValueError("Need at least 2 sentences to calculate similarity")
        
        # Calculate similarities for all pairs
        similarities = {}
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = util.cos_sim(embeddings[i], embeddings[j])
                similarities[f"{i+1}_{j+1}"] = {
                    'sentences': (sentences[i], sentences[j]),
                    'similarity': sim.item()
                }
        
        return similarities
    except Exception as e:
        logger.error(f"Failed to calculate similarities: {str(e)}")
        raise

def main():
    """Main function with proper resource management."""
    model = None
    try:
        # 1. Load a pretrained Sentence Transformer model
        model = load_model()
        
        # The sentences to encode
        sentences = [
            "The weather is lovely today.",
            "The weather is lovely today.",  # Duplicate sentence
            "My name is Ravi",
        ]
        
        # 2. Calculate embeddings by calling model.encode()
        embeddings, unique_sentences = encode_sentences(model, sentences)
        logger.info(f"Embeddings shape: {embeddings.shape}")
        
        # 3. Calculate similarities
        similarities = calculate_similarities(embeddings, unique_sentences)
        
        # 4. Print results
        for pair_key, data in similarities.items():
            sent1, sent2 = data['sentences']
            sim_score = data['similarity']
            print(f"Similarity between sentences {pair_key}: {sim_score:.4f}")
            print(f"  Sentence 1: {sent1}")
            print(f"  Sentence 2: {sent2}")
            print()
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
    finally:
        # Clean up GPU memory if using CUDA
        if model is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared")

if __name__ == "__main__":
    main()

