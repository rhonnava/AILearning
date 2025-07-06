from sentence_transformers import SentenceTransformer, util

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "The weather is lovely today.",
    "My name is Ravi",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences, convert_to_tensor=True)
print(embeddings.shape)
# [3, 384]

# Cosine similarity
similarity_1_2 = util.cos_sim(embeddings[0], embeddings[1])
similarity_1_3 = util.cos_sim(embeddings[0], embeddings[2])

print(f"Similarity between 1 and 2: {similarity_1_2.item():.4f}")
print(f"Similarity between 1 and 3: {similarity_1_3.item():.4f}")

