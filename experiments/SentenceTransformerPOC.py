from sentence_transformers import SentenceTransformer, util
import torch


def _scalar_cosine_similarity(embedding_a, embedding_b):
	"""Return cosine similarity as a Python float, robust across devices and shapes."""
	similarity_tensor = util.cos_sim(embedding_a, embedding_b)
	return similarity_tensor.squeeze().detach().cpu().item()


def main():
	# 1. Load a pretrained Sentence Transformer model
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
	model.eval()

	# The sentences to encode
	sentences = [
		"The weather is lovely today.",
		"The weather is lovely today.",
		"My name is Ravi",
	]

	# 2. Calculate embeddings by calling model.encode()
	with torch.inference_mode():
		embeddings = model.encode(sentences, convert_to_tensor=True, device=device)
	print(embeddings.shape)
	# [3, 384]

	# Cosine similarity
	similarity_1_2 = _scalar_cosine_similarity(embeddings[0], embeddings[1])
	similarity_1_3 = _scalar_cosine_similarity(embeddings[0], embeddings[2])

	print(f"Similarity between 1 and 2: {similarity_1_2:.4f}")
	print(f"Similarity between 1 and 3: {similarity_1_3:.4f}")


if __name__ == "__main__":
	main()

