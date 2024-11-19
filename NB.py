from math import log
import sys

def load_preprocessed_data(filepath):
    """
    Load preprocessed data from file where each line is:
    label word1:count1 word2:count2 ...
    """
    documents = []
    vocabulary = set()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
                
            # First part is the label
            label = parts[0]
            
            # Rest are word:count pairs
            word_counts = {}
            for word_count in parts[1:]:
                word, count = word_count.split(':')
                word_counts[word] = int(count)
                vocabulary.add(word)
                
            documents.append((word_counts, label))
    
    return documents, vocabulary

def train_naive_bayes(documents, classes):
    """
    Train Naive Bayes on preprocessed documents
    documents: list of (word_counts_dict, label) tuples
    classes: set of possible classes (e.g., {'pos', 'neg'})
    """
    # Initialize parameters
    logprior = {}
    loglikelihood = {}
    
    # Count documents in each class
    N_doc = len(documents)
    class_counts = {c: 0 for c in classes}
    for _, label in documents:
        class_counts[label] += 1
    
    # Get vocabulary from all documents
    V = set()
    for word_counts, _ in documents:
        V.update(word_counts.keys())
    
    # Calculate class priors (no smoothing for priors)
    for c in classes:
        logprior[c] = log(class_counts[c] / N_doc)
    
    # Create bigdoc for each class (total word counts per class)
    bigdoc = {c: {} for c in classes}
    for word_counts, label in documents:
        for word, count in word_counts.items():
            bigdoc[label][word] = bigdoc[label].get(word, 0) + count
    
    # Calculate word likelihoods with add-1 smoothing
    for c in classes:
        # Calculate total words in class (including smoothing)
        total_words = sum(bigdoc[c].values()) + len(V)  # Add V for smoothing
        
        # Calculate probability for each word
        for word in V:
            count = bigdoc[c].get(word, 0) + 1  # Add-1 smoothing
            loglikelihood[(word, c)] = log(count / total_words)
    
    return V, logprior, loglikelihood

def test_naive_bayes(word_counts, V, logprior, loglikelihood, classes):
    """
    Classify a single document using the trained model
    word_counts: dictionary of {word: count} for the document
    """
    # Initialize scores with prior probabilities
    scores = {c: logprior[c] for c in classes}
    
    # Add log likelihood for each word found in document
    for word, count in word_counts.items():
        if word in V:  # Only consider words in our vocabulary
            for c in classes:
                # Multiply likelihood by count of word in document
                # (In log space, this is adding the log likelihood count times)
                scores[c] += count * loglikelihood.get((word, c), 0)
    
    # Return class with highest score
    return max(scores.items(), key=lambda x: x[1])[0]

def save_model(V, logprior, loglikelihood, model_file):
    """Save model parameters to file"""
    with open(model_file, 'w', encoding='utf-8') as f:
        # Save vocabulary
        f.write("### Vocabulary ###\n")
        for word in sorted(V):
            f.write(f"{word}\n")
            
        # Save priors
        f.write("\n### Priors ###\n")
        for c, prior in logprior.items():
            f.write(f"{c} {prior}\n")
            
        # Save likelihoods
        f.write("\n### Likelihoods ###\n")
        for (word, c), likelihood in loglikelihood.items():
            f.write(f"{word} {c} {likelihood}\n")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python NB.py <train_file> <test_file> <model_file> <output_file>")
        sys.exit(1)
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    model_file = sys.argv[3]
    output_file = sys.argv[4]
    
    # Load training data
    train_docs, vocab = load_preprocessed_data(train_file)
    classes = {'pos', 'neg'}
    
    # Train model
    V, logprior, loglikelihood = train_naive_bayes(train_docs, classes)
    
    # Save model
    save_model(V, logprior, loglikelihood, model_file)
    
    # Test model and write predictions
    test_docs, _ = load_preprocessed_data(test_file)
    correct = 0
    total = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for word_counts, true_label in test_docs:
            predicted_label = test_naive_bayes(word_counts, V, logprior, loglikelihood, classes)
            f.write(f"{predicted_label}\n")
            
            # Update accuracy counting
            total += 1
            if predicted_label == true_label:
                correct += 1
        
        # Write accuracy as last line
        accuracy = (correct / total) * 100
        f.write(f"\nAccuracy: {accuracy:.2f}%")