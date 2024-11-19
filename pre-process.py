import os
import string
from collections import Counter

def read_vocabulary(vocab_path):
    """Read the IMDB vocabulary file"""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)

def preprocess_text(text):
    """Clean and tokenize text"""
    # Convert to lowercase
    text = text.lower()
    
    # Separate punctuation from words
    for punct in string.punctuation:
        text = text.replace(punct, ' ' + punct + ' ')
    
    # Split into words and remove empty strings
    words = [word.strip() for word in text.split()]
    return [word for word in words if word]

def process_directory(root_dir, vocab, output_file):
    """Process all reviews in directory and write to output file"""
    with open(output_file, 'w', encoding='utf-8') as out:
        # Process each class directory (pos/neg)
        for class_name in ['pos', 'neg']:
            class_dir = os.path.join(root_dir, class_name)
            
            # Process each review file
            for filename in os.listdir(class_dir):
                if filename.endswith('.txt'):
                    try:
                        # Read and preprocess the review
                        with open(os.path.join(class_dir, filename), 'r', encoding='utf-8') as f:
                            text = f.read()
                        
                        # Get word counts (only for words in vocabulary)
                        words = preprocess_text(text)
                        word_counts = Counter(word for word in words if word in vocab)
                        
                        # Create feature string (word:count pairs)
                        features = ' '.join(f'{word}:{count}' 
                                         for word, count in word_counts.items())
                        
                        # Write to output file
                        out.write(f'{class_name} {features}\n')
                        
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python pre-process.py <vocab_file> <input_dir> <output_file>")
        print("Example: python pre-process.py imdb.vocab train train_processed.txt")
        sys.exit(1)
    
    vocab_file = sys.argv[1]
    input_dir = sys.argv[2]
    output_file = sys.argv[3]
    
    # Read vocabulary
    vocab = read_vocabulary(vocab_file)
    
    # Process directory
    process_directory(input_dir, vocab, output_file)