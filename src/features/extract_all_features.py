import os
import time

from src.features.tfidf_features import extract_tfidf_features
from src.features.embedding_features import extract_embedding_features
from src.features.phobert_features import extract_phobert_features
from config import cfg


def main():
    print("="*60)
    print("FEATURE EXTRACTION FOR FAKE NEWS DETECTION")
    print("="*60)
    
    train_path = os.path.join(cfg.PATHS.splits_dir, 'train.csv')
    val_path = os.path.join(cfg.PATHS.splits_dir, 'val.csv')
    test_path = os.path.join(cfg.PATHS.splits_dir, 'test.csv')
    features_dir = cfg.PATHS.features_dir
    
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            print(f"Error: {path} not found!")
            print("Please run the data splitting script first.")
            return
    
    total_start = time.time()
    
    print("\n" + "="*60)
    print("Step 1/3: Extracting TF-IDF Features")
    print("="*60)
    
    start = time.time()
    tfidf_features = extract_tfidf_features(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        output_dir=cfg.PATHS.tfidf_dir,
        max_features=cfg.TFIDF.max_features,
        ngram_range=cfg.TFIDF.ngram_range   # Unigrams/bigrams/trigrams per config
    )
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Train shape: {tfidf_features['X_train'].shape}")
    print("\n" + "="*60)
    print("Step 2/3: Extracting Embedding Features")
    print("="*60)
    
    start = time.time()
    embedding_features = extract_embedding_features(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        output_dir=cfg.PATHS.embedding_dir,
        max_vocab_size=50000,
        max_seq_length=cfg.BILSTM.max_seq_length,
        min_freq=2
    )
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Vocabulary size: {embedding_features['extractor'].vocab_size}")
    
    print("\n" + "="*60)
    print("Step 3/3: Extracting PhoBERT Features")
    print("="*60)
    
    start = time.time()
    try:
        phobert_features = extract_phobert_features(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            output_dir=cfg.PATHS.phobert_dir,
            max_length=cfg.PHOBERT.max_seq_len
        )
        print(f"Time: {time.time() - start:.2f}s")
        print(f"Train shape: {phobert_features['train_encoded']['input_ids'].shape}")
    except (ImportError, RuntimeError, OSError) as e:
        print(f"Warning: PhoBERT feature extraction failed: {e}")
        print("You may need to install transformers: pip install transformers")
        print("Skipping PhoBERT features...")
    
    total_time = time.time() - total_start
    
    print("\n" + "="*60)
    print("FEATURE EXTRACTION COMPLETE!")
    print("="*60)
    print(f"\n  Total time: {total_time:.2f}s")
    print(f"\n Features saved to: {features_dir}")
    print("   ├── tfidf/")
    print("   │   ├── tfidf_vectorizer.pkl")
    print("   │   └── tfidf_features.pkl")
    print("   ├── embedding/")
    print("   │   ├── embedding_extractor.pkl")
    print("   │   └── embedding_features.pkl")
    print("   └── phobert/")
    print("       ├── phobert_config.pkl")
    print("       └── phobert_features.pkl")
    
    print("\n" + "="*60)
    print("NEXT STEP: Train models using:")
    print("   python src/training/train_lr.py      # Logistic Regression")
    print("   python src/training/train_svm.py     # SVM")
    print("   python src/training/train_bilstm.py  # BiLSTM")
    print("   python src/training/train_phobert.py # PhoBERT")
    print("="*60)


if __name__ == "__main__":
    main()
