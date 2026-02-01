"""
Master training script to train all models sequentially.

Usage:
    python src/training/train_all.py
"""

import os
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)


def main():
    """Train all models sequentially."""
    
    print("="*60)
    print("🚀 TRAINING ALL MODELS")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    total_start = time.time()
    
    # ============================================================
    # 1. Logistic Regression
    # ============================================================
    print("\n" + "="*60)
    print("📦 Model 1/4: Logistic Regression")
    print("="*60)
    
    try:
        start = time.time()
        from src.training.train_lr import main as train_lr
        lr_trainer, lr_metrics = train_lr()
        results['Logistic Regression'] = {
            'status': 'success',
            'test_accuracy': lr_metrics['accuracy'],
            'test_f1': lr_metrics['f1_macro'],
            'time': time.time() - start
        }
    except Exception as e:
        print(f"❌ Error training Logistic Regression: {e}")
        results['Logistic Regression'] = {'status': 'failed', 'error': str(e)}
    
    # ============================================================
    # 2. SVM
    # ============================================================
    print("\n" + "="*60)
    print("📦 Model 2/4: SVM")
    print("="*60)
    
    try:
        start = time.time()
        from src.training.train_svm import main as train_svm
        svm_trainer, svm_metrics = train_svm()
        results['SVM'] = {
            'status': 'success',
            'test_accuracy': svm_metrics['accuracy'],
            'test_f1': svm_metrics['f1_macro'],
            'time': time.time() - start
        }
    except Exception as e:
        print(f"❌ Error training SVM: {e}")
        results['SVM'] = {'status': 'failed', 'error': str(e)}
    
    # ============================================================
    # 3. BiLSTM
    # ============================================================
    print("\n" + "="*60)
    print("📦 Model 3/4: BiLSTM")
    print("="*60)
    
    try:
        start = time.time()
        from src.training.train_bilstm import main as train_bilstm
        bilstm_trainer, bilstm_metrics = train_bilstm()
        results['BiLSTM'] = {
            'status': 'success',
            'test_accuracy': bilstm_metrics['accuracy'],
            'test_f1': bilstm_metrics['f1_macro'],
            'time': time.time() - start
        }
    except Exception as e:
        print(f"❌ Error training BiLSTM: {e}")
        results['BiLSTM'] = {'status': 'failed', 'error': str(e)}
    
    # ============================================================
    # 4. PhoBERT
    # ============================================================
    print("\n" + "="*60)
    print("📦 Model 4/4: PhoBERT")
    print("="*60)
    
    try:
        start = time.time()
        from src.training.train_phobert import main as train_phobert
        phobert_trainer, phobert_metrics = train_phobert()
        results['PhoBERT'] = {
            'status': 'success',
            'test_accuracy': phobert_metrics['accuracy'],
            'test_f1': phobert_metrics['f1_macro'],
            'time': time.time() - start
        }
    except Exception as e:
        print(f"❌ Error training PhoBERT: {e}")
        results['PhoBERT'] = {'status': 'failed', 'error': str(e)}
    
    # ============================================================
    # Summary
    # ============================================================
    total_time = time.time() - total_start
    
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    
    print(f"\n{'Model':<25} {'Status':<10} {'Accuracy':<12} {'F1-Score':<12} {'Time':<10}")
    print("-"*70)
    
    for model, result in results.items():
        if result['status'] == 'success':
            print(f"{model:<25} {'✅':<10} {result['test_accuracy']:.4f}       {result['test_f1']:.4f}       {result['time']:.1f}s")
        else:
            print(f"{model:<25} {'❌':<10} {'N/A':<12} {'N/A':<12} {'N/A':<10}")
    
    print("-"*70)
    print(f"\n⏱️  Total training time: {total_time/60:.1f} minutes")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find best model
    best_model = None
    best_f1 = 0
    for model, result in results.items():
        if result['status'] == 'success' and result['test_f1'] > best_f1:
            best_f1 = result['test_f1']
            best_model = model
    
    if best_model:
        print(f"\n🏆 Best Model: {best_model} (F1: {best_f1:.4f})")
    
    print("\n" + "="*60)
    print("✅ ALL TRAINING COMPLETE!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = main()
