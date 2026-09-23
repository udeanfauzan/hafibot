"""
Script sederhana untuk testing model Fall Detection menggunakan video raw dari dataset.

Penggunaan:
1. Pilih video acak dari dataset (Fall atau ADL):
   python scripts/test_video_dataset.py

2. Pilih video acak khusus kategori Fall:
   python scripts/test_video_dataset.py --random Fall

3. Pilih video acak khusus kategori ADL:
   python scripts/test_video_dataset.py --random ADL

4. Test video spesifik (path relatif dari data/raw atau absolut):
   python scripts/test_video_dataset.py --video "Fall/Fall_ke_depan/Fall_ke_depan_1.mp4"
"""
import sys
import argparse
import random
from pathlib import Path

__test__ = False  # Prevent pytest from collecting this CLI tool as a test suite

# Setup path agar bisa import modul dari src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.inference import FallDetector, CONFIDENCE_THRESHOLD
except ImportError as e:
    print(f"[ERROR] Gagal import modul src: {e}")
    print("Pastikan Anda menjalankan script ini dari dalam virtual environment yang benar.")
    sys.exit(1)

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

def list_categories():
    if not RAW_DATA_DIR.exists():
        return []
    return [d.name for d in RAW_DATA_DIR.iterdir() if d.is_dir()]

def get_random_video(category=None):
    categories = list_categories()
    if not categories:
        return None
        
    if not category:
        category = random.choice(categories)
    elif category not in categories:
        print(f"[WARNING] Kategori '{category}' tidak ditemukan. Memilih kategori acak.")
        category = random.choice(categories)
    
    cat_dir = RAW_DATA_DIR / category
    subcats = [d for d in cat_dir.iterdir() if d.is_dir()]
    
    if not subcats:
        return None
        
    # Pilih subkategori acak
    subcat = random.choice(subcats)
    
    # Pilih video acak (.mp4)
    videos = [f for f in subcat.iterdir() if f.is_file() and f.suffix.lower() == '.mp4']
    if not videos:
        return None
        
    return random.choice(videos)

def main():
    parser = argparse.ArgumentParser(description="Test model Fall Detection dengan video dataset")
    parser.add_argument(
        "--video", type=str, 
        help="Path ke video spesifik (relatif terhadap data/raw/ atau path absolut)."
    )
    parser.add_argument(
        "--random", choices=['Fall', 'ADL', 'any'], default='any', 
        help="Pilih kategori video acak untuk ditest (default: any)"
    )
    parser.add_argument(
        "--threshold", type=float, default=CONFIDENCE_THRESHOLD, 
        help=f"Threshold probabilitas untuk klasifikasi Fall (default: {CONFIDENCE_THRESHOLD})"
    )
    
    args = parser.parse_args()
    video_path = None
    
    if args.video:
        path = Path(args.video)
        if path.is_absolute() and path.exists():
            video_path = path
        else:
            # Coba cari di dalam data/raw
            test_path = RAW_DATA_DIR / args.video
            if test_path.exists():
                video_path = test_path
            else:
                print(f"[ERROR] Video tidak ditemukan: {args.video}")
                print(f"Dicari di: {test_path}")
                sys.exit(1)
    else:
        # Pilih video acak sesuai kategori
        cat = None if args.random == 'any' else args.random
        video_path = get_random_video(cat)
        
        if not video_path:
            print(f"[ERROR] Tidak dapat menemukan video acak di direktori dataset ({RAW_DATA_DIR}).")
            sys.exit(1)
            
    try:
        rel_path = video_path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel_path = video_path

    print(f"\n========================================================")
    print(f"Mempersiapkan Test Video")
    print(f"========================================================")
    print(f"Video Target : {rel_path}")
    print(f"Threshold    : {args.threshold}")
    print(f"========================================================\n")
    
    # Inisialisasi dan jalankan detektor
    detector = FallDetector(confidence_threshold=args.threshold)
    detector.run(source=str(video_path))

if __name__ == "__main__":
    main()
