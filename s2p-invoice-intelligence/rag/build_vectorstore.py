# rag/build_vectorstore.py
# Shortcut to build the vectorstore — run this on Day 3
# Usage: python rag/build_vectorstore.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.pipeline import build_vectorstore

if __name__ == "__main__":
    build_vectorstore()
    print("\nVectorstore ready. Now run: python anomaly/detector.py")
