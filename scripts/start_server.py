import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.app import app

if __name__ == "__main__":
    print("=" * 55)
    print("   AEGIS INTRUSION DETECTION SYSTEM SERVER")
    print("   Architecture: CNN + Bi-LSTM + Transformer")
    print("   Dashboard:    http://127.0.0.1:8000")
    print("=" * 55)
    
    app.run(host="127.0.0.1", port=8000, debug=False)
