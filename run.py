import os
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import Flask app
from app.app import app

if __name__ == "__main__":
    print("Starting Document Intelligence System...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    app.run(debug=True) 