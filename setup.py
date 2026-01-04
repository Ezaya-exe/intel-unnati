"""
Setup script for NCERT Doubt Solver
Ensures all directories exist and dependencies are ready
"""

import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directory_structure():
    """Create required directories"""
    directories = [
        "data/raw_pdfs",
        "data/processed/ocr_output",
        "data/processed/images",
        "data/processed/chunks",
        "data/vector_db",
        "logs",
        "models",
        "configs"
    ]
    
    logger.info("Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"  ✓ {directory}")
    
    logger.info("✓ Directory structure created")


def create_env_template():
    """Create .env template file"""
    env_template = """# NCERT Doubt Solver - Environment Variables

# Optional: OpenAI API Key (for GPT models)
# OPENAI_API_KEY=your_openai_key_here

# Optional: Cohere API Key (for embeddings)
# COHERE_API_KEY=your_cohere_key_here

# Optional: HuggingFace Token (for gated models)
# HUGGINGFACE_TOKEN=your_hf_token_here

# Model Settings
MODEL_PROVIDER=huggingface  # Options: huggingface, openai, ollama
LLM_MODEL=google/flan-t5-base
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2

# Server Settings
API_HOST=0.0.0.0
API_PORT=8000

# Vector DB Settings
VECTOR_DB_PATH=data/vector_db
CHUNK_SIZE=512
CHUNK_OVERLAP=128
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(env_template)
        logger.info("✓ Created .env template file")
    else:
        logger.info("  .env file already exists")


def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Data files
data/raw_pdfs/*.pdf
data/processed/
data/vector_db/

# Models
models/*.bin
models/*.pkl

# Logs
logs/*.log

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
"""
    
    gitignore_file = Path(".gitignore")
    with open(gitignore_file, 'w') as f:
        f.write(gitignore_content)
    logger.info("✓ Created .gitignore file")


def check_dependencies():
    """Check if key dependencies can be imported"""
    logger.info("\nChecking dependencies...")
    
    dependencies = {
        'paddleocr': 'PaddleOCR',
        'cv2': 'OpenCV',
        'fitz': 'PyMuPDF',
        'chromadb': 'ChromaDB',
        'sentence_transformers': 'SentenceTransformers',
        'fastapi': 'FastAPI',
        'transformers': 'Transformers',
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            logger.info(f"  ✓ {name}")
        except ImportError:
            logger.warning(f"  ✗ {name} - Not installed")
            missing.append(name)
    
    if missing:
        logger.warning(f"\nMissing dependencies: {', '.join(missing)}")
        logger.warning("Run: pip install -r requirements.txt")
        return False
    else:
        logger.info("\n✓ All dependencies installed")
        return True


def create_sample_config():
    """Create sample configuration file"""
    config = {
        "pipeline": {
            "ocr": {
                "use_gpu": False,
                "supported_languages": ["english", "hindi", "sanskrit", "marathi"]
            },
            "chunking": {
                "chunk_size": 512,
                "chunk_overlap": 128
            },
            "embedding": {
                "model": "paraphrase-multilingual-mpnet-base-v2",
                "batch_size": 32
            },
            "llm": {
                "provider": "huggingface",
                "model": "google/flan-t5-base",
                "temperature": 0.7,
                "max_tokens": 500
            }
        },
        "subjects": [
            "Mathematics",
            "Science",
            "Social Science",
            "Hindi",
            "Sanskrit",
            "English"
        ],
        "grades": [5, 6, 7, 8, 9, 10]
    }
    
    import json
    config_file = Path("configs/pipeline_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("✓ Created configuration file")


def main():
    """Run complete setup"""
    logger.info("="*60)
    logger.info("NCERT Doubt Solver - Setup")
    logger.info("="*60)
    
    # Create directories
    create_directory_structure()
    
    # Create config files
    create_env_template()
    create_gitignore()
    create_sample_config()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    logger.info("\n" + "="*60)
    if deps_ok:
        logger.info("✓ Setup complete!")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("1. Add NCERT PDF files to data/raw_pdfs/")
        logger.info("2. Run the pipeline: python src/main_pipeline.py")
        logger.info("3. Start API server: python src/api/api_server.py")
    else:
        logger.info("⚠ Setup incomplete - install missing dependencies")
        logger.info("="*60)
        logger.info("\nRun: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
