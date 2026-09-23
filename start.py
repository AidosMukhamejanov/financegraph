"""Rebuild analytics before serving, including on deployment."""
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    load_dotenv(root / '.env', override=False, encoding='utf-8-sig')
    subprocess.run([sys.executable, str(root / 'pipeline.py'), '--data', str(root / 'data'),
                    '--out', os.getenv('GRAPH_OUT', str(root / 'out'))], check=True, cwd=root)
    import uvicorn
    uvicorn.run('api:app', host='0.0.0.0', port=int(os.getenv('PORT', '8000')), app_dir=str(root))
