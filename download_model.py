"""Download Google's official model next to this script."""
from pathlib import Path
from urllib.request import urlopen
import shutil
URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
if __name__ == '__main__':
    output = Path(__file__).resolve().parent / 'hand_landmarker.task'
    temporary = output.with_suffix('.download')
    try:
        with urlopen(URL, timeout=120) as source, temporary.open('wb') as target:
            shutil.copyfileobj(source, target)
        temporary.replace(output)
        print(output)
    finally:
        temporary.unlink(missing_ok=True)
