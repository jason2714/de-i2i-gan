from pathlib import Path
train_dir = Path('data/face/train')
val_dir = Path('data/face/val')
val_dir.mkdir(exist_ok=True, parents=True)
filepaths = [filepath for filepath in train_dir.iterdir()]
for filepath in filepaths[int(len(filepaths)*0.9):]:
    new_filepath = val_dir / filepath.name
    filepath.rename(new_filepath)