from pathlib import Path
base_dir = Path('data/face')
train_dir = base_dir / 'train'
val_dir = base_dir / 'val'
if not val_dir.exists():
    val_dir.mkdir(exist_ok=True, parents=True)
    filepaths = [filepath for filepath in train_dir.iterdir()]
    for filepath in filepaths[int(len(filepaths)*0.9):]:
        new_filepath = val_dir / filepath.name
        filepath.rename(new_filepath)