from pathlib import Path
import xml.etree.ElementTree as ET
import json

# label2idx = {'Background': 0,
#              'Crack': 1,
#              'Spallation': 2,
#              'Efflorescence': 3,
#              'ExposedBars': 4,
#              'CorrosionStain': 5}
label2idx_path = Path('data/codebrim/metadata/label2idx.json')
with label2idx_path.open('r') as fp:
    label2idx = json.load(fp)
path = Path('data/codebrim/metadata/defects.xml')
tree = ET.parse(path)
annotations = tree.getroot()
filenames = [child.attrib['name'] for child in annotations if 'name' in child.attrib]
labels = []
for defects in annotations:
    if defects.tag == 'Defect':
        image_name = defects.get('name')
        label = [0] * len(label2idx)
        for cls in defects:
            label[label2idx[cls.tag]] = int(cls.text)
        labels.append([image_name, label])
print(labels[:10])
