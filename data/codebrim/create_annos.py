from pathlib import Path
import xml.etree.ElementTree as ET
import json


def create_annos(anno_dir):
    label2idx_path = anno_dir / 'label2idx.json'
    for file_path in anno_dir.iterdir():
        if file_path.suffix == '.xml':

            # read from xml
            tree = ET.parse(file_path)
            annotations = tree.getroot()

            # get label2idx dictionary
            label2idx = None
            if not label2idx_path.exists():
                idx = 0
                while annotations[idx].tag != 'Defect':
                    idx += 1
                label2idx = {cls: idx
                             for idx, cls in enumerate(sorted([cls.tag for cls in annotations[0]]))}
                with label2idx_path.open('w') as fp:
                    json.dump(label2idx, fp, indent=2)
            else:
                with label2idx_path.open('r') as fp:
                    label2idx = json.load(fp)

            # read labels from xml
            labels = dict()
            for defects in annotations:
                if defects.tag == 'Defect':
                    image_name = defects.get('name')
                    label = [0] * len(label2idx)
                    for cls in defects:
                        label[label2idx[cls.tag]] = int(cls.text)
                    labels[image_name] = label

            # dump labels to json file
            json_labels_path = file_path.parent / f'{file_path.stem}.json'
            with json_labels_path.open('w') as fp:
                json.dump(labels, fp, indent=2)


if __name__ == '__main__':
    create_annos(Path('data/codebrim/metadata'))
