from tqdm import tqdm
import os
import shutil

def set_up_version():
    with open('version.txt', 'r+') as f:
        version = f.read()
        version = float(version)
        new_version = version + 0.1
        f.seek(0)
        f.write(str(round(new_version, 1)))
        f.truncate()
        f.close()
        os.mkdir('./versions/{}'.format(new_version))
        shutil.copyfile('image-classifier.py', './versions/{}/image-classifier.py'.format(new_version))

if __name__ == '__main__':
    set_up_version()