import os

salicon_dataset_root = '/home/lab-zhu.huansheng/workspace/dataset/SALICON'

images_files  = os.listdir(os.path.join(salicon_dataset_root,'images'))
print ("Total Number of Images: {}".format(len(images_files)))

os.makedirs('data/salicon/images/train')
os.makedirs('data/salicon/images/val')
os.makedirs('data/salicon/images/test')

for f in images_files:
    if 'test' in f:
        os.symlink(os.path.join(salicon_dataset_root,'images',f), os.path.join('data/salicon/images/test',f))
    elif 'val' in f:
        os.symlink(os.path.join(salicon_dataset_root,'images',f), os.path.join('data/salicon/images/val',f))
    else:
        os.symlink(os.path.join(salicon_dataset_root,'images',f), os.path.join('data/salicon/images/train',f))


os.makedirs('data/salicon/maps')
os.symlink(os.path.join(salicon_dataset_root,'maps','train'), os.path.join('data/salicon/maps/train'))
os.symlink(os.path.join(salicon_dataset_root,'maps','val'), os.path.join('data/salicon/maps/val'))