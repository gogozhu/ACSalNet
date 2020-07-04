import os
import glob

dataset_root = '/home/lab-zhu.huansheng/workspace/dataset/MIT1003'
dataset_root_test = '/home/lab-zhu.huansheng/workspace/dataset/MIT300'

images_files  = glob.glob(os.path.join(dataset_root,'ALLSTIMULI/*.jpeg'))
print ("Total Number of TrainSet: {}".format(len(images_files)))
images_files.sort()
images_files_test  = glob.glob(os.path.join(dataset_root_test,'BenchmarkIMAGES/*.jpg'))
print ("Total Number of TestSet: {}".format(len(images_files_test)))


os.makedirs('data/MIT1003/images/train')
os.makedirs('data/MIT1003/images/val')
os.makedirs('data/MIT1003/images/test')
os.makedirs('data/MIT1003/maps/train')
os.makedirs('data/MIT1003/maps/val')

for f in images_files[:-100]:
    os.symlink(f, os.path.join('data/MIT1003/images/train',f.split('/')[-1]))
    os.symlink(f.replace('ALLSTIMULI','ALLFIXATIONMAPS').replace('.jpeg','_fixMap.jpg'), os.path.join('data/MIT1003/maps/train',f.split('/')[-1].replace('.jpeg','.jpg')))
for f in images_files[-100:]:
    os.symlink(f, os.path.join('data/MIT1003/images/val',f.split('/')[-1]))
    os.symlink(f.replace('ALLSTIMULI','ALLFIXATIONMAPS').replace('.jpeg','_fixMap.jpg'), os.path.join('data/MIT1003/maps/val',f.split('/')[-1].replace('.jpeg','.jpg')))

for f in images_files_test:
    os.symlink(f, os.path.join('data/MIT1003/images/test',f.split('/')[-1]))