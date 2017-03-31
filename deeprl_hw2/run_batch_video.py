import os
import glob

paths = glob.glob('model-active/*/*.index')
for path in paths:
    model = path.split('/')[1]
    cmd = 'python main.py --model {0} --eval --video --model_dir {1}'.format(model, path)
    print cmd
    path_dir, path_file = path.strip().split('.')[0].rsplit('/', 1)
    check_dir = os.path.join(path_dir, path_file.rsplit('-', 1)[1])
    if os.path.exists(check_dir):
        print check_dir, 'exists'
    else:
        pass
        # os.system(cmd)
