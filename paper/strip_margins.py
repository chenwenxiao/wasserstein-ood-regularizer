import os
import subprocess

for parent, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.pdf'):
            src = os.path.join(parent, f)
            dst_dir = os.path.join('../dist.strip/', parent)
            dst = os.path.join(dst_dir, f)
            if not os.path.isdir(dst_dir):
                os.makedirs(dst_dir)
            if not os.path.exists(dst):
                print('* {} -> {}'.format(src, dst))
                subprocess.check_call(['pdfcrop', src ,dst])

