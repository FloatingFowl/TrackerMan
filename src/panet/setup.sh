export PATH=$PATH:/usr/local/cuda/bin
pip install -r requirements.txt
cd lib
sh make.sh

echo("Download weights from https://drive.google.com/file/d/1-pVZQ3GR6Aj7KJzH9nWoRQ-Lts8IcdMS/view?usp=sharing and save it in the root folder")
