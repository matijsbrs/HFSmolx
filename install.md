cd /home/matijs/smollm2&& rm -rf .venv
cd /home/matijs/ifier && python3.10 -m venv .venv
cd /home/matijs/git/ImageClassifier && .venv/bin/python --version
cd /home/matijs/git/ImageClassifier && .venv/bin/pip install --upgrade pip
cd /home/matijs/git/ImageClassifier && .venv/bin/pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117

