# Run with GPU
conda create -n torch-gpu python=3.8
conda activate torch-gpu
pip3 install -U --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip3 install numpy matplotlib tqdm

# for installation verification:
conda install -c conda-forge jupyter jupyterlab
python check-gpu.py