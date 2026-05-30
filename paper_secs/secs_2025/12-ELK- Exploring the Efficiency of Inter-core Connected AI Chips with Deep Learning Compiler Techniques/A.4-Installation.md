# A.4 Installation

(1) Start by downloading the Elk artifact from GitHub:

```
git clone https://github.com/platformxlab/elk.git
cd elk
```

(2) Please make sure all prerequisites are successfully installed:

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 tmux -y
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
python3.10 -m pip install -r requirements.txt
```

