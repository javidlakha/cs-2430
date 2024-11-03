# cs-2430

## Installation

``` bash
# Install virtual environment
python3.12 -m venv venv

# Activate virtual environment
source venv/bin/activate
echo "Python interpreter: $(which python)"

# Install required packages
pip install -r requirements.txt
```

## Usage

```bash
# Activate virtual environment
source venv/bin/activate
echo "Python interpreter: $(which python)"

# Add current directory to PYTHONPATH
export PYTHONPATH=\"\$PWD:\$PYTHONPATH\

# Run experiments
python run.py
python evaluate.py
```
