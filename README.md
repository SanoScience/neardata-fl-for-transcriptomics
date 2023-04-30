# Neardata Fl For Transcriptomics
-----

**Table of Contents**

- [Environment setup](#env-setup)
- [License](#license)

## Environment setup
### Compile the ```requirements.in``` file to get ```requirements.txt```
```bash
pip-compile requirements.in
```
### Create and activate your virtual environment.
```bash
python -m venv venv
source venv/bin/activate
```
### Now install packages from the requirements.txt file.
```bash
pip install -r requirements.txt
```
### Install also neardata-fl-for-transcriptomics from source to run scripts from the root dir.
```
pip install .
```
### We use NeptuneAI as the MLOps tool. The api token should be stored in .env file, so that python-dotenv package can load it as an env variable. The .env file should have the following entry, where 'xxx' is the api token:
```bash
NEPTUNE_API_TOKEN=xxx
```
## License


