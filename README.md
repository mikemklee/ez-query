# Query Zendesk articles with LLM

### How to run locally

1. Clone this repo

2. Create a virtual environment
    ```bash
    python -m venv env_name
    ```

3. Activate the virtual environment
    ```bash
    source env_name/bin/activate
    ```

4. Install dependencies
   ```bash
    pip install -r requirements.txt
   ```

5. Create a `.env` file on the project root, and enter in your OPENAI_API_KEY
   ```bash
    OPENAI_API_KEY=your_api_key
   ```

6. Run the app
   ```bash
    python main.py --question "What you want to find out" --subdomain "your_subdomain"
   ```

### For help
```bash
python main.py --help
```

### References
[llama-index](https://gpt-index.readthedocs.io/en/latest/getting_started/installation.html)