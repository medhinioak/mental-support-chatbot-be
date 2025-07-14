# Emotional Support Chatbot Backend

This is the backend for the mental support application. This app runs on Python 3.10.16. Behaviour in other versions may be unexpected.

---

## Features

- Finetuned distilBERT model to detect emotions
- Finetuned bert-base-uncased model to detect suicidality
- RESTful APIs for chat and export
- Notebooks for training the emotion and risk models
- Script to calculate BERTscore of the generated response and to evaluate the response using LLM-as-a-judge
- Feature to export chat using Mailgun

---

### Prerequisites

- 3.10.16
- `pip`
- mistral model running on port 11434. Run `ollama run mistral --verbose` on your local machine
- An access token created on Huggingfaces
- An account created on Mailgun. (We need a domain and a API key to run the export feature)

### Getting a Huggingfaces access token

This project uses the https://huggingface.co/datasets/AIMH/SWMH database which is protected because of its sensitive nature of the content. This requires you to authenticate yourself before you can use it. Therefore you will need a dev token to run this application. You can create one by creating a huggingfaces account clicking on Profile -> Access Tokens

Select `+Create new token` and name your token and select `Read access to contents of all repos under your personal namespace` and `Read access to contents of all public gated repos you can access` and click on `Create token`. Copy the value of this token and write it into `HUGGINGFACES_TOKEN` inside config.json

### Setting up a Mailgun account

Go to https://www.mailgun.com/ and select `Start for free`. Add your information and set up an account.

Go to dashboard, select `API keys` and click on `Add new key`. Add a short description and click on `Create key`. Copy the API key and add it into config.json

Go back to the dashboard and click on `Domains`. Select `Add new domain` and add the relavant details. Sandbox domains are restricted to 5 recipients so make sure you add the email to which you want to test the export feature on. Copy the domain name and add it into config.json

### Installation (Local)

```bash
# Clone the repo
git clone https://github.com/medhinioak/mental-support-chatbot-be.git
cd mental-support-chatbot-be

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn app.main:app --reload
```
