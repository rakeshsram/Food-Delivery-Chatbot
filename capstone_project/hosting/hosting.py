import os

from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
repo_id = "rakesh1715/Food-Delivery-Chatbot"
api.create_repo(
    repo_id=repo_id, 
    repo_type="space", 
    space_sdk="docker", 
    exist_ok=True
)
api.upload_folder(
        folder_path="capstone_project/deployment",  # the local folder containing your files
        repo_id=repo_id,  # the target repo
        repo_type="space",  # dataset, model, or space
        path_in_repo="",  # optional: subfolder path inside the repo
)

# ADD THE GROQ API KEY TO HF REPO SECRETS
api.add_space_secret(
    repo_id=repo_id,
    key="GROQ_API_KEY",
    value="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"
)
