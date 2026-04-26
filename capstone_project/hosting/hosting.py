import os

from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
        folder_path="capstone_project/deployment",  # the local folder containing your files
        repo_id="rakesh1715/Food-Delivery-Chatbot",  # the target repo
        repo_type="space",  # dataset, model, or space
        path_in_repo="",  # optional: subfolder path inside the repo
)

# ADD THE GROQ API KEY TO HF REPO SECRETS
api.add_space_secret(
    repo_id="rakesh1715/Food-Delivery-Chatbot",
    key="GROQ_API_KEY",
    value="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"
