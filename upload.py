import argparse
import os
import shutil
import tempfile
from huggingface_hub import HfApi, Repository
from huggingface_hub.utils._errors import HfHubHTTPError

def upload_model_to_hf(username, repo_name, ckpt_dir):
    # create a new repository on Hugging Face Hub if it doesn't exist
    api = HfApi()
    repo_id = f"{username}/{repo_name}"

    try:
        api.create_repo(repo_id=repo_id, private=False)
        print(f"Repository {repo_id} created successfully.")
    except HfHubHTTPError as e:
        if e.response.status_code == 409:
            print(f"Repository {repo_id} already exists.")
        else:
            raise e

    # Create a temporary directory for cloning the repository
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = Repository(local_dir=temp_dir, clone_from=repo_id)

        for file_name in os.listdir(ckpt_dir):
            full_file_name = os.path.join(ckpt_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, repo.local_dir)

        # push the files to the repository
        repo.push_to_hub()

        print(f"Model {repo_name} has been successfully uploaded to Hugging Face Hub.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload model checkpoint to Hugging Face Hub.")
    parser.add_argument('--repo_name', type=str, required=True,
                        help="The name of the Hugging Face repository.")
    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help="The directory of the model checkpoint.")
    parser.add_argument('--username', type=str, default="AI4Library",
                        help="Your Hugging Face username.")

    args = parser.parse_args()

    upload_model_to_hf(args.username, args.repo_name, args.ckpt_dir)
