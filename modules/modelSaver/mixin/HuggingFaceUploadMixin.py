
import os
from huggingface_hub import HfApi, create_repo

class HuggingFaceUploadMixin:
    def _upload_to_huggingface(
            self,
            model_path: str,
            repo_id: str,
            token: str,
            private: bool = True,
            commit_message: str = "Model update"
    ):
        """
        Upload model to Hugging Face

        Args:
            model_path: Path to the model to upload
            repo_id: Hugging Face repository ID (username/repo_name)
            token: Hugging Face API token
            private: Whether to make the repository private
            commit_message: Commit message
        """
        api = HfApi()

        try:
            # Create repository if it doesn't exist
            create_repo(
                repo_id,
                private=private,
                token=token,
                repo_type="model",
                exist_ok=True
            )

            if os.path.isdir(model_path):
                api.upload_folder(
                    folder_path=model_path,
                    repo_id=repo_id,
                    token=token,
                    commit_message=commit_message
                )
            else:
                api.upload_file(
                    path_or_fileobj=model_path,
                    path_in_repo=os.path.basename(model_path),
                    repo_id=repo_id,
                    token=token,
                    commit_message=commit_message
                )

            return True
        except Exception as e:
            print(f"Error occurred while uploading to Hugging Face: {str(e)}")
            return False
