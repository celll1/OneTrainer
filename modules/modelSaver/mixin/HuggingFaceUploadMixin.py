from huggingface_hub import HfApi, create_repo
import os
from pathlib import Path
from tqdm import tqdm


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

            # Upload model files
            path = Path(model_path)
            if path.is_file():
                files = [path]
                base_path = path.parent
            else:
                files = list(path.rglob("*"))
                base_path = path

            for file in tqdm(files, desc="Uploading files"):
                if file.is_file():
                    relative_path = str(file.relative_to(base_path))
                    try:
                        api.upload_file(
                            path_or_fileobj=str(file),
                            path_in_repo=relative_path,
                            repo_id=repo_id,
                            token=token,
                            commit_message=f"Upload {relative_path}"
                        )
                    except Exception as e:
                        print(f"Failed to upload {relative_path}: {str(e)}")
                        continue

            return True
        except Exception as e:
            print(f"Error occurred while uploading to Hugging Face: {str(e)}")
            return False
