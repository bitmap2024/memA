import os
import argparse
import subprocess
from typing import Optional
from huggingface_hub import snapshot_download


class ModelDownloader:
    def __init__(
        self,
        repo_id: str,
        local_dir: str,
        endpoint: Optional[str] = "https://hf-mirror.com",
        force_download: bool = False,
        resume_download: bool = True,
        max_workers: int = 4
    ) -> None:
        self.repo_id = repo_id
        self.local_dir = local_dir
        self.endpoint = endpoint
        self.force_download = force_download
        self.resume_download = resume_download
        self.max_workers = max_workers

    def configure_environment(self) -> None:
        if self.endpoint:
            os.environ["HF_ENDPOINT"] = self.endpoint

    def download_with_api(self) -> None:
        snapshot_download(
            repo_id=self.repo_id,
            endpoint=self.endpoint,
            local_dir=self.local_dir,
            force_download=self.force_download,
            max_workers=self.max_workers,
            ignore_patterns=["*.DS_Store", "imgs/*"],
        )

    def download_with_cli(self) -> None:
        command = [
            "hf",
            "download",
            self.repo_id,
            "--local-dir",
            self.local_dir,
            "--exclude",
            "*.DS_Store",
            "--exclude",
            "imgs/*",
        ]
        subprocess.run(command, check=True)

    def download(self) -> None:
        self.configure_environment()
        print(f"开始下载模型到: {self.local_dir}")
        print(f"使用镜像源: {os.environ.get('HF_ENDPOINT', 'default')}")
        try:
            self.download_with_api()
            print("模型下载完成！")
        except Exception as api_error:
            print(f"API下载失败: {api_error}")
            print("尝试使用 hf cli 下载...")
            try:
                self.download_with_cli()
                print("使用 hf cli 下载成功！")
            except FileNotFoundError:
                print("未找到 hf 命令，请运行: pip install -U huggingface_hub")
            except subprocess.CalledProcessError as cli_error:
                print(f"CLI下载也失败: {cli_error}")
                print("请检查网络连接或尝试手动下载")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 Hugging Face 下载模型到本地目录")
    parser.add_argument("--repo-id", required=True, help="Hugging Face 仓库标识，如 'IndexTeam/IndexTTS-2'")
    parser.add_argument("--local-dir", required=True, help="模型保存的本地目录路径")
    parser.add_argument("--endpoint", default="https://hf-mirror.com", help="Hugging Face 镜像端点")
    parser.add_argument("--max-workers", type=int, default=4, help="并发下载线程数")
    parser.add_argument("--force-download", action=argparse.BooleanOptionalAction, default=False, help="是否强制重新下载")
    parser.add_argument("--resume-download", action=argparse.BooleanOptionalAction, default=True, help="是否断点续传")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    downloader = ModelDownloader(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        endpoint=args.endpoint,
        force_download=args.force_download,
        resume_download=args.resume_download,
        max_workers=args.max_workers,
    )
    downloader.download()
