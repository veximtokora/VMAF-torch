from setuptools import setup

setup(
    name="vmaf_torch",
    version="0.1",
    url="https://gitcode.com/huaweicloud/VMAF-torch.git",
    author="Kirill Aistov",
    author_email="kirill.aistov1@huawei.com",
    description="VMAF Reimplementation on PyTorch",
    packages=[
        "vmaf_torch",
    ],
    install_requires=["torch>=1.0.0", "yuvio"],
)
