# china-ai-scientist
本项目基于github上的一个开源项目AI-Scientist（AI科学家）做二次修改，更适合中国宝宝体质。
在中国，由于大陆墙，我们无法访问国外AI大模型官网，无法直接使用API Key。并且在使用原项目踩到的一些坑都会写入这篇文档，帮助读者快速使用。

<h1 align="center">
  <a href="https://github.com/SakanaAI/AI-Scientist/blob/main/docs/logo_2.png">
    <img src="docs/logo_2.png" width="215" /></a><br>
  <b>AI科学家: 拥抱全自动化</b><br>
  <b>将科学发现开源到底 🧑‍🔬</b><br>
</h1>

<p align="center">
  📚 <a href="https://arxiv.org/abs/2408.06292">[论文]</a> |
  📝 <a href="https://sakana.ai/ai-scientist/">[官网]</a> |
  📂 <a href="https://drive.google.com/drive/folders/1G7A0wTqfXVa-cpexjk0oaXakaSJwffEt">[驱动文件夹]</a>
</p>

人工智能的一大挑战是开发能够进行科学研究和发现新知识的智能体。虽然前沿模型已经被用来辅助人类科学家，
例如用于头脑风暴或编写代码，但它们仍然需要广泛的人工监督，或者受限于特定的任务。

我们很高兴地介绍AI科学家，这是第一个全面的系统，能够实现完全自动化的科学发现，使得基础模型如大型语言模型（LLMs）能够独立进行研究。

我们进一步提供了我们论文中的所有运行和数据，您可以在
[这里](https://drive.google.com/drive/folders/1G7A0wTqfXVa-cpexjk0oaXakaSJwffEt?usp=sharing)
找到。我们对每个基础模型在每个模板上运行了大约50个创意。我们强烈建议阅读一些
[Claude论文](https://drive.google.com/drive/folders/1Mmpz6M1FK4q8e-SewgZcUzdeD0Q2zC39?usp=sharing)
（特别是扩散相关的），以了解其优缺点。以下是AI科学家生成的一些示例论文📝：

1. [DualScale Diffusion: Adaptive Feature Balancing for Low-Dimensional Generative Models](https://github.com/SakanaAI/AI-Scientist/blob/main/example_papers/adaptive_dual_scale_denoising.pdf)
2. [Multi-scale Grid Noise Adaptation: Enhancing Diffusion Models For Low-dimensional Data](https://github.com/SakanaAI/AI-Scientist/blob/main/example_papers/grid_based_noise_adaptation.pdf)
3. [GAN-Enhanced Diffusion: Boosting Sample Quality and Diversity](https://github.com/SakanaAI/AI-Scientist/blob/main/example_papers/gan_diffusion.pdf)
4. [DualDiff: Enhancing Mode Capture in Low-dimensional Diffusion Models via Dual-expert Denoising](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/dual_expert_denoiser.pdf) 
5. [StyleFusion: Adaptive Multi-style Generation in Character-Level Language Models](https://github.com/SakanaAI/AI-Scientist/blob/main/example_papers/multi_style_adapter.pdf)
6. [Adaptive Learning Rates for Transformers via Q-Learning](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/rl_lr_adaptation.pdf)
8. [Unlocking Grokking: A Comparative Study of Weight Initialization Strategies in Transformer Models](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/weight_initialization_grokking.pdf)
9. [Grokking Accelerated: Layer-wise Learning Rates for Transformer Generalization](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/layerwise_lr_grokking.pdf)
10. [Grokking Through Compression: Unveiling Sudden Generalization via Minimal Description Length](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/mdl_grokking_correlation.pdf)
11. [Accelerating Mathematical Insight: Boosting Grokking Through Strategic Data Augmentation](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/data_augmentation_grokking.pdf)

**小笔记**: 注意！此代码库将执行由大型语言模型（LLM）编写的代码。这种自主性伴随有各种风险和挑战。
这包括例如使用潜在危险的软件包、网络访问以及可能产生进程。请自行判断使用。请确保适当进行[容器化](#containerization)并限制网络访问。

<p align="center">
  <a href="https://github.com/SakanaAI/AI-Scientist/blob/main/example_papers/adaptive_dual_scale_denoising/adaptive_dual_scale_denoising.pdf"><img src="https://github.com/SakanaAI/AI-Scientist/blob/main/docs/anim-ai-scientist.gif" alt="Adaptive Dual Scale Denoising" width="80%" />
</p>

# 目录

1. [材料准备](#材料准备)
2. [论文选题](#论文选题)
3. [论文评审](#论文评审)
4. [论文模板](#论文模板)
5. [FQA](#faq)
6. [引用AI科学家](#citing-the-ai-scientist)
8. [容器化](#containerization)

# 材料准备
## 算力准备
为了减少您的资金消耗，本项目计划在两种环境（家中、云上）开展AI研究，请根据自己的条件选择合适的实验环境：
#### 在家中开展试验
1. 一台带有英伟达显卡的电脑，显卡要求：Geforce RTX30系列及以上版本，例如RTX3050、RTX4060、RTX5080。原因是项目需要进行bfloat16计算， 
需要支持sm_80架构的显卡。Tesla显卡中，T系列和P系列不支持sm_80，请不要购买，推荐Tesla A系列和H系列，例如A10、A100、H800。
2. 魔法网络：由于本项目会访问一些被墙的网址，为了避免试验中断，请给家里安装软路由。比如淘宝购买R2S、华硕梅林路由器等等，请参考
[我的博客](https://blog.csdn.net/qq_43626147/article/details/142746627)
#### 云上环境
1. 海外GPU服务器：想要不被墙、且有显卡，只能购买海外GPU服务器。文本测试过阿里云国外GPU服务器，显卡型号A10，region首尔，
系统Alibaba cloud linux，驱动driver550，cuda12.4，cudnn9.2，价格11.66元/小时，系统盘调成100GB。

## API key 准备
以下的每个key都必须创建，否则AI无法运行。
1. 申请一张美国银行卡：中国大陆可以登录 bewildcard.com ，花一点钱注册一个美国虚拟银行卡，就能愉快的使用所有美国的服务，包括AI和云计算等等。
2. 官方chatGPT4 key：chatGPT不允许大陆和香港登录，而且需要美国银行卡才能充值。去https://platform.openai.com上创建一个key，
并绑定银行卡作为支付方式，设置自动充值，并预存20美刀会员。
[详细操作视频](https://www.bilibili.com/video/BV1ax42197uA/?share_source=copy_web&vd_source=4cee0005e63af504f1a4e5f79e975468)
3. Semantic Scholar： 简称S2 key，目前官方key已经拿不到了，我用企业邮箱去官网填了申请表，美国佬不批的，回复说申请单积压，半年后再来。
目前国内做S2 key中转的只有一位叫阿杰的大哥。文档地址是 https://api.ominiai.cn/ ， 需要先加阿杰的微信，付款，对方会给你一个兑换卡，然后去
控制台---充值---兑换码里使用刚才拿到的兑换卡。最后在控制台---令牌这里创建一个key，这个就是我们需要的S2 api key。控制台地址是
 https://api.ominiai.cn/panel


# 运行环境准备

### 操作系统准备
AI科学家需要在Linux系统上运行，本项目在如下发行版上测试过： Debian 11，Debian 12，Ubuntu 24.04，RockyLinux 9.x，CentOS Steam 9.x。

本文使用环境： Debian 12.7。以下操作围绕Debian 12.7给出。

### 系统包安装
```bash
# 与显卡安装有关的包
apt update && apt upgrade -y
apt -y install gcc make vim git
apt -y install linux-headers-$(uname -r) build-essential libglvnd-dev pkg-config

# 与AI科学家相关的包
apt install -y texlive-full
```

### 显卡驱动安装
NVIDIA显卡驱动官网下载地址：  https://www.nvidia.cn/drivers/lookup/

我的试验环境是NVIDIA RTX 4070 laptop，安装过程如下：
```bash
# 禁用 nouveau
vim /etc/modprobe.d/blacklist-nouveau.conf
blacklist nouveau               
options nouveau modeset=0 

# 更新系统驱动
update-initramfs -u
# 重启
reboot

# 查看显卡型号
lspci|grep NVIDIA

# RTX 4070
cd /tmp && wget https://cn.download.nvidia.com/XFree86/Linux-x86_64/550.120/NVIDIA-Linux-x86_64-550.120.run
bash NVIDIA-Linux-x86_64-550.120.run

# 验证驱动是否安装成功，查看driver和cuda版本。
nvidia-smi

```

### Anaconda3安装
官网地址 https://www.anaconda.com/ ，下载最新版并安装，Anaconda可以管理多个版本的python和三方包，所以放心安装最新版。
安装过程演示；
```bash
# 将/opt作为工作目录
cd /opt

# 下载Anaconda3（国内从中科大源下载速度快）
wget https://mirrors.ustc.edu.cn/anaconda/archive/Anaconda3-2024.06-1-Linux-x86_64.sh

# 安装Anaconda3
bash Anaconda3-2024.06-1-Linux-x86_64.sh

第一步，阅读协议，一直按空格，最后输入yes
第二步，输入安装位置，/opt/anaconda3
第三步，开机自启动，输入no

```

### 克隆项目
```bash
cd /opt
git clone https://github.com/yangjiacheng1996/china-ai-scientist.git
git clone https://github.com/gregversteeg/NPEET.git
```

### 创建虚拟环境

```bash
# 创建一个conda虚拟环境（conda虚拟环境只保存在Anaconda安装目录下，和venv不一样）
# 启动conda
cd /opt/china-ai-scientist
source /opt/anaconda3/bin/activate

# 创建虚拟环境
conda create -n ai_scientist python=3.11

# 激活虚拟环境
conda activate ai_scientist

# Install pypi requirements
pip install -r requirements.txt
```



### 训练所需数据下载
```bash
# Prepare NanoGPT data
cd /opt/china-ai-scientist
python data/enwik8/prepare.py
python data/shakespeare_char/prepare.py
python data/text8/prepare.py
```

#### 训练模型
既然是AI科学家，那么就需要在本地训练一个小的AI模型来充当一个“人”。运行下方命令训练NanoGPT。
```bash
# Set up NanoGPT baseline run
cd /opt/china-ai-scientist
cd templates/nanoGPT && python experiment.py --out_dir run_0 && python plot.py
```
训练轻量NanoGPT
```bash
# NOTE: YOU MUST FIRST RUN THE PREPARE SCRIPTS ABOVE!
cd /opt/china-ai-scientist
cd templates/nanoGPT_lite && python experiment.py --out_dir run_0 && python plot.py
```
训练2D Diffusion
```bash
# Set up 2D Diffusion
cd /opt/NPEET
pip install .
pip install scikit-learn

# Set up 2D Diffusion baseline run
cd /opt/china-ai-scientist
cd templates/2d_diffusion && python experiment.py --out_dir run_0 && python plot.py
```

训练Grokking
```bash
# Set up Grokking
pip install einops

# Set up Grokking baseline run
cd /opt/china-ai-scientist
cd templates/grokking && python experiment.py --out_dir run_0 && python plot.py
```
全部训练完成后，记得备份你的AI科学家。如果在后续使用过程中出现意外，可以重头再来。
```bash
cp -R /opt/china-ai-scientist  /opt/china-ai-scientist-bak
zip -r china-ai-scientist.zip /opt/china-ai-scientist-bak
```
将zip包放到百度网盘里，或者NAS里。

# 论文选题
### 设置Semantic Scholar api key环境变量（以下简称S2 key）
如果你有Semantic Scholar官方的key，那么设置比较简单，只需要一个环境变量：
```bash
export S2_API_KEY="sk-xFDp6Ec9Y50q07r1C563F991Dd324bE1A45c5d2a4bC65602"
```
如果你采用国内的S2中转站的api key，则还需要给出中转站的地址。如下我给出了阿杰的omini站点的地址和key。
AI-Scientist原项目访问的S2官网地址是 https://api.semanticscholar.org/graph/v1/paper/search

我们需要把 https://api.semanticscholar.org 替换成国内 https://api.ominiai.cn/generalProxy/
如果你能购买到其他站点的S2 api，请根据实际情况替换。
```bash
# Semantic Scholar api key
export S2_API_URL="https://api.ominiai.cn/generalProxy/graph/v1/paper/search"
export S2_API_KEY="sk-xFDp6Ec9Y50q07r1C563F991Dd324bE1A45c5d2a4bC65602"

```

### 设置OpenRouter官方api key
本项目全程使用Openai官方key，如果你采用其他key，例如Claude、OpenRouter，则只能进行选题，无法跑完全部流程。
请根据自己的实际key，设置如下环境变量
```bash
# openai api key
export OPENAI_API_KEY="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"
```

### 开始论文选题
```bash
source /opt/anaconda3/bin/activate
conda activate ai_scientist
# Run the paper generation.
cd /opt/china-ai-scientist
python launch_scientist.py --model "gpt-4o-2024-05-13" --experiment nanoGPT --num-ideas 1
```
想要查看AI科学家支持哪些模型，可以查看帮助命令
```bash
python launch_scientist.py --help
```
不同的模型需要设置不同的key环境变量，详情请看AI-Scientist原项目文档。

如果你有超过 1 个 GPU，请使用 `parallel` 选项在多个 GPU 上并行生成选题。

这个选题命令一次运行不一定能产生一个合适的选题，所以需要多运行几次。如果遇到如下输出表示生成选题失败:
```bash
Completed idea: adaptive_block_size, Success: False
All ideas evaluated.
```
选题失败就重新执行选题命令，直到出现如下字样表示选题成功，程序会继续运行接下来的试验部分。
```bash
*Starting Experiments*
Based on the experiment description, I plan to run the following experiments: 
```
选题成功自动进入论文写作和试验环节，无需人工干预，耐心等待。

经过实测，4070显卡生成一篇论文pdf文件，平均花费15美元（100元人民币）。用时23小时。

### 论文评审
相当于预答辩，或者期刊评委问答。

选题命令执行结束后，会产生一个论文pdf初稿，已经可以拿去润色润色发表期刊了。
如果不放心，可以继续让ai评审一下。使用如下脚本进行评审。
pdf的文件路径在/opt/china-ai-scientist/results/nanoGPT中，进入对应时间戳目录中有一个
initialization_lr_interplay.pdf文件。将pdf绝对路径替换脚本中的report.pdf，并运行脚本。

```python
import openai
from ai_scientist.perform_review import load_paper, perform_review

client = openai.OpenAI()
model = "gpt-4o-2024-05-13"

# Load paper from pdf file (raw text)
paper_txt = load_paper("report.pdf")
# Get the review dict of the review
review = perform_review(
    paper_txt,
    model,
    client,
    num_reflections=5,
    num_fs_examples=1,
    num_reviews_ensemble=5,
    temperature=0.1,
)

# Inspect review results
review["Overall"]  # overall score 1-10
review["Decision"]  # ['Accept', 'Reject']
review["Weaknesses"]  # List of weaknesses (str)
```

运行批量分析，可以使用如下命令。

```bash
cd review_iclr_bench
python iclr_analysis.py --num_reviews 500  --batch_size 100 --num_fs_examples 1 --num_reflections 5 --temperature 0.1 --num_reviews_ensemble 5
```

## 论文模板

If there is an area of study you would like **The AI Scientist** to explore, it should be very easy to create your own templates. In general, follow the structure of the existing templates, which consists of:

- `experiment.py` -- This is a single file where the 'meat' of the content is. It takes in an argument for `out_dir`, which is where it should create the folder and save the relevant information from the run.
- `plot.py` -- This should take in the information from the `run` folders and create plots. The code should be clear and easy to edit.
- `prompt.json` -- Put information about your template here.
- `seed_ideas.json` -- Put example ideas here. You can also try to generate ideas without any examples, and then pick the best one or two to put here.
- `latex/template.tex` -- We recommend using our latex folder, but be sure to replace the pre-loaded citations with ones that you would expect to be more relevant.
   
## Template Resources

We provide 3 templates, which heavily use code from other repositories, which we credit below. (Normally, we would do this in the files themselves, but it's unclear how this would affect The AI Scientist since it would be visible).

The NanoGPT template used code from [NanoGPT](https://github.com/karpathy/nanoGPT) and this [PR](https://github.com/karpathy/nanoGPT/pull/254).

The 2D Diffusion template used code from [tiny-diffusion](https://github.com/tanelp/tiny-diffusion), [ema-pytorch](https://github.com/lucidrains/ema-pytorch), and [Datasaur](https://www.research.autodesk.com/publications/same-stats-different-graphs/).

The Grokking template used code from [Sea-Snell/grokking](https://github.com/Sea-Snell/grokking) and [danielmamay/grokking](https://github.com/danielmamay/grokking).

We would like to thank the developers of the open-source models and packages for their contributions and for making their work available.

## Citing The AI Scientist

If you use **The AI Scientist** in your research, please cite it as follows:

```
@article{lu2024aiscientist,
  title={The {AI} {S}cientist: Towards Fully Automated Open-Ended Scientific Discovery},
  author={Lu, Chris and Lu, Cong and Lange, Robert Tjarko and Foerster, Jakob and Clune, Jeff and Ha, David},
  journal={arXiv preprint arXiv:2408.06292},
  year={2024}
}
```

### FAQ

We recommend reading our paper in the first instance for any questions you have on The AI Scientist.

### Why am I missing files when running The AI Scientist?
Make sure you have completed all the setup and preparation steps before the main experiment script.

### Why has a PDF or a review not been generated?
The AI Scientist finishes an idea with a success rate that depends on both the template, the base foundation model, and the complexity of the idea. We advise referring to our main paper. The highest success rates are observed with Claude Sonnet 3.5.
Reviews are best done with GPT-4o, all other models have issues with positivity bias or failure to conform to required outputs.

### What is the cost of each idea generated?
Typically less than $15 per paper with Claude Sonnet 3.5. We recommend DeepSeek Coder V2 for a much more cost-effective approach. A good place to look for new models is the [Aider leaderboard](https://aider.chat/docs/leaderboards/).

### How do I change the base conference format associated with the write-ups?
Change the base `template.tex` files contained within each template.

### How do I run The AI Scientist for different subject fields?
Please refer to the instructions for different templates. In this current iteration, this is restricted to ideas that can be expressed in code. However, lifting this restriction would represent exciting future work! :)

### How do I add support for a new foundation model?
Please see this [PR](https://github.com/SakanaAI/AI-Scientist/pull/7) for an example of how to add a new model, e.g. this time for Claude via Bedrock.
We do not advise any model that is significantly weaker than GPT-4 level for The AI Scientist.

### Why do I need to run the baseline runs myself?
These appear as `run_0` and should be run per machine you execute The AI Scientist on for accurate run-time comparisons due to hardware differences.

### What if I have problems accessing the Semantic Scholar API?
We use the Semantic Scholar API to check ideas for novelty and collect citations for the paper write-up. You may be able to skip these phases in case you don't have an API key or the API is slow to access.

## Containerization

We include a [community-contributed](https://github.com/SakanaAI/AI-Scientist/pull/21) Docker image that may assist with your containerization efforts in `experimental/Dockerfile`.

You can use this image like this:

```bash
# Endpoint Script
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -v `pwd`/templates:/app/AI-Scientist/templates <AI_SCIENTIST_IMAGE> \
  --model gpt-4o-2024-05-13 \
  --experiment 2d_diffusion \
  --num-ideas 2
```

```bash
# Interactive
docker run -it -e OPENAI_API_KEY=$OPENAI_API_KEY \
  --entrypoint /bin/bash \
  <AI_SCIENTIST_IMAGE>
```

