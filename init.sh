#!/bin/bash

set -e  # 出错即退出
echo "📦 开始安装 pyenv 及 Python 3.10..."

# 安装依赖
sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
  libffi-dev liblzma-dev git

# 安装 pyenv
if [ ! -d "$HOME/.pyenv" ]; then
    echo "🐍 安装 pyenv..."
    curl https://pyenv.run | bash
else
    echo "✅ pyenv 已安装，跳过。"
fi

# 写入 shell 配置（支持 bash 和 zsh）
echo '🔧 写入 pyenv 环境变量配置...'
PYENV_CONFIG='
# >>> pyenv 初始化 >>>
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
# <<< pyenv 初始化 <<<
'

if [ -f "$HOME/.bashrc" ]; then
    echo "$PYENV_CONFIG" >> "$HOME/.bashrc"
    echo "✅ 写入 ~/.bashrc 完成"
fi

if [ -f "$HOME/.zshrc" ]; then
    echo "$PYENV_CONFIG" >> "$HOME/.zshrc"
    echo "✅ 写入 ~/.zshrc 完成"
fi

# 加载当前 shell 会话中的 pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# 安装 Python 3.10
echo "📥 安装 Python 3.10.13..."
pyenv install 3.10.13 -s
pyenv global 3.10.13

# 创建虚拟环境
echo "📦 创建 pyenv 虚拟环境 myenv310..."
pyenv virtualenv 3.10.13 myenv310
pyenv activate myenv310

# 显示结果
echo "✅ Python 和虚拟环境设置完成："
python --version
pip --version
which python
