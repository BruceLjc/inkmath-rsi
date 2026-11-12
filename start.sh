#!/bin/sh
# 文件职责：在项目独立目录启动本地原生笔迹实验窗口，不连接外部服务。
set -eu
cd -- "$(dirname -- "$0")"
exec .venv/bin/python -m inkmath.draw "$@"
