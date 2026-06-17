别忘了更改路径（需要更改路径的文件包括\controllers\new_ti_zi\new_ti_zi.py ,\python_scripts\Project_config.py ，\controllers\Train_main\Train_main.py 等等）

切换算法还是在Train_main.py里手动改，按需要保留要跑的算法入口，其他入口先注释掉或不调用。

大框架就是这么个框架，但有很多网络，经验池方面的问题.（详细的问ai，然后一个个优化）
## 本机路径配置

为了防止多人协作时 pull 之后又要反复改路径，我把 `controllers/Train_main/Train_main.py` 和 `controllers/new_ti_zi/new_ti_zi.py` 用到的本机路径统一放到了 `python_scripts/Project_config.py`。之后每个人主要维护自己本地的 `Project_config.py`，不要把自己的本机路径提交上去。

不同电脑运行前，先改 `Project_config.py` 里的这几个路径：

```python
WEBOTS_HOME = r"D:\Dev\Tools\Webots"
WEBOTS_CONDA_ENV = r"D:\Dev\Env\Python\Miniconda3\envs\webots"
PYTHON_37_EXPECTED = os.path.join(WEBOTS_CONDA_ENV, "python.exe")
```

同时检查 `path_list` 中的项目路径是否对应自己电脑上的项目位置，尤其是 `resetFlag`、日志目录和 checkpoint 目录，例如：

```python
'resetFlag': 'D:/Dev/Projects/RSA/Multi-Stage_Hybrid_Training/python_scripts/resetFlag.txt'
```

环境重置机制没有变：训练控制器和 Supervisor 控制器还是两个独立 Python 进程，它们不直接共享变量，而是通过 `resetFlag.txt` 文件通信。训练进程写入 `0` 请求重置，Supervisor 检测到 `0` 后重置仿真，再写回 `1` 表示完成。

如果多人共享同一个 GitHub 仓库，建议让 Git 停止跟踪 `python_scripts/`，避免每个人的本机路径互相覆盖。在项目根目录打开 Git Bash，然后执行：

```bash
git rm --cached python_scripts/Project_config.py
echo "python_scripts/Project_config.py" >> .gitignore
git add .gitignore
git commit -m "Stop tracking local project config"
git push
```

其他组员 pull 这次提交前，建议先备份自己的配置：

```bash
cp python_scripts/Project_config.py python_scripts/Project_config.backup.py
git pull
cp python_scripts/Project_config.backup.py python_scripts/Project_config.py
```

之后每个人只需要维护自己本地的 `python_scripts/Project_config.py`，不要再提交本机路径。
