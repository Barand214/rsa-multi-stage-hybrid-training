"""
美赛算法库检测脚本
检查所有需要的Python库是否已正确安装
"""

import sys

def check_package(package_name, import_name=None):
    """
    检查单个包是否安装
    
    Args:
        package_name: 包的显示名称
        import_name: 实际导入时使用的名称（如果与package_name不同）
    """
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', '未知版本')
        print(f"✅ {package_name:20s} - 已安装 (版本: {version})")
        return True
    except ImportError:
        print(f"❌ {package_name:20s} - 未安装")
        return False

def main():
    print("=" * 70)
    print("美赛算法库安装检测")
    print("=" * 70)
    print()
    
    packages = {
        # 基础科学计算
        "NumPy": "numpy",
        "Pandas": "pandas",
        "Matplotlib": "matplotlib",
        "Seaborn": "seaborn",
        "SciPy": "scipy",
        
        # 机器学习
        "Scikit-learn": "sklearn",
        
        # 时间序列分析
        "Statsmodels": "statsmodels",
        "pmdarima": "pmdarima",
        
        # 关联规则挖掘
        "mlxtend": "mlxtend",
        
        # 深度学习框架（至少需要一个）
        "PyTorch": "torch",
        "TensorFlow": "tensorflow",
    }
    
    print("【基础库检测】")
    print("-" * 70)
    installed = []
    missing = []
    
    for display_name, import_name in packages.items():
        if check_package(display_name, import_name):
            installed.append(display_name)
        else:
            missing.append((display_name, import_name))
    
    print()
    print("=" * 70)
    print(f"检测结果: {len(installed)}/{len(packages)} 个库已安装")
    print("=" * 70)
    print()
    
    # 检查深度学习框架
    has_pytorch = "PyTorch" in installed
    has_tensorflow = "TensorFlow" in installed
    
    if not has_pytorch and not has_tensorflow:
        print("⚠️  警告: 未检测到深度学习框架（PyTorch或TensorFlow），LSTM算法将无法使用")
        print()
    elif has_pytorch and has_tensorflow:
        print("✅ 检测到两个深度学习框架都已安装（PyTorch和TensorFlow）")
        print()
    elif has_pytorch:
        print("✅ 检测到PyTorch，可以使用LSTM算法")
        print()
    else:
        print("✅ 检测到TensorFlow，可以使用LSTM算法")
        print()
    
    # 算法可用性检查
    print("【算法可用性检查】")
    print("-" * 70)
    
    algorithms = {
        "ARIMA": ["Statsmodels"],
        "BP神经网络": ["Scikit-learn"],
        "LSTM": ["PyTorch", "TensorFlow"],  # 至少需要一个
        "GMM (高斯混合模型)": ["Scikit-learn"],
        "Apriori算法": ["mlxtend"],
        "随机森林": ["Scikit-learn"],
        "K-means聚类": ["Scikit-learn"],
    }
    
    for algo, required_libs in algorithms.items():
        if algo == "LSTM":
            # LSTM只需要PyTorch或TensorFlow其中之一
            if any(lib in installed for lib in required_libs):
                print(f"✅ {algo:25s} - 可用")
            else:
                print(f"❌ {algo:25s} - 不可用 (需要: {' 或 '.join(required_libs)})")
        else:
            if all(lib in installed for lib in required_libs):
                print(f"✅ {algo:25s} - 可用")
            else:
                missing_libs = [lib for lib in required_libs if lib not in installed]
                print(f"❌ {algo:25s} - 不可用 (缺少: {', '.join(missing_libs)})")
    
    print()
    
    # 如果有缺失的库，提供安装建议
    if missing:
        print("=" * 70)
        print("【安装建议】")
        print("=" * 70)
        print()
        print("请在Anaconda Prompt中激活webots环境后执行以下命令：")
        print()
        print("conda activate webots")
        print()
        
        # 分类安装命令
        conda_packages = []
        pip_packages = []
        
        for display_name, import_name in missing:
            if import_name in ['numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 'scipy', 'statsmodels']:
                conda_packages.append(import_name if import_name != 'sklearn' else 'scikit-learn')
            elif import_name in ['torch']:
                print("# 安装PyTorch (CPU版本):")
                print("conda install pytorch torchvision torchaudio cpuonly -c pytorch -y")
                print()
            elif import_name in ['tensorflow']:
                print("# 安装TensorFlow:")
                print("conda install tensorflow -y")
                print()
            else:
                pip_packages.append(import_name)
        
        if conda_packages:
            print(f"# 安装基础库:")
            print(f"conda install {' '.join(conda_packages)} -y")
            print()
        
        if pip_packages:
            print(f"# 安装其他库:")
            print(f"pip install {' '.join(pip_packages)}")
            print()
    else:
        print("=" * 70)
        print("🎉 恭喜！所有需要的库都已安装，可以开始美赛编程了！")
        print("=" * 70)
    
    print()
    print("提示: 如需重新检测，请再次运行此脚本")
    print()

if __name__ == "__main__":
    main()

