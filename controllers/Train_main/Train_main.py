import os
import subprocess
import sys


THIS_FILE = os.path.abspath(__file__)
CONTROLLER_DIR = os.path.dirname(THIS_FILE)
PROJECT_ROOT = os.path.abspath(os.path.join(CONTROLLER_DIR, "..", ".."))
WEBOTS_HOME = r"D:\Dev\Tools\Webots"
WEBOTS_CONDA_ENV = r"D:\Dev\Env\Python\Miniconda3\envs\webots"

sys.path.append(PROJECT_ROOT)


def _prepend_runtime_dll_paths():
    candidate_paths = [
        WEBOTS_CONDA_ENV,
        os.path.join(WEBOTS_CONDA_ENV, "Library", "mingw-w64", "bin"),
        os.path.join(WEBOTS_CONDA_ENV, "Library", "usr", "bin"),
        os.path.join(WEBOTS_CONDA_ENV, "Library", "bin"),
        os.path.join(WEBOTS_CONDA_ENV, "DLLs"),
        os.path.join(WEBOTS_CONDA_ENV, "Scripts"),
        os.path.join(WEBOTS_CONDA_ENV, "bin"),
        os.path.join(WEBOTS_CONDA_ENV, "Lib", "site-packages", "torch", "lib"),
    ]

    existing = os.environ.get("PATH", "")
    parts = existing.split(os.pathsep) if existing else []
    normalized = {os.path.normcase(path) for path in parts}

    prepend = []
    for path in candidate_paths:
        if os.path.isdir(path) and os.path.normcase(path) not in normalized:
            prepend.append(path)

    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + parts)
        print("Prepended runtime DLL paths:", flush=True)
        for path in prepend:
            print("  " + path, flush=True)


_prepend_runtime_dll_paths()

DARWIN_MANAGER_CANDIDATES = [
    os.path.join(WEBOTS_HOME, "projects", "robots", "robotis", "darwin-op", "libraries", "managers"),
    os.path.join(WEBOTS_HOME, "resources", "projects", "robots", "robotis", "darwin-op", "libraries", "managers"),
]
for manager_path in DARWIN_MANAGER_CANDIDATES:
    if os.path.isdir(manager_path):
        sys.path.append(manager_path)
        break

from python_scripts.Project_config import path_list


PYTHON_37_EXPECTED = r"D:\Dev\Env\Python\Miniconda3\envs\webots\python.exe"
DEFAULT_DP_MISSION_PYTHON = os.environ.get("DIFFUSION_POLICY_MISSION_PYTHON")
DEFAULT_ALGO = os.environ.get("TRAIN_ALGO", "dqn")


def _warn_python_runtime():
    expected = os.path.normcase(PYTHON_37_EXPECTED)
    current = os.path.normcase(sys.executable)
    if expected != current:
        print("Warning: current controller Python is not the expected Webots 3.7 environment.")
        print("  Expected: %s" % PYTHON_37_EXPECTED)
        print("  Current:  %s" % sys.executable)
    else:
        print("Webots controller Python matched: %s" % sys.executable)


def _ensure_flag_file(file_path, default_value="1"):
    folder = os.path.dirname(file_path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)

    if not os.path.isfile(file_path):
        with open(file_path, "w") as file:
            file.write(default_value)
        return

    with open(file_path, "r+") as file:
        file.seek(0)
        file.write(default_value)
        file.truncate()


def _bootstrap_project_files():
    reset_flag = path_list.get("resetFlag", "")
    reset_flag1 = path_list.get("resetFlag1", "")
    if reset_flag:
        _ensure_flag_file(reset_flag, default_value="1")
    if reset_flag1:
        _ensure_flag_file(reset_flag1, default_value="1")


def _normalize_algo_name(name):
    text = str(name).strip().lower().replace("-", "").replace("_", "")
    alias = {
        "dqn": "dqn",
        "diffwave": "diffwave",
        "diffwaveppo": "diffwave",
        "dp": "diffusionpolicy",
        "diffusionpolicy": "diffusionpolicy",
        "ppo": "diffwave",
        "sac": "sac",
    }
    return alias.get(text, "")


def _get_selected_algo():
    for arg in sys.argv[1:]:
        if arg.startswith("--algo="):
            selected = _normalize_algo_name(arg.split("=", 1)[1])
            if selected:
                return selected

    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        selected = _normalize_algo_name(sys.argv[1])
        if selected:
            return selected

    selected = _normalize_algo_name(DEFAULT_ALGO)
    return selected or "dqn"


def _run_dqn():
    from python_scripts.DQN.DQN_episoid import DQN_episoid

    print("Training with DQN.")
    DQN_episoid()


def _run_diffwave():
    from python_scripts.DiffWave.DiffWave_episoid_1 import DiffWave_episoid_1

    print("Training with DiffWave reward-weighted diffusion.")
    DiffWave_episoid_1()


def _run_diffusion_policy():
    from python_scripts.DiffusionPolicy.WebotsEnvServer import WebotsRPCServer

    host = os.environ.get("DP_HOST", "127.0.0.1")
    port = int(os.environ.get("DP_PORT", "8765"))
    episodes = int(os.environ.get("DP_EPISODES", "1000"))
    save_dir = os.environ.get(
        "DP_SAVE_DIR",
        os.path.join(PROJECT_ROOT, "python_scripts", "DiffusionPolicy", "checkpoint"),
    )
    mission_python = os.environ.get("DIFFUSION_POLICY_MISSION_PYTHON") or DEFAULT_DP_MISSION_PYTHON

    if not mission_python:
        raise RuntimeError(
            "Diffusion Policy mission Python is not configured. "
            "Set DIFFUSION_POLICY_MISSION_PYTHON before using TRAIN_ALGO=diffusionpolicy."
        )
    if not os.path.isfile(mission_python):
        raise RuntimeError("Diffusion Policy mission Python not found: %s" % mission_python)

    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = PROJECT_ROOT if not old_pythonpath else PROJECT_ROOT + os.pathsep + old_pythonpath
    env.setdefault("PYTHONIOENCODING", "utf-8")

    command = [
        mission_python,
        "-u",
        "-m",
        "python_scripts.DiffusionPolicy.dp_online_trainer",
        "--host",
        host,
        "--port",
        str(port),
        "--episodes",
        str(episodes),
        "--save-dir",
        save_dir,
    ]
    device = os.environ.get("DP_DEVICE")
    if device:
        command.extend(["--device", device])

    for env_name, arg_name in (
        ("DP_TEST_INTERVAL", "--test-interval"),
        ("DP_SAVE_INTERVAL", "--save-interval"),
        ("DP_NUM_TEST_EPISODES", "--num-test-episodes"),
    ):
        value = os.environ.get(env_name)
        if value:
            command.extend([arg_name, value])

    print("Training with pure online Diffusion Policy.")
    print("Mission Python: %s" % mission_python)
    print("Webots DP endpoint: %s:%d" % (host, port))
    print("DP episodes: %d" % episodes)
    print("DP save dir: %s" % save_dir)

    trainer_process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env)
    try:
        server = WebotsRPCServer(host=host, port=port)
        server.serve_forever()
    finally:
        if trainer_process.poll() is None:
            try:
                trainer_process.wait(timeout=10)
            except Exception:
                print("Warning: Diffusion Policy trainer is still running after server shutdown.")
        if trainer_process.poll() not in (None, 0):
            raise RuntimeError(
                "Diffusion Policy trainer exited with code %s" % trainer_process.returncode
            )


def _run_sac():
    from python_scripts.SAC.SAC_episoid import SAC_episoid

    print("Training with SAC.")
    SAC_episoid()


def main():
    _bootstrap_project_files()
    _warn_python_runtime()

    raw_algo = DEFAULT_ALGO
    if len(sys.argv) > 1:
        raw_algo = sys.argv[1].split("=", 1)[-1] if sys.argv[1].startswith("--algo=") else sys.argv[1]
    algo = _get_selected_algo()
    if str(raw_algo).strip().lower().replace("-", "").replace("_", "") in ("ppo", "diffwaveppo"):
        print("Compatibility note: 'ppo' now maps to DiffWave, not PPO.")

    print("Selected training algorithm: %s" % algo)
    print("Available algorithms: dqn / sac / diffwave / diffusionpolicy; ppo is a DiffWave compatibility alias.")

    if algo == "dqn":
        _run_dqn()
    elif algo == "diffwave":
        _run_diffwave()
    elif algo == "diffusionpolicy":
        _run_diffusion_policy()
    elif algo == "sac":
        _run_sac()
    else:
        raise RuntimeError("Unsupported algorithm: %s" % algo)


if __name__ == "__main__":
    print("_________")
    main()
