"""
External-process RPC client for the route-3 dual-Python architecture.

Python runtime target
---------------------
- External training process: Python 3.11.14

The transport is a length-prefixed pickle stream with protocol 4 so it remains
compatible with the Webots controller process running Python 3.7.12.
"""
from __future__ import annotations

import argparse
import pickle
import socket
import struct
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

PICKLE_PROTOCOL = 4
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


# ---------------------------------------------------------------------------
# Transport helpers
# ---------------------------------------------------------------------------
def _recv_exact(sock_obj: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock_obj.recv(size - len(data))
        if not chunk:
            raise EOFError("socket closed while receiving packet")
        data.extend(chunk)
    return bytes(data)



def recv_packet(sock_obj: socket.socket) -> Dict[str, Any]:
    header = _recv_exact(sock_obj, 8)
    payload_len = struct.unpack("!Q", header)[0]
    payload = _recv_exact(sock_obj, payload_len)
    return pickle.loads(payload)



def send_packet(sock_obj: socket.socket, obj: Dict[str, Any]) -> None:
    payload = pickle.dumps(obj, protocol=PICKLE_PROTOCOL)
    header = struct.pack("!Q", len(payload))
    sock_obj.sendall(header + payload)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class WebotsDPClient:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        timeout_s: float = 300.0,
        retry_attempts: int = 60,
        retry_sleep_s: float = 1.0,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.retry_attempts = int(retry_attempts)
        self.retry_sleep_s = float(retry_sleep_s)
        self.sock: Optional[socket.socket] = None

    # ------------------------------
    # Connection
    # ------------------------------
    def connect(self) -> None:
        if self.sock is not None:
            return

        last_error: Optional[BaseException] = None
        for _ in range(self.retry_attempts):
            try:
                sock_obj = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock_obj.settimeout(self.timeout_s)
                sock_obj.connect((self.host, self.port))
                self.sock = sock_obj
                return
            except OSError as exc:
                last_error = exc
                time.sleep(self.retry_sleep_s)

        raise ConnectionError(
            "failed to connect to WebotsEnvServer at %s:%d after %d attempts: %s"
            % (self.host, self.port, self.retry_attempts, last_error)
        )

    def close(self) -> Optional[Dict[str, Any]]:
        if self.sock is None:
            return None
        try:
            response = self.request("close")
            return response
        except Exception:
            return None
        finally:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    # ------------------------------
    # RPC
    # ------------------------------
    def request(self, cmd: str, **kwargs: Any) -> Dict[str, Any]:
        self.connect()
        assert self.sock is not None
        send_packet(self.sock, {"cmd": cmd, "kwargs": kwargs})
        reply = recv_packet(self.sock)
        if not reply.get("ok", False):
            raise RuntimeError(
                "WebotsEnvServer error for cmd=%s\n%s\n%s"
                % (
                    cmd,
                    reply.get("error", "unknown error"),
                    reply.get("traceback", ""),
                )
            )
        return reply["result"]

    # ------------------------------
    # Public env API
    # ------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        result = self.request("reset", seed=seed, options=options)
        return result["obs"], result["info"]

    def step_grasp(self, action: np.ndarray | list[float] | tuple[float, float]) -> Dict[str, Any]:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.shape[0] != 2:
            raise ValueError("step_grasp expects action_dim=2")
        return self.request("step_grasp", action=action_arr)

    def step_tai(self, action: np.ndarray | list[float] | tuple[float, float, float]) -> Dict[str, Any]:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.shape[0] != 3:
            raise ValueError("step_tai expects action_dim=3")
        return self.request("step_tai", action=action_arr)

    # ------------------------------
    # Context manager
    # ------------------------------
    def __enter__(self) -> "WebotsDPClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
def _smoke_test(host: str, port: int) -> None:
    with WebotsDPClient(host=host, port=port) as client:
        obs, info = client.reset(seed=0)
        print("[smoke] reset ok")
        print("[smoke] stage=", obs.get("stage"), "state_dim=", np.asarray(obs["robot_state"]).shape)
        print("[smoke] image_shape=", np.asarray(obs["image"]).shape)
        grasp_reply = client.step_grasp(np.zeros(2, dtype=np.float32))
        print("[smoke] step_grasp reward=", grasp_reply["reward"], "terminated=", grasp_reply["terminated"])



def main() -> None:
    parser = argparse.ArgumentParser(description="External client for WebotsEnvServer")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--mode", type=str, default="smoke", choices=["smoke"])
    args = parser.parse_args()

    if args.mode == "smoke":
        _smoke_test(args.host, args.port)


if __name__ == "__main__":
    main()
