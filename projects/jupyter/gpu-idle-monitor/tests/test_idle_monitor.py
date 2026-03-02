"""Unit tests for the GPU idle-shutdown monitor."""

import os
import sys
from unittest.mock import MagicMock, patch

import requests as req_lib

# Allow ``import idle_monitor`` from the parent directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import idle_monitor  # noqa: E402

# ---------------------------------------------------------------------------
# _read_token
# ---------------------------------------------------------------------------


class TestReadToken:
    def test_reads_and_strips_token_from_file(self, tmp_path):
        token_file = tmp_path / "token"
        token_file.write_text("my-secret-token\n")
        with patch.object(idle_monitor, "JUPYTER_TOKEN_FILE", str(token_file)):
            assert idle_monitor._read_token() == "my-secret-token"

    def test_returns_empty_string_when_file_missing(self):
        with patch.object(idle_monitor, "JUPYTER_TOKEN_FILE", "/nonexistent/path"):
            assert idle_monitor._read_token() == ""


# ---------------------------------------------------------------------------
# get_jupyter_activity
# ---------------------------------------------------------------------------


class TestGetJupyterActivity:
    def _resp(self, data, status=200):
        r = MagicMock()
        r.status_code = status
        r.json.return_value = data
        return r

    def test_busy_kernel_returns_true(self):
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._resp(
                [{"id": "k1", "execution_state": "busy"}]
            )
            assert idle_monitor.get_jupyter_activity("tok") is True

    def test_idle_kernel_no_terminals_returns_false(self):
        with patch("requests.get") as mock_get:
            mock_get.side_effect = [
                self._resp([{"id": "k1", "execution_state": "idle"}]),
                self._resp([]),
            ]
            assert idle_monitor.get_jupyter_activity("tok") is False

    def test_open_terminal_returns_true(self):
        with patch("requests.get") as mock_get:
            mock_get.side_effect = [
                self._resp([]),
                self._resp([{"name": "1"}]),
            ]
            assert idle_monitor.get_jupyter_activity("tok") is True

    def test_no_kernels_no_terminals_returns_false(self):
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._resp([])
            assert idle_monitor.get_jupyter_activity("tok") is False

    def test_request_error_returns_false(self):
        with patch("requests.get", side_effect=req_lib.ConnectionError("refused")):
            assert idle_monitor.get_jupyter_activity("tok") is False

    def test_non_200_status_not_counted(self):
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._resp([], status=403)
            assert idle_monitor.get_jupyter_activity("tok") is False

    def test_authorization_header_sent_when_token_provided(self):
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._resp([])
            idle_monitor.get_jupyter_activity("secret")
            for call in mock_get.call_args_list:
                assert call.kwargs["headers"]["Authorization"] == "token secret"

    def test_no_authorization_header_when_token_empty(self):
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._resp([])
            idle_monitor.get_jupyter_activity("")
            for call in mock_get.call_args_list:
                assert "Authorization" not in call.kwargs.get("headers", {})


# ---------------------------------------------------------------------------
# get_ssh_connections
# ---------------------------------------------------------------------------


class TestGetSshConnections:
    _HEADER = "  sl  local_address rem_address   st\n"

    def _write_tcp(self, tmp_path, lines):
        f = tmp_path / "tcp"
        f.write_text(self._HEADER + "".join(lines))
        return str(f)

    def test_counts_established_ssh_connections(self, tmp_path):
        # Port 22 = 0x0016; state 01 = ESTABLISHED
        path = self._write_tcp(
            tmp_path,
            [
                "   0: 00000000:0016 0A000002:D7F4 01 00000000:00000000\n",
                "   1: 00000000:0016 0A000003:C1F2 01 00000000:00000000\n",
                "   2: 00000000:1F90 0A000002:ABCD 01 00000000:00000000\n",  # port 8080
            ],
        )
        assert idle_monitor.get_ssh_connections(path) == 2

    def test_non_established_state_not_counted(self, tmp_path):
        path = self._write_tcp(
            tmp_path,
            [
                "   0: 00000000:0016 00000000:0000 0A 00000000:00000000\n",  # LISTEN
            ],
        )
        assert idle_monitor.get_ssh_connections(path) == 0

    def test_empty_table_returns_zero(self, tmp_path):
        path = self._write_tcp(tmp_path, [])
        assert idle_monitor.get_ssh_connections(path) == 0

    def test_missing_file_returns_zero(self):
        assert idle_monitor.get_ssh_connections("/nonexistent/tcp") == 0


# ---------------------------------------------------------------------------
# is_active
# ---------------------------------------------------------------------------


class TestIsActive:
    def test_active_when_jupyter_busy(self):
        with patch.object(idle_monitor, "get_jupyter_activity", return_value=True):
            with patch.object(idle_monitor, "get_ssh_connections", return_value=0):
                assert idle_monitor.is_active("tok") is True

    def test_active_when_ssh_connected(self):
        with patch.object(idle_monitor, "get_jupyter_activity", return_value=False):
            with patch.object(idle_monitor, "get_ssh_connections", return_value=2):
                assert idle_monitor.is_active("tok") is True

    def test_inactive_when_both_idle(self):
        with patch.object(idle_monitor, "get_jupyter_activity", return_value=False):
            with patch.object(idle_monitor, "get_ssh_connections", return_value=0):
                assert idle_monitor.is_active("tok") is False


# ---------------------------------------------------------------------------
# delete_pod
# ---------------------------------------------------------------------------


class TestDeletePod:
    def test_deletes_named_pod(self):
        with patch.object(idle_monitor, "POD_NAME", "my-pod"):
            with patch.object(idle_monitor, "POD_NAMESPACE", "jupyter"):
                with patch("kubernetes.config.load_incluster_config"):
                    mock_v1 = MagicMock()
                    with patch("kubernetes.client.CoreV1Api", return_value=mock_v1):
                        idle_monitor.delete_pod()
                        mock_v1.delete_namespaced_pod.assert_called_once_with(
                            name="my-pod", namespace="jupyter"
                        )

    def test_does_nothing_when_pod_name_unset(self):
        with patch.object(idle_monitor, "POD_NAME", ""):
            with patch("kubernetes.config.load_incluster_config") as mock_cfg:
                idle_monitor.delete_pod()
                mock_cfg.assert_not_called()

    def test_logs_error_on_kubernetes_exception(self, caplog):
        with patch.object(idle_monitor, "POD_NAME", "bad-pod"):
            with patch("kubernetes.config.load_incluster_config"):
                with patch(
                    "kubernetes.client.CoreV1Api",
                    side_effect=Exception("k8s unreachable"),
                ):
                    with caplog.at_level("ERROR"):
                        idle_monitor.delete_pod()
                    assert "Failed to delete pod" in caplog.text
