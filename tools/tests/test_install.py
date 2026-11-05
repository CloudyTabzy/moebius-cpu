"""Tests gimp-plugin/install.py's downloader against a local server that misbehaves."""
import http.server
import importlib.util
import os
import threading
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("install", Path(__file__).parents[2] / "gimp-plugin" / "install.py")
install = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(install)

BODY = bytes(range(256)) * 40_000  # 10 MB, so several chunks


class Flaky(http.server.BaseHTTPRequestHandler):
    """Serves BODY with Range support. Drops the connection after `drop_after` bytes, `drops` times."""

    drops = 0
    drop_after = 0
    status = None
    ignore_range = False
    requests = []

    def do_GET(self):
        cls = type(self)
        cls.requests.append(self.headers.get("Range"))
        if cls.status:
            self.send_error(cls.status)
            return
        start = 0
        ranged = self.headers.get("Range")
        if ranged and not cls.ignore_range:
            start = int(ranged.split("=")[1].rstrip("-"))
            if start >= len(BODY):
                self.send_error(416)
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(BODY) - 1}/{len(BODY)}")
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(BODY) - start))
        self.end_headers()
        end = len(BODY)
        if cls.drops:
            cls.drops -= 1
            end = min(end, start + cls.drop_after)
        self.wfile.write(BODY[start:end])
        self.close_connection = True

    def log_message(self, *args):
        pass


@pytest.fixture
def url():
    handler = type("Handler", (Flaky,), {"requests": []})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/file", handler
    server.shutdown()


def test_a_clean_download_leaves_only_the_file(url, tmp_path):
    address, _ = url
    target = tmp_path / "file.bin"
    install.fetch(address, str(target), "test", wait=0)
    assert target.read_bytes() == BODY
    assert not (tmp_path / "file.bin.part").exists()


def test_dropped_connections_resume_where_they_stopped(url, tmp_path):
    address, handler = url
    handler.drops, handler.drop_after = 3, 2_000_000
    target = tmp_path / "file.bin"
    install.fetch(address, str(target), "test", wait=0)
    assert target.read_bytes() == BODY
    assert handler.requests == [None, "bytes=2000000-", "bytes=4000000-", "bytes=6000000-"]


def test_an_existing_part_file_is_continued(url, tmp_path):
    address, handler = url
    target = tmp_path / "file.bin"
    (tmp_path / "file.bin.part").write_bytes(BODY[:1234])
    install.fetch(address, str(target), "test", wait=0)
    assert target.read_bytes() == BODY
    assert handler.requests == ["bytes=1234-"]


def test_a_finished_part_file_is_accepted(url, tmp_path):
    address, _ = url
    target = tmp_path / "file.bin"
    (tmp_path / "file.bin.part").write_bytes(BODY)
    install.fetch(address, str(target), "test", wait=0)
    assert target.read_bytes() == BODY


def test_a_server_that_ignores_ranges_restarts_the_file(url, tmp_path):
    address, handler = url
    handler.ignore_range = True
    target = tmp_path / "file.bin"
    (tmp_path / "file.bin.part").write_bytes(b"stale")
    install.fetch(address, str(target), "test", wait=0)
    assert target.read_bytes() == BODY


def test_a_finished_file_is_not_fetched_again(url, tmp_path):
    address, handler = url
    target = tmp_path / "file.bin"
    target.write_bytes(b"done")
    install.fetch(address, str(target), "test", wait=0)
    assert handler.requests == []


def test_a_missing_file_stops_at_once(url, tmp_path):
    address, handler = url
    handler.status = 404
    with pytest.raises(install.InstallError, match="404"):
        install.fetch(address, str(tmp_path / "file.bin"), "test", wait=0)
    assert len(handler.requests) == 1


def test_a_failing_server_gives_up_with_a_clear_message(url, tmp_path):
    address, handler = url
    handler.status = 503
    with pytest.raises(install.InstallError, match="kept failing"):
        install.fetch(address, str(tmp_path / "file.bin"), "test", attempts=3, wait=0)
    assert len(handler.requests) == 3


def test_no_gimp_is_explained(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert install.main([]) == 1
    assert "GIMP 3.2 was not found" in capsys.readouterr().out
    assert not os.listdir(tmp_path)
