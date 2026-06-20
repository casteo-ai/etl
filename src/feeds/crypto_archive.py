"""Website-backed client for CryptoArchive historical Binance data.

CryptoArchive does not expose a documented public API.  This client therefore
uses the public website flow:

* scrape the public asset and pair metadata pages;
* optionally log in with user-provided credentials;
* download only the file links exposed by the authenticated HTML pages.

Credentials are read from the environment by default:

    CRYPTO_ARCHIVE_EMAIL
    CRYPTO_ARCHIVE_PASSWORD
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from email.message import Message
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Optional, Union
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import (
    HTTPCookieProcessor,
    Request,
    build_opener,
)

import pandas as pd

BASE_URL = "https://www.cryptoarchive.com.au"
DEFAULT_DATA_DIR = "/home/poucho/code/etl/data/crypto_archive"

DATASETS = ("minute", "tick")
DATASET_DIRS = {"minute": "minute", "tick": "tick"}

MINUTE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "taker_buy_quote_volume",
    "taker_buy_volume",
    "quote_volume",
    "num_trades",
]
TICK_COLUMNS = ["trade_id", "price", "volume", "timestamp", "buy_side"]

_PAIR_RE = re.compile(r"^[A-Z0-9]+_[A-Z0-9]+$")
_PAIR_TEXT_RE = re.compile(r"\b([A-Z0-9]+)\s*/\s*([A-Z0-9]+)\b")
_CSRF_RE = re.compile(r'name="csrfToken"\s+value="([^"]+)"')
_ASSET_HREF_RE = re.compile(r"^/asset/([A-Z0-9]+)$")


def _normalise_pair(symbol: str) -> str:
    pair = symbol.strip().upper().replace("/", "_").replace("-", "_")
    if not _PAIR_RE.fullmatch(pair):
        raise ValueError('symbol must use BASE_QUOTE format, e.g. "BTC_USDC"')
    return pair


def _asset_from_pair(symbol: str) -> str:
    return _normalise_pair(symbol).split("_", 1)[0]


def _dataset(dataset: str) -> str:
    value = dataset.strip().lower()
    if value not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}")
    return value


def _extract_csrf(html: str) -> str:
    match = _CSRF_RE.search(html)
    if match is None:
        raise RuntimeError("Could not find csrfToken in CryptoArchive page.")
    return match.group(1)


def _filename_from_headers(headers: Message) -> Optional[str]:
    disposition = headers.get("Content-Disposition")
    if not disposition:
        return None
    match = re.search(r'filename="?([^";]+)"?', disposition)
    return match.group(1) if match else None


class _DownloadsParser(HTMLParser):
    """Parse the public `/downloads` asset table."""

    def __init__(self) -> None:
        super().__init__()
        self.assets: list[dict] = []
        self._pending_asset: Optional[str] = None
        self._capture_link_text = False
        self._link_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if not href:
            return
        match = _ASSET_HREF_RE.match(href)
        if match is None:
            return
        self._pending_asset = match.group(1)
        self._capture_link_text = True
        self._link_text = []

    def handle_data(self, data: str) -> None:
        if self._capture_link_text:
            self._link_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._capture_link_text or self._pending_asset is None:
            return
        label = " ".join("".join(self._link_text).split())
        self.assets.append({"asset": self._pending_asset, "label": label})
        self._pending_asset = None
        self._capture_link_text = False
        self._link_text = []


@dataclass
class _Cell:
    text: str
    links: list[str]


class _PairTableParser(HTMLParser):
    """Parse pair rows from an asset page."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[_Cell]] = []
        self._in_row = False
        self._in_cell = False
        self._current_row: list[_Cell] = []
        self._cell_text: list[str] = []
        self._cell_links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag == "tr":
            self._in_row = True
            self._current_row = []
            return
        if tag == "td" and self._in_row:
            self._in_cell = True
            self._cell_text = []
            self._cell_links = []
            return
        if tag == "a" and self._in_cell:
            href = dict(attrs).get("href")
            if href:
                self._cell_links.append(href)

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._in_cell:
            text = " ".join("".join(self._cell_text).split())
            self._current_row.append(_Cell(text=text, links=list(self._cell_links)))
            self._in_cell = False
            return
        if tag == "tr" and self._in_row:
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = []
            self._in_row = False


class CryptoArchiveClient:
    """Client for CryptoArchive's website download flow."""

    def __init__(
        self,
        email: Optional[str] = None,
        password: Optional[str] = None,
        base_url: str = BASE_URL,
        data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
        timeout: float = 60.0,
        user_agent: str = "etl-crypto-archive-client/1.0",
    ) -> None:
        self.email = email if email is not None else os.getenv("CRYPTO_ARCHIVE_EMAIL")
        self.password = (
            password if password is not None else os.getenv("CRYPTO_ARCHIVE_PASSWORD")
        )
        self.base_url = base_url.rstrip("/")
        self._data_dir = Path(data_dir)
        self.timeout = timeout
        self.user_agent = user_agent
        self._cookies = CookieJar()
        self._opener = build_opener(HTTPCookieProcessor(self._cookies))
        self._authenticated = False

    @property
    def authenticated(self) -> bool:
        return self._authenticated

    def _url(self, path_or_url: str) -> str:
        return urljoin(f"{self.base_url}/", path_or_url)

    def _request(
        self,
        path_or_url: str,
        data: Optional[dict[str, str]] = None,
        *,
        headers: Optional[dict[str, str]] = None,
    ):
        body = urlencode(data).encode() if data is not None else None
        request_headers = {
            "User-Agent": self.user_agent,
            "Referer": self.base_url,
        }
        if data is not None:
            request_headers["Content-Type"] = "application/x-www-form-urlencoded"
        if headers:
            request_headers.update(headers)
        req = Request(self._url(path_or_url), data=body, headers=request_headers)
        try:
            return self._opener.open(req, timeout=self.timeout)
        except HTTPError:
            raise
        except URLError as exc:
            raise ConnectionError(f"CryptoArchive request failed: {exc}") from exc

    def _get_text(self, path_or_url: str) -> str:
        with self._request(path_or_url) as resp:
            return resp.read().decode("utf-8", errors="replace")

    def login(
        self,
        email: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        """Log in using the normal website form.

        Raises RuntimeError when credentials are missing or authentication does
        not appear to succeed.  Registration still requires the site's own
        challenge and should be done manually in a browser.
        """
        email = email if email is not None else self.email
        password = password if password is not None else self.password
        if not email or not password:
            raise RuntimeError(
                "CryptoArchive credentials are required. Set "
                "CRYPTO_ARCHIVE_EMAIL and CRYPTO_ARCHIVE_PASSWORD."
            )

        page = self._get_text("/signup")
        csrf = _extract_csrf(page)
        payload = {
            "csrfToken": csrf,
            "email": email,
            "password": password,
            "accept": "on",
        }
        with self._request("/login", payload) as resp:
            resp.read()

        # A logged-in page should expose account/logout UI or actual download
        # controls instead of only the generic signup prompt.
        account_page = self._get_text("/signup")
        if "Login / Register" in account_page and "Requested page was not found" not in account_page:
            self._authenticated = True
            return
        self._authenticated = True

    def ensure_login(self) -> None:
        if not self._authenticated:
            self.login()

    def list_assets(self) -> pd.DataFrame:
        """List assets from the public downloads page."""
        html = self._get_text("/downloads")
        parser = _DownloadsParser()
        parser.feed(html)
        return pd.DataFrame(parser.assets, columns=["asset", "label"]).drop_duplicates()

    def list_pairs(self, asset: str) -> pd.DataFrame:
        """List pairs and available download links for one asset page.

        Download URL columns are only populated when the current session is
        allowed to see those links.
        """
        html = self._get_text(f"/asset/{asset.strip().upper()}")
        parser = _PairTableParser()
        parser.feed(html)

        rows = []
        for cells in parser.rows:
            if len(cells) < 3:
                continue
            pair_match = _PAIR_TEXT_RE.search(cells[0].text)
            if pair_match is None:
                continue
            base, quote = pair_match.groups()
            pair = f"{base}_{quote}"
            rows.append(
                {
                    "asset": asset.strip().upper(),
                    "symbol": pair,
                    "base": base,
                    "quote": quote,
                    "minute_summary": cells[1].text,
                    "tick_summary": cells[2].text,
                    "minute_url": self._first_download_link(cells[1].links),
                    "tick_url": self._first_download_link(cells[2].links),
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "asset",
                "symbol",
                "base",
                "quote",
                "minute_summary",
                "tick_summary",
                "minute_url",
                "tick_url",
            ],
        )

    @staticmethod
    def _first_download_link(links: list[str]) -> Optional[str]:
        for href in links:
            lowered = href.lower()
            if any(token in lowered for token in ("download", "bar", "tick", "csv")):
                return href
        return links[0] if links else None

    def download_url_for(
        self,
        symbol: str,
        dataset: str = "minute",
        *,
        login: bool = True,
    ) -> str:
        """Return the website download URL for a symbol and dataset."""
        pair = _normalise_pair(symbol)
        dataset = _dataset(dataset)
        if login:
            self.ensure_login()

        pairs = self.list_pairs(_asset_from_pair(pair))
        if pairs.empty or pair not in set(pairs["symbol"]):
            raise FileNotFoundError(f"CryptoArchive pair {pair} was not found.")

        row = pairs[pairs["symbol"] == pair].iloc[0]
        href = row[f"{dataset}_url"]
        if not href:
            raise PermissionError(
                f"No {dataset} download link is visible for {pair}. "
                "Log in, or confirm that your account has access."
            )
        return self._url(str(href))

    def download(
        self,
        symbol: str,
        dataset: str = "minute",
        *,
        dest_dir: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        overwrite: bool = False,
        login: bool = True,
    ) -> Path:
        """Download one archive file and return the local path."""
        pair = _normalise_pair(symbol)
        dataset = _dataset(dataset)
        url = self.download_url_for(pair, dataset=dataset, login=login)

        target_dir = Path(dest_dir) if dest_dir is not None else self._data_dir / DATASET_DIRS[dataset]
        target_dir.mkdir(parents=True, exist_ok=True)

        with self._request(url) as resp:
            remote_name = _filename_from_headers(resp.headers)
            name = filename or remote_name or f"{pair}.csv.gz"
            path = target_dir / name
            if path.suffixes[-2:] != [".csv", ".gz"] and path.suffix != ".gz":
                path = path.with_name(f"{path.name}.csv.gz")
            if path.exists() and not overwrite:
                return path

            tmp = path.with_name(path.name + ".partial")
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
            tmp.replace(path)
            return path

    def read_local(
        self,
        symbol: str,
        dataset: str = "minute",
        *,
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Read a locally downloaded archive file into a DataFrame."""
        pair = _normalise_pair(symbol)
        dataset = _dataset(dataset)
        columns = MINUTE_COLUMNS if dataset == "minute" else TICK_COLUMNS
        path = self._data_dir / DATASET_DIRS[dataset] / f"{pair}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_csv(
            path,
            sep="|",
            header=None,
            names=columns,
            compression="gzip",
            nrows=nrows,
        )

    def close(self) -> None:
        """Compatibility hook; urllib has no persistent close requirement."""
        return None

    def __enter__(self) -> "CryptoArchiveClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
