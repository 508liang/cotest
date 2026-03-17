"""
Scholar PDF 检索器
流程：
  1. 用 SERPAPI Google Scholar 搜索，提取 resources 中的 PDF 链接
  2. 下载 PDF 并本地缓存（pdf_cache/ 目录），下载后强制验证 PDF 魔数
  3. 用 PyMuPDF (fitz) 提取文字层；失败则尝试 pdfplumber
  4. 将正文切分为最多 150 段，BM25 排名，取前 10 段
    5. 若本轮 Scholar 结果均无可用 PDF，回退到普通 Google 搜索：
     优先使用 SERPAPI 已返回的 snippet（无需再请求），
     再尝试能正常访问的 HTML 页面
"""
import os
import re
import time
import hashlib
import requests
import concurrent.futures

from serpapi import GoogleSearch

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False
    print("[ScholarRetriever] 警告：未安装 PyMuPDF，请运行 pip install pymupdf")

try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

import html2text as _html2text

_text_extractor = _html2text.HTML2Text()
_text_extractor.ignore_links = True
_text_extractor.ignore_images = True
_text_extractor.ignore_tables = True
_text_extractor.ignore_emphasis = True

PDF_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "pdf_cache")
os.makedirs(PDF_CACHE_DIR, exist_ok=True)

_DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*;q=0.9",
    "Accept-Language": "en-US,en;q=0.9",
}

_HTML_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

PDF_DOWNLOAD_HARD_TIMEOUT_SECONDS = 5
PDF_CONNECT_TIMEOUT_SECONDS = 3
PDF_READ_TIMEOUT_SECONDS = 5
PDF_MAX_PAGES = 50
SCHOLAR_MAX_PER_REQUEST = 20


def scholar_search(query: str, api_key: str, num: int = 10, start: int = 0) -> list:
    """
    Scholar 检索支持分页拉取。
    SerpAPI 的 google_scholar 单次 num 有上限，超过部分需要通过 start 分页获取。
    """
    target = max(0, int(num or 0))
    if target == 0:
        return []

    offset = max(0, int(start or 0))
    collected = []
    seen_keys = set()

    while len(collected) < target:
        batch_size = min(SCHOLAR_MAX_PER_REQUEST, target - len(collected))
        params = {
            "engine": "google_scholar",
            "q": query,
            "num": batch_size,
            "start": offset,
            "api_key": api_key,
        }
        search = GoogleSearch(params)
        batch = search.get_dict().get("organic_results", [])
        if not batch:
            break

        for item in batch:
            key = (
                str(item.get("result_id") or "").strip()
                or str(item.get("link") or "").strip()
                or str(item.get("title") or "").strip()
            )
            if key and key in seen_keys:
                continue
            if key:
                seen_keys.add(key)
            collected.append(item)
            if len(collected) >= target:
                break

        if len(batch) < batch_size:
            break
        offset += batch_size

    return collected[:target]


def extract_pdf_links(scholar_results: list) -> list:
    pdf_links = []
    for result in scholar_results:
        title = result.get("title", "")
        for resource in result.get("resources", []):
            if str(resource.get("file_format", "")).upper() == "PDF":
                link = resource.get("link", "")
                if link:
                    pdf_links.append((title, link))
    return pdf_links


def _pdf_local_path(url: str) -> str:
    name = hashlib.md5(url.encode()).hexdigest() + ".pdf"
    return os.path.join(PDF_CACHE_DIR, name)


def _is_valid_pdf(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"%PDF"
    except Exception:
        return False


def download_pdf(
    url: str,
    connect_timeout: int = PDF_CONNECT_TIMEOUT_SECONDS,
    read_timeout: int = PDF_READ_TIMEOUT_SECONDS,
    hard_timeout_seconds: int = PDF_DOWNLOAD_HARD_TIMEOUT_SECONDS,
) -> str | None:
    local_path = _pdf_local_path(url)

    if os.path.exists(local_path):
        if _is_valid_pdf(local_path):
            return local_path
        os.remove(local_path)

    try:
        start = time.time()
        resp = requests.get(
            url,
            headers=_DOWNLOAD_HEADERS,
            timeout=(connect_timeout, read_timeout),
            stream=True,
        )
        if resp.status_code != 200:
            print(f"[ScholarRetriever] HTTP {resp.status_code}: {url}")
            return None

        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                if time.time() - start > hard_timeout_seconds:
                    resp.close()
                    f.close()
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    print(f"[ScholarRetriever] 下载超时（>{hard_timeout_seconds}s），已跳过: {url}")
                    return None
                f.write(chunk)

        if not _is_valid_pdf(local_path):
            os.remove(local_path)
            print(f"[ScholarRetriever] 非 PDF 内容，已丢弃: {url}")
            return None

        return local_path
    except Exception as e:
        if os.path.exists(local_path):
            os.remove(local_path)
        print(f"[ScholarRetriever] 下载失败 {url}: {e}")
        return None


def _extract_page_fitz(args: tuple) -> tuple:
    local_path, page_num = args
    try:
        doc = fitz.open(local_path)
        text = doc[page_num].get_text("text")
        doc.close()
        return page_num, text
    except Exception:
        return page_num, ""


def _extract_page_pdfplumber(args: tuple) -> tuple:
    local_path, page_num = args
    try:
        with pdfplumber.open(local_path) as pdf:
            text = pdf.pages[page_num].extract_text() or ""
        return page_num, text
    except Exception:
        return page_num, ""


def extract_text_from_pdf(local_path: str, max_pages: int = PDF_MAX_PAGES) -> str:
    if _HAS_FITZ:
        try:
            doc = fitz.open(local_path)
            total_pages = len(doc)
            num_pages = min(total_pages, max_pages)
            doc.close()
            if total_pages > max_pages:
                print(f"[ScholarRetriever] PDF 页数 {total_pages}，仅解析前 {max_pages} 页")
            workers = max(1, min(num_pages, 8))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(
                    _extract_page_fitz,
                    [(local_path, i) for i in range(num_pages)]
                ))
            results.sort(key=lambda x: x[0])
            combined = "\n".join(t for _, t in results).strip()
            if combined:
                return combined
        except Exception as e:
            print(f"[ScholarRetriever] PyMuPDF 并行提取失败: {e}")

    if _HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(local_path) as pdf:
                total_pages = len(pdf.pages)
                num_pages = min(total_pages, max_pages)
            if total_pages > max_pages:
                print(f"[ScholarRetriever] PDF 页数 {total_pages}，仅解析前 {max_pages} 页")
            workers = max(1, min(num_pages, 8))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(
                    _extract_page_pdfplumber,
                    [(local_path, i) for i in range(num_pages)]
                ))
            results.sort(key=lambda x: x[0])
            combined = "\n".join(t for _, t in results).strip()
            if combined:
                return combined
        except Exception as e:
            print(f"[ScholarRetriever] pdfplumber 并行提取失败: {e}")

    return ""


def _split_paragraphs(text: str, max_paragraphs: int = 150) -> list:
    _MAX_PARA_CHARS = 800

    raw_paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(raw_paras) < 10:
        raw_paras = [p.strip() for p in re.split(r"(?<=[。！？.!?])\s+", text) if p.strip()]

    result = []
    for p in raw_paras:
        if len(p) < 20:
            continue
        if len(p) <= _MAX_PARA_CHARS:
            result.append(p)
        else:
            sentences = re.split(r"(?<=[。！？.!?])\s*", p)
            chunk = ""
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if not chunk:
                    chunk = s
                elif len(chunk) + 1 + len(s) <= _MAX_PARA_CHARS:
                    chunk += " " + s
                else:
                    if len(chunk) >= 20:
                        result.append(chunk)
                    chunk = s
            if len(chunk.strip()) >= 20:
                result.append(chunk.strip())

    return result[:max_paragraphs]


def _bm25_top_indices(query: str, paragraphs: list, top_k: int = 10) -> list:
    if not _HAS_BM25 or not paragraphs:
        return list(range(min(top_k, len(paragraphs))))

    def tokenize(text):
        tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text.lower())
        return tokens if tokens else list(text)

    tokenized_corpus = [tokenize(p) for p in paragraphs]
    tokenized_query = tokenize(query)

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    )
    return top_indices


def bm25_rank(query: str, paragraphs: list, top_k: int = 10) -> list:
    indices = _bm25_top_indices(query, paragraphs, top_k)
    return [paragraphs[i] for i in indices]


def _fetch_html_text(url: str, max_length: int = 6000) -> str | None:
    try:
        resp = requests.get(url, headers=_HTML_HEADERS, timeout=(5, 8), stream=True)
        if resp.status_code in (403, 401, 429):
            return None
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "").lower()
        if content_type and "html" not in content_type and "text" not in content_type:
            resp.close()
            return None

        chunks = []
        total = 0
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                chunks.append(chunk)
                total += len(chunk)
                if total >= 200_000:
                    break
        resp.close()
        encoding = resp.encoding or "utf-8"
        raw_html = b"".join(chunks).decode(encoding, errors="replace")
        raw = _text_extractor.handle(raw_html)
        raw = "\n".join(raw.split("\n\n"))
        return raw[:max_length]
    except Exception:
        return None


def google_web_fallback(query: str, api_key: str, top_k_para: int = 10) -> list:
    params = {
        "q": query,
        "num": 12,
        "google_domain": "google.com",
        "api_key": api_key,
    }
    _t_gs = time.time()
    print(f"[ScholarRetriever][WebFallback] 开始 Google 普通搜索: {query!r}")
    results = GoogleSearch(params).get_dict().get("organic_results", [])
    print(f"[ScholarRetriever][WebFallback] Google 搜索完成，找到 {len(results)} 条结果，耗时 {time.time()-_t_gs:.2f}s")

    tagged: list[tuple[str, str, str, str]] = []

    for result in results[:12]:
        title = result.get("title", "")
        link = result.get("link", "")
        scholar_snippet = result.get("snippet", "").strip()

        if scholar_snippet and len(scholar_snippet) >= 20:
            tagged.append((scholar_snippet, title, link, scholar_snippet))

        if link:
            print(f"[ScholarRetriever][WebFallback] 正在抓取页面: {link[:60]}")
            text = _fetch_html_text(link)
            if text:
                paras_from_page = _split_paragraphs(text, max_paragraphs=30)
                print(f"[ScholarRetriever][WebFallback] 抓取成功，提取 {len(paras_from_page)} 段落")
                for p in paras_from_page:
                    tagged.append((p, title, link, scholar_snippet))
            else:
                print("[ScholarRetriever][WebFallback] 页面不可访问或内容为空，跳过")

    if not tagged:
        print("[ScholarRetriever][WebFallback] 未收集到任何段落，返回空结果")
        return []

    print(f"[ScholarRetriever][WebFallback] 共收集 {len(tagged)} 段落，开始 BM25 排名（top_k={top_k_para}）...")
    plain_paras = [t[0] for t in tagged]
    top_indices = _bm25_top_indices(query, plain_paras, top_k=top_k_para)

    seen: dict = {}
    for i in top_indices:
        para_text, title, link, scholar_snippet = tagged[i]
        if link not in seen:
            seen[link] = {
                "title": title,
                "link": link,
                "snippet": scholar_snippet or para_text[:150],
                "paragraph": para_text,
            }
        else:
            seen[link]["paragraph"] += "\n\n" + para_text
    return list(seen.values())


def retrieve_top_paragraphs(
    query: str,
    api_key: str,
    scholar_num: int = 10,
    max_paragraphs_per_pdf: int = 150,
    top_k: int = 10,
) -> tuple[list, str]:
    _t0 = time.time()
    print(f"[ScholarRetriever] 开始 Google Scholar 搜索: {query!r}")
    scholar_results = scholar_search(query, api_key, num=scholar_num)
    print(f"[ScholarRetriever] Scholar 搜索完成，找到 {len(scholar_results)} 条结果，耗时 {time.time()-_t0:.2f}s")
    pdf_links = extract_pdf_links(scholar_results)
    print(f"[ScholarRetriever] 找到 {len(pdf_links)} 个 PDF 链接")

    pdf_url_to_meta = {}
    for result in scholar_results:
        title = result.get("title", "")
        scholar_snippet = result.get("snippet", "")
        for resource in result.get("resources", []):
            if str(resource.get("file_format", "")).upper() == "PDF":
                purl = resource.get("link", "")
                if purl:
                    pdf_url_to_meta[purl] = {
                        "title": title,
                        "link": purl,
                        "scholar_snippet": scholar_snippet,
                    }

    tagged_paragraphs = []
    _pdf_ok = 0
    for title, url in pdf_links:
        meta = pdf_url_to_meta.get(url, {"title": title, "link": url, "scholar_snippet": ""})
        print(f"[ScholarRetriever] 正在下载 PDF: {title[:40]}...")
        _t_dl = time.time()
        local = download_pdf(url)
        if not local:
            print(f"[ScholarRetriever] PDF 下载失败，跳过: {title[:40]}")
            continue
        print(f"[ScholarRetriever] PDF 下载完成，耗时 {time.time()-_t_dl:.2f}s")
        text = extract_text_from_pdf(local)
        if not text:
            print(f"[ScholarRetriever] 无法提取文本: {title[:40]}")
            continue
        paras = _split_paragraphs(text, max_paragraphs=max_paragraphs_per_pdf)
        print(f"[ScholarRetriever] {title[:40]}... → 切分为 {len(paras)} 段")
        _pdf_ok += 1
        for p in paras:
            tagged_paragraphs.append((p, meta["title"], meta["link"], meta["scholar_snippet"]))

    if tagged_paragraphs:
        print(f"[ScholarRetriever] 共收集到 {len(tagged_paragraphs)} 段落（来自 {_pdf_ok} 篇 PDF），开始 BM25 排名...")
        plain_paras = [t[0] for t in tagged_paragraphs]
        top_indices_ordered = _bm25_top_indices(query, plain_paras, top_k=top_k)

        MAX_PARAGRAPH_CHARS = 2000
        seen: dict = {}
        for i in top_indices_ordered:
            para_text, title, pdf_url, scholar_snippet = tagged_paragraphs[i]
            if pdf_url not in seen:
                seen[pdf_url] = {
                    "title": title,
                    "link": pdf_url,
                    "snippet": scholar_snippet or para_text[:150],
                    "paragraph": para_text[:MAX_PARAGRAPH_CHARS],
                }
            else:
                current = seen[pdf_url]["paragraph"]
                if len(current) < MAX_PARAGRAPH_CHARS:
                    remaining = MAX_PARAGRAPH_CHARS - len(current)
                    seen[pdf_url]["paragraph"] += "\n\n" + para_text[:remaining]

        results = list(seen.values())
        print(f"[ScholarRetriever] Scholar 检索完成，BM25 筛选后共 {len(results)} 篇论文将发送给 LLM")
        return results, "scholar"

    print("[ScholarRetriever] 无 PDF 段落，回退到 Google Web（snippet + HTML）")
    _t_web = time.time()
    results = google_web_fallback(query, api_key, top_k_para=top_k)
    print(f"[ScholarRetriever] Google Web 回退完成，最终返回 {len(results)} 条结果，耗时 {time.time()-_t_web:.2f}s，将发送给 LLM")
    return results, "google_web"
