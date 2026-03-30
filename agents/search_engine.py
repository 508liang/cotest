try:
    # Old SDK layout
    from serpapi import GoogleSearch
except ImportError:
    # New SDK layout (serpapi>=1.0)
    from serpapi.google_search import GoogleSearch


class SearchEngine:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_search_results(self, query):
        params = self._get_search_params(query)
        results = self._get_google_search_results(params)
        return self._format_search_results(results)

    def get_academic_results(self, query):
        """
        使用 Google Scholar（通过 SerpAPI）获取偏学术/论文的搜索结果，
        用于选题场景下的“来源”展示，减少无关网站。
        对每条结果会尽量补全 link（主 link 为空时用 versions/resources/cited_by 等回退）。
        """
        params = {
            "engine": "google_scholar",
            "q": query,
            "num": 10,
            "api_key": self.api_key,
        }
        search = GoogleSearch(params)
        results = search.get_dict().get("organic_results", [])
        formatted = self._format_academic_results(results)

        # 优先保留常见学术站点 / 论文源，如果为空再退回全部结果
        preferred_domains = [
            "scholar.google.",
            "springer",
            "sciencedirect",
            "ieee.org",
            "acm.org",
            "nature.com",
            "wiley.com",
            "tandfonline.com",
            "elsevier.com",
            "arxiv.org",
            "ssrn.com",
            "cnki.net",
            "wanfangdata",
        ]
        academic_results = [
            r for r in formatted
            if any(domain in r["link"] for domain in preferred_domains)
        ]
        return academic_results or formatted

    def _get_search_params(self, query):
        return {
            "q": query,
            "num": 10,
            "google_domain": "google.com",
            "api_key": self.api_key
        }

    def _get_google_search_results(self, params):
        search = GoogleSearch(params)
        return search.get_dict().get("organic_results", [])

    def _format_search_results(self, results):
        return [
            {
                "title": result.get("title", ""),
                "link": result.get("link", ""),
                "snippet": result.get("snippet", "")
            }
            for result in results
        ]

    def _pick_academic_link(self, result):
        """
        从 SerpAPI Google Scholar 单条结果中选取可用的 URL。
        主 link 为空时依次回退：versions 页、resources[0]（如 PDF）、cited_by 页。
        """
        link = (result.get("link") or "").strip()
        if link:
            return link
        il = result.get("inline_links") or {}
        ver = il.get("versions") or {}
        if isinstance(ver, dict):
            link = (ver.get("link") or "").strip()
            if link:
                return link
        resources = result.get("resources") or []
        if resources and isinstance(resources[0], dict):
            link = (resources[0].get("link") or "").strip()
            if link:
                return link
        cited = il.get("cited_by") or {}
        if isinstance(cited, dict):
            link = (cited.get("link") or "").strip()
            if link:
                return link
        return ""

    def _format_academic_results(self, results):
        """学术结果格式化：统一用 _pick_academic_link 保证每条尽量有可点击 link。"""
        return [
            {
                "title": result.get("title", ""),
                "link": self._pick_academic_link(result),
                "snippet": result.get("snippet", "")
            }
            for result in results
        ]