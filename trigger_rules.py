import re


CONFUSION_CUES = (
    "听不懂",
    "不懂",
    "看不懂",
    "能解释",
    "解释一下",
    "什么意思",
    "啥是",
    "是啥",
    "什么是",
    "是什么",
    "啥",
)

CONFLICT_KEYWORDS = (
    "不同意",
    "不认可",
    "不行",
    "有问题",
    "不成立",
    "不合理",
    "反对",
    "争议",
)

DECISION_HINTS = (
    "还是",
    "是否",
    "哪个好",
    "怎么选",
    "该用",
    "是先",
    "要不要",
)

LOW_INFO_JUDGMENT_CUES = (
    "选择哪个",
    "选哪个",
    "哪个好",
    "怎么选",
    "怎么办",
    "该选哪个",
)


def has_confusion_cue(query: str) -> bool:
    return any(cue in query for cue in CONFUSION_CUES)


def clean_query_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^[\s,，。.!！？:：;；、~\-]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def is_conflict_like_message(query: str) -> bool:
    return any(keyword in query for keyword in CONFLICT_KEYWORDS)


def is_decision_like_message(query: str) -> bool:
    clean = clean_query_text(query)
    if not clean:
        return False
    if "？" in clean or "?" in clean:
        return True
    return any(hint in clean for hint in DECISION_HINTS)


def is_low_information_judgment_query(query: str) -> bool:
    clean = clean_query_text(query)
    if not clean:
        return True
    if any(cue in clean for cue in LOW_INFO_JUDGMENT_CUES):
        return len(clean) <= 14
    return len(clean) <= 6


def extract_candidate_terms(query: str, limit: int = 5) -> list[str]:
    english_terms = re.findall(
        r"(?<![A-Za-z\d])[A-Z]{2,}(?![A-Za-z\d])|(?<![A-Za-z\d])[A-Za-z]{6,}(?![A-Za-z\d])",
        query,
    )
    chinese_terms = re.findall(
        r"[\u4e00-\u9fff]{2,}(?:模型|算法|机制|范式|矩阵|网络|优化|动力学|控制|证明|定理)",
        query,
    )

    terms = []
    for term in english_terms + chinese_terms:
        clean = term.strip().lower()
        if clean and clean not in terms:
            terms.append(clean)
    return terms[:limit]
