"""Routing and language-detection tests. No model weights are loaded: `Router.route` is pure."""
import sys
import os
import threading
import time as _time
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import QTYPES, TEMP_MAX, TEMP_MIN, clamp_temperature, temp_bucket  # noqa: E402
from laya.lang import analyse, detect_script, guess_latin_language, is_english, state_text  # noqa: E402
from laya.router import (  # noqa: E402
    BUNDLE_REPO,
    DEFAULT_MODELS,
    STANDALONE_MODELS,
    _english_from_code,
    _repo_str,
    _split,
    Router,
    match_typed_decisions_workflow,
    normalise_name,
    resolve_model_spec,
)

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


# --------------------------------------------------------------------- script detection
SCRIPTS = [
    ("english", "The customer was charged twice and wants a refund.", "latin"),
    ("armenian", "Հայերեն", "armenian"),
    ("armenian uppercase", "ՀԱՅԵՐԵՆ", "armenian"),
    ("armenian punctuation only", "։֊", "unknown"),
    ("azerbaijani lone schwa", "ə", "latin"),
    ("azerbaijani uppercase", "MÜŞTƏRİ İLƏ ƏLAQƏ SAXLAYIN", "latin"),
    ("french", "Le client a été facturé deux fois et demande un remboursement.", "latin"),
    ("hindi", "ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।", "devanagari"),
    ("japanese", "お客様は二重に請求されたため返金を希望しています。", "kana"),
    ("chinese", "客户被重复扣款要求退款", "han"),
    ("korean", "고객이 두 번 청구되어 환불을 원합니다", "hangul"),
    ("arabic", "تم خصم المبلغ مرتين من العميل ويريد استرداد الأموال", "arabic"),
    ("tamil", "வாடிக்கையாளரிடம் இருமுறை கட்டணம் வசூலிக்கப்பட்டது", "tamil"),
    ("russian", "С клиента дважды сняли деньги и он хочет возврат", "cyrillic"),
    ("thai", "ลูกค้าถูกเรียกเก็บเงินสองครั้งและต้องการเงินคืน", "thai"),
    ("greek", "Ο πελάτης χρεώθηκε δύο φορές και θέλει επιστροφή χρημάτων", "greek"),
    ("hebrew", "הלקוח חויב פעמיים ורוצה החזר כספי", "hebrew"),
    ("empty", "", "unknown"),
    ("digits only", "12345 6789", "unknown"),
]
for label, text, want in SCRIPTS:
    check("script/" + label, detect_script(text), want)


# --------------------------------------------------------------------- english vs not
for label, text, want in [
    ("plain english", "Please refund the duplicate charge on invoice 4411 today.", True),
    ("armenian", "Հայերեն", False),
    ("azerbaijani no diacritics", "Sifarisim gelmedi ve pulum geri qaytarilmadi, zehmet olmasa yoxlayin", False),
    ("azerbaijani few diacritics", "Mən sizin xidmətinizdən razı deyiləm və pulumu geri istəyirəm", False),
    ("english short", "refund me", True),
    ("hindi", "ग्राहक से दो बार शुल्क लिया गया", False),
    ("japanese", "お客様は二重に請求されました", False),
    ("russian", "С клиента дважды сняли деньги", False),
    ("french long", "Le client a été facturé deux fois et il demande un remboursement pour la "
                    "facture qui a été payée le mois dernier avec la carte de crédit", False),
    ("german long", "Der Kunde wurde zweimal belastet und möchte eine Rückerstattung für die "
                    "Rechnung die nicht korrekt ist und auch nicht bezahlt wurde", False),
    # Latin-script languages with no stopword list of their own: reported in #35, where Romanian
    # states were handed to the English checkpoint (0.330 accuracy, 0.658 ECE on `ro`) instead of
    # the multilingual one. An unidentified language must never be assumed English.
    ("romanian", "Gătește-mi o rețetă de sarmale de post pentru mâine.", False),
    ("romanian invoice", "Am fost taxat de două ori pentru factura din luna martie și vreau banii", False),
    ("polish", "Klient został obciążony dwukrotnie i chce zwrot pieniędzy za fakturę", False),
    ("czech", "Zákazníkovi byla částka účtována dvakrát a žádá o vrácení peněz", False),
    ("turkish", "Müşteriden iki kez ücret alındı ve para iadesi istiyor lütfen yardım", False),
    ("vietnamese", "Khách hàng đã bị thu phí hai lần và muốn được hoàn tiền ngay", False),
    # English with the odd loanword must not tip over into the multilingual checkpoint
    ("english with loanwords", "We visited a cafe in Zurich and the naive assumption about the "
                               "invoice was wrong, so please refund the duplicate charge", True),
]:
    check("is_english/" + label, is_english(text), want)

# Undecided is reported as undecided rather than dressed up as a detection: a single shared
# function word used to name a language ("para" in Turkish text was called Spanish).
check("latin/undecided is flagged", analyse("Müşteriden iki kez ücret alındı ve para iadesi istiyor")["language_undecided"], True)
check("latin/undecided names no language", analyse("Müşteriden iki kez ücret alındı ve para iadesi istiyor")["language"], None)
check("latin/english is not undecided", analyse("Please refund the duplicate charge on the invoice")["language_undecided"], False)
check("latin/diacritic rate reported", analyse("Gătește-mi o rețetă de sarmale")["diacritic_rate"] > 0.02, True)
check("latin/english has no diacritics", analyse("Please refund the duplicate charge today")["diacritic_rate"], 0.0)
# every branch of analyse() reports the same keys, so a caller can read one without guarding
_KEYS = {"script", "script_profile", "language", "is_english", "language_undecided",
         "diacritic_rate", "non_latin_fraction", "mixed_segment"}
for label, text in [("english", "Please refund the duplicate charge"), ("hindi", "ग्राहक से दो बार"),
                    ("romanian", "Gătește-mi o rețetă de sarmale"), ("no letters", "12345 ???")]:
    check("analyse/keys " + label, set(analyse(text)), _KEYS)
# A 0-0 tie between non-English stopword lists is no evidence for any of them
check("latin_lang/zero tie invents nothing", guess_latin_language("Cât e ora acum la Tokyo"), None)

# Known limitation, kept visible on purpose: Romanian short enough to carry no diacritics and an
# English function word ("in") still reads as English. A real LID model is the fix, not more
# stopwords -- see the discussion in #35.
check("latin/KNOWN GAP romanian without diacritics", is_english("Care este ora in Tokyo?"), True)


# --------------------------------------------------------------------- Latin language guess
for label, text, want in [
    ("english", "The customer was charged twice and wants a refund for this invoice", "en"),
    ("french", "Le client a ete facture deux fois et il demande un remboursement pour la facture", "fr"),
    ("german", "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung", "de"),
    ("spanish", "El cliente fue cobrado dos veces y quiere que le devuelvan el dinero por la factura", "es"),
    ("azerbaijani", "Zəhmət olmasa, sifarişim üçün pulu geri qaytarın, çünki məhsul gəlmədi", "az"),
    # one shared function word is not enough to name a language
    ("azerbaijani single hit stays undecided",
     "Müştəridən iki dəfə pul alınıb və o, geri qaytarılmasını istəyir", None),
    ("azerbaijani uppercase dotted I", "MÜŞTƏRİ İLƏ ƏLAQƏ SAXLAYIN VƏ PULU GERİ QAYTARIN", "az"),
    ("too short", "refund", None),
]:
    check("latin_lang/" + label, guess_latin_language(text), want)
# a non-English guess must never fire on ordinary English
check("latin_lang/long english stays en",
      guess_latin_language("Please refund the duplicate charge on invoice 4411 today because "
                           "we have been waiting for three days and nobody has replied to us"), "en")


# --------------------------------------------------------------------- dotted tokens (#177)
# `com` is Portuguese ("with") and `o` is its article, and `_WORD` splits `github.com` into
# `github` + `com`, so every domain in a state scored a Portuguese hit: two of them crossed the
# margin and sent an English state with links in it to the multilingual checkpoint, reported as
# Portuguese. A token whose dot or @ joins word characters is an identifier, not prose.
_r_dotted = Router()
for label, state in [
    ("url and email fields", {"url": "github.com", "email": "user@acme.com"}),
    ("two bare domains", "github.com acme.com"),
    ("link list", {"links": ["example.com", "example.co.uk", "docs.readthedocs.io"]}),
]:
    check("latin_lang/dotted " + label, analyse(state)["language"], None)
    check("is_english/dotted " + label, is_english(state), True)
    check("route/dotted " + label, _r_dotted.route(state).model, "english")
# versions, decimals and dotted abbreviations are identifiers too, and were never prose
for text, label in [("build 1.2.3 on 12.30 with ratio 0.5", "version and decimal"),
                    ("Report by Smith et al., e.g. the U.S.A. office", "dotted abbreviation")]:
    check("is_english/dotted " + label, is_english(text), True)
    check("route/dotted " + label, _r_dotted.route(text).model, "english")
# masking identifiers must not cost the prose around them its language, and a full stop ends a
# sentence rather than joining an identifier: the word before it keeps its letters.
check("latin_lang/portuguese prose with a link",
      guess_latin_language("O cliente nao recebeu o produto, mas quer o dinheiro para a conta, "
                           "veja example.com"), "pt")
check("latin_lang/portuguese sentence with a full stop",
      guess_latin_language("O cliente nao recebeu o produto, mas quer o dinheiro para a conta."), "pt")
check("route/english with a link stays english",
      _r_dotted.route({"body": "Please check example.com and acme.com for the invoice"}).model, "english")


# --------------------------------------------------------------------- state flattening
check("state_text/dict", "charged twice" in state_text({"body": "charged twice", "n": 3}), True)
check("state_text/nested", "deep" in state_text({"a": {"b": ["deep"]}}), True)
check("state_text/list", "x" in state_text(["x", {"y": "z"}]), True)
check("state_text/none", state_text(None), "")
# keys must not drive detection: English keys around Hindi content stay non-English
check("state_text/keys ignored",
      analyse({"subject": "नमस्ते", "body": "ग्राहक से दो बार शुल्क लिया गया"})["is_english"], False)

# --------------------------------------------------------------------- dict-state language (#384)
# The same German sentence must route the same way as a string and as a dict value.
# A mapping that is not a builtin dict, a bytes value, and English sibling fields used to
# hide that value: detection then reported language_undecided (or English) and the router
# fell through to the English checkpoint.
_DE = "Mein Konto wurde zweimal belastet"
_r_de = Router()
_de_str = _r_de.route(_DE, {})
_de_dict = _r_de.route({"message": _DE}, {})
check("route/dict german matches string", _de_dict.model, _de_str.model)
check("route/dict german is multilingual", _de_dict.model, "multilingual")
check("route/dict german names de", _de_dict["detection"]["language"], "de")
check("route/dict german is not undecided", _de_dict["detection"]["language_undecided"], False)
check("analyse/dict german matches string", analyse({"message": _DE})["language"], analyse(_DE)["language"])
# English notes must not outvote the message, and a long note must not push it out of the window.
_de_ticket = {
    "ticket_id": "TCK-88213",
    "channel": "web chat",
    "agent_notes": "Please check the shipping status and refund the customer if the charge was duplicated.",
    "message": _DE,
}
check("route/dict german beside english notes", _r_de.route(_de_ticket, {}).model, "multilingual")
check("route/dict german beside english notes is not undecided",
      _r_de.route(_de_ticket, {})["detection"]["language_undecided"], False)
_de_long = {
    "agent_notes": "Please check the shipping status and tell the customer about the refund. " * 80,
    "message": _DE,
}
check("route/dict german after long english note", _r_de.route(_de_long, {}).model, "multilingual")
check("route/dict german after long english note names de",
      _r_de.route(_de_long, {})["detection"]["language"], "de")
# The leaf scan #384 added runs a full `_analyse_text` pass on every line of every value. A line
# too short to be selected is now skipped before that happens. The #384 checks above pin the
# answers; these add the cost side, the constant itself, and the invariant a cheaper scan must
# not trade away.
#
# Ratios to prose of the same size, never wall-clock, so a loaded runner inflates both sides.
# Warm-up then best of nine: the numerators are ~2 ms windows against a ~24 ms denominator, and
# at five reps a single descheduled run moved the 10 000-field ratio to 0.385. Nine reps holds it.
# Previous / ceiling / current, each ceiling the geometric mean of the pair it separates:
#   one-char lines, 50 000 chars       2.85 / 0.80 / 0.23
#   six-char lines, 50 000 chars       1.24 / 0.35 / 0.10
#   10 000 six-character fields        1.85 / 0.68 / 0.25
# Worst current ratio over six trials was 0.232 / 0.098 / 0.249 idle and 0.231 / 0.097 / 0.246
# with twelve busy processes on the box, so every ceiling keeps at least 2.7x of room under load;
# reverting gives 3.19 / 1.25 / 2.27 under that same load, 3.3x or more the other way. The
# measured ratio goes in the check name, so a red CI log says what it was, not only "got False".
def _ms(state, reps=9):
    analyse(state)
    analyse(state)
    best = float("inf")
    for _ in range(reps):
        t = _time.perf_counter()
        analyse(state)
        best = min(best, _time.perf_counter() - t)
    return best * 1000


# Prose LINES, not one long line: `_leaf_non_english` slices each line to 4000 characters, so a
# 50 000-character one-liner does a fraction of the work and would be a meaningless denominator
# (2.2 ms against 24.1 ms for the same characters as lines). Every state here is a dict on
# purpose: `analyse` returns a plain string's verdict before the leaf scan runs.
_PROSE_50K = {"body": "\n".join(["The customer was billed twice and wants a refund."] * 1020)[:50_000]}
_prose_ms = _ms(_PROSE_50K)
for _name, _state, _ceiling in (
    ("one-char lines", {"body": ("a\n" * 25_000)[:50_000]}, 0.80),
    ("six-char lines", {"body": ("abcdef\n" * 7_142)[:50_000]}, 0.35),
    ("10k six-char fields", {"f%d" % i: "abcdef" for i in range(10_000)}, 0.68),
):
    _ratio = _ms(_state) / _prose_ms
    check("leaf scan/%s vs prose = %.2f, ceiling %.2f" % (_name, _ratio, _ceiling),
          _ratio < _ceiling, True)

# The constant. Seven is the largest sound threshold: each branch of `_leaf_non_english` needs
# four `_WORD` tokens -- maximal runs of letters and letter-like numerals, so four need three
# separators between them -- or ten letters. "é à ü ö" is exactly seven characters and four
# tokens, and must still be read: this is the check that pins the threshold, and raising it to 8
# turns that check red. "é à üö" is six characters and three tokens; no six-character line can be
# selected at any threshold, so that check is a behaviour pin rather than a second bound -- it
# passes on unguarded code too, and is here to catch a future change that makes short lines
# selectable. Both values sit past 4000 characters of English, which is what makes the leaf scan
# the code under test: before that the segment scan has already answered.
_EN_PAST_SEGMENT_CAP = "The customer was billed twice and wants a refund. " * 120
check("leaf scan/a 7-character foreign line is still read",
      analyse({"note": _EN_PAST_SEGMENT_CAP, "msg": "é à ü ö"})["is_english"], False)
check("leaf scan/a 6-character line stays English",
      analyse({"note": _EN_PAST_SEGMENT_CAP, "msg": "é à üö"})["is_english"], True)

# #384's invariant, which a budget shared across the state would break: a long earlier value must
# not stop a later one from being read. Cheap to keep, and it is the one way a future attempt to
# bound this scan by total characters would go wrong silently -- the state below would route to
# the English checkpoint, which BENCHMARKS.md shows collapsing off English.
check("leaf scan/a later value is still read after a 50k earlier one",
      analyse({"pad": "x " * 25_000, "msg": "我们三月份被重复收费了两次。"})["is_english"], False)

from collections import UserDict  # noqa: E402
from types import MappingProxyType  # noqa: E402
check("route/userdict german", _r_de.route(UserDict({"message": _DE}), {}).model, "multilingual")
check("route/userdict german is not undecided",
      _r_de.route(UserDict({"message": _DE}), {})["detection"]["language_undecided"], False)
check("route/mappingproxy german",
      _r_de.route(MappingProxyType({"message": _DE}), {}).model, "multilingual")
check("route/bytes german value",
      _r_de.route({"message": _DE.encode("utf-8")}, {}).model, "multilingual")
# A name beside an English request is not a second message.
check("route/dict cyrillic name stays english",
      _r_de.route({"name": "Антон Павлович Чехов",
                   "body": "Please refund the duplicate charge on invoice 4411 today."}, {}).model,
      "english")
check("route/dict jose stays english",
      _r_de.route({"name": "José",
                   "body": "Please refund the duplicate charge on invoice 4411 today."}, {}).model,
      "english")
check("route/dict english body stays english",
      _r_de.route({"body": "Please refund the duplicate charge on invoice 4411 today."}, {}).model,
      "english")


# --------------------------------------------------------------------- workflow signatures
check("profile/armenian", analyse("Հայերեն")["script_profile"], {"armenian": 1.0})
check("profile/armenian mixed with Latin", analyse("Հայերեն abc")["non_latin_fraction"], 0.7)
check("profile/azerbaijani schwa is latin", analyse("ələ")["script_profile"], {"latin": 1.0})

TD = {
    "agent_trace_observability": ["action", "needs_review", "outcome", "risk", "urgency"],
    "customer_service": ["action", "category", "churn_risk", "needs_human", "urgency"],
    "invoice_processing": ["discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"],
    "security_incidents": ["credential_compromise", "disposition", "severity", "true_positive", "urgency"],
}
for wf, ids in TD.items():
    check("workflow/" + wf, match_typed_decisions_workflow({i: {} for i in ids}), wf)
check("workflow/partial overlap", match_typed_decisions_workflow({"urgency": {}, "category": {}}), None)
check("workflow/superset", match_typed_decisions_workflow({i: {} for i in TD["customer_service"] + ["extra"]}), None)
check("workflow/empty", match_typed_decisions_workflow({}), None)


# --------------------------------------------------------------------- name normalisation
for alias, want in [("en", "english"), ("laya", "english"), ("multi", "multilingual"),
                    ("ML", "multilingual"), ("typed", "typed-decisions"),
                    ("typed_decisions", "typed-decisions"), ("English", "english"),
                    ("convaiinnovations/laya".split("/")[-1], "english")]:
    check("alias/" + alias, normalise_name(alias), want)
try:
    normalise_name("nope")
    FAIL.append("alias/unknown: should have raised")
except ValueError:
    PASS.append("alias/unknown raises")


# --------------------------------------------------------------------- registry spec resolution (#780)
# `load()` resolves a name or alias through this table instead of forwarding it to the Hub as a
# repo id, so both entry points read one registry. Anything the registry does not know -- a repo
# id, a local path, an ONNX export -- resolves to None and is left alone.
for name, want in [("english", ("convaiinnovations/laya", None)),
                   ("laya", ("convaiinnovations/laya", None)),
                   ("typed-decisions", ("convaiinnovations/laya", "typed-decisions")),
                   ("typed", ("convaiinnovations/laya", "typed-decisions")),
                   ("ml", ("convaiinnovations/laya", "multilingual")),
                   ("MULTI", ("convaiinnovations/laya", "multilingual")),
                   (" typed-decisions ", ("convaiinnovations/laya", "typed-decisions"))]:
    check("spec/" + name, resolve_model_spec(name), want)
check("spec/agrees with normalise_name",
      resolve_model_spec("decisions"), tuple(_split(DEFAULT_MODELS[normalise_name("decisions")])))
for unknown in ("convaiinnovations/laya", "test/custom-model", "/tmp/checkpoint", "./local",
                "nope", "", "convaiinnovations/laya-typed-decisions"):
    check("spec/not a name: " + (unknown or "<empty>"), resolve_model_spec(unknown), None)


# --------------------------------------------------------------------- routing decisions
r = Router()
Q_GENERIC = {"dept": {"type": "choice", "instructions": "Which team?",
                      "criteria": {"billing": None, "tech": None}}}
Q_TD = {i: {"type": "noul", "instructions": "x"} for i in TD["customer_service"]}

cases = [
    ("english text", {"body": "I was charged twice, please refund."}, Q_GENERIC, {}, "english"),
    ("armenian text", {"body": "Հայերեն"}, Q_GENERIC, {}, "multilingual"),
    ("armenian explicit override", {"body": "Հայերեն"}, Q_GENERIC, {"model": "english"}, "english"),
    ("azerbaijani no diacritics", {"body": "Sifarisim gelmedi ve pulum geri qaytarilmadi, zehmet olmasa yoxlayin"},
     Q_GENERIC, {}, "multilingual"),
    ("azerbaijani explicit override", {"body": "Müştəridən iki dəfə pul alınıb"}, Q_GENERIC,
     {"model": "english"}, "english"),
    ("hindi text", {"body": "मुझसे दो बार शुल्क लिया गया"}, Q_GENERIC, {}, "multilingual"),
    ("japanese text", {"body": "二重に請求されました"}, Q_GENERIC, {}, "multilingual"),
    ("korean text", {"body": "두 번 청구되었습니다"}, Q_GENERIC, {}, "multilingual"),
    ("arabic text", {"body": "تم خصم المبلغ مرتين"}, Q_GENERIC, {}, "multilingual"),
    # Latin brand names are the letter plurality here, but the request itself is CJK
    ("chinese with a brand", {"body": "我的 iPhone 15 Pro Max 订单还没到"}, Q_GENERIC, {}, "multilingual"),
    ("japanese with brands", {"body": "Amazonで買ったiPhoneが届かない"}, Q_GENERIC, {}, "multilingual"),
    ("korean with a brand", {"body": "Samsung Galaxy 주문이 아직 안 왔어요"}, Q_GENERIC, {}, "multilingual"),
    ("english with a han name", {"body": "My name is 王小明 and my order is late"}, Q_GENERIC, {}, "english"),
    # an English wrapper dilutes the share, but the request is still CJK
    ("chinese in a ticket", {"ticket_id": "TCK-88213", "channel": "web chat",
                             "agent_notes": "Customer asked about a delayed order. Please check shipping status.",
                             "message": "我的订单已经两个星期了还没有到"}, Q_GENERIC, {}, "multilingual"),
    ("korean after english turns", [{"role": "agent", "text": "Hello! Thanks for contacting support."},
                                    {"role": "agent", "text": "Could you share your order number please?"},
                                    {"role": "user", "text": "주문번호는 5521이고 아직 배송이 안 됐어요"}],
     Q_GENERIC, {}, "multilingual"),
    ("english with greek symbols", {"request": "Compute the mean μ and variance σ of X, then P(|X-μ| > 2σ)."},
     Q_GENERIC, {}, "english"),
    # English prose that names someone in their own script: the name is not the request, and a
    # capitalised run, a lone symbol and a pronunciation are all annotation rather than content.
    ("english prose, russian name",
     {"body": "Anton Pavlovich Chekhov (Russian: Антон Павлович Чехов) was a playwright."},
     Q_GENERIC, {}, "english"),
    ("english prose, name with IPA",
     {"body": "Vladimir Nabokov (Russian: Влади́мир Набо́ков [vlɐˈdʲimʲɪr nɐˈbokəf]) wrote Lolita "
              "and taught literature at Cornell for more than a decade."},
     Q_GENERIC, {}, "english"),
    ("english prose, greek name",
     {"body": "Eleftherios Venizelos (Greek: Ελευθέριος Βενιζέλος) served as prime minister."},
     Q_GENERIC, {}, "english"),
    ("english prose, hebrew name",
     {"body": "Amos Oz (Hebrew: עמוס עוז), born Amos Klausner, was an Israeli writer and professor "
              "of literature at Ben-Gurion University of the Negev in Beersheba."},
     Q_GENERIC, {}, "english"),
    ("german text", {"body": "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung "
                             "fuer die Rechnung die nicht korrekt ist"}, Q_GENERIC, {}, "multilingual"),
    ("explicit model", {"body": "anything"}, Q_GENERIC, {"model": "multilingual"}, "multilingual"),
    ("explicit model overrides script", {"body": "मुझसे दो बार"}, Q_GENERIC,
     {"model": "english"}, "english"),
    ("explicit task", {"body": "x"}, Q_GENERIC, {"task": "typed_decisions"}, "typed-decisions"),
    ("explicit lang en", {"body": "मुझसे दो बार"}, Q_GENERIC, {"lang": "en"}, "english"),
    ("explicit lang de", {"body": "hello there"}, Q_GENERIC, {"lang": "de"}, "multilingual"),
    ("td workflow, auto OFF", {"body": "I was charged twice"}, Q_TD, {}, "english"),
    ("empty state", {}, Q_GENERIC, {}, "english"),
    ("none state", None, Q_GENERIC, {}, "english"),
]
for label, state, qs, kw, want in cases:
    check("route/" + label, r.route(state, qs, **kw)["model"], want)

# auto task detection is opt-in
r_auto = Router(auto_task_detection=True)
check("route/td workflow, auto ON",
      r_auto.route({"body": "I was charged twice"}, Q_TD)["model"], "typed-decisions")
check("route/auto ON but generic questions",
      r_auto.route({"body": "I was charged twice"}, Q_GENERIC)["model"], "english")
# explicit model still beats auto-detected workflow
check("route/explicit beats workflow",
      r_auto.route({"body": "x"}, Q_TD, model="multilingual")["model"], "multilingual")
# An auto-detected workflow reports `repo` like every other branch does. It used to hand back the raw
# (repo, subfolder) spec, which serialises to a JSON list instead of the "repo/subfolder" string.
check("route/auto workflow repo is a string",
      r_auto.route({"body": "I was charged twice"}, Q_TD)["repo"], "convaiinnovations/laya/typed-decisions")
check("route/auto workflow repo matches explicit task",
      r_auto.route({"body": "I was charged twice"}, Q_TD)["repo"],
      r_auto.route({"body": "I was charged twice"}, Q_TD, task="typed_decisions")["repo"])
check("route/auto workflow repo standalone",
      Router(auto_task_detection=True, standalone_repos=True).route({"body": "x"}, Q_TD)["repo"],
      "convaiinnovations/laya-typed-decisions")

# decision payload shape
d = r.route({"body": "मुझसे दो बार शुल्क लिया गया"}, Q_GENERIC)
check("decision/has repo", d["repo"], "convaiinnovations/laya/multilingual")
check("decision/has reason", isinstance(d["reason"], str) and len(d["reason"]) > 0, True)
check("decision/detection script", d["detection"]["script"], "devanagari")
check("decision/.model property", d.model, "multilingual")
check("decision/is dict", isinstance(d, dict), True)

# default override
check("route/custom default", Router(default="multilingual").route("12345", Q_GENERIC)["model"],
      "multilingual")


# --------------------------------------------------------------------- unknown-Latin routing (#35)
_r_lat = Router()
for label, text in [
    ("romanian", "Gătește-mi o rețetă de sarmale de post pentru mâine."),
    ("romanian agent request", "Exportă APK-ul pentru Android și pune-l pe Drive ca să-l instalez."),
    ("polish", "Klient został obciążony dwukrotnie i chce zwrot pieniędzy za fakturę"),
    ("turkish", "Müşteriden iki kez ücret alındı ve para iadesi istiyor lütfen yardım"),
]:
    check("route/unknown latin " + label, _r_lat.route(text).model, "multilingual")
# and the reason must say what it actually routed on, not report a language it did not identify
check("route/undecided reason mentions letters",
      "not identified" in _r_lat.route("Müşteriden iki kez ücret alındı ve para iadesi istiyor").reason, True)
check("route/english still english",
      _r_lat.route("Please refund the duplicate charge on invoice 4411 today.").model, "english")
check("route/short english still english", _r_lat.route("refund me").model, "english")

# Undecided Latin text follows `default`, as a state with no letters already did. Short messages made
# only of content words carry nothing that names their language, and hard-coding English for them
# sent every short Portuguese message to the checkpoint that is 0.97 confident at 0.47 accuracy on
# `pt`, whatever the router was configured with.
_r_ml = Router(default="multilingual")
for text in ["Quero cancelar", "Esqueci minha senha", "Fui cobrado duas vezes",
             "Produto veio quebrado, quero trocar", "refund me"]:
    check("route/undecided follows default " + text[:24], _r_ml.route(text).model, "multilingual")
    check("route/undecided stock default " + text[:24], _r_lat.route(text).model, "english")
check("route/undecided reason names the default",
      "using default (multilingual)" in _r_ml.route("Esqueci minha senha").reason, True)
# identified English is not undecided, so a non-English default leaves it alone
for text in ["Please refund the duplicate charge", "Please refund the duplicate charge on invoice 4411 today."]:
    check("route/identified english ignores default " + text[:24], _r_ml.route(text).model, "english")
check("route/english reason unchanged",
      _r_lat.route("Please refund the duplicate charge on invoice 4411 today.").reason, "English Latin text")


# --------------------------------------------------------------------- mixed states
# A Portuguese ticket carrying an English stack trace, error payload or form template read as English
# as a whole -- the English part is longer -- and went to the checkpoint that cannot read the
# customer's own words. Any line or field that on its own is named a non-English language now wins.
_TRACE = ("O sistema caiu de novo hoje de manhã, segue o log:\n"
          "Traceback (most recent call last):\n"
          "  File \"/app/main.py\", line 42, in handler\n"
          "    return self.process(request)\n"
          "ConnectionError: the connection to the database was refused because the pool is "
          "exhausted and there is no available slot for this request")
for label, state, segment in [
    ("portuguese ticket + english traceback", _TRACE, "O sistema caiu de novo hoje de manhã, segue o log:"),
    ("portuguese field + english error payload",
     {"descricao": "O pagamento não foi processado",
      "error": {"code": "card_declined", "message": "Your card was declined. Please try again with a "
                "different card or contact your bank for more information."}},
     "O pagamento não foi processado"),
    ("english form template + portuguese body",
     {"subject": "New ticket from the web form", "body": "Quero cancelar meu plano"},
     "Quero cancelar meu plano"),
    ("parenthesis in prose is not code",
     {"subject": "Urgent: production is down for all customers since the last deploy and the "
                 "status page is red for the whole region",
      "body": "Deu erro (500) no login, alguém pode ver isso agora?"},
     "Deu erro (500) no login, alguém pode ver isso agora?"),
    # the rule runs both ways: an English ticket that pastes a foreign log goes to multilingual too
    ("english ticket + portuguese error log",
     "Our Brazilian branch cannot issue invoices since this morning. The system shows this message:\n"
     "ERRO: Não foi possível emitir a nota fiscal, o certificado digital está vencido\n"
     "Can you help us before the end of the day?",
     "ERRO: Não foi possível emitir a nota fiscal, o certificado digital está vencido"),
    ("english ticket + german error log",
     "The nightly sync to the Munich server keeps failing and we lose the whole batch.\n"
     "Fehler: Die Verbindung zum Server wurde unterbrochen, bitte versuchen Sie es spaeter noch einmal\n"
     "Please check the firewall rules on your side.",
     "Fehler: Die Verbindung zum Server wurde unterbrochen, bitte versuchen Sie es spaeter noch einmal"),
    ("english ticket + spanish error payload",
     {"subject": "Payment failed for a customer in Madrid",
      "description": "The customer tried three times with the same card and each attempt was declined by "
                     "the gateway, so we would like to know whether the problem is on our side or with the bank.",
      "error": {"code": "card_declined",
                "message": "La tarjeta fue rechazada por el banco emisor, contacte con su banco"}},
     "La tarjeta fue rechazada por el banco emisor, contacte con su banco"),
    # acronyms are dropped only from mixed-case text: a line written all in capitals keeps its words
    ("all-caps portuguese line",
     "This is the fourth email I have sent about the same order and nobody has answered any of them.\n"
     "The customer wrote this in the chat and then closed the window:\n"
     "QUERO MEU DINHEIRO DE VOLTA AGORA\n"
     "Could someone from the billing team look at order 5512 today?",
     "QUERO MEU DINHEIRO DE VOLTA AGORA"),
]:
    check("mixed/is not english: " + label, is_english(state), False)
    check("mixed/segment reported: " + label, analyse(state)["mixed_segment"], segment)
    check("mixed/routes multilingual: " + label, _r_lat.route(state).model, "multilingual")
check("mixed/reason names the segment",
      "a line or field reads as 'pt'" in _r_lat.route(_TRACE).reason, True)
# English stays English: several English lines, a short foreign sign-off, and code pasted into a request
# (`os.path` reads as Portuguese, `round(el, 2)` as Spanish, `np.mean(na)` as Portuguese)
for label, state in [
    ("multi-line english", "Hi team,\nThe export failed again last night.\nCan you check the logs?\nThanks"),
    ("short portuguese sign-off", "Please resend the invoice for March, the amount is wrong.\nAtenciosamente, Joao"),
    ("os.path", "The build broke after the refactor.\nREPO = os.path.dirname(os.path.dirname(__file__))\n"
                "Please take a look at the import paths when you can."),
    ("round(el)", "The latency script crashes on large runs.\nmix[key] = {\"total_s\": round(el, 2)}\n"
                  "Can you check why the stream is empty?"),
    ("np.mean(na)", "The summary is wrong for empty suites.\nif na: non[m] = round(float(np.mean(na)), 4)\n"
                    "Please guard the empty case."),
    ("english json", {"status": "open", "priority": "high",
                      "message": "The customer was charged twice and wants a refund"}),
    # a line carries far less text than a state, so its evidence must be two different words and no
    # acronyms or slash compounds. A ham-radio listing on 20 Newsgroups (misc.forsale/76512) went to
    # multilingual on `COM ... COM` alone; hockey picks on the team codes, OS/2 on `os` and `dos`.
    ("same word twice (Nav/Com, COM)",
     "I'm looking for good deals on the following (used or new):\nAviation Headsets (with mic).\n"
     "Handheld Nav/Com tranciever (may consider COM only).\nPortable GPS or Loran Navigator."),
    ("team codes", "Round two predictions for the pool, as promised.\nQUE  vs MON:  MON  in 7.\n"
                   "PIT  vs NYI:  PIT  in 5."),
    ("slash compound", "I need a converter for these image formats.\n"
                       "DOS, OS/2 or platform independent programs if possible.\nThanks in advance."),
    ("backslash path", "My modem stopped answering after the upgrade.\nC:\\DOS\\mode COM1:9600,n,8,1,p\n"
                       "Is that the right line for a 9600 baud connection?"),
]:
    check("mixed/english stays english: " + label, is_english(state), True)
    check("mixed/no segment: " + label, analyse(state)["mixed_segment"], None)
# a state is user input: one long line with no joiner took 43 s at 40,000 characters when compounds
# were stripped with an open-ended regex; the segment check now reads at most the 4,000-character cap
import time as _time
_t0 = _time.perf_counter()
analyse({"subject": "The export failed again last night for the whole region", "body": "a" * 200_000})
check("mixed/long single-line field stays fast", _time.perf_counter() - _t0 < 5.0, True)
check("mixed/segment check reads at most the cap",
      analyse({"log": "The export failed again last night for the whole region. " * 80,
               "body": "Quero cancelar meu plano agora mesmo"})["mixed_segment"], None)


# --------------------------------------------------------------------- plain-ASCII Romance (#172)
# A state that lost its accents carries no diacritic rate for the non-English signal to read, and
# mail clients and ticket systems strip them routinely, so the function-word lists are the only
# evidence left. Spanish and Italian complaints were reported `is_english=True` and handed to the
# English checkpoint -- the one that collapses off English -- while the accented spelling of the
# same text reached multilingual. The lists for fr/es/pt/it held mostly accented words (`está`,
# `más`, `è`, `être`, `não`) plus a handful of unaccented ones, so a stripped state matched one
# word or none and fell under the two-hit margin below.
for lang, text in [
    ("es", "El pedido llego roto y nadie responde cuando escribo al soporte"),
    ("es", "Quiero cancelar mi plan y pedir un reembolso"),
    ("es", "La factura tiene un error en el importe total"),
    ("es", "Necesito que me devuelvan el dinero de la compra duplicada"),
    ("it", "Il cliente e stato addebitato due volte e vuole un rimborso"),
    ("it", "Voglio cancellare il mio abbonamento e chiedere un rimborso"),
    ("it", "La fattura contiene un errore nell importo totale"),
    ("pt", "O cliente foi cobrado duas vezes e quer o dinheiro de volta"),
    ("fr", "Le client a ete facture deux fois et demande un remboursement"),
    ("fr", "Je ne peux pas acceder a mon compte et j ai besoin d aide"),
]:
    check("latin_lang/plain ascii " + lang + " " + text[:32], guess_latin_language(text), lang)
    check("is_english/plain ascii " + lang + " " + text[:32], is_english(text), False)
    check("route/plain ascii " + lang + " " + text[:32], _r_lat.route(text).model, "multilingual")
# the accented spellings must keep working: those route on the diacritic rate
for lang, text in [
    ("es", "La facturación tiene un error y necesito una corrección urgente"),
    ("it", "La fattura è sbagliata, devo avere un rimborso per il pagamento"),
    ("fr", "La commande est arrivée cassée et personne ne répond au support"),
]:
    check("route/accented " + lang, _r_lat.route(text).model, "multilingual")

# English must not move for this. The added words are ordinary English tokens as well -- `de facto`,
# `et al.`, `e.g.`, `la carte`, `UN`, `MI5`, `DOS` -- and a state carrying none of them is the case
# that has to keep routing to English.
for text in [
    "The customer was charged twice and wants a refund for this invoice",
    "Please cancel my subscription and refund the duplicate charge today",
    "The report by Smith et al. shows the de facto standard, e.g. the LA office and Rio",
    "Our MI5 and UN contacts discussed the DOS attack in LA last month",
    "No refund was issued, so I am writing to you again about invoice 4411",
    "no refund no reply",
    "The son of the director filed a complaint about the duplicate invoice",
]:
    check("is_english/romance control " + text[:32], is_english(text), True)
    check("route/romance control " + text[:32], _r_lat.route(text).model, "english")

# A word several lists claim (`la`, `e`, `o`) says "not English" without saying *which* language, so
# it may not name one on its own -- the same rule as the 0-0 tie above, which is why the sample
# below still names nothing even though `la` and `e` now count for Italian: it routes to multilingual
# on the Romanian diacritic, not on a guessed language.
check("latin_lang/shared words alone name nothing",
      analyse("Cât e ora acum la Tokyo")["language"], None)
check("route/shared words still multilingual",
      _r_lat.route("Cât e ora acum la Tokyo").model, "multilingual")
# ...but a distinctive word in the same state is enough to name the language it belongs to.
check("latin_lang/distinctive word names the language",
      guess_latin_language("La fattura contiene un errore nell importo totale"), "it")
check("latin_lang/shared hits still count toward a named language",
      guess_latin_language("La factura tiene un error en el importe total"), "es")


# --------------------------------------------------------------------- Brazilian support text
# Short Brazilian messages lean on `você`/`vc`, the unaccented `nao`/`voce` and `gostaria`, none of
# which the `pt` list held, so each matched one word, fell under the two-hit margin and went to the
# English checkpoint -- which on `pt` reports 0.97 mean confidence at 0.47 accuracy (ECE 0.51).
for text in [
    "Boa tarde, gostaria de cancelar o plano",
    "Voce pode me mandar a nota fiscal?",
    "Você pode me mandar a nota fiscal?",
    "Nao consigo fazer login no app",
    "Pix nao caiu na conta",
    "Gostaria de saber o prazo de entrega",
    "Estou esperando faz uma semana",
    "Vc pode cancelar pra mim?",
    # a bug report whose jargon is English keeps only these words to say it is Portuguese
    "Deu erro 500 no endpoint de login depois do update",
    "Depois da atualizacao ninguem consegue logar",
    "Antes funcionava, agora deu pau",
    "Estava tudo certo ate a migracao",
    "Entao o sistema travou de novo",
]:
    check("latin_lang/pt-br " + text[:32], guess_latin_language(text), "pt")
    check("route/pt-br " + text[:32], _r_lat.route(text).model, "multilingual")
# each added word that is also an English token must not move English text
for text in [
    "Our Sao Paulo office still has not received the invoice",
    "My VC asked for the cap table and the invoice",
    "Nossa Cafe charged my card twice this month",
    "The Boa Vista branch reported an outage this morning",
    "The pra team will review the claim tomorrow",
]:
    check("route/pt-br control " + text[:32], _r_lat.route(text).model, "english")


# --------------------------------------------------------------------- romanized Bangla
# Bangla is often typed in Latin letters ("Banglish") when no Bengali keyboard is at hand. It has no
# diacritics and matched no stopword list, so it was reported `is_english=True` and handed to the
# English checkpoint, which scores 0.08 on Bangla MASSIVE at 0.94 confidence. The Bengali-script
# spelling of the same text already routed on script alone.
for text in [
    "amar kach theke duibar taka kata hoyeche, doya kore ferot din",
    "ami invoice er jonno duibar charge peyechi, refund chai",
    "Ami ei product ta niye khub hotash, ekhon e cancel korte chai",
    "apnara keno amar call dhorchen na? ajke kichu ekta korun",
    "bhai amar account e login korte parchi na",
    "taka ekhono ferot paini, kobe pabo?",
    "order ta kobe asbe bolte parben?",
]:
    check("latin_lang/banglish " + text[:32], guess_latin_language(text), "bn")
    check("is_english/banglish " + text[:32], is_english(text), False)
    check("route/banglish " + text[:32], _r_lat.route(text).model, "multilingual")
check("route/bengali script", _r_lat.route("আমার কাছ থেকে দুইবার টাকা কাটা হয়েছে").model, "multilingual")

# English must not move. `chai`, `ar`, `ami`, `koto`, `kore`, `oi` are Bangla words that also turn
# up in English text as a drink, an acronym or a name; one of them next to English function words
# stays English.
for text in [
    "Chai latte order was charged twice, please refund the extra amount",
    "The AR team says the ETA for the fix is Friday",
    "Ami Patel from the Koto office sent the invoice to Kore Ltd",
    "Our AR and VR demo in Oi Bahia went well, the client wants a quote",
    "Take the age of the account into account before you refund",
]:
    check("is_english/banglish control " + text[:32], is_english(text), True)
    check("route/banglish control " + text[:32], _r_lat.route(text).model, "english")
# ...and no other language may move either: the `bn` list claims no word another list holds, and
# leaves out Romance words such as `ora`, `nei`, `vai`.
from laya.lang import _STOP  # noqa: E402
check("latin_lang/bn list shares no word with another list",
      sorted(w for w in _STOP.get("bn", ()) for lg, words in _STOP.items() if lg != "bn" and w in words), [])
check("latin_lang/romanian with ei stays romanian",
      guess_latin_language("Ei nu sunt de acord cu factura, vreau o corecție"), "ro")


# --------------------------------------------------------------------- plain-ASCII German (#54)
# No umlaut for the diacritic rate to catch, and `in`/`was` counted for English alone, so these were
# labelled English and handed to the checkpoint that cannot read them.
for text in ["trage diesen termin in meinen kalender ein",
             "wie lautet die temperatur in fulda in hessen",
             "schalte das licht im wohnzimmer aus",
             "was ist die aktuelle zeit"]:
    check("latin_lang/ascii german " + text, guess_latin_language(text), "de")
    check("route/ascii german " + text, _r_lat.route(text).model, "multilingual")
check("route/english control for #54",
      _r_lat.route("I would like to book a flight to Berlin tomorrow").model, "english")
# English that shares words with the German list stays English. `in` and `den` leave the first, a real
# en-US MASSIVE utterance, one German hit short of flipping; the chat line carries `im` and flips if
# any one of the English words `am`, `an` or `so` joins the German list.
for text in ["turn off smart lamp in den", "im so sorry, am an hour late, stuck in traffic"]:
    check("route/english sharing german words " + text, _r_lat.route(text).model, "english")
# German words that Spanish (`es`) or French (`du`) also claim would stop naming those languages
check("latin_lang/spanish es stays evidence", guess_latin_language("que hora es en australia"), "es")
check("latin_lang/french du stays evidence", guess_latin_language("baisse le volume du haut-parleur"), "fr")

# ------------------------------------------------------------------ accented loanwords in English (#337)
# The diacritic rate is measured over every character, so one `é` in a short English sentence
# clears the 0.02 floor and used to veto the English resolution outright: plain English with a
# loanword or foreign name (`café`, `résumé`, `José`, `Zürich`) went to the checkpoint the
# README says collapses on English-heavy Latin text. English wins the veto back only through the
# word rescue: at least two distinct function words no other list holds, and at most one word
# carrying a non-English letter.
for text in ["Please send me the café menu today please",
             "Could you email me your résumé before the meeting",
             "Send the invoice to José before Friday",
             "We visited Zürich last summer and loved it"]:
    check("latin_lang/loanword english stays english " + text, guess_latin_language(text), "en")
    check("route/loanword english stays english " + text, _r_lat.route(text).model, "english")
# Genuinely non-English accented text keeps its multilingual routing: a German sentence with
# umlauts and no English function word is not rescued.
check("latin_lang/accented german stays non-english",
      is_english("Grüße aus Köln, wir melden uns wegen der Rechnung"), False)
check("route/accented german stays multilingual",
      _r_lat.route("Grüße aus Köln, wir melden uns wegen der Rechnung").model, "multilingual")
# Danish still has no list here, and its accented sentences pick up just one or two English-shaped
# words (`i`, `at`, `for`, `have`), which is not the two-distinct-word English the rescue requires.
for text in ["sluk lyset i soveværelset",                          # da
             "kan jeg få en refundering for det dobbelte beløb",   # da
             "stäng av ljuset i sovrummet"]:                        # sv
    check("latin_lang/nordic accented stays non-english " + text, is_english(text), False)
    check("route/nordic accented stays multilingual " + text, _r_lat.route(text).model, "multilingual")

# Swedish-specific words now identify both normal text and ASCII-normalised support prose. The
# second sample is the real false-negative shape: `få` alone was too weak to stop its two English-
# shaped words (`i`, `kan`) from pulling the request onto the English checkpoint.
for text in ["Om ni inte kan få tillbaka de raderade filerna i dag avslutar jag mitt abonnemang.",
             "Om ni inte kan fa tillbaka de raderade filerna i dag avslutar jag mitt abonnemang.",
             "jag vill att ni hjalper mig med detta"]:
    check("latin_lang/swedish is named " + text, guess_latin_language(text), "sv")
    check("route/swedish uses multilingual " + text, _r_lat.route(text).model, "multilingual")
    check("route/swedish detection reports sv " + text, _r_lat.route(text)["detection"]["language"], "sv")
# Common Swedish support phrasing should identify the language across billing, account, technical
# and delivery messages, both with and without Swedish diacritics.
for text in ["Kan ni hjälpa mig?", "Min faktura är fel", "Jag behöver hjälp med betalningen",
             "Kan ni skicka kvittot igen?", "Jag vill byta lösenord",
             "Appen kraschar när jag öppnar inställningarna",
             "Var hittar jag inställningen för tvåfaktorsinloggning?",
             "Vi har debiterats två gånger för mars",
             "Var är mitt paket? Spårningen har inte uppdaterats"]:
    check("latin_lang/swedish support is named " + text, guess_latin_language(text), "sv")
    check("route/swedish support uses multilingual " + text,
          _r_lat.route(text).model, "multilingual")
for text in ["min faktura ar fel", "jag behover hjalp med betalningen",
             "kan ni skicka kvittot igen", "jag vill byta losenord",
             "appen kraschar nar jag oppnar installningarna",
             "var hittar jag installningen for tva faktorsinloggning",
             "vi har blivit debiterade tva ganger for mars",
             "var ar mitt paket sparningen har inte uppdaterats"]:
    check("latin_lang/ascii swedish support is named " + text, guess_latin_language(text), "sv")
    check("route/ascii swedish support uses multilingual " + text,
          _r_lat.route(text).model, "multilingual")
# Short login failures often contain English technical vocabulary and can arrive without Swedish
# diacritics. Swedish `kan` + `inte` must outweigh the incidental English token `in`.
for text in ["Kan inte logga in", "kan inte logga in", "Jag kan inte logga in", "Vi kan inte logga in"]:
    check("latin_lang/short swedish login is named " + text, guess_latin_language(text), "sv")
    check("route/short swedish login uses multilingual " + text,
          _r_lat.route(text).model, "multilingual")
    check("route/short swedish login detection reports sv " + text,
          _r_lat.route(text)["detection"]["language"], "sv")
# Two- and three-word Swedish support fragments do not reach the general four-word evidence
# threshold. Only distinctly Swedish terms should name them; generic words and Nordic controls
# remain undecided rather than being guessed as Swedish.
for text in ["Ingen åtkomst", "Ingen atkomst", "Fakturan är fel", "Betalningen nekades",
             "Behöver hjälp", "Behover hjalp", "Glömt lösenord", "Glomt losenord",
             "Felmeddelande igen", "Kvitto saknas", "Inloggningen fungerar"]:
    check("latin_lang/short Swedish support is named " + text, guess_latin_language(text), "sv")
    check("route/short Swedish support uses multilingual " + text,
          _r_lat.route(text).model, "multilingual")
    check("route/short Swedish support detection reports sv " + text,
          _r_lat.route(text)["detection"]["language"], "sv")
for text in ["Ingen adgang", "Fakturaen feil", "Glemt passord", "Pakken forsinket",
             "No account access", "Password forgotten"]:
    check("latin_lang/short non-Swedish stays undecided " + text,
          guess_latin_language(text), None)
check("latin_lang/english login stays english",
      guess_latin_language("I cannot login to my account"), "en")
check("route/english login stays english",
      _r_lat.route("I cannot login to my account").model, "english")
check("latin_lang/danish login is not called swedish",
      guess_latin_language("Jeg kan ikke logge inn"), None)
check("latin_lang/danish account login is not called swedish",
      guess_latin_language("Jeg kan ikke logge ind på min konto"), None)
# Two non-English-letter words is a running non-English vocabulary, not one loanword: the rescue
# does not fire even with English function words present.
check("latin_lang/two diacritic words are not one loanword",
      is_english("The naïve façade needs a fresh coat of paint"), False)


# --------------------------------------------------------------------- temperature clamp (#35)
# A fitted temperature below 1 sharpens logits. The shipped `choice:11+` bucket is 0.1006, which
# turned a 0.24 top probability into 0.99 confidence on 13-option skill routing.
check("clamp/pathological sharpening", clamp_temperature(0.1006), 0.5)
check("clamp/shipped choice:11+ is rejected", clamp_temperature(0.10058280825614929), TEMP_MIN)
check("clamp/legitimate value untouched", clamp_temperature(1.7601518630981445), 1.7601518630981445)
check("clamp/neutral untouched", clamp_temperature(1.0), 1.0)
check("clamp/upper bound", clamp_temperature(9.0), TEMP_MAX)
check("clamp/zero", clamp_temperature(0.0), TEMP_MIN)
check("clamp/negative", clamp_temperature(-3.0), TEMP_MIN)
check("clamp/none falls back to neutral", clamp_temperature(None), 1.0)
check("clamp/garbage falls back to neutral", clamp_temperature("x"), 1.0)
check("clamp/nan falls back to neutral", clamp_temperature(float("nan")), 1.0)
check("clamp/inf falls back to neutral", clamp_temperature(float("inf")), 1.0)
check("clamp/bools are not temperatures", clamp_temperature(True), 1.0)
check("clamp/False is not a sharpening zero", clamp_temperature(False), 1.0)
check("clamp/bounds are sane", TEMP_MIN <= 1.0 <= TEMP_MAX, True)
# 13 options is the bucket the reported skill-router landed in
check("clamp/13 options is the 11+ bucket", temp_bucket(QTYPES["choice"], 13), "choice:11+")


# --------------------------------------------------------------------- LRU bookkeeping
class _Stub:
    def __init__(self, name):
        self.name = name

    def system_one(self, state, questions):
        return {"model": self.name, "answers": {}, "usage": {}}


def stubbed_router(max_loaded):
    rr = Router(max_loaded=max_loaded)
    rr.load = lambda n, _r=rr: _load_stub(_r, n)
    return rr


def _load_stub(rr, name):
    key = normalise_name(name)
    if key in rr._agents:
        rr._touch(key)
        return rr._agents[key]
    rr._agents[key] = _Stub(key)
    rr._order.append(key)
    rr._evict()
    return rr._agents[key]


rr = stubbed_router(1)
rr.load("english"); rr.load("multilingual")
check("lru/cap 1 keeps newest", rr.loaded, ["multilingual"])
check("lru/cap 1 agents match order", sorted(rr._agents), ["multilingual"])

rr = stubbed_router(2)
rr.load("english"); rr.load("multilingual"); rr.load("typed-decisions")
check("lru/cap 2 evicts oldest", rr.loaded, ["multilingual", "typed-decisions"])

rr = stubbed_router(2)
rr.load("english"); rr.load("multilingual"); rr.load("english")   # touch english
rr.load("typed-decisions")
check("lru/touch protects", sorted(rr.loaded), ["english", "typed-decisions"])

rr.unload("english")
check("lru/unload one", "english" in rr.loaded, False)
rr.unload()
check("lru/unload all", rr.loaded, [])


# --------------------------------------------------------------------- default cap (#172)
# #172 measured a workload that alternates languages at 20-23 s per request on CPU (reloading a
# checkpoint every request) against 49-136 ms with both resident. Automatic routing only ever
# chooses between `english` and `multilingual`, so the default holds both, and a deployment that
# never alternates never builds the second.
def counting_router(cap=None):
    """Router whose loader records which checkpoints it had to build."""
    rr = Router() if cap is None else Router(max_loaded=cap)
    built = []

    def load(name, _rr=rr, _built=built):
        key = normalise_name(name)
        if key in _rr._agents:
            _rr._touch(key)
            return _rr._agents[key]
        _built.append(key)
        _rr._agents[key] = _Stub(key)
        _rr._order.append(key)
        _rr._evict()
        return _rr._agents[key]

    rr.load = load
    return rr, built


check("lru/default is two", Router().max_loaded, 2)
_en = {"body": "I was charged twice for invoice 4411, please refund."}
_ml = {"body": "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung"}
for cap, want_built in ((1, 20), (2, 2)):
    cr, built = counting_router(cap)
    for _ in range(10):                          # the reported alternating workload
        cr.predict(_en, Q_GENERIC)
        cr.predict(_ml, Q_GENERIC)
    check("lru/alternating traffic, cap=%d builds" % cap, len(built), want_built)
# and the default costs a single-language deployment nothing at all
cr, built = counting_router()
for _ in range(5):
    cr.predict(_en, Q_GENERIC)
check("lru/single-language traffic builds one checkpoint", built, ["english"])


# --------------------------------------------------------------------- bundle vs standalone
check("bundle/english is repo root", DEFAULT_MODELS["english"], (BUNDLE_REPO, None))
check("bundle/multilingual subfolder", DEFAULT_MODELS["multilingual"], (BUNDLE_REPO, "multilingual"))
check("bundle/typed subfolder", DEFAULT_MODELS["typed-decisions"], (BUNDLE_REPO, "typed-decisions"))
check("repo_str/root", _repo_str((BUNDLE_REPO, None)), "convaiinnovations/laya")
check("repo_str/sub", _repo_str((BUNDLE_REPO, "multilingual")), "convaiinnovations/laya/multilingual")
check("repo_str/plain string", _repo_str("some/repo"), "some/repo")

r_bundle = Router()
r_alone = Router(standalone_repos=True)
check("bundle/default router uses bundle",
      r_bundle.route({"m": "मुझसे दो बार"}, Q_GENERIC)["repo"], "convaiinnovations/laya/multilingual")
check("standalone/opt-in uses own repo",
      r_alone.route({"m": "मुझसे दो बार"}, Q_GENERIC)["repo"], "convaiinnovations/laya-multilingual")
check("standalone/english unchanged",
      r_alone.route({"m": "I was charged twice"}, Q_GENERIC)["repo"], "convaiinnovations/laya")
check("standalone map complete", sorted(STANDALONE_MODELS), sorted(DEFAULT_MODELS))
# a local-path override must still work (the Space and tests rely on it)
r_local = Router(models={"english": "/tmp/en", "multilingual": "/tmp/ml"})
check("override/local path kept", r_local.route({"m": "मुझसे दो बार"}, Q_GENERIC)["repo"], "/tmp/ml")


# --------------------------------------------------------------------- preload
# Exercise the real preload/load/LRU paths; only checkpoint construction is stubbed.
with patch("laya.agent.Agent", side_effect=lambda repo, **kw: _Stub(repo)) as build:
    rp = Router(preload=True)
    check("preload/all three stay resident", sorted(rp.loaded),
          ["english", "multilingual", "typed-decisions"])
    check("preload/max_loaded raised", rp.max_loaded, 3)
    check("preload/builds each model once", build.call_count, 3)
    check("preload/returns router", rp.preload() is rp, True)
    check("preload/repeated call reuses models", build.call_count, 3)

    empty = Router()
    empty.preload([])
    check("preload/empty selection leaves models unloaded", empty.loaded, [])
    check("preload/empty selection builds nothing", build.call_count, 3)

    rp2 = Router()
    rp2.preload(["english", "multilingual"])
    check("preload/subset stays resident", sorted(rp2.loaded), ["english", "multilingual"])
    check("preload/subset capacity", rp2.max_loaded, 2)
    rp2.load("english")
    check("preload/touch does not evict", sorted(rp2.loaded), ["english", "multilingual"])
    check("preload/touch does not rebuild", build.call_count, 5)

    incremental = Router()
    incremental.preload(["english"])
    english = incremental.load("english")
    incremental.preload(["multilingual"])
    check("preload/incremental keeps both models", incremental.loaded, ["english", "multilingual"])
    check("preload/incremental capacity", incremental.max_loaded, 2)
    check("preload/incremental reuses original", incremental.load("english") is english, True)
    check("preload/incremental avoids rebuilds", build.call_count, 7)

    incremental.preload(["en", "english", "multi", "ml"])
    check("preload/aliases do not inflate capacity", incremental.max_loaded, 2)
    check("preload/aliases reuse models", build.call_count, 7)
    incremental.preload(["english", "typed-decisions"])
    check("preload/overlap preserves unrequested models", sorted(incremental.loaded),
          ["english", "multilingual", "typed-decisions"])
    check("preload/overlap capacity", incremental.max_loaded, 3)
    for name in DEFAULT_MODELS:
        incremental.predict("hello", Q_GENERIC, model=name)
    check("preload/predictions never rebuild", build.call_count, 8)

    attached = Router()
    original = _Stub("already-built")
    attached.attach("english", original)
    attached.preload(["multilingual"])
    check("preload/keeps attached model", attached.load("english") is original, True)
    check("preload/attached and new stay resident", sorted(attached.loaded), ["english", "multilingual"])
    check("preload/attached capacity", attached.max_loaded, 2)
    check("preload/attached model not rebuilt", build.call_count, 9)

    roomy = Router(max_loaded=5)
    roomy.preload(["en", "english", "multi"])
    check("preload/larger capacity is preserved", roomy.max_loaded, 5)
    check("preload/duplicates build once", build.call_count, 11)


# --------------------------------------------------------------------- attach
ra = stubbed_router(1)
sentinel = _Stub("already-built")
ra.attach("english", sentinel)
check("attach/registers under the name", ra._agents["english"], sentinel)
check("attach/counts as resident", "english" in ra.loaded, True)
# `max_loaded` starts at `max(1, max_loaded)`, so one attach to a cap-1 router cannot
# move it and `ra.max_loaded >= 1` held before `attach` was ever called. The cap only
# rises on the attach that would not otherwise fit.
check("attach/first attach leaves the cap alone", ra.max_loaded, 1)
ra.attach("multilingual", _Stub("second"))
check("attach/raises max_loaded to hold the extra one", ra.max_loaded, 2)
ra.unload("multilingual")
# attaching then loading another must not evict the attached one
_load_stub(ra, "multilingual")
check("attach/survives a later load", sorted(ra.loaded), ["english", "multilingual"])
check("attach/still the same object", ra._agents["english"] is sentinel, True)
check("attach/accepts aliases", stubbed_router(1).attach("en", _Stub("x")) is not None, True)


# --------------------------------------------------------------------- thread safety (issue #95)

def _concurrent_load_dedup():
    """Concurrent load() of the same checkpoint must build one Agent, shared by all callers."""
    import laya.agent as _agent_mod
    constructions = []
    cl = threading.Lock()

    class _SlowAgent:
        def __init__(self, *args, **kwargs):
            _time.sleep(0.05)  # widen the check-then-build window
            with cl:
                constructions.append(1)

        def system_one(self, state, questions):
            return {"model": "fake", "answers": {}, "usage": {}}

    old = _agent_mod.Agent
    _agent_mod.Agent = _SlowAgent
    try:
        r = Router()
        got = []

        def _worker():
            got.append(r.load("english"))

        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return len({id(x) for x in got}), len(constructions), len(r._order), sorted(r._agents)
    finally:
        _agent_mod.Agent = old

unique, built, order_len, agents = _concurrent_load_dedup()
check("threads/8 concurrent loads share one Agent", unique, 1)
check("threads/Agent constructed exactly once", built, 1)
check("threads/LRU views stay consistent", (order_len == 1 and agents == ["english"]), True)


def _concurrent_hotpath():
    """Concurrent hot-path loads of an already-cached model must keep _order/_agents consistent."""
    import laya.agent as _agent_mod

    class _Agent:
        def __init__(self, *args, **kwargs):
            pass

        def system_one(self, state, questions):
            return {"model": "fake", "answers": {}, "usage": {}}

    old = _agent_mod.Agent
    _agent_mod.Agent = _Agent
    try:
        r = Router(max_loaded=3)
        r.load("english")  # warm the cache

        def _worker():
            r.load("english")

        threads = [threading.Thread(target=_worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return len(r._order), len(r._agents), r._order
    finally:
        _agent_mod.Agent = old

order_len, agents_len, order = _concurrent_hotpath()
check("threads/hot-path loads keep one entry", order_len, 1)
check("threads/hot-path loads keep agents consistent", agents_len, 1)
check("threads/hot-path order intact", order, ["english"])


def _cold_build_does_not_hold_lifecycle_lock():
    """A cold build must not stall `loaded` or a resident checkpoint for its whole duration.

    The build runs a download plus construction, seconds to minutes. Held under `_lock`, it made
    `GET /health` (which reads `loaded`) and every request for an already-resident checkpoint wait
    for it, so a liveness probe timed out during a lazy load.
    """
    import laya.agent as _agent_mod
    building, release = threading.Event(), threading.Event()
    constructions = []

    class _BlockingAgent:
        def __init__(self, *args, **kwargs):
            constructions.append(1)
            building.set()
            release.wait(5)

    old = _agent_mod.Agent
    _agent_mod.Agent = _BlockingAgent
    try:
        r = Router(max_loaded=3)
        resident = _Stub("english")
        r.attach("english", resident)
        loaders = [threading.Thread(target=r.load, args=("multilingual",)) for _ in range(2)]
        for t in loaders:
            t.start()
        building.wait(5)
        timings = {}
        for label, call in (("loaded", lambda: r.loaded),
                            ("resident load", lambda: r.load("english"))):
            done = threading.Event()
            threading.Thread(target=lambda c=call, d=done: (c(), d.set()), daemon=True).start()
            timings[label] = done.wait(1.0)
        release.set()
        for t in loaders:
            t.join(5)
        loaded_after, built = sorted(r.loaded), len(constructions)
        # An unload issued while a build is in flight waits for it, so the build cannot land after it.
        building.clear()
        release.clear()
        threading.Thread(target=r.load, args=("typed-decisions",), daemon=True).start()
        building.wait(5)
        unloader = threading.Thread(target=r.unload, args=("typed-decisions",))
        unloader.start()
        _time.sleep(0.1)
        release.set()
        unloader.join(5)
        timings["unload waits for the build"] = "typed-decisions" not in r.loaded
        return timings, built, loaded_after
    finally:
        release.set()
        _agent_mod.Agent = old

timings, built, resident_after = _cold_build_does_not_hold_lifecycle_lock()
check("threads/loaded answers during a cold build", timings["loaded"], True)
check("threads/resident checkpoint answers during a cold build", timings["resident load"], True)
check("threads/concurrent cold loads still build once", built, 1)
check("threads/cold build lands in the LRU", resident_after, ["english", "multilingual"])
check("threads/unload during a cold build frees it", timings["unload waits for the build"], True)


def _on_load_reenters_router(target, hooks_concurrent):
    """An `on_load` hook may call `router.load()` again without deadlocking.

    `_build_lock` is not re-entrant, so this holds only while `load()` dispatches `on_load`
    after releasing it. Returns whether the outer load finished, the agent the hook got back,
    the outer agent, and what ended up loaded.
    """
    import laya.agent as _agent_mod

    class _FastAgent:
        def __init__(self, *args, **kwargs):
            pass

    got = []

    class _ReentrantHook:
        def on_load(self, ctx):
            if ctx.model == "multilingual" and not got:
                got.append(ctx.router.load(target))

    old = _agent_mod.Agent
    _agent_mod.Agent = _FastAgent
    try:
        r = Router(max_loaded=3, hooks=[_ReentrantHook()], hooks_concurrent=hooks_concurrent)
        outer = []
        t = threading.Thread(target=lambda: outer.append(r.load("multilingual")), daemon=True)
        t.start()
        t.join(5)
        return not t.is_alive(), got[0] if got else None, outer[0] if outer else None, sorted(r.loaded)
    finally:
        _agent_mod.Agent = old

for concurrent in (True, False):
    tag = "concurrent" if concurrent else "serialised"
    finished, inner, outer, loaded = _on_load_reenters_router("multilingual", concurrent)
    check(f"threads/on_load reloading the same checkpoint does not deadlock ({tag})", finished, True)
    check(f"threads/on_load reload returns the resident agent ({tag})",
          inner is not None and inner is outer, True)
    finished, inner, outer, loaded = _on_load_reenters_router("typed-decisions", concurrent)
    check(f"threads/on_load loading another checkpoint does not deadlock ({tag})", finished, True)
    check(f"threads/on_load can load another checkpoint ({tag})", loaded, ["multilingual", "typed-decisions"])


def _test_per_checkpoint_unload_granularity():
    """unload("english") must not wait for an in-flight build of multilingual.

    unload("multilingual") must wait for its own in-flight build, and resident access
    (r.loaded and r.load("english")) must remain non-blocking while multilingual builds.
    """
    import laya.agent as _agent_mod
    multi_building = threading.Event()
    multi_release = threading.Event()

    class _ControllableAgent:
        def __init__(self, repo, *args, **kwargs):
            if "multilingual" in repo or kwargs.get("subfolder") == "multilingual":
                multi_building.set()
                multi_release.wait(10)

        def system_one(self, state, questions):
            return {"model": "fake", "answers": {}, "usage": {}}

    old = _agent_mod.Agent
    _agent_mod.Agent = _ControllableAgent
    try:
        r = Router(max_loaded=3)
        resident = _Stub("english")
        r.attach("english", resident)

        multi_loader = threading.Thread(target=r.load, args=("multilingual",), daemon=True)
        multi_loader.start()
        multi_building.wait(5)

        # 1. unload("english") must complete immediately without waiting for multilingual build
        english_unload_done = threading.Event()
        threading.Thread(target=lambda: (r.unload("english"), english_unload_done.set()), daemon=True).start()
        unloaded_english_fast = english_unload_done.wait(1.0)
        english_resident = "english" in r.loaded

        # 7. loaded and resident access remain non-blocking
        r.attach("english", resident)
        loaded_done = threading.Event()
        threading.Thread(target=lambda: (r.loaded, loaded_done.set()), daemon=True).start()
        loaded_fast = loaded_done.wait(1.0)

        resident_load_done = threading.Event()
        threading.Thread(target=lambda: (r.load("english"), resident_load_done.set()), daemon=True).start()
        resident_load_fast = resident_load_done.wait(1.0)

        # 2. unload("multilingual") waits for multilingual build
        multi_unload_done = threading.Event()
        multi_unloader = threading.Thread(
            target=lambda: (r.unload("multilingual"), multi_unload_done.set()), daemon=True
        )
        multi_unloader.start()
        unloaded_multi_early = multi_unload_done.wait(0.1)

        multi_release.set()
        multi_loader.join(5)
        multi_unloader.join(5)

        unloaded_multi_finished = multi_unload_done.wait(1.0)
        multi_resident_after = "multilingual" in r.loaded

        return (
            unloaded_english_fast,
            english_resident,
            loaded_fast,
            resident_load_fast,
            unloaded_multi_early,
            unloaded_multi_finished,
            multi_resident_after,
        )
    finally:
        multi_release.set()
        _agent_mod.Agent = old

(
    unloaded_en_fast,
    en_resident,
    loaded_fast,
    resident_load_fast,
    unloaded_multi_early,
    unloaded_multi_finished,
    multi_resident_after,
) = _test_per_checkpoint_unload_granularity()
check("threads/unload english does not wait for multilingual build", unloaded_en_fast, True)
check("threads/english is evicted after unload", en_resident, False)
check("threads/loaded property is non-blocking during build", loaded_fast, True)
check("threads/resident load is non-blocking during build", resident_load_fast, True)
check("threads/unload multilingual waits for in-flight build", unloaded_multi_early, False)
check("threads/unload multilingual completes after build", unloaded_multi_finished, True)
check("threads/multilingual not resident after unload", multi_resident_after, False)


def _test_concurrent_load_same_checkpoint_dedup():
    """Concurrent loads for the same checkpoint must construct exactly one Agent."""
    import laya.agent as _agent_mod
    build_started = threading.Event()
    build_release = threading.Event()
    constructions = []

    class _SlowAgent:
        def __init__(self, *args, **kwargs):
            constructions.append(1)
            build_started.set()
            build_release.wait(5)

    old = _agent_mod.Agent
    _agent_mod.Agent = _SlowAgent
    try:
        r = Router(max_loaded=3)
        got = []
        threads = [
            threading.Thread(target=lambda: got.append(r.load("multilingual")), daemon=True)
            for _ in range(5)
        ]
        for t in threads:
            t.start()

        build_started.wait(5)
        build_release.set()
        for t in threads:
            t.join(5)

        return len(constructions), len({id(a) for a in got}), len(got)
    finally:
        build_release.set()
        _agent_mod.Agent = old

built_count, unique_agents, total_callers = _test_concurrent_load_same_checkpoint_dedup()
check("threads/concurrent loads construct exactly one Agent", built_count, 1)
check("threads/concurrent callers receive identical Agent instance", unique_agents, 1)
check("threads/all concurrent callers complete", total_callers, 5)


def _test_attach_during_build_wins():
    """attach() called while a build is in flight must win without being overwritten."""
    import laya.agent as _agent_mod
    build_started = threading.Event()
    build_release = threading.Event()

    class _ControllableAgent:
        def __init__(self, *args, **kwargs):
            build_started.set()
            build_release.wait(5)

    old = _agent_mod.Agent
    _agent_mod.Agent = _ControllableAgent
    try:
        r = Router(max_loaded=3)
        attached_agent = _Stub("english")
        loader_result = []

        loader = threading.Thread(target=lambda: loader_result.append(r.load("english")), daemon=True)
        loader.start()

        build_started.wait(5)
        r.attach("english", attached_agent)

        build_release.set()
        loader.join(5)

        subsequent = r.load("english")
        return (
            loader_result[0] is attached_agent,
            subsequent is attached_agent,
            r._agents.get("english") is attached_agent,
        )
    finally:
        build_release.set()
        _agent_mod.Agent = old

loader_won, subsequent_won, resident_won = _test_attach_during_build_wins()
check("threads/attach during build wins for in-flight loader", loader_won, True)
check("threads/attach during build wins for subsequent load", subsequent_won, True)
check("threads/attach during build remains resident", resident_won, True)


def _test_failed_build_cleans_inflight_and_allows_retry():
    """Failed builds clean up in-flight state, propagate exceptions, and allow retry."""
    import laya.agent as _agent_mod
    fail_first = [True]

    class _FailingAgent:
        def __init__(self, *args, **kwargs):
            if fail_first[0]:
                raise ValueError("corrupted weights download")
            self.model = "ok"

    old = _agent_mod.Agent
    _agent_mod.Agent = _FailingAgent
    try:
        r = Router(max_loaded=3)
        errors = []

        def _loader(err_list):
            try:
                r.load("english")
            except ValueError as e:
                err_list.append(str(e))

        t1 = threading.Thread(target=_loader, args=(errors,), daemon=True)
        t2 = threading.Thread(target=_loader, args=(errors,), daemon=True)
        t1.start()
        t2.start()
        t1.join(5)
        t2.join(5)

        inflight_cleared = "english" not in r._loading
        not_resident = "english" not in r._agents

        fail_first[0] = False
        retried_agent = r.load("english")
        retried_resident = "english" in r.loaded

        return (
            len(errors),
            errors[0] if errors else None,
            inflight_cleared,
            not_resident,
            retried_agent.model,
            retried_resident,
        )
    finally:
        _agent_mod.Agent = old

err_count, first_err, inflight_cleared, not_res, retried_model, retried_res = (
    _test_failed_build_cleans_inflight_and_allows_retry()
)
check("threads/failed build raises to concurrent loaders", err_count, 2)
check("threads/failed build error message preserved", first_err, "corrupted weights download")
check("threads/failed build cleans _loading registry", inflight_cleared, True)
check("threads/failed build does not leave model resident", not_res, True)
check("threads/subsequent load retries and succeeds", retried_model, "ok")
check("threads/retried load lands in resident set", retried_res, True)


def _test_waiting_callers_no_deadlock():
    """Concurrent mix of loads and unloads on same and different models must not deadlock."""
    import laya.agent as _agent_mod

    class _FastAgent:
        def __init__(self, *args, **kwargs):
            _time.sleep(0.005)

    old = _agent_mod.Agent
    _agent_mod.Agent = _FastAgent
    try:
        r = Router(max_loaded=3)
        models = ["english", "multilingual", "typed-decisions"]
        threads = []
        for _ in range(6):
            for m in models:
                threads.append(threading.Thread(target=r.load, args=(m,)))
                threads.append(threading.Thread(target=r.unload, args=(m,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        all_finished = not any(t.is_alive() for t in threads)
        loading_clean = len(r._loading) == 0
        return all_finished, loading_clean
    finally:
        _agent_mod.Agent = old

no_hang, loading_empty = _test_waiting_callers_no_deadlock()
check("threads/concurrent load and unload callers do not deadlock", no_hang, True)
check("threads/all in-flight markers cleared after completion", loading_empty, True)


def _test_unload_reload_sequence_no_resurrection():
    """An older in-flight build cannot resurrect the model after an unload/reload sequence."""
    import laya.agent as _agent_mod
    first_build_started = threading.Event()
    first_build_release = threading.Event()
    agent_instances = []

    class _SequencedAgent:
        def __init__(self, *args, **kwargs):
            agent_instances.append(self)
            if len(agent_instances) == 1:
                first_build_started.set()
                first_build_release.wait(5)

    old = _agent_mod.Agent
    _agent_mod.Agent = _SequencedAgent
    try:
        r = Router(max_loaded=3)
        t_load = threading.Thread(target=r.load, args=("english",), daemon=True)
        t_load.start()
        first_build_started.wait(5)

        unload_done = threading.Event()
        t_unload = threading.Thread(target=lambda: (r.unload("english"), unload_done.set()), daemon=True)
        t_unload.start()

        first_build_release.set()
        t_load.join(5)
        t_unload.join(5)

        unloaded_ok = unload_done.is_set()
        not_resident_after_unload = "english" not in r.loaded

        c_agent = r.load("english")
        is_new_agent = c_agent is agent_instances[1] and c_agent is not agent_instances[0]
        resident_final = "english" in r.loaded

        return unloaded_ok, not_resident_after_unload, is_new_agent, resident_final
    finally:
        first_build_release.set()
        _agent_mod.Agent = old

unloaded_ok, not_res_after, is_new_agent, res_final = _test_unload_reload_sequence_no_resurrection()
check("threads/unload in-flight build finishes successfully", unloaded_ok, True)
check("threads/model not resident after unload completes", not_res_after, True)
check("threads/reload builds new instance without resurrection", is_new_agent, True)
check("threads/reload lands in resident set", res_final, True)



# --------------------------------------------------------------------- unlisted scripts
# `detect_script` counts an alphabetic character only when one of `_SCRIPT_RANGES` claims
# it. Those ranges cover the scripts the checkpoints were measured on, and most of Unicode
# is outside them -- 68% of alphabetic codepoints, including the CJK extensions, the kana
# supplements, bopomofo, halfwidth katakana and dozens of smaller scripts. An unclaimed
# character used to be counted nowhere, so text written only in such a script produced a
# total of 0, was reported as "unknown", and `analyse` treats "unknown" as English. It was
# therefore routed to the English checkpoint, which has no tokens for it at all, with the
# reason "no letters detected in state" -- for text that plainly has letters.
#
# Every case below reported unknown / is_english=True / model "english" before this change.
for label, text in (
    ("halfwidth katakana", "ｱﾘｶﾞﾄｳ"),
    ("bopomofo", "ㄆㄇㄈㄉ"),
    ("kana supplement", "\U0001B000\U0001B001"),
    ("CJK Ext-B", "\U00020000\U00020001"),
    ("hangul jamo ext-A", "\ua960\ua961"),
    ("Cherokee", "ᏣᎳᎩ"),
    ("Mongolian", "ᠮᠣᠩᠭᠣᠯ"),
    ("Syriac", "ܫܠܡܐ"),
    ("Thaana", "ދިވެހި"),
    ("Tifinagh", "ⵜⴰⵎⴰⵣⵉⵖⵜ"),
    ("Yi", "ꆈꌠ"),
):
    check("unlisted/" + label + " is not called English", analyse(text)["is_english"], False)
    check("unlisted/" + label + " routes to multilingual",
          Router().route(text)["model"], "multilingual")

# Fullwidth Latin is Latin, not an unlisted script.
check("unlisted/fullwidth latin is latin", detect_script("ＨＥＬＬＯ"), "latin")

# A state with no letters at all must keep behaving exactly as before.
for label, text in (("empty", ""), ("digits only", "12345 67890"), ("emoji only", "😀😀😀")):
    check("unlisted/letterless " + label + " is still unknown",
          analyse(text)["script"], "unknown")
    check("unlisted/letterless " + label + " keeps the default",
          Router().route(text)["model"], "english")

# The scripts the table does name must be untouched.
for label, text, script in (
    ("english", "please cancel my subscription", "latin"),
    ("german", "Mein Konto wurde zweimal belastet, bitte erstatten Sie den Betrag", "latin"),
    ("hindi", "यह एक हिंदी वाक्य है", "devanagari"),
    ("chinese", "请取消我的订阅", "han"),
    ("japanese", "ありがとう", "kana"),
    ("korean", "감사합니다", "hangul"),
    ("russian", "Мой аккаунт был списан дважды", "cyrillic"),
    ("arabic", "تم خصم حسابي مرتين", "arabic"),
    ("greek", "Η χρέωση έγινε δύο φορές", "greek"),
    ("armenian", "Իմ հաշիվը գանձվել է երկու անգամ", "armenian"),
):
    check("unlisted/regression " + label + " script", detect_script(text), script)

# ------------------------------------------------- lang codes that name no language (#359)
# `C`, `POSIX` and `C.UTF-8` are valid `$LANG` values that identify nothing, and `C.UTF-8` is
# the default in the official Python image -- which is where `laya-serve` runs. Passing one to
# `Router(lang=...)` used to resolve to "not English" and pin every request to the multilingual
# checkpoint, before detection ever ran. The blank-string case below already abstained; these
# codes are the same kind of non-answer, so they abstain too. The ISO 639-2 special codes say
# the same thing in the standard's vocabulary.
_ENGLISH_STATE = "Please refund the duplicate charge on invoice 4411"
for code in ("C", "POSIX", "C.UTF-8", "c.utf8", "c", "posix",
             "und", "zxx", "mul", "UND", "Zxx", " und "):
    check("lang-code/%r abstains at the code level" % code, _english_from_code(code), None)
    decision = Router().route(_ENGLISH_STATE, lang=code)
    check("lang-code/%r lets detection name the checkpoint" % code, decision.model, "english")
    check("lang-code/%r says the hint was not used" % code,
          "explicit" in str(decision.reason), False)

# An empty hint is the case this mirrors, so it must still behave the same way.
check("lang-code/empty string still abstains", _english_from_code(""), None)
check("lang-code/None still abstains", _english_from_code(None), None)
check("lang-code/whitespace still abstains", _english_from_code("   "), None)

# The change must not touch codes that do name a language: English still routes now, and a
# non-English code still forces the multilingual checkpoint rather than being second-guessed.
for code in ("en", "eng", "english", "EN", "en-US", "en_US.UTF-8"):
    check("lang-code/%r is still decisive English" % code, _english_from_code(code), True)
    check("lang-code/%r routes without detection" % code,
          Router().route(_ENGLISH_STATE, lang=code).reason.count("explicit"), 1)
for code in ("de", "fr", "zh", "ja", "pt-BR", "de_DE.UTF-8"):
    check("lang-code/%r is still decisive non-English" % code, _english_from_code(code), False)
    check("lang-code/%r routes to the multilingual checkpoint" % code,
          Router().route(_ENGLISH_STATE, lang=code).model, "multilingual")

# A subtag after an agnostic primary is not itself agnostic: `C` names nothing, but the primary
# subtag is what is compared, so a hypothetical `C-something` also abstains and is not treated
# as a language by accident.
check("lang-code/agnostic primary wins over its subtag", _english_from_code("C.UTF-8"), None)
check("lang-code/posix with a modifier abstains", _english_from_code("POSIX-1"), None)


# --------------------------------------------------------------------- registered checkpoints
# A Router serves the checkpoints a caller registers beside the built-in three: named in `model=`
# like them, loaded, evicted and unloaded like them, and selected without `model=` by the set of
# question ids a request carries, the way the typed-decisions workflows have always been.
from laya.router import Signature, canonical_name, instructions_digest  # noqa: E402

check("registry/canonical lowercases and aliases", canonical_name(" ML "), "multilingual")
check("registry/canonical leaves an unknown name as typed", canonical_name("Papers"), "papers")

_SECTION_Q = {"section": {"type": "choice", "instructions": "Which section?", "criteria": {"a": "A", "b": "B"}}}
_TWO_Q = {"section": _SECTION_Q["section"], "subsection": _SECTION_Q["section"]}
_CLAIM = "Transparency improved operator performance in 11 of 17 studies."
r = Router(models={"papers": "/tmp/laya-papers", "tone": ("acme/laya-tone", None)},
           signatures={"papers": [["section"], ["subsection", "section"]]})
check("registry/registered names resolve", r.resolve("papers"), "papers")
check("registry/registered names resolve case-insensitively", r.resolve(" PAPERS "), "papers")
check("registry/built-ins still resolve", r.resolve("ml"), "multilingual")
check("registry/models holds built-ins plus registered", sorted(r.models),
      ["english", "multilingual", "papers", "tone", "typed-decisions"])
check("registry/signatures are Signature objects per checkpoint", r.signatures,
      {"papers": [Signature(["section"]), Signature(["section", "subsection"])]})
try:
    r.resolve("nope")
    check("registry/unknown name raises", False, True)
except ValueError as e:
    check("registry/unknown name lists the registered ones too", "'papers'" in str(e), True)

d = r.route(_CLAIM, _SECTION_Q)
check("signature/one id set routes to the registered checkpoint", d.model, "papers")
check("signature/reason names the checkpoint", "signature registered for 'papers'" in d.reason, True)
check("signature/workflow label is the checkpoint name", d["workflow"], "papers")
check("signature/no detection ran", d["detection"], None)
check("signature/repo is the registered source", d["repo"], "/tmp/laya-papers")
check("signature/a second id set routes to the same checkpoint", r.route(_CLAIM, _TWO_Q).model, "papers")
check("signature/a superset of ids is not a match",
      r.route(_CLAIM, {**_SECTION_Q, "extra": _SECTION_Q["section"]}).model, "english")
check("signature/a subset of ids is not a match",
      r.route(_CLAIM, {"subsection": _SECTION_Q["section"]}).model, "english")
check("signature/explicit model= outranks it", r.route(_CLAIM, _SECTION_Q, model="tone").model, "tone")
check("signature/explicit task= outranks it", r.route(_CLAIM, _SECTION_Q, task="tone").model, "tone")
check("signature/registered source shows as repo", r.route(_CLAIM, _SECTION_Q, model="tone")["repo"], "acme/laya-tone")
check("signature/registering is the opt-in: auto_task_detection stays off", r.auto_task_detection, False)
check("signature/route_batch keeps the mix in order",
      [x.model for x in r.route_batch([{"state": _CLAIM, "questions": _SECTION_Q},
                                       {"state": _CLAIM, "questions": {"q": _SECTION_Q["section"]}},
                                       {"state": _CLAIM, "questions": _TWO_Q, "model": "english"}])],
      ["papers", "english", "english"])

# The built-in typed-decisions workflows keep their opt-in, and a registered signature that
# happens to equal one of them wins, because registering it is the more specific instruction.
_CS = {q: _SECTION_Q["section"] for q in ("action", "category", "churn_risk", "needs_human", "urgency")}
check("signature/built-in workflow still needs auto_task_detection", r.route(_CLAIM, _CS).model, "english")
check("signature/built-in workflow with auto_task_detection",
      Router(auto_task_detection=True).route(_CLAIM, _CS)["workflow"], "customer_service")
r_cs = Router(models={"papers": "/tmp/laya-papers"}, signatures={"papers": [list(_CS)]}, auto_task_detection=True)
check("signature/registered set equal to a built-in one wins", r_cs.route(_CLAIM, _CS).model, "papers")
check("signature/match_signature reports the pair", r_cs.match_signature(_CS), ("papers", "papers"))
check("signature/match_signature on no questions", r_cs.match_signature({}), None)
check("signature/match_signature on a built-in set", Router(auto_task_detection=True).match_signature(_CS),
      ("typed-decisions", "customer_service"))
check("signature/match_signature takes the workflow already computed",
      Router(auto_task_detection=True).match_signature(_CS, "customer_service"), ("typed-decisions", "customer_service"))

# R1-8: the workflow sets are walked once per route, whether or not the questions match one.
import laya.router as _router_mod
for _label, _qs in (("matching", _CS), ("non-matching", {"x": {}}), ("none", None)):
    _calls = []
    _real_match = _router_mod.match_typed_decisions_workflow
    with patch.object(_router_mod, "match_typed_decisions_workflow",
                      lambda q, _f=_real_match: _calls.append(1) or _f(q)):
        Router(preload=False).route(_CLAIM, _qs)
    check("signature/route walks the workflow sets once for %s questions" % _label, len(_calls), 1)

# R1-14: a checkpoint registered after construction is not read as an artifact map of a nested env.
_old_env = os.environ.get("LAYA_SHA256_DIGESTS")
os.environ["LAYA_SHA256_DIGESTS"] = '{"english": {"model.safetensors": "%s"}}' % ("c" * 64)
try:
    _r_late = Router()
    _r_late.register("mine", "/tmp/laya-mine")
    check("register/nested LAYA_SHA256_DIGESTS gives a later checkpoint the empty placeholder",
          _r_late.sha256_digests["mine"], {})
finally:
    if _old_env is None:
        os.environ.pop("LAYA_SHA256_DIGESTS", None)
    else:
        os.environ["LAYA_SHA256_DIGESTS"] = _old_env

# R1-15: a source starting with ~ is a local path.
check("register/~ in a source is expanded", Router(models={"mine": "~/laya-mine"}).models["mine"],
      os.path.expanduser("~/laya-mine"))

# R1-16: `auto` is the routing word everywhere, so it cannot name a checkpoint.
try:
    Router().register("auto", "/tmp/laya-auto")
    check("register/auto is refused", True, False)
except ValueError as _e:
    check("register/auto is refused", "auto" in str(_e), True)

# R1-5: routing while another thread registers signatures must not see a dict change size.
import threading as _threading
_r_race = Router(models={"papers": "/tmp/laya-papers"}, signatures={"papers": [["a"]]})
_race_errors = []
_stop = _threading.Event()
def _route_loop():
    try:
        while not _stop.is_set():
            _r_race.match_signature({"zz": 1})
    except Exception as _e:  # noqa: BLE001
        _race_errors.append(_e)
_t = _threading.Thread(target=_route_loop)
_t.start()
for _i in range(300):
    _r_race.register("m%d" % _i, "/tmp/m%d" % _i, signatures=[["q%d" % _i]])
_stop.set()
_t.join()
check("signature/match_signature survives a concurrent register", _race_errors, [])

# R1-6: an attach()ed agent with no source is not part of a whole-router preload.
with patch("laya.agent.Agent", side_effect=lambda repo, **kw: _Stub(repo)) as _build_att:
    _r_att = Router()
    _r_att.attach("mine", _Stub("mine"))
    _r_att.unload("mine")
    _r_att.preload()
    check("preload/skips a name with no source", sorted(_r_att.loaded),
          ["english", "multilingual", "typed-decisions"])

# register(): the same thing after construction.
r2 = Router()
check("register/returns the canonical name", r2.register("Tone", "acme/laya-tone", signatures=[["mood"]]), "tone")
check("register/routes by its signature", r2.route(_CLAIM, {"mood": _SECTION_Q["section"]}).model, "tone")
check("register/a second call replaces the source", (r2.register("tone", "/tmp/tone-v2"), r2.models["tone"])[1], "/tmp/tone-v2")
check("register/replacing the source keeps the signatures", r2.signatures["tone"], [Signature(["mood"])])
check("register/the same signature twice is one entry",
      (r2.register("tone", "/tmp/tone-v2", signatures=[["mood"]]), r2.signatures["tone"])[1], [Signature(["mood"])])
check("register/a built-in name re-points that checkpoint",
      Router(models={"english": "/tmp/my-english"}).models["english"], "/tmp/my-english")
check("register/an alias re-points its built-in", Router(models={"en": "/tmp/mine"}).models["english"], "/tmp/mine")
for bad, why in (("Bad/Name", "slash"), ("", "empty"), ("-dash", "leading dash"), ("a b", "space")):
    try:
        Router().register(bad, "/tmp/x")
        check("register/refuses %s" % why, False, True)
    except ValueError:
        check("register/refuses %s" % why, True, True)
try:
    Router(models={"papers": "/x", "tone": "/y"}, signatures={"papers": [["q"]], "tone": [["q"]]})
    check("register/one id set cannot select two checkpoints", False, True)
except ValueError as e:
    check("register/one id set cannot select two checkpoints", "'papers'" in str(e), True)
try:
    Router(models={"papers": "/x"}, signatures={"papers": [[]]})
    check("register/an empty signature is refused", False, True)
except ValueError:
    check("register/an empty signature is refused", True, True)
try:
    Router(signatures={"papers": [["q"]]})
    check("register/signatures for an unregistered name are refused", False, True)
except ValueError:
    check("register/signatures for an unregistered name are refused", True, True)
try:
    Router(models={"papers": "/x"}, signatures={"papers": "section"})
    check("register/a bare string is not a list of sets", False, True)
except TypeError:
    check("register/a bare string is not a list of sets", True, True)
try:
    Router(models={"papers": 42})
    check("register/a source must be a path, repo or pair", False, True)
except TypeError:
    check("register/a source must be a path, repo or pair", True, True)

# Every name-taking option accepts a registered name, the way it accepts a built-in one.
r3 = Router(models={"papers": "/tmp/laya-papers"}, revisions={"papers": "abc123"}, default="papers",
            sha256_digests={"papers": {"model.safetensors": "a" * 64}})
check("register/default may be a registered name", r3.default, "papers")
check("register/revisions keyed by a registered name", r3.revisions, {"papers": "abc123"})
check("register/sha256_digests keyed by a registered name", r3.sha256_digests["papers"], {"model.safetensors": "a" * 64})
check("register/undecided text falls back to the registered default",
      r3.route("Quero cancelar", {"q": _SECTION_Q["section"]}).model, "papers")
_old_env = os.environ.get("LAYA_SHA256_DIGESTS")
os.environ["LAYA_SHA256_DIGESTS"] = '{"papers": {"model.safetensors": "%s"}}' % ("b" * 64)
try:
    check("register/LAYA_SHA256_DIGESTS may name a registered checkpoint",
          Router(models={"papers": "/tmp/laya-papers"}).sha256_digests["papers"], {"model.safetensors": "b" * 64})
finally:
    if _old_env is None:
        os.environ.pop("LAYA_SHA256_DIGESTS", None)
    else:
        os.environ["LAYA_SHA256_DIGESTS"] = _old_env

# attach() under a new name registers it with no source: resident now, not reloadable later.
r4 = Router()
r4.attach("adhoc", _Stub("adhoc"))
check("attach/new name becomes resident", "adhoc" in r4.loaded, True)
check("attach/new name is routable", r4.route(_CLAIM, _SECTION_Q, model="adhoc").model, "adhoc")
check("attach/new name has no source", r4.models["adhoc"], None)
r4.unload("adhoc")
try:
    r4.load("adhoc")
    check("attach/reload without a source is refused", False, True)
except ValueError as e:
    check("attach/reload without a source is refused", "no source" in str(e), True)

# load() hands a registered local path or repo to Agent as a source, not as a registry name.
import laya.agent as _agent_for_registry
_built = []


class _RecordingAgent:
    def __init__(self, repo, **kwargs):
        _built.append((repo, kwargs.get("subfolder")))


_prev_agent = _agent_for_registry.Agent
_agent_for_registry.Agent = _RecordingAgent
try:
    r5 = Router(models={"papers": "/tmp/laya-papers", "tone": ("acme/laya-tone", "v2")}, max_loaded=3)
    r5.load("papers")
    r5.load("tone")
    check("load/a registered path reaches Agent unchanged", _built[0], ("/tmp/laya-papers", None))
    check("load/a registered (repo, subfolder) pair reaches Agent", _built[1], ("acme/laya-tone", "v2"))
    check("load/registered checkpoints are resident like built-ins", r5.loaded, ["papers", "tone"])
    r5.preload(["papers", "english"])
    check("load/preload accepts registered names", "english" in r5.loaded and "papers" in r5.loaded, True)
    r5.unload("papers")
    check("load/unload accepts registered names", "papers" in r5.loaded, False)
finally:
    _agent_for_registry.Agent = _prev_agent


# --------------------------------------------------------------------- pinned signatures
# A signature may pin a question's instructions as well as its id, so two workflows that both ask
# a `section` question are told apart by what they ask. The model never sees the id, only the
# instructions and options, which is why the instructions are the stable part of a question kind.
_ASK = "Which section of the brief does this claim belong to?"
_ASK_REFLOWED = "  Which section   of the brief\ndoes this claim belong to?  "
_OTHER_ASK = "Which rule of the style guide does this post break?"


def _q(instructions, qid="section"):
    return {qid: {"type": "choice", "instructions": instructions, "criteria": {"a": "A", "b": "B"}}}


check("digest/prefix", instructions_digest(_ASK)[:5], "sha1:")
check("digest/whitespace is collapsed", instructions_digest(_ASK_REFLOWED), instructions_digest(_ASK))
check("digest/different text, different digest", instructions_digest(_OTHER_ASK) == instructions_digest(_ASK), False)
check("digest/a digest passes through", instructions_digest(instructions_digest(_ASK)), instructions_digest(_ASK))
check("digest/structured instructions are canonical JSON",
      instructions_digest({"question": "q", "focus": "f"}), instructions_digest({"focus": "f", "question": "q"}))
check("digest/None has no digest", instructions_digest(None), None)

rp = Router(models={"research": "/tmp/r", "linkedin": "/tmp/l", "generic": "/tmp/g"},
            signatures={"research": [{"section": _ASK}],
                        "linkedin": [{"section": instructions_digest(_OTHER_ASK)}],
                        "generic": [["section"]]})
check("pinned/text pin routes the matching question", rp.route("claim", _q(_ASK)).model, "research")
check("pinned/reflowed instructions still match", rp.route("claim", _q(_ASK_REFLOWED)).model, "research")
check("pinned/a digest pin routes too", rp.route("post", _q(_OTHER_ASK)).model, "linkedin")
check("pinned/same id, unknown instructions falls to the id-only signature",
      rp.route("x", _q("Something else entirely?")).model, "generic")
check("pinned/same id, no instructions falls to the id-only signature",
      rp.route("x", {"section": {"type": "choice", "criteria": {"a": "A"}}}).model, "generic")
check("pinned/a different id set matches nothing", rp.route("x", _q(_ASK, qid="sect")).model, "english")
check("pinned/explicit model still wins", rp.route("x", _q(_ASK), model="generic").model, "generic")
check("pinned/registered reports the digest",
      rp.registered["research"]["signatures"], [{"ids": ["section"], "instructions": {"section": instructions_digest(_ASK)}}])
check("pinned/registered reports an id-only signature as a list", rp.registered["generic"]["signatures"], [["section"]])
check("pinned/Signature equality ignores pin order",
      Signature(["a", "b"], {"a": "sha1:x", "b": "sha1:y"}) == Signature(["b", "a"], {"b": "sha1:y", "a": "sha1:x"}), True)

# Without the id-only fallback, an unpinned question under a pinned id routes by language.
rq = Router(models={"research": "/tmp/r"}, signatures={"research": [{"section": _ASK, "subsection": None}]})
two = {**_q(_ASK), **_q("Which subsection?", qid="subsection")}
check("pinned/a null pin leaves that question unconstrained", rq.route("claim", two).model, "research")
check("pinned/the pinned question must still match",
      rq.route("claim", {**_q(_OTHER_ASK), **_q("Which subsection?", qid="subsection")}).model, "english")
check("pinned/partial id set does not match", rq.route("claim", _q(_ASK)).model, "english")

# Conflicts: the same ids with the same pins cannot select two checkpoints; different pins may.
try:
    Router(models={"a": "/a", "b": "/b"}, signatures={"a": [{"section": _ASK}], "b": [{"section": _ASK}]})
    check("pinned/identical pinned signatures on two checkpoints are refused", False, True)
except ValueError as e:
    check("pinned/identical pinned signatures on two checkpoints are refused", "already registered for 'a'" in str(e), True)
try:
    Router(models={"a": "/a", "b": "/b"}, signatures={"a": [{"section": _ASK}], "b": [{"section": _OTHER_ASK}]})
    check("pinned/different pins on the same id are allowed", True, True)
except ValueError:
    check("pinned/different pins on the same id are allowed", False, True)
try:
    Router(models={"a": "/a", "b": "/b"}, signatures={"a": [{"section": _ASK}], "b": [["section"]]})
    check("pinned/an id-only signature beside a pinned one is allowed", True, True)
except ValueError:
    check("pinned/an id-only signature beside a pinned one is allowed", False, True)
for bad, why in (({"section": 7}, "non-text pin"), ({"section": ""}, "blank pin"), ({}, "empty mapping"), ({"": _ASK}, "blank id")):
    try:
        Router(models={"a": "/a"}, signatures={"a": [bad]})
        check("pinned/refuses %s" % why, False, True)
    except (TypeError, ValueError):
        check("pinned/refuses %s" % why, True, True)
check("pinned/register accepts the mapping form",
      (Router().register("tone", "/t", signatures=[{"mood": _OTHER_ASK}]), True)[1], True)


import hashlib  # noqa: E402

# Nested pins: the signature with more pins is tried first; pins that neither contain nor contradict
# each other cannot be told apart and are refused.
_TWO = {**_q(_ASK), **_q(_OTHER_ASK, qid="title")}
for order in (("a", "b"), ("b", "a")):
    sigs = {"a": [{"section": _ASK, "title": None}], "b": [{"section": _ASK, "title": _OTHER_ASK}]}
    rn = Router(models={"a": "/a", "b": "/b"}, signatures={k: sigs[k] for k in order})
    check("pinned/the more specific signature wins in either order (%s)" % "".join(order),
          rn.route("x", _TWO).model, "b")
for first, second in (({"section": _ASK, "title": None}, {"section": None, "title": _OTHER_ASK}),
                      ({"section": _ASK, "title": None}, {"section": _ASK, "title": None})):
    try:
        Router(models={"a": "/a", "b": "/b"}, signatures={"a": [first], "b": [second]})
        check("pinned/indistinguishable pins on one id set are refused", False, True)
    except ValueError as e:
        check("pinned/indistinguishable pins on one id set are refused", "already registered for 'a'" in str(e), True)

# The shape `registered` reports is accepted back, so it round-trips.
rr = Router(models={"a": "/a"}, signatures={"a": [{"section": _ASK}, ["title"]]})
rt = Router(models={"a": "/a"}, signatures={"a": rr.registered["a"]["signatures"]})
check("pinned/registered round-trips into signatures", rt.registered["a"]["signatures"], rr.registered["a"]["signatures"])
check("pinned/a round-tripped signature still routes", rt.route("x", _q(_ASK)).model, "a")
try:
    Router(models={"a": "/a"}, signatures={"a": [{"ids": ["section"], "instructions": {"other": _ASK}}]})
    check("pinned/a reported shape pinning a foreign id is refused", False, True)
except ValueError:
    check("pinned/a reported shape pinning a foreign id is refused", True, True)

# Digests: well-formed ones pass through (case-folded), malformed ones are refused up front.
_HEX = instructions_digest(_ASK)
check("digest/uppercase hex is normalised", instructions_digest("SHA1:" + _HEX[5:].upper()), _HEX)
for bad in ("sha1:3f2a", "sha1:" + "g" * 40, "sha1:" + "0" * 41):
    try:
        Router(models={"a": "/a"}, signatures={"a": [{"section": bad}]})
        check("digest/malformed pin %r is refused" % bad[:12], False, True)
    except ValueError:
        check("digest/malformed pin %r is refused" % bad[:12], True, True)
check("digest/a request whose instructions look like a bad digest just does not match",
      rp.route("x", _q("sha1:zz")).model, "generic")

# Known digests are fixed: changing the normalisation would invalidate every digest stored in a manifest.
check("digest/known text value", instructions_digest("Which section?"), "sha1:" + hashlib.sha1(b"Which section?").hexdigest())
check("digest/known text digest literal", instructions_digest("Which  section?\n"),
      "sha1:b65b4dbcfc4a7efd6cf009d0da00ceed3328efc9")
check("digest/known structured value", instructions_digest({"b": 1, "a": "é"}),
      "sha1:" + hashlib.sha1('{"a": "é", "b": 1}'.encode("utf-8")).hexdigest())


# Signature objects and the described shape get the same checks as the mapping form.
_d = instructions_digest(_ASK)
_r = Router(models={"a": "/tmp/a"}, signatures={"a": [Signature(["section"], {"section": _ASK})]})
check("pinned/Signature text pin is hashed", _r.registered["a"]["signatures"],
      [{"ids": ["section"], "instructions": {"section": _d}}])
check("pinned/Signature text pin routes", _r.route("x", _q(_ASK)).model, "a")
_r = Router(models={"a": "/tmp/a"}, signatures={"a": [Signature(["section"], {"section": _d.upper().replace("SHA1", "sha1")})]})
check("pinned/Signature uppercase digest normalised", _r.route("x", _q(_ASK)).model, "a")
for _bad, _why in ((Signature(["section"], {"title": _d}), "foreign pin"),
                   (Signature(["section"], {"section": ""}), "blank pin"),
                   (Signature(["section"], {"section": "sha1:3f2a"}), "malformed digest")):
    try:
        Router(models={"a": "/tmp/a"}, signatures={"a": [_bad]})
        check("pinned/Signature %s refused" % _why, False, True)
    except (ValueError, TypeError):
        check("pinned/Signature %s refused" % _why, True, True)
for _bad, _why in (({"ids": [1, 2], "instructions": {}}, "non-text ids"),
                   ({"ids": ["m", None], "instructions": {}}, "null id"),
                   ({"ids": ["m"], "instructions": {"m": ""}}, "blank pin"),
                   ({"ids": ["m"], "instructions": {"m": 7}}, "non-text pin")):
    try:
        Router(models={"a": "/tmp/a"}, signatures={"a": [_bad]})
        check("pinned/described %s refused" % _why, False, True)
    except (ValueError, TypeError):
        check("pinned/described %s refused" % _why, True, True)
_same = Router(models={"a": "/tmp/a"}, signatures={"a": [{"section": _ASK, "title": None},
                                                         {"section": None, "title": _OTHER_ASK}]})
check("pinned/non-nested pins on one checkpoint are accepted", len(_same.signatures["a"]), 2)
_at = Router(models={"b": "/b"}, signatures={"b": [{"section": None, "title": "Y"}]})
try:
    _at.register("c", "/c", signatures=[["mood"], {"section": "X", "title": None}], description="tone")
    _at_raised = False
except ValueError:
    _at_raised = True
check("register/refused call raises", _at_raised, True)
check("register/refused call leaves no checkpoint", "c" in _at.models or "c" in _at.signatures, False)
_at2 = Router(models={"a": "/a", "b": "/b"}, signatures={"b": [["t"]]})
try:
    _at2.register("a", "/new", signatures=[["t"]])
except ValueError:
    pass
check("register/refused call keeps the old source", _at2.registered["a"]["source"], "/a")

# An explicit `sha256_digests[name] = None` is a present entry that opts that checkpoint out of
# digest checks; a refused register() must restore it, not drop it (dropping it would hand the
# checkpoint back to the environment's digest map).
_at3 = Router(models={"a": "/a", "b": "/b"}, sha256_digests={"a": None}, signatures={"b": [["t"]]})
try:
    _at3.register("a", "/a2", signatures=[["t"]])
    check("register/refused call raises on a taken signature", False, True)
except ValueError:
    check("register/refused call raises on a taken signature", True, True)
check("register/refused call keeps an opted-out digest entry", ("a" in _at3.sha256_digests, _at3.sha256_digests.get("a", "gone")),
      (True, None))
check("register/refused call keeps the old source after a digest opt-out", _at3.models["a"], "/a")
# register(signatures=...) on a registered name replaces its signatures; unregister() removes a name.
_rp = Router(models={"a": "/a", "b": "/b"}, signatures={"a": [["t"]], "b": [["u"]]})
_rp.register("a", "/a", signatures=[["v"]])
check("register/signatures replace the earlier ones", [sig.describe() for sig in _rp.signatures["a"]], [["v"]])
check("register/a replaced signature no longer routes", _rp.match_signature({"t": {}}), None)
check("register/a replacement routes", _rp.match_signature({"v": {}}), ("a", "a"))
_rp.register("a", "/a2")
check("register/no signatures argument keeps them", [sig.describe() for sig in _rp.signatures["a"]], [["v"]])
try:
    _rp.register("a", "/a", signatures=[["u"]])
    check("register/replacement still refuses another checkpoint's signature", False, True)
except ValueError:
    check("register/replacement still refuses another checkpoint's signature", True, True)
check("register/a refused replacement keeps the old signatures", [sig.describe() for sig in _rp.signatures["a"]], [["v"]])
_rp.descriptions["a"] = "A"
_rp.sha256_digests["a"] = None
_rp.revisions["a"] = "main"
_rp.attach("a", object())
_rp.unregister("a")
check("unregister/removes the name everywhere",
      ("a" in _rp.models, "a" in _rp.signatures, "a" in _rp.descriptions, "a" in _rp.sha256_digests,
       "a" in _rp.revisions, "a" in _rp.loaded, "a" in _rp.registered), (False,) * 7)
check("unregister/its signature is free again", _rp.match_signature({"v": {}}), None)
check("unregister/others are untouched", sorted(_rp.registered), ["b"])
for _bad in ("english", "ml", "never-registered"):
    try:
        _rp.unregister(_bad)
        check("unregister/%s refused" % _bad, False, True)
    except ValueError:
        check("unregister/%s refused" % _bad, True, True)
_rp.register("a", "/a", signatures=[["v"]])
check("unregister/the name can be registered again", _rp.match_signature({"v": {}}), ("a", "a"))

# Routing while another thread re-registers and unregisters never fails or answers from a torn state.
_rc = Router(models={"a": "/a"}, signatures={"a": [["t"]]})
_stop = threading.Event()
_bad = []


def _churn():
    while not _stop.is_set():
        _rc.register("a", "/a", signatures=[["t"]])
        _rc.register("c", "/c", signatures=[["u"]])
        _rc.unregister("c")


def _route():
    for _ in range(2000):
        try:
            _got = _rc.match_signature({"t": {}})
            if _got != ("a", "a"):
                _bad.append(_got)
            _rc.registered
        except Exception as exc:  # noqa: BLE001
            _bad.append(exc)


_th = [threading.Thread(target=_churn)] + [threading.Thread(target=_route) for _ in range(3)]
for _t in _th:
    _t.start()
for _t in _th[1:]:
    _t.join()
_stop.set()
_th[0].join()
check("register+unregister while routing stay consistent", _bad[:3], [])


def _test_unregister_during_build_leaves_no_zombie():
    """A load that starts right after unregister's unload must not leave an agent under the dropped name."""
    import laya.agent as _agent_mod
    started, release = threading.Event(), threading.Event()

    class _Slow:
        def __init__(self, *args, **kwargs):
            started.set()
            release.wait(5)

    old = _agent_mod.Agent
    _agent_mod.Agent = _Slow
    try:
        r = Router(models={"tone": "/tone"})
        loaders = []

        def _then_load(real):
            def hooked(*args):
                out = real(*args)
                if not loaders:     # the instant after the unload: a request starts loading the name
                    loaders.append(threading.Thread(target=_swallow, args=(r.load, "tone"), daemon=True))
                    loaders[0].start()
                    started.wait(1)
                return out
            return hooked

        for _hook in ("unload", "_unload_key"):     # whichever step frees the resident agent
            if hasattr(r, _hook):
                setattr(r, _hook, _then_load(getattr(r, _hook)))
        _timer = threading.Timer(0.3, release.set)
        _timer.start()
        r.unregister("tone")
        for _loader in loaders:
            _loader.join(5)
        release.set()
        _timer.cancel()
        r.register("tone", "/other")
        return ("tone" in r.loaded, "tone" in r._agents)
    finally:
        release.set()
        _agent_mod.Agent = old


def _swallow(fn, *args):
    try:
        fn(*args)
    except Exception:  # noqa: BLE001
        pass


check("unregister/a build in flight leaves no resident agent behind",
      _test_unregister_during_build_leaves_no_zombie(), (False, False))

def _test_unregister_drops_signatures_with_the_model():
    """While unregister waits for an in-flight build, no request may match the name into a missing model."""
    import laya.agent as _agent_mod
    started, release = threading.Event(), threading.Event()

    class _Slow:
        def __init__(self, *args, **kwargs):
            started.set()
            release.wait(5)

    old = _agent_mod.Agent
    _agent_mod.Agent = _Slow
    try:
        r = Router(models={"tone": "/tone"}, signatures={"tone": [["section"]]})
        loader = threading.Thread(target=_swallow, args=(r.load, "tone"), daemon=True)
        loader.start()
        started.wait(2)
        remover = threading.Thread(target=_swallow, args=(r.unregister, "tone"), daemon=True)
        remover.start()
        _time.sleep(0.2)     # unregister is now waiting for the build
        try:
            out = ("tone" in r.models, "tone" in r.signatures, r.match_signature({"section": {}}))
        finally:
            release.set()
            loader.join(5)
            remover.join(5)
        return out
    finally:
        release.set()
        _agent_mod.Agent = old


check("unregister/signatures leave with the model while a build is in flight",
      _test_unregister_drops_signatures_with_the_model(), (False, False, None))

def _test_unregister_between_match_and_route():
    r = Router(models={"tone": "/fake/tone"}, signatures={"tone": [["section"]]})
    real = r.match_signature

    def _match_then_unregister(*args, **kwargs):
        out = real(*args, **kwargs)
        r.unregister("tone")        # lands between the match and the route
        return out

    r.match_signature = _match_then_unregister
    try:
        return r.route("hello there", questions={"section": {"type": "choice", "choices": ["a", "b"]}}).model
    except KeyError:
        return "KeyError"


check("route/an unregister between the signature match and the route falls back to language routing",
      _test_unregister_between_match_and_route(), "english")

def _test_unregister_right_after_the_models_read():
    class _Models(dict):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.owner = None

        def __contains__(self, key):
            out = super().__contains__(key)
            self._drop(key)
            return out

        def get(self, key, default=None):
            out = super().get(key, default)
            self._drop(key)
            return out

        def __getitem__(self, key):
            out = super().__getitem__(key)
            self._drop(key)
            return out

        def _drop(self, key):
            if self.owner is not None and key == "tone":
                owner, self.owner = self.owner, None
                owner.unregister("tone")     # lands right after the first read of the entry

    r = Router(models={"tone": "/fake/tone"}, signatures={"tone": [["section"]]})
    r.models = _Models(r.models)
    r.models.owner = r
    try:
        return r.route("hello there", questions={"section": {"type": "choice", "choices": ["a", "b"]}}).model
    except KeyError:
        return "KeyError"


check("route/an unregister right after the models read cannot raise KeyError",
      _test_unregister_right_after_the_models_read(), "tone")


def _test_sourceless_signature_still_routes():
    class _Fake:
        pass
    r = Router(models={"tone": None}, signatures={"tone": [["section"]]})
    r.attach("tone", _Fake())
    return r.route("hello there", questions={"section": {"type": "choice", "choices": ["a", "b"]}}).model


check("route/a signature registered for a sourceless resident checkpoint still routes to it",
      _test_sourceless_signature_still_routes(), "tone")

try:
    Signature("section")
    _bare = False
except TypeError:
    _bare = True
check("Signature/bare string ids refused", _bare, True)

# --------------------------------------------------------------------- option pins and purpose
# A signature may also pin the answer set: the sorted labels of a choice question, or the level
# count of a score question. Labels, not descriptions, so rewording a description keeps the
# checkpoint valid and adding or removing a label does not. `for` is documentation only.
from laya.router import options_key, parse_signature  # noqa: E402

_SCREEN = "Read or skip this reference?"
def _choice(labels, qid="screen", instructions=_SCREEN):
    return {qid: {"type": "choice", "instructions": instructions, "criteria": {k: "about " + k for k in labels}}}
def _score(levels, qid="urgency"):
    return {qid: {"type": "score", "instructions": "How urgent?", "criteria": list(levels)}}

check("options_key/choice gives sorted labels", options_key(_choice(["skip", "read"])["screen"]), ["read", "skip"])
check("options_key/score gives the level count", options_key(_score(["calm", "firm", "angry"])["urgency"]), 3)
check("options_key/noul has nothing to pin", options_key({"type": "noul", "instructions": "x"}), None)
check("options_key/not a question", options_key("nope"), None)
check("options_key/list-form choice gives sorted labels",
      options_key({"type": "choice", "criteria": ["b", "a", "c"]}), ["a", "b", "c"])
check("options_key/noul with criteria has nothing to pin",
      options_key({"type": "noul", "criteria": {"true": "", "false": ""}}), None)
_rl = Router(models={"tri": "/t", "scr": "/s"},
             signatures={"tri": [{"ids": ["q"], "options": {"q": ["a", "b", "c"]}}],
                         "scr": [{"ids": ["q"], "options": {"q": 3}}]})
check("options pin/list-form choice with other labels is not a score",
      _rl.match_signature({"q": {"type": "choice", "instructions": "x", "criteria": ["x", "y", "z"]}}), None)
check("options pin/list-form choice reaches the label pin",
      _rl.match_signature({"q": {"type": "choice", "instructions": "x", "criteria": ["c", "a", "b"]}}), ("tri", "tri"))
check("options pin/score still matches the count",
      _rl.match_signature({"q": {"type": "score", "instructions": "x", "criteria": ["x", "y", "z"]}}), ("scr", "scr"))
for _bad in ({"ids": ["section"], "options": {"section": ["a", "b"]}, "purpose": "x"},
             {"ids": "section", "options": {"section": ["a", "b"]}},
             {"ids": ["section"], "options": None}, {"ids": ["section"], "instructions": None}):
    try:
        parse_signature("k", _bad)
        _ok = False
    except (TypeError, ValueError):
        _ok = True
    check("parse_signature refuses a malformed faceted object %r" % (_bad,), _ok, True)

ro = Router(models={"screen": "/s", "triage": "/t", "open": "/o"},
            signatures={"screen": [{"ids": ["screen"], "instructions": {"screen": _SCREEN},
                                    "options": {"screen": ["read", "skip"]}, "for": "read or skip a reference"}],
                        "triage": [{"ids": ["urgency"], "options": {"urgency": 3}}],
                        "open": [["section"]]})
check("options/matching labels route", ro.route("x", _choice(["skip", "read"])).model, "screen")
check("options/descriptions may change", ro.route("x", {"screen": {**_choice(["read", "skip"])["screen"], "criteria": {"read": "worth it", "skip": "not"}}}).model, "screen")
check("options/an added label is no match", ro.route("x", _choice(["read", "skip", "maybe"])).model, "english")
check("options/a removed label is no match", ro.route("x", _choice(["read"])).model, "english")
check("options/a wrong instructions text is no match", ro.route("x", _choice(["read", "skip"], instructions="Keep it?")).model, "english")
check("options/score level count matches", ro.route("x", _score(["a", "b", "c"])).model, "triage")
check("options/score level count mismatch", ro.route("x", _score(["a", "b"])).model, "english")
check("options/noul under a pinned id is no match", ro.route("x", {"urgency": {"type": "noul", "instructions": "Urgent?"}}).model, "english")
check("options/registered reports all facets", ro.registered["screen"]["signatures"],
      [{"ids": ["screen"], "instructions": {"screen": instructions_digest(_SCREEN)},
        "options": {"screen": ["read", "skip"]}, "for": "read or skip a reference"}])
check("options/registered reports options alone", ro.registered["triage"]["signatures"], [{"ids": ["urgency"], "options": {"urgency": 3}}])
check("options/registered keeps the list form for id-only", ro.registered["open"]["signatures"], [["section"]])
_rt = Router(models={"screen": "/s"}, signatures={"screen": ro.registered["screen"]["signatures"]})
check("options/the reported shape round-trips", _rt.registered["screen"]["signatures"], ro.registered["screen"]["signatures"])
check("options/purpose takes no part in equality",
      Signature(["a"], options={"a": ["x"]}, purpose="one") == Signature(["a"], options={"a": ["x"]}, purpose="two"), True)
check("options/order of labels does not matter", Signature(["a"], options={"a": ["y", "x"]}) == Signature(["a"], options={"a": ["x", "y"]}), True)

# Specificity and conflicts across facets.
rs = Router(models={"a": "/a", "b": "/b"},
            signatures={"a": [{"ids": ["screen"], "instructions": {"screen": _SCREEN}}],
                        "b": [{"ids": ["screen"], "instructions": {"screen": _SCREEN}, "options": {"screen": ["read", "skip"]}}]})
check("options/the signature with more facets wins", rs.route("x", _choice(["read", "skip"])).model, "b")
check("options/the less specific one takes the rest", rs.route("x", _choice(["read", "skip", "maybe"])).model, "a")
try:
    Router(models={"a": "/a", "b": "/b"},
           signatures={"a": [{"ids": ["screen"], "options": {"screen": ["read", "skip"]}}],
                       "b": [{"ids": ["screen"], "instructions": {"screen": _SCREEN}}]})
    check("options/disjoint facets on two checkpoints are refused", False, True)
except ValueError:
    check("options/disjoint facets on two checkpoints are refused", True, True)
try:
    Router(models={"a": "/a", "b": "/b"},
           signatures={"a": [{"ids": ["screen"], "options": {"screen": ["read", "skip"]}}],
                       "b": [{"ids": ["screen"], "options": {"screen": ["keep", "drop"]}}]})
    check("options/different option sets on two checkpoints are allowed", True, True)
except ValueError:
    check("options/different option sets on two checkpoints are allowed", False, True)
for bad, why in (({"ids": ["screen"], "options": {"screen": "read"}}, "a bare string"),
                 ({"ids": ["screen"], "options": {"screen": []}}, "an empty label list"),
                 ({"ids": ["screen"], "options": {"screen": ["read", "read"]}}, "a duplicate label"),
                 ({"ids": ["screen"], "options": {"screen": 0}}, "a zero level count"),
                 ({"ids": ["screen"], "options": {"screen": True}}, "a boolean"),
                 ({"ids": ["screen"], "options": {"other": ["a"]}}, "a pin outside the ids"),
                 ({"ids": ["screen"], "for": ""}, "a blank purpose"),
                 ({"ids": ["screen"], "extra": 1}, "an unknown facet")):
    try:
        Router(models={"a": "/a"}, signatures={"a": [bad]})
        check("options/refuses %s" % why, False, True)
    except (TypeError, ValueError):
        check("options/refuses %s" % why, True, True)
for _p in ("", "  ", 5):
    try:
        Router(models={"a": "/a"}, signatures={"a": [Signature(["q"], purpose=_p)]})
        check("options/Signature object with purpose %r is refused like the faceted form" % (_p,), False, True)
    except TypeError:
        check("options/Signature object with purpose %r is refused like the faceted form" % (_p,), True, True)
check("parse_signature/module-level form agrees with the Router's",
      parse_signature("a", {"ids": ["screen"], "options": {"screen": ["skip", "read"]}}) == Router._parse_signature("a", {"ids": ["screen"], "options": {"screen": ["read", "skip"]}}), True)


# --------------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all routing tests passed")
sys.exit(1 if FAIL else 0)
