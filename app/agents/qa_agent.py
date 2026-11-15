"""Bounded plan/search/check loop over original, team-filtered source turns."""

import json
import re
from itertools import zip_longest

from app.agents.prompts import QA_PROMPT_VERSION, QA_PLAN, QA_COVERAGE, QA_DRAFT, QA_REVIEW
from app.citations import quote_with_context, verify_citation, verify_quote
from app.config import ALL_TEAMS, PARSED_DIR, project_path
from app.generation import NvidiaGenerationClient
from app.ingest import normalise_text
from app.memory import source_turns
from app.retriever import search_hybrid, validate_search, validate_meeting_filter
from app.retrieval_plan import requested_meeting


def require_fields(value: dict, fields: set) -> None:
    """Reject malformed model contracts instead of guessing missing fields."""

    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Model response has unexpected or missing fields")


def checked_text(value: str, max_words: int = 100) -> str:
    """Keep generated questions and queries short and nonempty."""

    if not isinstance(value, str) or not value.strip() or len(value.split()) > max_words:
        raise ValueError("Model response contains invalid or excessive text")
    return value.strip()


def _is_answer_constraint(value: str) -> bool:
    """Recognise imperative exclusions that are not source-verifiable facts."""

    text = " ".join(value.lower().split())
    return (text.startswith(("exclude ", "do not ", "don't ", "never ",
                             "ensure excluded ", "ensure that excluded "))
            or " must be excluded" in text
            or " must not be guessed" in text
            or " should be excluded" in text)


def citation_options(sources: dict) -> dict:
    """Build stable IDs for exact, overlapping excerpts from supplied turns only."""

    options = {}
    for (meeting_id, turn_id), source in sorted(sources.items()):
        words = list(re.finditer(r"\S+", source["text"]))
        if len(words) < 6:
            continue
        starts = [0] if len(words) <= 60 else list(range(0, len(words) - 59, 40))
        if len(words) > 60 and starts[-1] != len(words) - 60:
            starts.append(len(words) - 60)
        for start in starts:
            end = min(start + 60, len(words))
            quote = source["text"][words[start].start():words[end - 1].end()]
            if not verify_quote(quote, {"utterances": [{"original_text": source["text"]}]}):
                raise ValueError("Source excerpt failed exact quote verification")
            identifier = f"{meeting_id}_t{turn_id}_w{start}"
            options[identifier] = {"meeting_id": meeting_id, "turn_id": turn_id, "quote": quote}
    return options


def _turn_query_score(turn: dict, query_terms) -> tuple[int, int]:
    """Rank original turns by literal query overlap without judging meaning."""

    if not isinstance(query_terms, list) or any(not isinstance(term, str) for term in query_terms):
        return (0, 0)
    words = re.findall(r"[^\W_]+", normalise_text(turn["original_text"]).lower())
    unique_words = set(words)
    matched = set()
    occurrences = 0
    for term in query_terms:
        # A small prefix allowance mirrors common FTS stemming cases such as
        # battery/batteries and price/prices while avoiding short-word noise.
        forms = {word for word in unique_words if word == term or (
            min(len(word), len(term)) >= 5 and word[:5] == term[:5]
        )}
        if forms:
            matched.add(term)
            occurrences += sum(words.count(form) for form in forms)
    return (len(matched), occurrences)


class QuestionAnsweringAgent:
    """Use bounded searches, one coverage repair, and one draft repair."""

    def __init__(self, client=None, search=None, parsed_dir=PARSED_DIR,
                 max_rounds=2, limit=5, max_context_words=6000, on_event=None):

        if type(max_rounds) is not int or not 1 <= max_rounds <= 3:
            raise ValueError("max_rounds must be between one and three")
        if type(limit) is not int or not 1 <= limit <= 10:
            raise ValueError("limit must be between one and ten")
        if type(max_context_words) is not int or not 1000 <= max_context_words <= 6000:
            raise ValueError("max_context_words must be between 1000 and 6000")
        self.client = client or NvidiaGenerationClient()
        self.search = search or search_hybrid
        self.parsed_dir = project_path(parsed_dir)
        self.max_rounds = max_rounds
        self.limit = limit
        self.max_context_words = max_context_words
        self.on_event = on_event

    def emit(self, stage, **details):
        """Publish observable steps without model reasoning or credentials."""

        if self.on_event is not None:
            self.on_event({"stage": stage, "details": details})

    def answer(self, question: str, team_id: str, as_of: str = "d") -> dict:
        """Answer one question or return an explicit failure without guessing."""

        selected_team = team_id
        team_id = validate_search(question, team_id, as_of, self.limit)
        question = checked_text(question, 200)
        meeting_number = requested_meeting(question)
        validate_meeting_filter(meeting_number, as_of)
        base = {"prompt_version": QA_PROMPT_VERSION, "question": question,
                "team_id": team_id, "as_of": as_of,
                "target_meeting": "abcd"[meeting_number - 1] if meeting_number else None,
                "time_scope": "as_of is an upper bound, not the meeting requested by the question"}
        if team_id == ALL_TEAMS:
            base["scope"] = ("All teams. Sources may come from different teams; a meeting_id starts with its "
                             "team ID (e.g. ES2002b is team ES2002). Never merge different teams' decisions; "
                             "name the team in the answer when it matters.")
        self.emit("planning", question=question, team_id=team_id, selected_team=selected_team,
                  as_of=as_of, target_meeting=base["target_meeting"])
        plan = self.client.generate_json(QA_PLAN, base)
        require_fields(plan, {"parts"})
        if not isinstance(plan["parts"], list) or not 1 <= len(plan["parts"]) <= 3:
            raise ValueError("Planner must return one to three question parts")
        parts = []
        for index, part in enumerate(plan["parts"], 1):
            require_fields(part, {"question", "query", "requirements"})
            item = {"id": f"p{index}", "question": checked_text(part["question"]),
                    "query": checked_text(part["query"])}
            requirements = part["requirements"]
            if not isinstance(requirements, list) or not 1 <= len(requirements) <= 4:
                raise ValueError("Each part needs one to four evidence requirements")
            requirements = [checked_text(text) for text in requirements]
            requirements = [text for text in requirements if not _is_answer_constraint(text)]
            if not requirements:
                raise ValueError("Each part needs at least one factual evidence requirement")
            item["requirements"] = [{"id": f"p{index}_r{number}", "question": text}
                                    for number, text in enumerate(requirements, 1)]
            parts.append(item)
        self.emit("plan_ready", parts=parts)
        meetings, sources, seen_queries, trace = {}, {}, set(), []
        coverage_repaired = False
        pending = {part["id"]: part["query"] for part in parts}
        context = dict(base, parts=parts)
        for round_number in range(1, self.max_rounds + 1):
            for part_id, query in pending.items():
                if query.casefold() in seen_queries:
                    continue
                seen_queries.add(query.casefold())
                self.emit("searching", round=round_number, part_id=part_id, query=query)
                filters = {"as_of": as_of, "limit": self.limit}
                if meeting_number is not None:
                    filters["meeting_number"] = meeting_number
                results = self.search(query, team_id, **filters)
                admission = {}
                added = self.add_sources(results, meetings, sources, team_id, as_of,
                                         meeting_number=meeting_number, diagnostics=admission)
                self.emit("retrieved", round=round_number, part_id=part_id, source_turns_added=added,
                          admission=admission,
                          chunks=[{key: chunk.get(key) for key in
                                   ("chunk_id", "meeting_id", "team_id", "meeting_number", "score",
                                    "retrieval_scores", "retrieval_reason", "start_timestamp",
                                    "end_timestamp", "speaker_ids", "turn_ids", "original_text")}
                                  for chunk in results])
                trace.append({"round": round_number, "part_id": part_id, "query": query,
                              "target_meeting": base["target_meeting"],
                              "source_turns_added": added, "admission": admission})
            if not sources:
                return self.failure("insufficient_evidence", "No usable source turns were retrieved", trace)
            context["sources"] = sorted(sources.values(), key=lambda item: (item["meeting_id"], item["turn_id"]))
            context["citation_options"] = [dict(value, citation_id=identifier)
                                           for identifier, value in citation_options(sources).items()]
            coverage_context = context
            self.emit("sources_ready", sources=context["sources"], citation_options=context["citation_options"])
            while True:
                self.emit("checking_coverage", round=round_number)
                coverage = self.client.generate_json(QA_COVERAGE, coverage_context)
                try:
                    coverage, pending = self.check_coverage(coverage, parts, meetings, sources, team_id, as_of)
                    self.emit("coverage_checked", round=round_number, coverage=coverage, missing_parts=list(pending))
                    break
                except ValueError as error:
                    self.emit("coverage_repair", error=str(error), repair_available=not coverage_repaired)
                    trace.append({"round": round_number, "stage": "coverage", "validation_error": str(error)})
                    if coverage_repaired:
                        return self.failure("invalid_evidence", "Coverage did not pass validation after one repair", trace)
                    coverage_repaired = True
                    coverage_context = dict(context, previous_coverage=coverage, correction=str(error))
            requirement_questions = {item["id"]: item["question"] for part in parts for item in part["requirements"]}
            trace.append({"round": round_number, "missing_parts": list(pending),
                          "requirement_checks": [
                              {"id": item["id"], "question": requirement_questions[item["id"]],
                               "supported": bool(item["evidence"]),
                               "source_turns": [{"meeting_id": cite["meeting_id"], "turn_id": cite["turn_id"]}
                                                for cite in item["evidence"]]}
                              for part in coverage["parts"] for item in part["requirements"]]})
            if not pending:
                context["coverage"] = coverage
                return self.draft(context, meetings, sources, team_id, as_of, trace)
        return self.failure("insufficient_evidence", "Some question parts still lack evidence", trace)

    def add_sources(self, results, meetings, sources, team_id, as_of, meeting_number=None, diagnostics=None):
        """Admit bounded original turns; independently enforce team and cutoff."""

        used = sum(len(json.dumps(item, ensure_ascii=False).split()) for item in sources.values())
        allowance = min(1000, self.max_context_words - used)
        diagnostics = diagnostics if diagnostics is not None else {}
        diagnostics.update(admitted=[], dropped=[], duplicates=[], words_before=used,
                           search_word_budget=allowance, total_word_budget=self.max_context_words)
        visited = set()
        added = 0
        candidates = []
        team_pattern = r"[A-Z]{2}\d{4}" if team_id == ALL_TEAMS else re.escape(team_id)
        for chunk in results:
            meeting_id = chunk.get("meeting_id")
            if not isinstance(meeting_id, str) or not re.fullmatch(team_pattern + "[a-d]", meeting_id):
                raise ValueError("Retrieval returned a different team or invalid meeting")
            # Each source must belong to its own meeting's team (the selected one unless ALL).
            source_team = meeting_id[:-1]
            if meeting_id[-1] > as_of or chunk.get("team_id") != source_team:
                raise ValueError("Retrieval violated the team or meeting cutoff")
            if meeting_number is not None and "abcd".index(meeting_id[-1]) + 1 != meeting_number:
                raise ValueError("Retrieval violated the requested meeting")
            if meeting_id not in meetings:
                path = self.parsed_dir / f"{meeting_id}.json"
                meeting = json.loads(path.read_text(encoding="utf-8"))
                if (meeting.get("meeting_id") != meeting_id or meeting.get("team_id") != source_team
                        or meeting.get("meeting_sequence") != meeting_id[-1]
                        or meeting.get("meeting_number") != "abcd".index(meeting_id[-1]) + 1):
                    raise ValueError("Parsed source metadata does not match retrieval")
                meetings[meeting_id] = meeting
            turns = list(source_turns(meetings[meeting_id], chunk.get("turn_ids")).values())
            query_terms = chunk.get("_query_terms", [])
            if query_terms:
                turns = sorted(enumerate(turns),
                               key=lambda item: (*(-value for value in _turn_query_score(item[1], query_terms)),
                                                 item[0]))
                turns = [turn for _, turn in turns]
            candidates.append([(meeting_id, turn) for turn in turns])
        # Give each ranked passage a turn before admitting more from the first.
        for row in zip_longest(*candidates):
            for candidate in row:
                if candidate is None:
                    continue
                meeting_id, turn = candidate
                key = (meeting_id, turn["utterance_id"])
                if key in visited:
                    continue
                visited.add(key)
                reference = {"meeting_id": meeting_id, "turn_id": turn["utterance_id"]}
                if key in sources:
                    diagnostics["duplicates"].append(reference)
                    continue
                item = {"meeting_id": meeting_id, "turn_id": turn["utterance_id"],
                        "speaker": "Speaker " + turn["speaker_id"],
                        "timestamp": turn["timestamp"], "text": turn["original_text"]}
                words = len(json.dumps(item, ensure_ascii=False).split())
                if words > allowance:
                    diagnostics["dropped"].append(dict(reference, reason="word_budget", words=words))
                    continue
                sources[key] = item
                allowance -= words
                added += 1
                diagnostics["admitted"].append(reference)
        diagnostics["words_after"] = sum(len(json.dumps(item, ensure_ascii=False).split()) for item in sources.values())
        return added

    def check_evidence(self, citation, meetings, sources, team_id, as_of):
        """Return verified evidence, adding original context to short exact spans."""

        if isinstance(citation, dict) and set(citation) == {"citation_id"}:
            identifier = citation["citation_id"]
            options = citation_options(sources)
            if not isinstance(identifier, str) or identifier not in options:
                raise ValueError("Select a citation_id from the supplied citation_options")
            citation = options[identifier]
        require_fields(citation, {"meeting_id", "turn_id", "quote"})
        meeting_id, turn_id = citation["meeting_id"], citation["turn_id"]
        if not isinstance(meeting_id, str) or type(turn_id) is not int:
            raise ValueError("Citation has invalid source identifiers")
        if not isinstance(citation["quote"], str) or not citation["quote"].strip() or len(citation["quote"].split()) > 80:
            raise ValueError("Citation quote must contain six to eighty words")
        source = sources.get((meeting_id, turn_id))
        if not source:
            raise ValueError(f"Citation {meeting_id} turn {turn_id} must refer to a supplied turn")
        turn = {"original_text": source["text"]}
        checked = dict(citation)
        if len(checked["quote"].split()) < 6:
            try:
                checked["quote"] = quote_with_context(checked["quote"], turn)
            except ValueError as error:
                raise ValueError(f"Citation {meeting_id} turn {turn_id}: {error}") from None
        if (not verify_citation(checked, meetings, None if team_id == ALL_TEAMS else team_id, as_of)
                or not verify_quote(checked["quote"], {"utterances": [turn]})):
            raise ValueError(f"Citation {meeting_id} turn {turn_id} must copy 6–80 exact words from the supplied turn, "
                             "including original punctuation and underscores")
        return checked

    def check_coverage(self, response, parts, meetings, sources, team_id, as_of):
        """Require support evidence or a follow-up query for every planned part."""

        require_fields(response, {"parts"})
        if not isinstance(response["parts"], list) or len(response["parts"]) != len(parts):
            raise ValueError("Coverage must include every question part exactly once")
        expected = {part["id"] for part in parts}
        seen, pending, checked_parts = set(), {}, []
        for part in response["parts"]:
            if not isinstance(part, dict):
                raise ValueError("Invalid coverage part")
            part_id = part.get("id")
            if not isinstance(part_id, str) or part_id not in expected or part_id in seen:
                raise ValueError("Coverage contains an unknown or repeated part")
            seen.add(part_id)
            planned = next(item for item in parts if item["id"] == part_id)
            checked = self.check_requirements(part, planned, meetings, sources, team_id, as_of)
            if not checked["supported"]:
                pending[part_id] = checked["next_query"]
            checked_parts.append(checked)
        return {"parts": checked_parts}, pending

    def check_requirements(self, part, planned, meetings, sources, team_id, as_of):
        """Compute completeness from evidence for every planned fact, not a boolean."""

        require_fields(part, {"id", "requirements", "next_query"})
        expected = {item["id"] for item in planned["requirements"]}
        if not isinstance(part["requirements"], list) or len(part["requirements"]) != len(expected):
            raise ValueError("Coverage must check every evidence requirement")
        seen, checks, evidence = set(), [], []
        for requirement in part["requirements"]:
            require_fields(requirement, {"id", "evidence"})
            identifier = requirement["id"]
            if not isinstance(identifier, str) or identifier not in expected or identifier in seen:
                raise ValueError("Unknown or repeated evidence requirement")
            seen.add(identifier)
            if not isinstance(requirement["evidence"], list) or len(requirement["evidence"]) > 3:
                raise ValueError("Each requirement accepts zero to three evidence citations")
            verified = [self.check_evidence(citation, meetings, sources, team_id, as_of)
                        for citation in requirement["evidence"]]
            checks.append({"id": identifier, "evidence": verified})
            for citation in verified:
                if citation not in evidence:
                    evidence.append(citation)
        supported = all(check["evidence"] for check in checks)
        if supported:
            # Completeness is derived exclusively from verified requirement evidence.
            # Some JSON-mode models still emit a redundant follow-up string; it cannot
            # weaken or expand that evidence, so canonicalise it instead of consuming
            # the single repair budget on a harmless control-field inconsistency.
            next_query = None
        else:
            next_query = checked_text(part["next_query"])
        return {"id": part["id"], "supported": supported, "evidence": evidence,
                "requirements": checks, "next_query": next_query}

    def validate_draft(self, draft, context, meetings, sources, team_id, as_of):
        """Enforce format, exact quotes, source boundaries, and citation coverage."""

        require_fields(draft, {"answer", "citations"})
        text = checked_text(draft["answer"], 160)
        # Small JSON-mode models occasionally leak the adjacent schema field name
        # as a final prose token. This is structural noise, not answer content.
        text = re.sub(r"\s+(?:citation_options|citations)\s*$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*\[[A-Z]{2}\d{4}[a-d]_t\d+_w\d+\]", "", text)
        text = re.sub(r"\s*\ufffd?[A-Z]{2}\d{4}[a-d]_t\d+_w\d+", "", text)
        text = re.sub(r"\s*\[citations:\s*\[\{.*?\}\]\]", "", text, flags=re.IGNORECASE)
        text = re.sub(
            r'\s*(?:Citation)?[\u200b\ufffd]*\[\{"citation_id":"[^"]*",'
            r'"part_ids":\[[^\]]*\]\}\]', "", text, flags=re.IGNORECASE)
        text = checked_text(text, 160)
        sentences = [sentence for sentence in re.split(r"[.!?](?:\s+|$)", text) if sentence.strip()]
        if "\n" in text or not 1 <= len(sentences) <= 3:
            raise ValueError("Answer must contain one to three plain sentences")
        if not isinstance(draft["citations"], list) or not 1 <= len(draft["citations"]) <= 3:
            raise ValueError("Answer requires one to three citations")
        expected = {part["id"] for part in context["parts"]}
        requirement_parents = {requirement["id"]: part["id"]
                               for part in context["parts"] for requirement in part["requirements"]}
        covered, seen, checked_citations = set(), set(), []
        for citation in draft["citations"]:
            if isinstance(citation, dict) and set(citation) == {"citation_id", "part_ids"}:
                reference = {"citation_id": citation["citation_id"]}
            else:
                require_fields(citation, {"meeting_id", "turn_id", "quote", "part_ids"})
                reference = {name: citation[name] for name in ("meeting_id", "turn_id", "quote")}
            checked = self.check_evidence(reference,
                                          meetings, sources, team_id, as_of)
            ids = citation["part_ids"]
            if not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids):
                raise ValueError("Citation must map to known question parts")
            ids = list(dict.fromkeys(requirement_parents.get(identifier, identifier) for identifier in ids))
            if any(identifier not in expected for identifier in ids):
                raise ValueError("Citation must map to known question parts")
            key = (checked["meeting_id"], checked["turn_id"], " ".join(checked["quote"].lower().split()))
            if key in seen:
                raise ValueError("Duplicate citation")
            seen.add(key)
            covered.update(ids)
            checked_citations.append(dict(checked, part_ids=list(ids)))
        if covered != expected:
            raise ValueError("Every question part must have a citation")
        for part in context.get("coverage", {}).get("parts", []):
            for requirement in part.get("requirements", []):
                if not any(part["id"] in cite["part_ids"]
                           and cite["meeting_id"] == evidence["meeting_id"]
                           and cite["turn_id"] == evidence["turn_id"]
                           and verify_quote(evidence["quote"], {"utterances": [{"original_text": cite["quote"]}]})
                           for cite in checked_citations for evidence in requirement["evidence"]):
                    raise ValueError(f"Final citations omit evidence for requirement {requirement['id']}")
        return {"answer": text, "citations": checked_citations}

    def draft(self, context, meetings, sources, team_id, as_of, trace):
        """Allow two bounded repairs, then return a checked answer or failure."""

        for attempt in range(3):
            self.emit("drafting", attempt=attempt + 1)
            candidate = self.client.generate_json(QA_DRAFT, context)
            try:
                candidate = self.validate_draft(candidate, context, meetings, sources, team_id, as_of)
                self.emit("verifying_answer", attempt=attempt + 1, citations=candidate["citations"])
                review = self.client.generate_json(QA_REVIEW, dict(context, candidate=candidate))
                require_fields(review, {"approved", "issues"})
                if (type(review["approved"]) is not bool or not isinstance(review["issues"], list)
                        or any(not isinstance(issue, str) for issue in review["issues"])
                        or review["approved"] != (not review["issues"])):
                    raise ValueError("Invalid semantic review response")
                if not review["approved"]:
                    raise ValueError("Semantic review: " + "; ".join(review["issues"]))
                self.emit("answer_verified", attempt=attempt + 1, review=review)
                trace.append({"draft_attempt": attempt + 1, "review": "approved"})
                return {"status": "answered", "answer": candidate["answer"],
                        "citations": [{"meeting_id": cite["meeting_id"], "quote": cite["quote"]}
                                      for cite in candidate["citations"]], "trace": trace}
            except ValueError as error:
                self.emit("draft_repair", attempt=attempt + 1, error=str(error))
                trace.append({"draft_attempt": attempt + 1, "validation_error": str(error)})
                context = dict(context, previous_draft=candidate, correction=str(error))
        return self.failure("invalid_answer", "Answer did not pass validation after two repairs", trace)

    @staticmethod
    def failure(status, reason, trace):
        """Keep failed runs distinguishable from valid submission answers."""

        return {"status": status, "answer": None, "citations": [], "reason": reason, "trace": trace}
