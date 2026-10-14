(() => {
  "use strict";

  const API_CHAT = "/api/chat";
  const API_READY = "/ready";
  const API_EVAL_LATEST = "/api/eval/latest";

  const translations = {
    fa: {
      title: "دستیار هوشمند eplanet",
      subtitle: "پاسخ‌گویی هوشمند بر پایه دانش تأییدشده",
      checking: "در حال بررسی…",
      ready: "سرویس آماده است",
      unavailable: "سرویس در دسترس نیست",
      connectionError: "اتصال برقرار نشد",
      newChat: "گفتگوی جدید",
      send: "ارسال",
      placeholder: "سؤالت رو اینجا بنویس…",
      footerNote: "پاسخ‌ها بر اساس پایگاه دانش تأییدشده eplanet تولید می‌شوند.",
      welcome: "سلام 👋 من دستیار هوش مصنوعی ای‌پلنت هستم. چطور می‌تونم کمکتون کنم؟",
      requestFailed: "درخواست ناموفق بود. لطفاً دوباره تلاش کن.",
      emptyAnswer: "برای این سؤال پاسخ قابل اتکایی پیدا نکردم. لطفاً سؤال را کمی دقیق‌تر مطرح کن.",
      handoff: "برای این مورد به اطلاعات حساب یا بررسی پشتیبانی نیاز است.",
      hideTrace: "بستن جزئیات",
      showTrace: "نمایش جزئیات",
      traceKicker: "پشت‌صحنه",
      traceTitle: "عملکرد بات",
      liveTrace: "Live trace",
      traceEmptyTitle: "هنوز پیامی پردازش نشده",
      traceEmptyText: "بعد از ارسال اولین سؤال، جزئیات درک پیام، بازیابی دانش، فیلترها و زمان اجرای هر مرحله اینجا نمایش داده می‌شود.",
      understanding: "درک پیام",
      resolvedQuery: "پرسش نهایی",
      context: "Context مؤثر",
      retrieval: "بازیابی دانش",
      candidates: "اسناد نهایی",
      blocked: "اسناد فیلترشده",
      blockedHelp: "این اسناد پیدا شدند اما به‌دلیل قوانین دسترسی یا context وارد پاسخ نشدند.",
      validation: "اعتبارسنجی پاسخ",
      performance: "زمان اجرای مراحل",
      noContext: "Context خاصی برای این پاسخ استفاده نشد.",
      noCandidates: "سند نهایی وجود ندارد.",
      language: "زبان",
      domain: "دامنه",
      task: "وظیفه",
      understandingMode: "حالت درک",
      rerankerMode: "Reranker",
      candidateCount: "تعداد اسناد",
      topKnowledge: "سند اول",
      topScore: "امتیاز اول",
      citationValid: "Citation",
      groundingValid: "Grounding",
      abstained: "Abstained",
      valid: "معتبر",
      invalid: "نامعتبر",
      yes: "بله",
      no: "خیر",
      notRun: "اجرا نشده",
      viewChat: "گفتگو",
      viewDashboard: "داشبورد دقت",
      dashboardTitle: "دقت و کیفیت پاسخ‌گویی بات",
      dashboardLoading: "در حال بارگذاری آخرین نتایج ارزیابی…",
      dashboardError: "نتایج ارزیابی هنوز در دسترس نیست.",
      dashboardLastRun: "آخرین اجرا",
      dashboardModel: "مدل",
      dashboardEnv: "محیط",
      dashboardOverallPass: "نرخ موفقیت کلی پیام‌های تست‌شده",
      dashboardTotalTurns: "کل پیام‌ها",
      dashboardPassed: "موفق",
      dashboardFailed: "ناموفق",
      dashboardEnabledCases: "سناریوی فعال",
      dashboardSkippedCases: "سناریوی غیرفعال",
      metricTop1: "دقت نتیجه اول (Top-1)",
      metricRecall5: "پوشش در ۵ نتیجه اول (Recall@5)",
      metricMrr: "میانگین معکوس رتبه (MRR)",
      metricClarification: "دقت تشخیص نیاز به سؤال شفاف‌ساز",
      metricContext: "دقت تشخیص بستر (کشور/تابعیت)",
      metricCitation: "نرخ اعتبار استناد",
      metricGrounding: "نرخ صحت مستندسازی پاسخ",
      metricAbstention: "دقت تصمیم به امتناع",
      metricLeakage: "نرخ نشت اطلاعات کشور نامرتبط",
      metricLatency: "میانگین زمان پاسخ (شامل فراخوانی مدل)",
      statusGood: "خوب",
      statusWarning: "قابل بررسی",
      statusCritical: "نیازمند توجه",
      statusNeutral: "معیار سرعت",
      dashboardCasesTitle: "نتیجه تک‌تک سناریوهای تست",
      colScenario: "سناریو",
      colResult: "نتیجه",
      colTopDoc: "سند اول",
      colRankMode: "مسیر رتبه‌بندی",
      colLatency: "زمان",
      colFailures: "خطاها",
      turnLabel: "نوبت",
    },
    en: {
      title: "eplanet AI Assistant",
      subtitle: "Smart answers grounded in approved knowledge",
      checking: "Checking…",
      ready: "Service is ready",
      unavailable: "Service is unavailable",
      connectionError: "Could not connect",
      newChat: "New chat",
      send: "Send",
      placeholder: "Ask your question here…",
      footerNote: "Answers are generated from eplanet's approved knowledge base.",
      welcome: "Hi 👋 I'm the eplanet AI Assistant. How can I help?",
      requestFailed: "The request failed. Please try again.",
      emptyAnswer: "I couldn't find a reliable answer for that. Please make your question a little more specific.",
      handoff: "This request needs account-specific information or support review.",
      hideTrace: "Hide details",
      showTrace: "Show details",
      traceKicker: "Behind the scenes",
      traceTitle: "Assistant trace",
      liveTrace: "Live trace",
      traceEmptyTitle: "No request processed yet",
      traceEmptyText: "After the first message, this panel will show understanding, retrieval, eligibility filters, validation, and stage timings.",
      understanding: "Understanding",
      resolvedQuery: "Resolved query",
      context: "Effective context",
      retrieval: "Knowledge retrieval",
      candidates: "Final candidates",
      blocked: "Filtered candidates",
      blockedHelp: "These documents were found but excluded by eligibility or context rules.",
      validation: "Answer validation",
      performance: "Stage timings",
      noContext: "No special context was used for this response.",
      noCandidates: "No final candidates.",
      language: "Language",
      domain: "Domain",
      task: "Task",
      understandingMode: "Understanding mode",
      rerankerMode: "Reranker",
      candidateCount: "Candidate count",
      topKnowledge: "Top knowledge",
      topScore: "Top score",
      citationValid: "Citation",
      groundingValid: "Grounding",
      abstained: "Abstained",
      valid: "Valid",
      invalid: "Invalid",
      yes: "Yes",
      no: "No",
      notRun: "Not run",
      viewChat: "Chat",
      viewDashboard: "Accuracy dashboard",
      dashboardTitle: "Answer accuracy & quality",
      dashboardLoading: "Loading the latest evaluation results…",
      dashboardError: "No evaluation results are available yet.",
      dashboardLastRun: "Last run",
      dashboardModel: "Model",
      dashboardEnv: "Environment",
      dashboardOverallPass: "Overall pass rate across tested turns",
      dashboardTotalTurns: "Total turns",
      dashboardPassed: "Passed",
      dashboardFailed: "Failed",
      dashboardEnabledCases: "Enabled cases",
      dashboardSkippedCases: "Skipped cases",
      metricTop1: "Top-1 accuracy",
      metricRecall5: "Recall@5",
      metricMrr: "Mean reciprocal rank (MRR)",
      metricClarification: "Clarification-need accuracy",
      metricContext: "Context accuracy (country/nationality)",
      metricCitation: "Citation validity rate",
      metricGrounding: "Grounding pass rate",
      metricAbstention: "Abstention accuracy",
      metricLeakage: "Cross-country leakage rate",
      metricLatency: "Average response time (incl. model calls)",
      statusGood: "Good",
      statusWarning: "Needs review",
      statusCritical: "Needs attention",
      statusNeutral: "Speed metric",
      dashboardCasesTitle: "Per-scenario test results",
      colScenario: "Scenario",
      colResult: "Result",
      colTopDoc: "Top document",
      colRankMode: "Ranking path",
      colLatency: "Latency",
      colFailures: "Failures",
      turnLabel: "turn",
    },
  };

  const timingLabels = {
    understanding_ms: "Understanding",
    context_requirement_ms: "Context requirement",
    bm25_ms: "BM25",
    vector_ms: "Vector search",
    hybrid_ms: "Hybrid merge",
    eligibility_ms: "Eligibility",
    reranker_ms: "Reranker",
    conversation_retrieval_ms: "Retrieval total",
    generation_ms: "Generation",
    citation_validation_ms: "Citation validation",
    grounding_ms: "Grounding",
    total_ms: "Total",
  };

  const state = {
    uiLanguage: localStorage.getItem("eplanet-demo-language") || "fa",
    conversationId: null,
    sending: false,
    traceVisible: localStorage.getItem("eplanet-demo-trace") !== "hidden",
    lastDebug: null,
    activeView: "chat",
    dashboardData: null,
    dashboardLoading: false,
  };

  const $ = (selector) => document.querySelector(selector);

  function t(key) {
    return translations[state.uiLanguage][key] || key;
  }

  function newConversationId() {
    if (window.crypto && crypto.randomUUID) {
      return `demo-${crypto.randomUUID()}`;
    }
    return `demo-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function applyLanguage() {
    const isFa = state.uiLanguage === "fa";
    document.documentElement.lang = isFa ? "fa" : "en";
    document.documentElement.dir = isFa ? "rtl" : "ltr";

    document.querySelectorAll("[data-i18n]").forEach((node) => {
      const key = node.dataset.i18n;
      if (key === "hideTrace" && !state.traceVisible) {
        node.textContent = t("showTrace");
      } else {
        node.textContent = t(key);
      }
    });

    document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
      node.placeholder = t(node.dataset.i18nPlaceholder);
    });

    document.querySelectorAll(".lang-btn").forEach((button) => {
      button.classList.toggle("active", button.dataset.lang === state.uiLanguage);
    });

    if (state.lastDebug) renderTrace(state.lastDebug);
    if (state.dashboardData) renderDashboard(state.dashboardData);
  }

  function applyTraceVisibility() {
    const workspace = $("#workspace");
    workspace.classList.toggle("trace-collapsed", !state.traceVisible);
    const label = $("#traceToggleBtn [data-i18n]");
    if (label) label.textContent = state.traceVisible ? t("hideTrace") : t("showTrace");
  }

  function createRow(role) {
    const row = document.createElement("div");
    row.className = `msg-row ${role}`;
    return row;
  }

  // Answers come back as lightweight markdown (## headings, **bold**,
  // [text](url) links) rather than plain text. Everything below builds
  // DOM nodes directly (textContent / createTextNode, never innerHTML
  // with message content) so this stays safe even though the source
  // text ultimately comes from an LLM-composed answer.
  const INLINE_PATTERN = /(\*\*([^*]+)\*\*)|(\[([^\]]+)\]\((https?:\/\/[^\s)]+)(?:\s+"[^"]*")?\))/g;

  function appendInlineFormatted(container, text) {
    let lastIndex = 0;
    let match;

    while ((match = INLINE_PATTERN.exec(text)) !== null) {
      if (match.index > lastIndex) {
        container.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
      }

      if (match[1] !== undefined) {
        const strong = document.createElement("strong");
        strong.textContent = match[2];
        container.appendChild(strong);
      } else {
        const link = document.createElement("a");
        link.href = match[5];
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = match[4];
        container.appendChild(link);
      }

      lastIndex = INLINE_PATTERN.lastIndex;
    }

    if (lastIndex < text.length) {
      container.appendChild(document.createTextNode(text.slice(lastIndex)));
    }
  }

  function appendMultilineFormatted(container, text) {
    const lines = text.split("\n");
    lines.forEach((line, index) => {
      if (index > 0) container.appendChild(document.createElement("br"));
      appendInlineFormatted(container, line);
    });
  }

  function appendRichText(container, text) {
    const blocks = String(text ?? "").split(/\n{2,}/);

    blocks.forEach((block) => {
      const trimmed = block.trim();
      const headingMatch = /^(#{1,6})\s+(.+)$/.exec(trimmed);

      if (headingMatch && !trimmed.includes("\n")) {
        const level = Math.min(headingMatch[1].length, 3);
        const heading = document.createElement("div");
        heading.className = `msg-heading msg-heading-${level}`;
        appendInlineFormatted(heading, headingMatch[2]);
        container.appendChild(heading);
        return;
      }

      const paragraph = document.createElement("div");
      paragraph.className = "msg-paragraph";
      appendMultilineFormatted(paragraph, block);
      container.appendChild(paragraph);
    });
  }

  // RTL-script Unicode ranges (Arabic, Persian, Kurdish Sorani all
  // share this block in practice).
  const RTL_CHAR_PATTERN = /[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]/;

  function detectTextDirection(text) {
    return RTL_CHAR_PATTERN.test(String(text ?? "")) ? "rtl" : "ltr";
  }

  // Each bubble's direction must follow its OWN content, not the
  // demo's UI-chrome language toggle - those are independent (e.g.
  // the operator can run the interface in English while chatting in
  // Persian). Reusing document.documentElement.dir for every bubble
  // left Persian/Arabic replies in an LTR bidi context whenever the
  // UI language didn't match the conversation language, scrambling
  // the on-screen order of numbers and words embedded in the RTL
  // text (reported live, with a screenshot). `language` is the
  // response's own classified language from the API when known (bot
  // replies); otherwise direction is detected from the text's own
  // script.
  function bubbleDirection(language, text) {
    const normalized = String(language ?? "").trim().toLowerCase();
    if (normalized) {
      return ["fa", "ar", "ku"].includes(normalized) ? "rtl" : "ltr";
    }
    return detectTextDirection(text);
  }

  function appendMessage(role, text, citations = [], language = null) {
    const row = createRow(role);
    const wrapper = document.createElement("div");
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.dir = bubbleDirection(language, text);
    appendRichText(bubble, text);
    wrapper.appendChild(bubble);

    if (role === "bot" && citations.length) {
      const citeWrap = document.createElement("div");
      citeWrap.className = "citations";
      citations.forEach((citation) => {
        const pill = document.createElement("span");
        pill.className = "citation-pill";
        pill.textContent = citation.knowledge_key;
        citeWrap.appendChild(pill);
      });
      wrapper.appendChild(citeWrap);
    }

    row.appendChild(wrapper);
    $("#messages").appendChild(row);
    scrollToBottom();
    return row;
  }

  function showTyping() {
    const row = createRow("bot");
    row.id = "typingRow";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    const typing = document.createElement("div");
    typing.className = "typing";
    typing.innerHTML = "<span></span><span></span><span></span>";
    bubble.appendChild(typing);
    row.appendChild(bubble);
    $("#messages").appendChild(row);
    scrollToBottom();
  }

  function hideTyping() {
    const row = $("#typingRow");
    if (row) row.remove();
  }

  function scrollToBottom() {
    const messages = $("#messages");
    messages.scrollTop = messages.scrollHeight;
  }

  function resetTrace() {
    state.lastDebug = null;
    $("#traceEmpty").hidden = false;
    $("#traceContent").hidden = true;
    $("#blockedSection").hidden = true;
  }

  function startNewConversation() {
    state.conversationId = newConversationId();
    $("#messages").innerHTML = "";
    resetTrace();
    appendMessage("bot", t("welcome"));
    $("#messageInput").focus();
  }

  function traceStat(key, value) {
    const item = document.createElement("div");
    item.className = "trace-stat";

    const k = document.createElement("span");
    k.className = "k";
    k.textContent = key;

    const v = document.createElement("span");
    v.className = "v";
    v.textContent = value == null || value === "" ? "—" : String(value);
    v.title = v.textContent;

    item.append(k, v);
    return item;
  }

  function formatScore(value) {
    if (value == null || Number.isNaN(Number(value))) return "—";
    return Number(value).toFixed(4);
  }

  function boolLabel(value, kind = "valid") {
    if (value == null) return { text: t("notRun"), cls: "neutral" };
    if (kind === "abstained") {
      return value
        ? { text: t("yes"), cls: "bad" }
        : { text: t("no"), cls: "good" };
    }
    return value
      ? { text: t("valid"), cls: "good" }
      : { text: t("invalid"), cls: "bad" };
  }

  function renderUnderstanding(debug) {
    const root = $("#traceUnderstanding");
    root.innerHTML = "";
    root.append(
      traceStat(t("language"), debug.language),
      traceStat(t("domain"), debug.domain),
      traceStat(t("task"), debug.task),
      traceStat(t("understandingMode"), debug.understanding_mode),
    );
    $("#traceResolvedQuery").textContent = debug.resolved_query || "—";
  }

  function renderContext(debug) {
    const root = $("#traceContext");
    root.innerHTML = "";
    const entries = Object.entries(debug.safe_context || {});

    if (!entries.length) {
      const empty = document.createElement("span");
      empty.className = "context-empty";
      empty.textContent = t("noContext");
      root.appendChild(empty);
      return;
    }

    entries.forEach(([key, value]) => {
      const chip = document.createElement("span");
      chip.className = "context-chip";
      chip.textContent = `${key}: ${value}`;
      root.appendChild(chip);
    });
  }

  function renderRetrieval(debug) {
    const root = $("#traceRetrieval");
    root.innerHTML = "";
    root.append(
      traceStat(t("rerankerMode"), debug.reranker_mode),
      traceStat(t("candidateCount"), debug.final_candidate_count ?? 0),
      traceStat(t("topKnowledge"), debug.top_knowledge_key),
      traceStat(t("topScore"), formatScore(debug.top_rerank_score)),
    );
  }

  function renderCandidates(debug) {
    const root = $("#traceCandidates");
    root.innerHTML = "";
    const candidates = debug.candidates || [];

    if (!candidates.length) {
      const empty = document.createElement("div");
      empty.className = "context-empty";
      empty.textContent = t("noCandidates");
      root.appendChild(empty);
      return;
    }

    const scores = candidates
      .map((item) => Number(item.rerank_score))
      .filter((value) => Number.isFinite(value));
    const maxScore = scores.length ? Math.max(...scores) : 1;
    const minScore = scores.length ? Math.min(...scores) : 0;
    const spread = Math.max(maxScore - minScore, 0.000001);

    candidates.forEach((candidate) => {
      const row = document.createElement("div");
      row.className = "candidate-row";

      const rank = document.createElement("span");
      rank.className = "candidate-rank";
      rank.textContent = `#${candidate.rank ?? "?"}`;

      const main = document.createElement("div");
      main.className = "candidate-main";
      const key = document.createElement("span");
      key.className = "candidate-key";
      key.textContent = candidate.knowledge_key || "—";
      key.title = key.textContent;
      const bar = document.createElement("div");
      bar.className = "score-bar";
      const fill = document.createElement("div");
      fill.className = "score-fill";
      const numeric = Number(candidate.rerank_score);
      const width = Number.isFinite(numeric)
        ? 35 + ((numeric - minScore) / spread) * 65
        : 0;
      fill.style.width = `${Math.max(0, Math.min(100, width))}%`;
      bar.appendChild(fill);
      main.append(key, bar);

      const score = document.createElement("span");
      score.className = "candidate-score";
      score.textContent = formatScore(candidate.rerank_score);

      row.append(rank, main, score);
      root.appendChild(row);
    });
  }

  function renderBlocked(debug) {
    const section = $("#blockedSection");
    const root = $("#traceBlocked");
    root.innerHTML = "";
    const blocked = debug.eligibility_blocked || [];
    section.hidden = !blocked.length;

    blocked.forEach((item) => {
      const row = document.createElement("div");
      row.className = "blocked-row";

      const key = document.createElement("div");
      key.className = "blocked-key";
      key.textContent = item.knowledge_key || "—";
      row.appendChild(key);

      const reasons = document.createElement("div");
      reasons.className = "blocked-reasons";
      (item.reasons || []).forEach((reason) => {
        const chip = document.createElement("span");
        chip.className = "reason-chip";
        chip.textContent = reason;
        reasons.appendChild(chip);
      });
      row.appendChild(reasons);
      root.appendChild(row);
    });
  }

  function renderValidation(debug) {
    const root = $("#traceValidation");
    root.innerHTML = "";

    const rows = [
      [t("citationValid"), boolLabel(debug.citation_valid)],
      [t("groundingValid"), boolLabel(debug.grounding_valid)],
      [t("abstained"), boolLabel(debug.abstained, "abstained")],
    ];

    rows.forEach(([name, value]) => {
      const row = document.createElement("div");
      row.className = "validation-row";
      const label = document.createElement("span");
      label.className = "validation-name";
      label.textContent = name;
      const badge = document.createElement("span");
      badge.className = `validation-value ${value.cls}`;
      badge.textContent = value.text;
      row.append(label, badge);
      root.appendChild(row);
    });
  }

  function renderTiming(debug) {
    const root = $("#traceTiming");
    root.innerHTML = "";
    const performance = debug.performance || {};

    Object.entries(timingLabels).forEach(([key, labelText]) => {
      const value = performance[key];
      if (value == null) return;

      const row = document.createElement("div");
      row.className = "timing-row";
      const name = document.createElement("span");
      name.className = "timing-name";
      name.textContent = labelText;
      const number = document.createElement("span");
      number.className = "timing-value";
      number.textContent = `${Number(value).toFixed(1)} ms`;
      row.append(name, number);
      root.appendChild(row);
    });
  }

  function renderTrace(debug) {
    if (!debug) {
      resetTrace();
      return;
    }

    state.lastDebug = debug;
    $("#traceEmpty").hidden = true;
    $("#traceContent").hidden = false;

    renderUnderstanding(debug);
    renderContext(debug);
    renderRetrieval(debug);
    renderCandidates(debug);
    renderBlocked(debug);
    renderValidation(debug);
    renderTiming(debug);
  }

  function formatPercent(value) {
    if (value == null || Number.isNaN(Number(value))) return "—";
    return `${Math.round(Number(value) * 100)}%`;
  }

  function formatLatency(value) {
    if (value == null || Number.isNaN(Number(value))) return "—";
    const ms = Number(value);
    return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
  }

  function statusForRatio(value, invert = false) {
    if (value == null) return "neutral";
    if (invert) {
      if (value <= 0.02) return "good";
      if (value <= 0.1) return "warning";
      return "critical";
    }
    if (value >= 0.9) return "good";
    if (value >= 0.75) return "warning";
    return "critical";
  }

  const metricDefs = [
    ["top1_accuracy", "metricTop1", false],
    ["recall_at_5", "metricRecall5", false],
    ["mrr", "metricMrr", false],
    ["clarification_accuracy", "metricClarification", false],
    ["context_accuracy", "metricContext", false],
    ["citation_validity_rate", "metricCitation", false],
    ["grounding_pass_rate", "metricGrounding", false],
    ["abstention_accuracy", "metricAbstention", false],
    ["leakage_rate", "metricLeakage", true],
  ];

  function renderDashboard(data) {
    state.dashboardData = data;
    const metrics = data.metrics || {};
    const meta = data.metadata || {};
    const results = data.results || [];

    $("#dashboardLoading")?.remove();
    const body = $("#dashboardBody");
    body.innerHTML = "";

    const metaParts = [];
    if (meta.generated_at) {
      metaParts.push(`${t("dashboardLastRun")}: ${new Date(meta.generated_at).toLocaleString()}`);
    }
    if (meta.model) metaParts.push(`${t("dashboardModel")}: ${meta.model}`);
    if (meta.environment) metaParts.push(`${t("dashboardEnv")}: ${meta.environment}`);
    $("#dashboardMeta").textContent = metaParts.join("  ·  ");

    const hero = document.createElement("div");
    hero.className = "hero-tile";

    const heroMain = document.createElement("div");
    const heroValue = document.createElement("div");
    heroValue.className = "hero-value";
    heroValue.textContent = formatPercent(metrics.turn_pass_rate);
    const heroLabel = document.createElement("div");
    heroLabel.className = "hero-label";
    heroLabel.textContent = t("dashboardOverallPass");
    heroMain.append(heroValue, heroLabel);
    hero.appendChild(heroMain);

    const breakdown = document.createElement("div");
    breakdown.className = "hero-breakdown";
    [
      [metrics.total_turns, "dashboardTotalTurns"],
      [metrics.passed_turns, "dashboardPassed"],
      [metrics.failed_turns, "dashboardFailed"],
      [metrics.enabled_cases, "dashboardEnabledCases"],
      [metrics.skipped_cases, "dashboardSkippedCases"],
    ].forEach(([value, key]) => {
      const item = document.createElement("div");
      const n = document.createElement("div");
      n.className = "n";
      n.textContent = value == null ? "—" : String(value);
      const l = document.createElement("div");
      l.className = "l";
      l.textContent = t(key);
      item.append(n, l);
      breakdown.appendChild(item);
    });
    hero.appendChild(breakdown);
    body.appendChild(hero);

    const statusLabels = {
      good: t("statusGood"),
      warning: t("statusWarning"),
      critical: t("statusCritical"),
      neutral: t("statusNeutral"),
    };

    const grid = document.createElement("div");
    grid.className = "metric-grid";

    metricDefs.forEach(([key, labelKey, invert]) => {
      const value = metrics[key];
      const status = statusForRatio(value, invert);

      const tile = document.createElement("div");
      tile.className = "metric-tile";

      const label = document.createElement("div");
      label.className = "label";
      label.textContent = t(labelKey);
      tile.appendChild(label);

      const row = document.createElement("div");
      row.className = "value-row";
      const val = document.createElement("span");
      val.className = "value";
      val.textContent = formatPercent(value);
      const tag = document.createElement("span");
      tag.className = `status-tag ${status}`;
      tag.textContent = statusLabels[status];
      row.append(val, tag);
      tile.appendChild(row);

      if (value != null) {
        const track = document.createElement("div");
        track.className = "meter-track";
        const fill = document.createElement("div");
        fill.className = "meter-fill";
        fill.style.width = `${Math.max(2, Math.min(100, value * 100))}%`;
        const meterColor = { good: "success", warning: "warning", critical: "danger" }[status] || "success";
        fill.style.background = `var(--${meterColor})`;
        track.appendChild(fill);
        tile.appendChild(track);
      }

      grid.appendChild(tile);
    });

    const latencyTile = document.createElement("div");
    latencyTile.className = "metric-tile";
    const latencyLabel = document.createElement("div");
    latencyLabel.className = "label";
    latencyLabel.textContent = t("metricLatency");
    const latencyRow = document.createElement("div");
    latencyRow.className = "value-row";
    const latencyValue = document.createElement("span");
    latencyValue.className = "value";
    latencyValue.textContent = formatLatency(metrics.average_latency_ms);
    const latencyTag = document.createElement("span");
    latencyTag.className = "status-tag neutral";
    latencyTag.textContent = t("statusNeutral");
    latencyRow.append(latencyValue, latencyTag);
    latencyTile.append(latencyLabel, latencyRow);
    grid.appendChild(latencyTile);

    body.appendChild(grid);

    const sectionTitle = document.createElement("div");
    sectionTitle.className = "dashboard-section-title";
    sectionTitle.textContent = t("dashboardCasesTitle");
    body.appendChild(sectionTitle);

    const tableWrap = document.createElement("div");
    tableWrap.className = "results-table-wrap";
    const table = document.createElement("table");
    table.className = "results-table";

    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    [
      t("colScenario"),
      t("colResult"),
      t("colTopDoc"),
      t("colRankMode"),
      t("colLatency"),
      t("colFailures"),
    ].forEach((label) => {
      const th = document.createElement("th");
      th.textContent = label;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    results.forEach((row) => {
      const tr = document.createElement("tr");

      const tdCase = document.createElement("td");
      const caseName = document.createElement("div");
      caseName.textContent = row.case_id;
      const turnInfo = document.createElement("div");
      turnInfo.className = "context-empty";
      turnInfo.textContent = `${t("turnLabel")} ${row.turn_index}`;
      tdCase.append(caseName, turnInfo);
      tr.appendChild(tdCase);

      const tdPass = document.createElement("td");
      const passIcon = document.createElement("span");
      passIcon.className = `pass-icon ${row.passed ? "ok" : "bad"}`;
      passIcon.textContent = row.passed ? "✓" : "✗";
      tdPass.appendChild(passIcon);
      tr.appendChild(tdPass);

      const tdKey = document.createElement("td");
      const keySpan = document.createElement("span");
      keySpan.className = "mono";
      keySpan.textContent = row.top_knowledge_key || "—";
      tdKey.appendChild(keySpan);
      tr.appendChild(tdKey);

      const tdMode = document.createElement("td");
      tdMode.textContent = row.reranker_mode || "—";
      tr.appendChild(tdMode);

      const tdLatency = document.createElement("td");
      tdLatency.className = "num-cell";
      tdLatency.textContent = formatLatency(row.latency_ms);
      tr.appendChild(tdLatency);

      const tdFailures = document.createElement("td");
      if (row.failures && row.failures.length) {
        row.failures.forEach((failure) => {
          const chip = document.createElement("span");
          chip.className = "failure-chip";
          chip.textContent = String(failure).split(":")[0];
          tdFailures.appendChild(chip);
        });
      } else {
        tdFailures.textContent = "—";
      }
      tr.appendChild(tdFailures);

      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    tableWrap.appendChild(table);
    body.appendChild(tableWrap);
  }

  async function loadDashboard() {
    if (state.dashboardData || state.dashboardLoading) return;
    state.dashboardLoading = true;

    try {
      const response = await fetch(API_EVAL_LATEST, { cache: "no-store" });
      const data = await response.json().catch(() => null);
      if (!response.ok) throw new Error((data && data.detail) || `HTTP ${response.status}`);
      renderDashboard(data);
    } catch (error) {
      const body = $("#dashboardBody");
      body.innerHTML = "";
      const message = document.createElement("p");
      message.className = "dashboard-error";
      message.textContent = `${t("dashboardError")} (${error.message})`;
      body.appendChild(message);
    } finally {
      state.dashboardLoading = false;
    }
  }

  function switchView(view) {
    if (state.activeView === view) return;
    state.activeView = view;

    document.querySelectorAll(".view-btn").forEach((button) => {
      button.classList.toggle("active", button.dataset.view === view);
    });

    $("#workspace").hidden = view !== "chat";
    $("#dashboardView").hidden = view !== "dashboard";

    if (view === "dashboard") loadDashboard();
  }

  function initViewToggle() {
    document.querySelectorAll(".view-btn").forEach((button) => {
      button.addEventListener("click", () => switchView(button.dataset.view));
    });
  }

  async function sendMessage(text) {
    if (state.sending) return;
    state.sending = true;
    $("#sendButton").disabled = true;
    appendMessage("user", text);
    showTyping();

    try {
      const response = await fetch(API_CHAT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation_id: state.conversationId,
          message: text,
          language: state.uiLanguage.toUpperCase(),
        }),
      });

      const data = await response.json().catch(() => null);
      if (!response.ok) throw new Error(data?.detail || `HTTP ${response.status}`);

      hideTyping();

      let answer = (data.answer || "").trim();
      if (!answer && data.needs_handoff) answer = t("handoff");
      if (!answer) answer = t("emptyAnswer");

      appendMessage("bot", answer, data.citations || [], data.language);
      renderTrace(data.debug || null);
    } catch (error) {
      hideTyping();
      appendMessage("system", t("requestFailed"));
      console.error(error);
    } finally {
      state.sending = false;
      $("#sendButton").disabled = false;
      $("#messageInput").focus();
    }
  }

  async function pollStatus() {
    const dot = $("#statusDot");
    const text = $("#statusText");
    try {
      const response = await fetch(API_READY, { cache: "no-store" });
      const data = await response.json();
      if (response.ok && data.status === "ready") {
        dot.className = "status-dot ok";
        text.textContent = t("ready");
      } else {
        dot.className = "status-dot bad";
        text.textContent = t("unavailable");
      }
    } catch (_) {
      dot.className = "status-dot bad";
      text.textContent = t("connectionError");
    }
  }

  function initComposer() {
    const form = $("#composer");
    const textarea = $("#messageInput");

    textarea.addEventListener("input", () => {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 140)}px`;
    });

    textarea.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const text = textarea.value.trim();
      if (!text) return;
      textarea.value = "";
      textarea.style.height = "auto";
      await sendMessage(text);
    });
  }

  function initLanguageToggle() {
    document.querySelectorAll(".lang-btn").forEach((button) => {
      button.addEventListener("click", () => {
        if (button.dataset.lang === state.uiLanguage) return;
        state.uiLanguage = button.dataset.lang;
        localStorage.setItem("eplanet-demo-language", state.uiLanguage);
        applyLanguage();
        startNewConversation();
        pollStatus();
      });
    });
  }

  function initTraceToggle() {
    $("#traceToggleBtn").addEventListener("click", () => {
      state.traceVisible = !state.traceVisible;
      localStorage.setItem("eplanet-demo-trace", state.traceVisible ? "visible" : "hidden");
      applyTraceVisibility();
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    applyLanguage();
    applyTraceVisibility();
    initComposer();
    initLanguageToggle();
    initTraceToggle();
    initViewToggle();
    $("#newChatBtn").addEventListener("click", startNewConversation);
    startNewConversation();
    pollStatus();
    setInterval(pollStatus, 15000);
  });
})();
