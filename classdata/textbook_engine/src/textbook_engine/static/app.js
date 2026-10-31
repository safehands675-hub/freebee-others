"use strict";

const classSelect = document.querySelector("#class-select");
const bookSelect = document.querySelector("#book-select");
const pageInput = document.querySelector("#page-input");
const bookWarning = document.querySelector("#book-warning");
const processBookButton = document.querySelector("#process-book-button");
const processStatus = document.querySelector("#process-status");
const classContext = document.querySelector("#class-context");
const connectionStatus = document.querySelector("#connection-status");
const messageList = document.querySelector("#message-list");
const errorBanner = document.querySelector("#error-banner");
const chatForm = document.querySelector("#chat-form");
const questionInput = document.querySelector("#question-input");
const sendButton = document.querySelector("#send-button");
const clearButton = document.querySelector("#clear-button");

let catalog = [];
let history = [];
let isSending = false;
let isProcessing = false;

function appendInlineFormatting(parent, text) {
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let lastIndex = 0;

  for (const match of text.matchAll(pattern)) {
    const index = match.index ?? 0;
    parent.append(document.createTextNode(text.slice(lastIndex, index)));
    const token = match[0];
    const formatted = document.createElement(token.startsWith("**") ? "strong" : "code");
    formatted.textContent = token.slice(token.startsWith("**") ? 2 : 1, - (token.startsWith("**") ? 2 : 1));
    parent.append(formatted);
    lastIndex = index + token.length;
  }
  parent.append(document.createTextNode(text.slice(lastIndex)));
}

function renderAnswer(container, answer) {
  const lines = answer.replace(/\r\n?/g, "\n").split("\n");
  let paragraph = [];
  let list = null;

  function flushParagraph() {
    if (paragraph.length === 0) return;
    const element = document.createElement("p");
    appendInlineFormatting(element, paragraph.join(" "));
    container.append(element);
    paragraph = [];
  }

  for (const line of lines) {
    const trimmed = line.trim();
    const listMatch = trimmed.match(/^(?:[-*]\s+|\d+[.)]\s+)(.+)$/);
    if (!trimmed) {
      flushParagraph();
      list = null;
      continue;
    }
    if (listMatch) {
      flushParagraph();
      if (!list) {
        list = document.createElement(/^\d/.test(trimmed) ? "ol" : "ul");
        container.append(list);
      }
      const item = document.createElement("li");
      appendInlineFormatting(item, listMatch[1]);
      list.append(item);
      continue;
    }
    list = null;
    paragraph.push(trimmed.replace(/^#{1,3}\s+/, ""));
  }
  flushParagraph();
}

function addMessage(role, text, options = {}) {
  const message = document.createElement("article");
  message.className = `message ${role === "user" ? "user-message" : "assistant-message"}`;

  if (role !== "user") {
    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = "✦";
    message.append(avatar);
  }

  const content = document.createElement("div");
  content.className = "message-content";
  const author = document.createElement("span");
  author.className = "message-author";
  author.textContent = role === "user" ? "You" : "Study Buddy";
  content.append(author);

  const bubble = document.createElement("div");
  bubble.className = role === "user" ? "bubble" : "bubble answer-markdown";
  if (options.typing) {
    if (options.typingText) {
      const label = document.createElement("span");
      label.className = "typing-label";
      label.textContent = options.typingText;
      bubble.append(label);
    }
    const indicator = document.createElement("span");
    indicator.className = "typing-indicator";
    indicator.setAttribute("aria-label", "Thinking");
    for (let index = 0; index < 3; index += 1) indicator.append(document.createElement("span"));
    bubble.append(indicator);
  } else if (role === "user") {
    bubble.textContent = text;
  } else {
    renderAnswer(bubble, text);
  }
  content.append(bubble);

  if (options.sources?.length) {
    const sourceList = document.createElement("div");
    sourceList.className = "source-list";
    sourceList.setAttribute("aria-label", "Textbook sources");
    for (const source of options.sources) {
      const link = document.createElement(source.page_url ? "a" : "span");
      link.className = "source-link";
      if (source.page_url) {
        link.href = source.page_url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
      }
      link.textContent = source.book;
      const page = document.createElement("span");
      page.textContent = source.page_number_type === "printed"
        ? `Printed page ${source.page} · PDF ${source.pdf_page}`
        : `PDF page ${source.pdf_page ?? source.page}`;
      link.append(page);
      sourceList.append(link);
    }
    content.append(sourceList);

    if (options.showImage && !options.images?.length && options.sources[0].page_url) {
      const figure = document.createElement("figure");
      figure.className = "source-preview";
      const image = document.createElement("img");
      image.src = options.sources[0].page_url;
      image.alt = `Textbook page ${options.sources[0].page} from ${options.sources[0].book}`;
      image.loading = "lazy";
      const caption = document.createElement("figcaption");
      caption.textContent = `Page image used for this question · ${options.sources[0].book}, printed page ${options.sources[0].page}, PDF page ${options.sources[0].pdf_page ?? options.sources[0].page}`;
      figure.append(image, caption);
      content.append(figure);
    }
  }

  for (const reference of options.images ?? []) {
    const figure = document.createElement("figure");
    figure.className = "source-preview";
    const image = document.createElement("img");
    image.src = reference.url;
    image.alt = reference.caption;
    image.loading = "lazy";
    const caption = document.createElement("figcaption");
    caption.textContent = `${reference.caption} · ${reference.book}${reference.pdf_page ? `, PDF page ${reference.pdf_page}` : ""}`;
    figure.append(image, caption);
    content.append(figure);
  }

  for (const warning of options.warnings ?? []) {
    const note = document.createElement("p");
    note.className = "message-note";
    note.textContent = warning;
    content.append(note);
  }
  if (options.note) {
    const note = document.createElement("p");
    note.className = "message-note";
    note.textContent = options.note;
    content.append(note);
  }

  message.append(content);
  messageList.append(message);
  messageList.scrollTop = messageList.scrollHeight;
  return message;
}

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.hidden = false;
}

function clearError() {
  errorBanner.textContent = "";
  errorBanner.hidden = true;
}

function selectedClass() {
  return catalog.find((item) => item.class === Number(classSelect.value));
}

function updateBooks() {
  const selected = selectedClass();
  const corpusBooks = selected?.corpus_books ?? [];
  const pendingBooks = corpusBooks.filter((book) => !book.processed);
  pageInput.value = "";
  bookSelect.replaceChildren();
  const chooseBook = document.createElement("option");
  chooseBook.value = "";
  chooseBook.textContent = "Choose a textbook";
  bookSelect.append(chooseBook);

  for (const book of selected?.books ?? []) {
    const option = document.createElement("option");
    option.value = book.source_id;
    option.textContent = book.page_json
      ? `${book.title} · page JSON`
      : book.prepared
      ? `${book.title} · prepared content`
      : book.printed_pages_indexed
      ? book.title
      : `${book.title} · page index needed`;
    bookSelect.append(option);
  }
  for (const book of pendingBooks) {
    const option = document.createElement("option");
    option.value = book.source_id;
    option.textContent = `${book.title} · not processed`;
    bookSelect.append(option);
  }
  const firstBookId = selected?.books[0]?.source_id ?? pendingBooks[0]?.source_id;
  if (firstBookId) {
    bookSelect.value = firstBookId;
  }

  bookWarning.textContent = "";
  bookWarning.hidden = true;
  if (!selected) {
    classContext.textContent = "No textbook PDFs were found in the source folder.";
  } else if (!selected.chat_supported) {
    classContext.textContent =
      `${corpusBooks.length} textbook${corpusBooks.length === 1 ? "" : "s"} listed · Class ${selected.class} chat is not supported yet.`;
  } else if (selected.available) {
    const pendingNote = pendingBooks.length
      ? ` · ${pendingBooks.length} more will prepare on first question`
      : "";
    classContext.textContent =
      `${selected.books.length} textbook${selected.books.length === 1 ? "" : "s"} ready for chat${pendingNote}`;
  } else if (corpusBooks.length) {
    classContext.textContent =
      `${corpusBooks.length} textbook${corpusBooks.length === 1 ? "" : "s"} listed · the selected book prepares on your first question`;
  } else {
    classContext.textContent = `No textbook PDFs found for Class ${selected.class}.`;
  }
  const selectedBook = selected?.books.find((book) => book.source_id === bookSelect.value);
  const selectedPendingBook = pendingBooks.some(
    (book) => book.source_id === bookSelect.value,
  );
  pageInput.disabled = !selected?.chat_supported
    || (!selectedBook?.page_lookup_available && !selectedPendingBook);
  bookSelect.disabled = !selected || (!selected.books.length && !pendingBooks.length);
  updateComposerAvailability();
  updateSelectedBookStatus();
}

function updateComposerAvailability() {
  const selected = selectedClass();
  const canChat = Boolean(
    selected?.chat_supported && (selected.available || selected.corpus_books?.length),
  );
  questionInput.disabled = !canChat || isSending || isProcessing;
  sendButton.disabled = !canChat || isSending || isProcessing;
}

function updateSelectedBookStatus() {
  const selected = selectedClass();
  const book = selected?.books.find((item) => item.source_id === bookSelect.value);
  const discovered = selected?.corpus_books?.find(
    (item) => item.source_id === bookSelect.value,
  );
  const canProcess = Boolean(
    selected?.chat_supported && discovered && !discovered.processed,
  );
  processBookButton.hidden = !canProcess;
  processBookButton.disabled = isSending || isProcessing;
  pageInput.disabled = !selected?.chat_supported
    || (!book?.page_lookup_available && !(discovered && !discovered.processed));
  if (!book) {
    if (discovered && !discovered.processed) {
      bookWarning.textContent =
        "The first question will prepare this textbook and its page index. The first response may take longer.";
      bookWarning.hidden = false;
    }
    return;
  }

  if (book.page_json) {
    bookWarning.textContent = book.text_quality_warning
      || "Page-level source text and AI descriptions are ready. Enter the printed textbook page number.";
    bookWarning.hidden = false;
    return;
  }
  if (book.prepared) {
    bookWarning.textContent = book.page_lookup_available
      ? "This book uses curated chapter JSON. Enter a printed page number or ask about a chapter."
      : "This book uses curated chapter JSON and linked images. Ask about a chapter (for example, “Explain chapter 1”); printed-page lookup is not configured for this prepared content.";
    bookWarning.hidden = false;
    return;
  }
  if (!book.printed_pages_indexed) {
    bookWarning.textContent = "Printed page lookup is not ready for this book yet. The local printed-page index must be built first.";
    bookWarning.hidden = false;
    return;
  }
  if (book.text_quality_warning) {
    bookWarning.textContent = book.text_quality_warning;
    bookWarning.hidden = false;
    return;
  }
  bookWarning.textContent = "";
  bookWarning.hidden = true;
}

async function processSelectedBook() {
  if (isProcessing || isSending) return;
  const selected = selectedClass();
  const source = selected?.corpus_books?.find(
    (item) => item.source_id === bookSelect.value && !item.processed,
  );
  if (!selected?.chat_supported || !source) {
    showError("Choose an unprocessed textbook from the available PDF list first.");
    return;
  }

  clearError();
  isProcessing = true;
  processBookButton.disabled = true;
  processStatus.hidden = false;
  processStatus.textContent = `Processing ${source.title}. This may take a while for large PDFs.`;
  updateComposerAvailability();
  let processingCompleted = false;
  try {
    const result = await requestJson("/api/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        class: selected.class,
        source_id: source.source_id,
      }),
    });
    processingCompleted = true;
    await refreshCatalog(selected.class, source.filename);
    const warnings = result.processing?.warnings ?? [];
    processStatus.textContent = warnings.length
      ? `Textbook is searchable. Limitations: ${warnings.join(" ")}`
      : `${source.title} is processed and ready.`;
    processStatus.hidden = false;
    resetConversation();
  } catch (error) {
    processStatus.textContent = processingCompleted
      ? "Processing completed, but the textbook list could not be refreshed. Reload the page before retrying."
      : "Processing did not finish. The original PDF is unchanged; review the error and retry.";
    processStatus.hidden = false;
    showError(error instanceof Error ? error.message : "The textbook could not be processed.");
  } finally {
    isProcessing = false;
    updateComposerAvailability();
    updateSelectedBookStatus();
  }
}

function populateCatalog(response, preferredClass = "", preferredFilename = "") {
  catalog = response.classes ?? [];
  classSelect.replaceChildren();
  const listedClasses = catalog.filter(
    (item) => item.available || item.corpus_books?.length,
  );
  for (const item of listedClasses) {
    const option = document.createElement("option");
    option.value = item.class;
    const bookCount = Math.max(item.books.length, item.corpus_books?.length ?? 0);
    option.textContent = item.chat_supported
      ? `Class ${item.class} (${bookCount} book${bookCount === 1 ? "" : "s"})`
      : `Class ${item.class} (${bookCount} books · chat not supported)`;
    classSelect.append(option);
  }
  if (listedClasses.length === 0) {
    const option = document.createElement("option");
    option.textContent = "No textbook PDFs found";
    option.disabled = true;
    classSelect.append(option);
  } else {
    const preferred = listedClasses.find(
      (item) => item.class === Number(preferredClass),
    );
    classSelect.value = String(preferred?.class ?? listedClasses[0].class);
  }

  updateBooks();
  if (!preferredFilename) return;

  const filename = preferredFilename.split(/[\\/]/).pop()?.toLowerCase();
  const selected = selectedClass();
  const processedBook = selected?.books.find(
    (book) => book.filename.toLowerCase() === filename,
  );
  const discoveredBook = selected?.corpus_books?.find(
    (book) => book.filename === preferredFilename,
  );
  const preferredSourceId = discoveredBook?.processed
    ? processedBook?.source_id
    : discoveredBook?.source_id ?? processedBook?.source_id;
  if (preferredSourceId) {
    bookSelect.value = preferredSourceId;
    updateSelectedBookStatus();
  }
}

async function refreshCatalog(preferredClass, preferredFilename) {
  const response = await requestJson("/api/catalog");
  populateCatalog(response, preferredClass, preferredFilename);
}

function resetConversation() {
  history = [];
  messageList.replaceChildren();
  const intro = document.createElement("article");
  intro.className = "message assistant-message welcome-message";
  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = "✦";
  const content = document.createElement("div");
  content.className = "message-content";
  const author = document.createElement("span");
  author.className = "message-author";
  author.textContent = "Study Buddy";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const selected = selectedClass();
  const first = document.createElement("p");
  first.textContent = selected?.chat_supported && selected.available
    ? `Ready to learn, Class ${selected.class}! Ask me about a lesson, a word, or a sum.`
    : selected?.chat_supported
    ? `Ready to prepare the selected Class ${selected.class} textbook when you ask your first question.`
    : `Class ${selected?.class ?? ""} books are listed, but chat is not supported for this class yet.`;
  const second = document.createElement("p");
  second.textContent = selected?.chat_supported && selected.available
    ? "Choose your textbook and use the number printed on the page—not the PDF page number."
    : selected?.chat_supported
    ? "Choose a textbook and ask your question. Its text and printed-page index will be prepared on demand."
    : "Choose a processed Class 1–8 textbook to start a chat.";
  bubble.append(first, second);
  content.append(author, bubble);
  intro.append(avatar, content);
  messageList.append(intro);

  const suggestions = document.createElement("div");
  suggestions.className = "suggestions";
  suggestions.setAttribute("aria-label", "Example questions");
  for (const suggestionText of ["Explain a tricky word", "Help me understand a lesson", "What can you see on this page?"]) {
    const suggestion = document.createElement("button");
    suggestion.type = "button";
    suggestion.className = "suggestion";
    suggestion.textContent = suggestionText;
    suggestion.disabled = !selected?.chat_supported
      || (!selected.available && !selected.corpus_books?.length);
    suggestions.append(suggestion);
  }
  messageList.append(suggestions);
  clearError();
}

function resizeQuestionBox() {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 125)}px`;
}

async function requestJson(path, options = {}) {
  const response = await fetch(path, options);
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error("The local tutor returned an unreadable response.");
  }
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
  return data;
}

async function sendQuestion(question) {
  if (isSending || isProcessing) return;
  const selected = selectedClass();
  if (!selected?.chat_supported) {
    showError(`Class ${selected?.class ?? ""} chat is not supported yet.`);
    return;
  }

  if (!selected.available) {
    if (!selected.corpus_books?.length) {
      showError("There are no textbook PDFs available for this class.");
      return;
    }
    if (!bookSelect.value) {
      showError("Choose a textbook first. It will be prepared when you ask your question.");
      bookSelect.focus();
      return;
    }
  }
  const sourceId = bookSelect.value || null;
  const selectedFilename = selected.corpus_books?.find(
    (book) => book.source_id === sourceId,
  )?.filename ?? selected.books.find(
    (book) => book.source_id === sourceId,
  )?.filename ?? "";
  const selectedSource = selected.corpus_books?.find(
    (book) => book.source_id === sourceId && !book.processed,
  );
  const page = pageInput.value.trim() ? Number(pageInput.value) : null;
  if (page !== null && !sourceId) {
    showError("Choose your textbook before entering its printed page number.");
    bookSelect.focus();
    return;
  }
  if (page !== null && (!Number.isSafeInteger(page) || page < 1)) {
    showError("Enter a valid printed page number, or leave it blank.");
    pageInput.focus();
    return;
  }

  clearError();
  addMessage("user", question);
  isSending = true;
  sendButton.disabled = true;
  questionInput.disabled = true;
  const typingMessage = addMessage("assistant", "", {
    typing: true,
    typingText: selectedSource
      ? "Preparing this textbook and its page index…"
      : undefined,
  });

  try {
    const answer = await requestJson("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        class: Number(classSelect.value),
        source_id: sourceId,
        page,
        question,
        history: history.slice(-8),
      }),
    });
    typingMessage.remove();
    addMessage("assistant", answer.answer, {
      sources: answer.sources,
      showImage: answer.visual_context_used,
      images: answer.images,
      warnings: answer.warnings,
      note: answer.images?.length
        ? "The tutor received original textbook images through their prepared image IDs. Image descriptions and object locations are not independently verified."
        : answer.visual_context_used
        ? `${answer.provider === "gemini" ? "Gemini inspected this page image." : "The tutor looked at this page image."} Any object locations in the answer are descriptions, not verified circles or labels.`
        : "",
    });
    history.push({
      role: "user",
      content: question,
      page_context: answer.page_context ?? null,
    });
    history.push({ role: "assistant", content: answer.answer });
    history = history.slice(-8);
    if (answer.sources?.length) {
      const currentBook = selected?.books.find((book) => book.source_id === answer.sources[0].source_id);
      if (currentBook?.text_quality_warning) {
        bookWarning.textContent = currentBook.text_quality_warning;
        bookWarning.hidden = false;
      }
    }
  } catch (error) {
    typingMessage.remove();
    showError(error instanceof Error ? error.message : "The tutor could not answer right now.");
  } finally {
    try {
      await refreshCatalog(selected.class, selectedFilename);
    } catch (error) {
      showError(
        `Could not refresh the textbook catalog: ${
          error instanceof Error ? error.message : "unknown error"
        }`,
      );
    }
    isSending = false;
    updateComposerAvailability();
    questionInput.focus();
  }
}

processBookButton.addEventListener("click", () => {
  void processSelectedBook();
});

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  questionInput.value = "";
  resizeQuestionBox();
  void sendQuestion(question);
});

questionInput.addEventListener("input", resizeQuestionBox);
questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

classSelect.addEventListener("change", () => {
  updateBooks();
  resetConversation();
});
bookSelect.addEventListener("change", () => {
  pageInput.value = "";
  updateSelectedBookStatus();
  resetConversation();
});
clearButton.addEventListener("click", resetConversation);
messageList.addEventListener("click", (event) => {
  const button = event.target.closest(".suggestion");
  if (!button) return;
  questionInput.value = button.textContent;
  resizeQuestionBox();
  questionInput.focus();
});

async function initialize() {
  try {
    const [health, response] = await Promise.all([
      requestJson("/api/health"),
      requestJson("/api/catalog"),
    ]);
    populateCatalog(response);
    connectionStatus.classList.toggle("online", health.ok && health.groq_configured);
    connectionStatus.classList.toggle("offline", !health.groq_configured);
    connectionStatus.lastElementChild.textContent = health.groq_configured
      ? `Tutor ready · ${health.model}`
      : "Add your Groq key to connect";
    if (!health.groq_configured) {
      showError("The demo is running, but no Groq API key is set. Add GROQ_API_KEY to the project .env file and restart the server.");
    }
  } catch (error) {
    connectionStatus.classList.add("offline");
    connectionStatus.lastElementChild.textContent = "Tutor is not available";
    showError(error instanceof Error ? error.message : "Could not connect to the local tutor.");
  }
}

void initialize();
