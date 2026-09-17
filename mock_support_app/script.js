// Mock Support Ticket System
// No backend, no database — pure client-side simulation for Playwright practice.

const form = document.getElementById("ticket-form");
const confirmation = document.getElementById("confirmation");
const errorBox = document.getElementById("error");
const resetButton = document.getElementById("reset-form");

// Maps the <select> value to the human-readable label shown in the confirmation.
const CATEGORY_LABELS = {
  account_access: "Account Access",
  hardware: "Hardware",
  software: "Software",
  network: "Network",
};

function generateTicketId() {
  // 5-digit numeric ID, e.g. 10000–99999. Stable format for test assertions
  // like expect(ticketId).toMatch(/^\d{5}$/).
  return String(Math.floor(10000 + Math.random() * 90000));
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
  confirmation.classList.add("hidden");
}

function clearError() {
  errorBox.textContent = "";
  errorBox.classList.add("hidden");
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  clearError();

  const issue = document.getElementById("issue").value.trim();
  const category = document.getElementById("category").value;
  const resolution = document.getElementById("resolution").value.trim();

  if (!issue || !category || !resolution) {
    showError("Please fill out all fields before submitting.");
    return;
  }

  const ticketId = generateTicketId();

  document.getElementById("ticket-id").textContent = ticketId;
  document.getElementById("ticket-category").textContent =
    CATEGORY_LABELS[category] || category;
  document.getElementById("ticket-resolution").textContent = resolution;

  form.classList.add("hidden");
  confirmation.classList.remove("hidden");
});

resetButton.addEventListener("click", () => {
  form.reset();
  form.classList.remove("hidden");
  confirmation.classList.add("hidden");
  clearError();
});
