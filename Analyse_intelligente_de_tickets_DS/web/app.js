// web/app.js
// Dashboard + Add page (safe on both pages)

async function apiGetTickets() {
  const r = await fetch("/api/tickets");
  if (!r.ok) throw new Error("Impossible de charger les tickets");
  const data = await r.json();
  // support: either {tickets:[...]} or [...]
  return Array.isArray(data) ? data : (data.tickets || []);
}

function esc(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function normalizeUrgence(u) {
  const v = (u || "").toLowerCase().trim();
  if (v === "haute") return "Haute";
  if (v === "moyenne") return "Moyenne";
  return "Basse";
}

function badgeColor(cat) {
  // color stable per category string
  const s = (cat || "Autre").trim();
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  const hue = h % 360;
  return `hsl(${hue} 70% 45%)`;
}

function cardHtml(t) {
  const cat = t.categorie || "Autre";
  const color = badgeColor(cat);
  const urgence = normalizeUrgence(t.urgence);
  const statue = (t.statue || "en_attente").trim();

  return `
    <div class="card" data-id="${esc(t.id)}">
      <div class="card-top">
        <div class="card-title">${esc(t.titre)}</div>
        <span class="cat-badge" style="background:${color}">${esc(cat)}</span>
      </div>

      <div class="card-text">${esc(t.texte)}</div>

      <div class="card-meta">
        <span class="pill">${esc(urgence)}</span>
        <span class="pill">${esc(t.type_ticket || "")}</span>
        <span class="pill">Temps: ${esc(t.temps_resolution)} h</span>
        <span class="pill statue ${statue === "termine" ? "s-done" : "s-wait"}">
          ${statue === "termine" ? "Terminé" : "En attente"}
        </span>
      </div>

      <div class="card-actions">
        ${
          statue === "termine"
            ? ""
            : `<button class="btn done" data-action="done">Marquer terminé</button>`
        }
        <button class="btn danger" data-action="delete">Supprimer</button>
      </div>
    </div>
  `;
}

function urgencyOrder(u) {
  // for "same category sort" you asked earlier, but now we group by urgency columns:
  // still useful for consistent ordering inside a column
  const v = normalizeUrgence(u);
  if (v === "Haute") return 0;
  if (v === "Moyenne") return 1;
  return 2;
}

function sortTickets(tickets) {
  // newest first, then urgency
  return [...tickets].sort((a, b) => {
    const da = new Date(a.created_at || 0).getTime();
    const db = new Date(b.created_at || 0).getTime();
    if (db !== da) return db - da;
    return urgencyOrder(a.urgence) - urgencyOrder(b.urgence);
  });
}

function renderDashboard(tickets) {
  const gridHigh = document.getElementById("gridHigh");
  const gridMid = document.getElementById("gridMid");
  const gridLow = document.getElementById("gridLow");
  const doneGrid = document.getElementById("doneGrid");
  const doneCount = document.getElementById("doneCount");

  // If we are not on dashboard page, do nothing safely
  if (!gridHigh || !gridMid || !gridLow || !doneGrid) return;

  const all = sortTickets(tickets);

  const waiting = all.filter(t => (t.statue || "en_attente") !== "termine");
  const done = all.filter(t => (t.statue || "") === "termine");

  const high = waiting.filter(t => normalizeUrgence(t.urgence) === "Haute");
  const mid = waiting.filter(t => normalizeUrgence(t.urgence) === "Moyenne");
  const low = waiting.filter(t => normalizeUrgence(t.urgence) === "Basse");

  gridHigh.innerHTML = high.map(cardHtml).join("") || `<div class="empty">Aucun ticket</div>`;
  gridMid.innerHTML = mid.map(cardHtml).join("") || `<div class="empty">Aucun ticket</div>`;
  gridLow.innerHTML = low.map(cardHtml).join("") || `<div class="empty">Aucun ticket</div>`;

  doneGrid.innerHTML = done.map(cardHtml).join("") || `<div class="empty">Aucun ticket terminé</div>`;
  if (doneCount) doneCount.textContent = String(done.length);
}

async function apiDelete(id) {
  const r = await fetch(`/api/tickets/${encodeURIComponent(id)}`, { method: "DELETE" });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || "Erreur suppression");
  return data;
}

async function apiDone(id) {
  const r = await fetch(`/api/tickets/${encodeURIComponent(id)}/done`, { method: "POST" });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || "Erreur mise à jour");
  return data;
}

function attachDashboardHandlers() {
  const root = document.getElementById("dashboardRoot");
  if (!root) return;

  root.addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-action]");
    if (!btn) return;

    const card = btn.closest(".card");
    if (!card) return;

    const id = card.getAttribute("data-id");
    const action = btn.getAttribute("data-action");

    try {
      if (action === "delete") {
        const ok = confirm("Supprimer ce ticket ? (action irréversible)");
        if (!ok) return;
        await apiDelete(id);
      } else if (action === "done") {
        await apiDone(id);
      } else {
        return;
      }

      // Refresh UI
      const tickets = await apiGetTickets();
      renderDashboard(tickets);

    } catch (err) {
      alert("Erreur: " + err.message);
    }
  });
}

async function initDashboard() {
  const gridHigh = document.getElementById("gridHigh");
  const gridMid = document.getElementById("gridMid");
  const gridLow = document.getElementById("gridLow");
  const doneGrid = document.getElementById("doneGrid");

  // Not dashboard page
  if (!gridHigh || !gridMid || !gridLow || !doneGrid) return;

  try {
    const tickets = await apiGetTickets();
    renderDashboard(tickets);
    attachDashboardHandlers();
  } catch (e) {
    // show a safe error message somewhere if exists
    const errBox = document.getElementById("errorBox");
    if (errBox) errBox.textContent = "Erreur chargement tickets: " + e.message;
    else console.error(e);
  }
}

async function initAddForm() {
  const form = document.getElementById("ticketForm");
  if (!form) return; // Not add page

  const msg = document.getElementById("msg");
  const titreEl = document.getElementById("titre");
  const texteEl = document.getElementById("texte");
  const submitBtn = form.querySelector("button[type='submit']");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const titre = (titreEl?.value || "").trim();
    const texte = (texteEl?.value || "").trim();

    if (!titre || !texte) {
      if (msg) msg.textContent = "❌ Titre et texte sont obligatoires.";
      return;
    }

    if (msg) msg.textContent = "Analyse en cours...";
    if (submitBtn) submitBtn.disabled = true;

    try {
      const r = await fetch("/api/tickets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ titre, texte })
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.error || "Erreur serveur");
      window.location.href = "/";
    } catch (err) {
      if (msg) msg.textContent = "❌ " + err.message;
      if (submitBtn) submitBtn.disabled = false;
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initDashboard();
  initAddForm();
});
