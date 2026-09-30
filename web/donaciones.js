const $ = (id) => document.getElementById(id);

function params() {
  const data = new FormData($("filters"));
  const query = new URLSearchParams();
  for (const [key, value] of data.entries()) {
    const text = String(value).trim();
    if (!text || ((key === "min" || key === "max") && Number(text) === 0)) continue;
    query.set(key, text);
  }
  query.set("top", "12");
  return query;
}

function bars(id, rows, countKey = "cantidad") {
  const root = $(id);
  root.innerHTML = (rows || []).map((row) => `
    <div class="bar">
      <span>${escapeHtml(row.etiqueta)}</span>
      <i style="width:${Math.max(row.pct, 2)}%"></i>
      <em>${row[countKey] ?? row.cantidad} · ${escapeHtml(row.suma_txt)}</em>
    </div>
  `).join("") || "<p>Sin datos.</p>";
}

function ranks(id, rows) {
  $(id).innerHTML = (rows || []).map((row) =>
    `<li><strong>${escapeHtml(row.nombre)}</strong><br><small>${row.cantidad} don. · ${escapeHtml(row.suma)}</small></li>`
  ).join("") || "<li>Sin datos.</li>";
}

function table(id, items) {
  if (!items || !items.length) {
    $(id).innerHTML = "<p>No hay donaciones para este filtro.</p>";
    return;
  }
  $(id).innerHTML = `
    <table>
      <thead><tr><th>Usuario</th><th>Fecha</th><th>Monto</th><th>Mensaje</th></tr></thead>
      <tbody>
        ${items.map((row) => `
          <tr>
            <td>${escapeHtml(row.nombre)}</td>
            <td>${escapeHtml(row.fecha)}</td>
            <td class="monto">${escapeHtml(row.monto)}</td>
            <td class="msg">${escapeHtml(row.mensaje)}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function load() {
  const res = await fetch(`/api/reporte?${params()}`);
  const data = await res.json();
  const est = data.estado || {};
  $("status").textContent = est.loading
    ? `Leyendo… ${est.progress || 0}`
    : `${data.cantidad} donaciones precargadas`;
  $("reload").disabled = Boolean(est.loading);

  $("kpis").innerHTML = (data.kpis || []).map((kpi) =>
    `<div class="kpi"><span>${escapeHtml(kpi.label)}</span><strong>${escapeHtml(kpi.value)}</strong></div>`
  ).join("");

  $("insights").innerHTML = (data.insights || []).map((note) => `<li>${escapeHtml(note)}</li>`).join("");

  const det = data.detalle || {};
  $("detalle").innerHTML = [
    ["Mínimo", det.minimo],
    ["Máximo", det.maximo],
    ["Desvío", det.desvio],
    ["P10", det.p10],
    ["P90", det.p90],
    ["P99", det.p99],
    ["Recurrentes", det.recurrentes],
    ["Una vez", det.una_vez],
    ["Privadas", det.privadas],
    ["Con link", det.con_link],
    ["Outliers", det.outliers],
    ["Pareto 80%", det.pareto],
  ].map(([k, v]) => `<div><dt>${k}</dt><dd>${escapeHtml(v)}</dd></div>`).join("");

  bars("rangos", data.rangos);
  bars("periodos", data.periodos);
  bars("frecuencia", data.frecuencia);
  ranks("top-monto", data.top_monto);
  ranks("top-cantidad", data.top_cantidad);
  table("mayores", data.mayores);
  const total = data.tabla?.total || data.cantidad || 0;
  $("pageinfo").textContent = total ? `${total} filas` : "0";
  const query = params().toString();
  $("open-listado").href = `/listado?${query}`;
  $("open-listado").textContent = total ? `Ver las ${total}` : "Ver listado";
  $("csv").href = `/api/listado.csv?${query}`;
  if (est.loading) setTimeout(load, 1200);
}

$("filters").addEventListener("submit", (event) => {
  event.preventDefault();
  load();
});
$("clear").addEventListener("click", () => {
  $("filters").reset();
  load();
});
$("reload").addEventListener("click", async () => {
  $("reload").disabled = true;
  await fetch("/api/cargar", { method: "POST" });
  load();
});

load();
