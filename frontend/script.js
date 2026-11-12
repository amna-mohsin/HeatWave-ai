const form = document.getElementById("f"), errBox = document.getElementById("err"), box = document.getElementById("result");

function showError(msg) { errBox.textContent = msg; errBox.hidden = false; box.hidden = true; }

function show(r) {
  errBox.hidden = true; box.hidden = false;
  const b = document.getElementById("badge");
  b.textContent = r.risk_band;
  b.className = "badge " + (r.prediction === 1 ? "danger" : r.risk_band === "Elevated" ? "elevated" : "safe");
  document.getElementById("prob").textContent = (r.probability * 100).toFixed(1) + "%";
  document.getElementById("meta").innerHTML =
    `<li>Decision threshold: ${(r.threshold * 100).toFixed(1)}%</li><li>Model: ${r.model}</li>` +
    `<li>Issue date: ${r.date}</li><li>Source: ${r.source}</li><li>Forecast window: days 5&ndash;7, Karachi</li>`;
}

async function call(url, options) {
  try {
    const res = await fetch(url, options);
    const data = await res.json();
    if (!res.ok) {
      const d = data.detail;
      const msg = Array.isArray(d) ? d.map(e => `${e.loc.slice(1).join(".")}: ${e.msg}`).join("; ") : d;
      return showError(msg || "Request failed.");
    }
    show(data);
  } catch (e) { showError("Cannot reach the server. Is uvicorn running?"); }
}

form.addEventListener("submit", ev => {
  ev.preventDefault();
  const body = {};
  for (const [k, v] of new FormData(form)) { if (v !== "") body[k] = Number(v); }
  call("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
});
document.getElementById("live").addEventListener("click", () => call("/predict/live"));
