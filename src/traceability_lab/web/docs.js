// De documentatie gebruikt alleen lokale bestanden en de gegenereerde specificatie.
const response = await fetch("/openapi.json");
const specification = await response.json();
for (const [path, methods] of Object.entries(specification.paths)) {
  for (const [method, operation] of Object.entries(methods)) {
    const row = document.createElement("p");
    row.textContent = `${method.toUpperCase()} ${path} — ${operation.summary || ""}`;
    document.getElementById("routes").append(row);
  }
}
document.getElementById("schemas").textContent = JSON.stringify(specification.components.schemas, null, 2);
