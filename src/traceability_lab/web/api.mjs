// Keep HTTP errors readable even when a proxy or server returns HTML or no body.
export async function api(path, body, method = "POST") {
  const response = await fetch(`/api${path}`, body === undefined ? {} : {
    method, headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
  });
  const text = await response.text();
  let result;
  try { result = text ? JSON.parse(text) : null; }
  catch {
    const error = new Error(response.ok
      ? `Invalid JSON response (HTTP ${response.status}).`
      : `HTTP ${response.status}${response.statusText ? ` ${response.statusText}` : ""}`);
    error.status = response.status;
    throw error;
  }
  if (!response.ok) {
    const detail = result?.detail;
    const error = new Error(typeof detail === "string" ? detail
      : detail != null ? JSON.stringify(detail) : `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return result;
}
