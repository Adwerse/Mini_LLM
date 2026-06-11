const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    throw new Error(readErrorMessage(payload, response.status));
  }

  return payload;
}

function readErrorMessage(payload, status) {
  if (typeof payload === "string" && payload.trim()) {
    return payload;
  }

  if (payload?.detail) {
    if (Array.isArray(payload.detail)) {
      return payload.detail
        .map((item) => {
          const location = Array.isArray(item.loc) ? item.loc.join(".") : "request";
          return `${location}: ${item.msg}`;
        })
        .join("; ");
    }
    return payload.detail;
  }

  return `Request failed with status ${status}`;
}

export function getHealth() {
  return request("/health");
}

export function getModelInfo() {
  return request("/model/info");
}

export function generateText({ prompt, maxNewTokens, temperature, topP }) {
  return request("/generate", {
    method: "POST",
    body: JSON.stringify({
      prompt,
      max_new_tokens: Number(maxNewTokens),
      temperature: Number(temperature),
      top_p: Number(topP),
    }),
  });
}

export { API_BASE_URL };
