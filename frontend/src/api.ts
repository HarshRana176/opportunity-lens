import type { JobSearchResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export interface JobMatchParams {
  file: File;
  what: string;
  where: string;
  limit: string;
  searchOnline: boolean;
}

export async function fetchJobMatches(params: JobMatchParams): Promise<JobSearchResponse> {
  const formData = new FormData();
  formData.append("file", params.file);
  formData.append("search_online", String(params.searchOnline));
  if (params.what.trim()) formData.append("what", params.what.trim());
  if (params.where.trim()) formData.append("where", params.where.trim());
  if (params.limit.trim()) formData.append("limit", params.limit.trim());

  const response = await fetch(`${API_BASE}/job-matches`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // response wasn't JSON; keep the generic message
    }
    throw new Error(detail);
  }

  return response.json();
}
