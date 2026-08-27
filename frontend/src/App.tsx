import { useState } from "react";
import type { DragEvent, FormEvent } from "react";
import "./App.css";
import { fetchJobMatches } from "./api";
import type { JobSearchResponse, MatchStatus } from "./types";

type Status = "idle" | "loading" | "success" | "error";

function statusClass(status: MatchStatus): string {
  if (status === "pass") return "badge badge-pass";
  if (status === "fail") return "badge badge-fail";
  if (status === "partial") return "badge badge-unknown";
  return "badge badge-unknown";
}

function componentLabel(name: string): string {
  return name
    .split("_")
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(" ");
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [what, setWhat] = useState("");
  const [where, setWhere] = useState("");
  const [limit, setLimit] = useState("10");
  const [searchOnline, setSearchOnline] = useState(true);
  const [status, setStatus] = useState<Status>("idle");
  const [errorMessage, setErrorMessage] = useState("");
  const [result, setResult] = useState<JobSearchResponse | null>(null);
  const [expandedJobId, setExpandedJobId] = useState<number | null>(null);

  function handleFileChange(selected: File | null) {
    if (selected && selected.type !== "application/pdf") {
      setErrorMessage("Only PDF files are supported.");
      return;
    }
    setErrorMessage("");
    setFile(selected);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    const dropped = event.dataTransfer.files?.[0] ?? null;
    handleFileChange(dropped);
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!file) {
      setErrorMessage("Please select a resume PDF first.");
      return;
    }

    setStatus("loading");
    setErrorMessage("");
    setResult(null);

    try {
      const data = await fetchJobMatches({ file, what, where, limit, searchOnline });
      setResult(data);
      setStatus("success");
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Something went wrong.");
      setStatus("error");
    }
  }

  return (
    <div className="page">
      <header className="header">
        <h1>OpportunityLens</h1>
        <p>Upload a resume, find and rank matching jobs.</p>
      </header>

      <form className="upload-form" onSubmit={handleSubmit}>
        <div
          className={`dropzone${isDragging ? " dropzone-active" : ""}${file ? " dropzone-filled" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => document.getElementById("resume-input")?.click()}
        >
          <input
            id="resume-input"
            type="file"
            accept="application/pdf"
            hidden
            onChange={(e) => handleFileChange(e.target.files?.[0] ?? null)}
          />
          {file ? (
            <p>Selected: <strong>{file.name}</strong></p>
          ) : (
            <p>Drag & drop a resume PDF here, or click to browse</p>
          )}
        </div>

        <div className="controls-grid">
          <label>
            Job role
            <input
              type="text"
              placeholder="e.g. Backend Engineer"
              value={what}
              onChange={(e) => setWhat(e.target.value)}
            />
          </label>
          <label>
            Location
            <input
              type="text"
              placeholder="e.g. Remote, Bengaluru"
              value={where}
              onChange={(e) => setWhere(e.target.value)}
            />
          </label>
          <label>
            Limit
            <input
              type="number"
              min={1}
              value={limit}
              onChange={(e) => setLimit(e.target.value)}
            />
          </label>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={searchOnline}
              onChange={(e) => setSearchOnline(e.target.checked)}
            />
            Search online for new jobs
          </label>
        </div>

        <button type="submit" disabled={status === "loading"}>
          {status === "loading" ? "Matching..." : "Find matching jobs"}
        </button>
      </form>

      {status === "loading" && <p className="status-line">Parsing resume and scoring jobs, this can take a minute...</p>}

      {status === "error" && <p className="status-line status-error">{errorMessage}</p>}

      {status === "success" && result && (
        <section className="results">
          <p className="discovery-line">
            Online discovery: {result.discovery.status}
            {result.discovery.status === "ok" &&
              ` — fetched ${result.discovery.fetched}, added ${result.discovery.newly_ingested}`}
            {result.discovery.detail ? ` (${result.discovery.detail})` : ""}
          </p>

          {result.matches.length === 0 ? (
            <p className="status-line">No matching jobs found yet. Try a different role or location.</p>
          ) : (
            <ul className="job-list">
              {result.matches.map((match) => {
                const expanded = expandedJobId === match.job_id;
                return (
                  <li className="job-card" key={match.job_id}>
                    <div className="job-card-header">
                      <div>
                        <h3>{match.job_title}</h3>
                        <p className="job-meta">
                          {[match.company, match.location].filter(Boolean).join(" · ") || "Details unavailable"}
                        </p>
                      </div>
                      <div className="score-block">
                        <span className="score">{Math.round(match.result.overall_score * 100)}%</span>
                        <span className={statusClass(match.result.evidence.eligibility)}>
                          {match.result.evidence.eligibility}
                        </span>
                      </div>
                    </div>

                    <div className="job-card-actions">
                      <button
                        type="button"
                        className="link-button"
                        onClick={() => setExpandedJobId(expanded ? null : match.job_id)}
                      >
                        {expanded ? "Hide match details" : "View match details"}
                      </button>
                      {match.job_url && (
                        <a className="view-job-button" href={match.job_url} target="_blank" rel="noreferrer">
                          View Job
                        </a>
                      )}
                    </div>

                    {expanded && (
                      <div className="match-details">
                        <p>
                          Skills matched: {match.result.evidence.skills.matched_required}/
                          {match.result.evidence.skills.total_required} required,{" "}
                          {match.result.evidence.skills.matched_preferred}/
                          {match.result.evidence.skills.total_preferred} preferred
                        </p>
                        <table>
                          <thead>
                            <tr>
                              <th>Component</th>
                              <th>Status</th>
                              <th>Contribution</th>
                            </tr>
                          </thead>
                          <tbody>
                            {match.result.components.map((component) => (
                              <tr key={component.name}>
                                <td>{componentLabel(component.name)}</td>
                                <td>
                                  <span className={statusClass(component.status)}>{component.status}</span>
                                </td>
                                <td>{component.contribution.toFixed(2)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        {match.result.evidence.unresolved_notes.length > 0 && (
                          <ul className="notes">
                            {match.result.evidence.unresolved_notes.map((note, idx) => (
                              <li key={idx}>{note}</li>
                            ))}
                          </ul>
                        )}
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </section>
      )}
    </div>
  );
}

export default App;
