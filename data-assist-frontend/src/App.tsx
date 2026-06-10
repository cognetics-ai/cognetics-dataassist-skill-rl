import { FormEvent, useCallback, useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { EventTimeline } from './components/EventTimeline';
import { ProgressRing } from './components/ProgressRing';
import { Stepper } from './components/Stepper';
import { useEventStream } from './hooks/useEventStream';
import { cancelQuery, draftSql, getMe, getResults, listRunHistory, login, runQuery, validateSql } from './lib/api';
import { QueryRunHistoryItem, ResultsResponse, StatusType, StreamEvent, ValidateResponse } from './types';

const MAX_EVENTS = 200;
type TabName = 'results' | 'explain' | 'warnings' | 'plan';

export function App() {
  const [authSoedId, setAuthSoedId] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [authError, setAuthError] = useState<string | null>(null);
  const [activeSoedId, setActiveSoedId] = useState<string | null>(null);

  const [datasource, setDatasource] = useState('starburst');
  const [inputMode, setInputMode] = useState<'auto' | 'sql' | 'nl'>('auto');
  const [search, setSearch] = useState('');
  const [prompt, setPrompt] = useState('');
  const [showContext, setShowContext] = useState(false);
  const [editable, setEditable] = useState(false);
  const [draftSqlText, setDraftSqlText] = useState('');
  const [draftWarnings, setDraftWarnings] = useState<string[]>([]);
  const [contextRefs, setContextRefs] = useState<Array<Record<string, unknown>>>([]);
  const [validation, setValidation] = useState<ValidateResponse | null>(null);
  const [currentStep, setCurrentStep] = useState('Building context');
  const [runId, setRunId] = useState<string | null>(null);
  const [liveStatus, setLiveStatus] = useState<StatusType>('idle');
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [progress, setProgress] = useState(0);
  const [engineState, setEngineState] = useState('Idle');
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);
  const [activeTab, setActiveTab] = useState<TabName>('results');
  const [selectedPastQueryId, setSelectedPastQueryId] = useState<string | null>(null);
  const [eventStreamVersion, setEventStreamVersion] = useState(0);

  const loginMutation = useMutation({
    mutationFn: async ({ soedId, password }: { soedId: string; password: string }) => {
      if (!soedId.trim()) {
        throw new Error('SOEID is required');
      }
      if (!password.trim()) {
        throw new Error('Password is required');
      }
      return login(soedId, password);
    },
  });

  const meQuery = useQuery({
    queryKey: ['me', activeSoedId],
    queryFn: () => getMe(activeSoedId as string),
    enabled: Boolean(activeSoedId),
  });

  const runHistoryQuery = useQuery({
    queryKey: ['run-history', activeSoedId],
    queryFn: () => listRunHistory(activeSoedId as string, 150),
    enabled: Boolean(activeSoedId),
    refetchInterval: 6000,
  });

  const resultsQuery = useQuery({
    queryKey: ['results', runId],
    queryFn: () => getResults(runId as string),
    enabled: Boolean(runId),
    refetchInterval: (query) => {
      const data = query.state.data as ResultsResponse | undefined;
      if (!data) {
        return 1200;
      }
      return ['succeeded', 'failed', 'cancelled'].includes(data.status) ? false : 1200;
    },
  });

  useEffect(() => {
    if (resultsQuery.data?.status) {
      setLiveStatus(resultsQuery.data.status);
      if (resultsQuery.data.status === 'succeeded') {
        setProgress(100);
      }
    }
  }, [resultsQuery.data?.status]);

  useEffect(() => {
    if (!activeSoedId) {
      return;
    }
    void runHistoryQuery.refetch();
  }, [activeSoedId, historyRefreshKey, runHistoryQuery.refetch]);

  const validateMutation = useMutation({ mutationFn: validateSql });
  const draftMutation = useMutation({ mutationFn: draftSql });
  const runMutation = useMutation({ mutationFn: runQuery });
  const cancelMutation = useMutation({ mutationFn: cancelQuery });

  const historyRuns = runHistoryQuery.data?.runs || [];
  const filteredRuns = useMemo(() => {
    const needle = search.trim().toLowerCase();
    if (!needle) {
      return historyRuns;
    }
    return historyRuns.filter((run) => {
      const corpus = `${run.natural_language_query || ''} ${run.submitted_text} ${run.submitted_sql || ''} ${run.submitted_prompt || ''} ${run.final_sql || ''}`.toLowerCase();
      return corpus.includes(needle);
    });
  }, [historyRuns, search]);
  const selectedPastRun = useMemo(
    () => historyRuns.find((run) => run.run_id === selectedPastQueryId) || null,
    [historyRuns, selectedPastQueryId],
  );

  const onEvent = useCallback((event: StreamEvent) => {
    setEvents((prev) => [event, ...prev].slice(0, MAX_EVENTS));

    if (event.event_type === 'ENGINE_STATE') {
      const state = String(event.payload.state || '').toUpperCase();
      if (state) {
        setEngineState(state);
        setLiveStatus(mapEngineStateToStatus(state));
      }

      const progressValue = parseProgressValue(
        event.payload.progressPercentage ?? ((event.payload.stats as Record<string, unknown> | undefined)?.progressPercentage as unknown),
      );
      if (state === 'RUNNING' && progressValue !== null) {
        setProgress(progressValue);
      } else if (state === 'FINISHED') {
        setProgress(100);
      }
    }
    if (event.event_type === 'RUN_SUCCEEDED') {
      setLiveStatus('succeeded');
      setProgress(100);
      setEngineState('FINISHED');
      setActiveTab('results');
      setHistoryRefreshKey((prev) => prev + 1);
    }
    if (event.event_type === 'RUN_FAILED') {
      setLiveStatus('failed');
      setEngineState('FAILED');
      setActiveTab('warnings');
      setHistoryRefreshKey((prev) => prev + 1);
    }
    if (event.event_type === 'RUN_CANCELLED') {
      setLiveStatus('cancelled');
      setEngineState('CANCELLED');
      setHistoryRefreshKey((prev) => prev + 1);
    }
  }, []);

  const streamState = useEventStream(runId, onEvent, eventStreamVersion);

  const directSqlMode = inputMode === 'sql' || (inputMode === 'auto' && looksLikeSql(prompt));

  async function handleSubmit() {
    if (!activeSoedId) {
      return;
    }

    const promptText = prompt.trim();
    if (!promptText && !draftSqlText.trim()) {
      return;
    }

    const selectedPastRunSql = (selectedPastRun?.final_sql || selectedPastRun?.submitted_sql || '').trim();
    const loadedSuccessfulRun = Boolean(
      selectedPastRun?.status === 'succeeded' &&
        selectedPastRunSql &&
        draftSqlText.trim() === selectedPastRunSql,
    );
    if (loadedSuccessfulRun) {
      setCurrentStep('Ready to run');
      setProgress(100);
      setLiveStatus('succeeded');
      setEngineState('FINISHED');
      setActiveTab('results');
      return;
    }

    setEvents([]);
    setValidation(null);
    setRunId(null);
    setEventStreamVersion((prev) => prev + 1);
    setLiveStatus('idle');

    if (directSqlMode) {
      const sqlText = promptText || draftSqlText.trim();
      setDraftSqlText(sqlText);
      setDraftWarnings([]);
      setContextRefs([]);
      setCurrentStep('Validating');
      setProgress(25);

      const validated = await validateMutation.mutateAsync({
        soeid: activeSoedId,
        sql: sqlText,
        engine: datasource,
      });
      setValidation(validated);
      setCurrentStep('Ready to run');
      setProgress(validated.is_valid ? 70 : 30);
      setActiveTab(validated.is_valid ? 'explain' : 'warnings');
      return;
    }

    setCurrentStep('Building context');
    setProgress(5);
    try {
      setCurrentStep('Generating SQL');
      const draft = await draftMutation.mutateAsync({
        soeid: activeSoedId,
        prompt: promptText,
        engine_preference: datasource,
      });
      setDraftSqlText(draft.draft_sql);
      setDraftWarnings(draft.warnings || []);
      setContextRefs(draft.context_refs || []);

      setCurrentStep('Validating');
      setProgress(20);
      const validated = await validateMutation.mutateAsync({
        soeid: activeSoedId,
        sql: draft.draft_sql,
        engine: datasource,
      });
      setValidation(validated);

      setCurrentStep('Optimizing');
      setProgress(45);
      await new Promise((resolve) => setTimeout(resolve, 350));
      setCurrentStep('Ready to run');
      setProgress(validated.is_valid ? 70 : 30);
      setActiveTab(validated.is_valid ? 'explain' : 'warnings');
    } catch {
      setLiveStatus('failed');
      setCurrentStep('Validating');
    }
  }

  async function handleRunQuery() {
    if (!activeSoedId) {
      return;
    }

    const promptText = prompt.trim();
    const draftText = draftSqlText.trim();
    const sqlForRun = draftText || (directSqlMode ? promptText : '');
    const promptForRun = !directSqlMode ? promptText : '';
    const rerunExistingRunId = canRunSelectedPastRun ? selectedPastRun?.run_id : undefined;

    if (!sqlForRun && !promptForRun) {
      return;
    }

    setLiveStatus('queued');
    setProgress(0);
    setEngineState('QUEUED');
    setEvents([]);

    const run = await runMutation.mutateAsync({
      soeid: activeSoedId,
      run_id: rerunExistingRunId,
      sql: sqlForRun || undefined,
      prompt: (rerunExistingRunId ? selectedPastRun?.natural_language_query || selectedPastRun?.submitted_prompt || promptForRun : promptForRun) || undefined,
      engine: rerunExistingRunId ? selectedPastRun?.engine || datasource : datasource,
      input_mode: sqlForRun ? 'sql' : 'nl',
    });
    setRunId(run.run_id);
    setEventStreamVersion((prev) => prev + 1);
    if (run.run_id === runId) {
      void resultsQuery.refetch();
    }
    setActiveTab('results');
    void runHistoryQuery.refetch();
  }

  async function handleCancelQuery() {
    if (!runId || !activeSoedId) {
      return;
    }
    await cancelMutation.mutateAsync({ soeid: activeSoedId, run_id: runId });
  }

  function handleNewQuery() {
    setSelectedPastQueryId(null);
    setPrompt('');
    setDraftSqlText('');
    setDraftWarnings([]);
    setContextRefs([]);
    setValidation(null);
    setRunId(null);
    setEvents([]);
    setLiveStatus('idle');
    setEngineState('Idle');
    setProgress(0);
    setCurrentStep('Building context');
    setActiveTab('results');
    setInputMode('auto');
  }

  function handlePromptChange(nextValue: string) {
    setPrompt(nextValue);

    const normalizedNext = nextValue.trim();
    const normalizedDraft = draftSqlText.trim();
    const hasDivergedFromDraft = Boolean(normalizedNext && normalizedDraft && normalizedNext !== normalizedDraft);
    if (hasDivergedFromDraft) {
      setDraftSqlText('');
      setDraftWarnings([]);
      setContextRefs([]);
      setValidation(null);
      setCurrentStep('Building context');
      setProgress(0);
      setRunId(null);
      setEvents([]);
      setLiveStatus('idle');
      setEngineState('Idle');
    }
  }

  function handleDraftSqlChange(nextValue: string) {
    setDraftSqlText(nextValue);
    setSelectedPastQueryId(null);
    setValidation(null);
    setRunId(null);
    setEvents([]);
    setLiveStatus('idle');
    setEngineState('Idle');
    setProgress(0);
    setCurrentStep('Validating');
  }

  function loadPastRun(run: QueryRunHistoryItem) {
    if (!activeSoedId) {
      return;
    }

    setSelectedPastQueryId(run.run_id);
    setPrompt(run.natural_language_query || run.submitted_prompt || run.submitted_text || '');
    setDraftSqlText(run.final_sql || run.submitted_sql || '');
    setDatasource(run.engine || datasource);
    setDraftWarnings([]);
    setContextRefs([]);
    setValidation(null);
    const mode = (run.input_mode || 'auto').toLowerCase();
    setInputMode(mode === 'sql' || mode === 'nl' || mode === 'auto' ? mode : 'auto');
    setCurrentStep('Ready to run');
    setRunId(run.run_id);
    setLiveStatus(run.status);
    setEngineState((run.route_mode || run.status || 'Idle').toUpperCase());
    setProgress(run.status === 'succeeded' ? 100 : run.status === 'running' ? 40 : 0);
    setActiveTab(run.status === 'failed' ? 'warnings' : 'results');
  }

  async function handleSignIn(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setAuthError(null);
    try {
      const me = await loginMutation.mutateAsync({ soedId: authSoedId.trim(), password: authPassword });
      setActiveSoedId(me.soeid);
      setAuthPassword('');
    } catch (err) {
      setAuthError((err as Error).message || 'Sign in failed');
    }
  }

  if (!activeSoedId) {
    return <SignInPage soedId={authSoedId} password={authPassword} setSoedId={setAuthSoedId} setPassword={setAuthPassword} onSubmit={handleSignIn} error={authError} pending={loginMutation.isPending} />;
  }

  const validationMessages = validation?.policy_findings?.map((item) => String(item.message || 'Validation finding')).filter(Boolean) || [];
  const combinedWarnings = [...draftWarnings, ...validationMessages, ...(validation?.fixes || [])];
  const latestEventText = events[0]
    ? String(events[0].payload?.message || events[0].payload?.text || events[0].payload?.state || events[0].event_type)
    : `Waiting for activity... (${engineState})`;

  const statusBadgeClass = `status-badge ${liveStatus}`;
  const hasInputForRun = Boolean(draftSqlText.trim() || prompt.trim());
  const selectedPastRunSql = (selectedPastRun?.final_sql || selectedPastRun?.submitted_sql || '').trim();
  const canRunSelectedPastRun = Boolean(
    selectedPastRun?.status === 'succeeded' &&
      selectedPastRunSql &&
      draftSqlText.trim() === selectedPastRunSql,
  );
  const canRun = draftSqlText.trim() ? Boolean(validation?.is_valid || canRunSelectedPastRun) : hasInputForRun;
  const runInProgress = liveStatus === 'running' || liveStatus === 'queued';

  return (
    <div className="app-shell">
      <header className="topbar glass">
        <div className="brand-wrap">
          <h1>Hermes</h1>
          <p>Data Assist for enterprise SQL intelligence</p>
        </div>

        <div className="topbar-controls">
          <label className="field-inline">
            <span>Datasource</span>
            <select value={datasource} onChange={(event) => setDatasource(event.target.value)}>
              <option value="starburst">Starburst</option>
              <option value="trino">Trino</option>
              <option value="mock">Mock</option>
            </select>
          </label>

          <label className="field-inline">
            <span>Run mode</span>
            <select value={inputMode} onChange={(event) => setInputMode(event.target.value as 'auto' | 'sql' | 'nl')}>
              <option value="auto">Auto route</option>
              <option value="sql">Direct SQL</option>
              <option value="nl">Natural language workflow</option>
            </select>
          </label>

          <div className="profile-chip">
            <strong>{meQuery.data?.display_name || activeSoedId}</strong>
            <span>{meQuery.data?.job_title || 'Job title unavailable'}</span>
            <span>
              {meQuery.data?.soeid || activeSoedId} · {meQuery.data?.role || 'role'}
            </span>
          </div>

          <div className={statusBadgeClass}>{liveStatus}</div>
        </div>
      </header>

      <main className="layout-grid">
        <aside className="left-panel glass">
          <button className="new-query-btn" onClick={handleNewQuery}>
            New Query
          </button>

          <div className="search-wrap">
            <input type="text" placeholder="Search past queries" value={search} onChange={(event) => setSearch(event.target.value)} />
          </div>

          <div className="history-list">
            {filteredRuns.map((run) => (
              <button key={run.run_id} className={`history-item ${selectedPastQueryId === run.run_id ? 'active' : ''}`} onClick={() => loadPastRun(run)}>
                <div className="history-top">
                  <span className={`dot ${run.status}`} />
                  <span>{new Date(run.query_start_time || run.created_at).toLocaleString()}</span>
                </div>
                <strong>{summarizeRun(run)}</strong>
                <p>{run.engine} · {run.status} · rows {run.row_count}</p>
              </button>
            ))}
            {filteredRuns.length === 0 ? <p className="history-empty">No runs yet for this user.</p> : null}
          </div>
        </aside>

        <section className="workspace glass">
          <div className="ask-row">
            <textarea value={prompt} onChange={(event) => handlePromptChange(event.target.value)} placeholder="Ask in natural language, or paste direct SQL..." />
          </div>
          <div className="submit-row">
            <button className="primary" onClick={handleSubmit} disabled={draftMutation.isPending || validateMutation.isPending}>
              {draftMutation.isPending || validateMutation.isPending ? 'Submitting...' : 'Submit'}
            </button>
          </div>

          <Stepper currentStep={currentStep} directSqlMode={directSqlMode} />

          <div className="toggle-row">
            <label>
              <input type="checkbox" checked={showContext} onChange={() => setShowContext((prev) => !prev)} /> Show context used
            </label>
            <label>
              <input type="checkbox" checked={editable} onChange={() => setEditable((prev) => !prev)} /> Edit SQL
            </label>
          </div>

          {showContext ? (
            <div className="context-panel">
              {contextRefs.length === 0 ? <p>No context references for this request.</p> : null}
              {contextRefs.map((ref, index) => (
                <pre key={`${index}-${String(ref.query_id || 'ref')}`}>{JSON.stringify(ref, null, 2)}</pre>
              ))}
            </div>
          ) : null}

          <div className="sql-card">
            <div className="sql-card-head">
              <h3>SQL Draft</h3>
              <div className="sql-actions">
                <button className="primary" disabled={!canRun || runMutation.isPending || runInProgress} onClick={handleRunQuery}>
                  Run
                </button>
                <button className="ghost" disabled={!runInProgress} onClick={handleCancelQuery}>
                  Cancel
                </button>
              </div>
            </div>

            {editable ? (
              <textarea className="sql-editor" value={draftSqlText} onChange={(event) => handleDraftSqlChange(event.target.value)} />
            ) : (
              <pre className="sql-preview">{draftSqlText || '-- SQL appears here after submit --'}</pre>
            )}

            {combinedWarnings.length > 0 ? (
              <div className="warnings-inline">
                {combinedWarnings.map((warning, index) => (
                  <p key={`${warning}-${index}`}>{warning}</p>
                ))}
              </div>
            ) : null}
          </div>

          <div className="progress-grid">
            <div className="progress-card">
              <ProgressRing value={progress} label={runInProgress ? engineState : 'Readiness'} />
              <p className="now-doing">Now doing: {latestEventText}</p>
              <p className="stream-status">SSE: {streamState.connected ? 'Connected' : streamState.error || 'Idle'}</p>
            </div>
            <EventTimeline events={events} />
          </div>

          <div className="tabs">
            <div className="tab-headers">
              <button className={activeTab === 'results' ? 'active' : ''} onClick={() => setActiveTab('results')}>
                Results
              </button>
              <button className={activeTab === 'explain' ? 'active' : ''} onClick={() => setActiveTab('explain')}>
                Explain Summary
              </button>
              <button className={activeTab === 'warnings' ? 'active' : ''} onClick={() => setActiveTab('warnings')}>
                Warnings / Guardrails
              </button>
              <button className={activeTab === 'plan' ? 'active' : ''} onClick={() => setActiveTab('plan')}>
                Query Plan
              </button>
            </div>

            <div className="tab-body">
              {activeTab === 'results' ? <ResultsTable data={resultsQuery.data} /> : null}
              {activeTab === 'explain' ? <pre>{JSON.stringify(validation?.explain_summary || {}, null, 2)}</pre> : null}
              {activeTab === 'warnings' ? (
                <div>
                  {combinedWarnings.length === 0 ? <p>No warnings.</p> : null}
                  {combinedWarnings.map((warning, index) => (
                    <p key={`${warning}-${index}`}>{warning}</p>
                  ))}
                </div>
              ) : null}
              {activeTab === 'plan' ? <p>Query Plan visualization is coming in a later iteration.</p> : null}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

function SignInPage({
  soedId,
  password,
  setSoedId,
  setPassword,
  onSubmit,
  error,
  pending,
}: {
  soedId: string;
  password: string;
  setSoedId: (value: string) => void;
  setPassword: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  error: string | null;
  pending: boolean;
}) {
  return (
    <div className="signin-shell">
      <div className="signin-hero">
        <h2>Welcome to Hermes</h2>
        <p>Enterprise SQL intelligence with guarded execution and live Starburst telemetry.</p>
      </div>
      <div className="signin-card">
        <div className="signin-brand">
          <h1>Hermes</h1>
          <p>Secure enterprise data access</p>
        </div>
        <form className="signin-form" onSubmit={onSubmit}>
          <label>
            <span>SOEID</span>
            <input value={soedId} onChange={(event) => setSoedId(event.target.value)} placeholder="Enter SOEID" autoComplete="username" />
          </label>
          <label>
            <span>Password</span>
            <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} placeholder="Enter Password" autoComplete="current-password" />
          </label>
          <p className="signin-note">Temporary non-SSO password for testing is `test`.</p>
          {error ? <p className="signin-error">{error}</p> : null}
          <button type="submit" className="signin-btn" disabled={pending}>
            {pending ? 'Signing In...' : 'Sign In'}
          </button>
        </form>
      </div>
    </div>
  );
}

function ResultsTable({ data }: { data: ResultsResponse | undefined }) {
  if (!data) {
    return <p>No results yet.</p>;
  }

  if (data.status === 'failed') {
    return <p>Execution failed: {data.error_message || 'Unknown error'}</p>;
  }

  if (data.status === 'cancelled') {
    return <p>Execution cancelled.</p>;
  }

  if (!data.schema.length) {
    return <p>Waiting for result rows...</p>;
  }

  return (
    <div className="results-wrap">
      <table>
        <thead>
          <tr>
            {data.schema.map((col) => (
              <th key={`${col.name}-${col.type}`}>{col.name}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row, rowIndex) => (
            <tr key={`row-${rowIndex}`}>
              {row.map((cell, cellIndex) => (
                <td key={`cell-${rowIndex}-${cellIndex}`}>{String(cell)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function looksLikeSql(value: string): boolean {
  const text = value.trim();
  if (!text) {
    return false;
  }
  return /^(select|with|insert|update|delete|merge|create|drop|alter|explain)\b/i.test(text);
}

function parseProgressValue(value: unknown): number | null {
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return null;
  }
  return Math.max(0, Math.min(100, Math.round(numeric)));
}

function mapEngineStateToStatus(state: string): StatusType {
  if (state === 'QUEUED' || state === 'PLANNING') {
    return 'queued';
  }
  if (state === 'RUNNING' || state === 'STARTING' || state === 'FINISHING') {
    return 'running';
  }
  if (state === 'FINISHED') {
    return 'succeeded';
  }
  if (state === 'FAILED') {
    return 'failed';
  }
  if (state === 'CANCELLED') {
    return 'cancelled';
  }
  return 'idle';
}

function summarizeRun(run: QueryRunHistoryItem): string {
  const text = (run.natural_language_query || run.submitted_prompt || run.submitted_text || run.submitted_sql || '').trim();
  if (!text) {
    return run.run_id;
  }
  return text.length > 80 ? `${text.slice(0, 80)}...` : text;
}
