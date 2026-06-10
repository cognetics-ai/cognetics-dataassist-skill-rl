import { useEffect, useRef, useState } from 'react';
import { apiBaseUrl } from '../lib/api';
import { StreamEvent } from '../types';

interface EventStreamState {
  connected: boolean;
  error: string | null;
}

export function useEventStream(
  runId: string | null,
  onEvent: (event: StreamEvent) => void,
  reconnectKey = 0,
): EventStreamState {
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!runId) {
      setConnected(false);
      setError(null);
      sourceRef.current?.close();
      sourceRef.current = null;
      return;
    }

    const source = new EventSource(`${apiBaseUrl}/events/stream?run_id=${encodeURIComponent(runId)}`);
    sourceRef.current = source;

    source.onopen = () => {
      setConnected(true);
      setError(null);
    };

    source.onmessage = (rawEvent) => {
      try {
        const parsed = JSON.parse(rawEvent.data) as StreamEvent;
        onEvent(parsed);
      } catch (err) {
        setError(`Failed to parse SSE event: ${(err as Error).message}`);
      }
    };

    source.onerror = () => {
      setConnected(false);
      setError('Event stream connection dropped');
      source.close();
    };

    return () => {
      source.close();
    };
  }, [runId, onEvent, reconnectKey]);

  return { connected, error };
}
