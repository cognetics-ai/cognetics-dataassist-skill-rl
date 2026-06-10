import { StreamEvent } from '../types';

interface EventTimelineProps {
  events: StreamEvent[];
}

function describeEvent(event: StreamEvent): string {
  const payload = event.payload || {};
  if (typeof payload.message === 'string') {
    return payload.message;
  }
  if (typeof payload.state === 'string') {
    return `Engine state: ${payload.state}`;
  }
  if (typeof payload.text === 'string') {
    return payload.text;
  }
  return event.event_type;
}

export function EventTimeline({ events }: EventTimelineProps) {
  return (
    <div className="timeline">
      {events.length === 0 ? <div className="timeline-empty">No events yet</div> : null}
      {events.map((event) => (
        <div className="timeline-item" key={event.event_id}>
          <div className="timeline-header">
            <strong>{event.event_type}</strong>
            <span>{new Date(event.timestamp).toLocaleTimeString()}</span>
          </div>
          <p>{describeEvent(event)}</p>
        </div>
      ))}
    </div>
  );
}
