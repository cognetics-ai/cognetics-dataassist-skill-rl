interface ProgressRingProps {
  value: number;
  label: string;
}

export function ProgressRing({ value, label }: ProgressRingProps) {
  const clamped = Math.max(0, Math.min(100, value));
  const radius = 52;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (clamped / 100) * circumference;

  return (
    <div className="ring-wrap">
      <svg className="ring" viewBox="0 0 140 140" role="img" aria-label={`Progress ${clamped}%`}>
        <circle className="ring-track" cx="70" cy="70" r={radius} />
        <circle className="ring-bar" cx="70" cy="70" r={radius} strokeDasharray={circumference} strokeDashoffset={offset} />
      </svg>
      <div className="ring-center">
        <strong>{clamped}%</strong>
        <span>{label}</span>
      </div>
    </div>
  );
}
