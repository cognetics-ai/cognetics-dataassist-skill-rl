const STEPS = ['Building context', 'Generating SQL', 'Validating', 'Optimizing', 'Ready to run'];

interface StepperProps {
  currentStep: string;
  directSqlMode?: boolean;
}

export function Stepper({ currentStep, directSqlMode = false }: StepperProps) {
  const currentIndex = Math.max(0, STEPS.findIndex((step) => step === currentStep));

  return (
    <div className="stepper">
      {STEPS.map((step, index) => {
        const isSkipped = directSqlMode && index < 2;
        const status = isSkipped ? 'skipped' : index < currentIndex ? 'done' : index === currentIndex ? 'active' : 'todo';
        return (
          <div key={step} className={`step ${status}`}>
            <span className="step-dot" />
            <span>{isSkipped ? `${step} (Skipped)` : step}</span>
          </div>
        );
      })}
    </div>
  );
}
