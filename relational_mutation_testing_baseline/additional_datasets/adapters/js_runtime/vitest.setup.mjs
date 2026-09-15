// Registers the jest-dom matchers (toBeInTheDocument, toHaveValue, ...) that the
// RefactorBench-JS holdout suites assume are available. Without this, a React fixture
// fails with "Invalid Chai property: toBeInTheDocument" on nearly every case —
// 89 of travel_itinerary_planner's 105 failures were exactly this.
import '@testing-library/jest-dom/vitest';
