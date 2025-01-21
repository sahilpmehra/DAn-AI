import { writable } from "svelte/store";

type AnalysisConfig = {
  isConfigured: boolean;
  decision?: "accept" | "reject" | "customize";
  selectedKeyVars: string[];
  selectedProbVars: string[];
};

const defaultConfig: AnalysisConfig = {
  isConfigured: false,
  decision: undefined,
  selectedKeyVars: [],
  selectedProbVars: [],
};

// Create the store with default values
export const analysisConfigStore = writable<AnalysisConfig>(defaultConfig);
