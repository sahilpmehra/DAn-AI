import { writable } from "svelte/store";

export const sessionId = writable<string | null>(null);
export const isFileUploaded = writable<boolean>(false);

// Define types
export interface TableData {
  headers: string[];
  data: Record<string, any>[];
}

export interface StatsData {
  headers: string[];
  data: Record<string, any>[];
}

export interface AnalysisData {
  summary: string;
  keyVariables: string[];
  problematicVariables: string[];
}

// Create stores
export const tableDataStore = writable<TableData>({ headers: [], data: [] });
export const statsDataStore = writable<StatsData>({ headers: [], data: [] });
export const analysisDataStore = writable<AnalysisData>({
  summary: "",
  keyVariables: [],
  problematicVariables: [],
});
