import { writable } from "svelte/store";

export const sessionId = writable<string | null>(null);
