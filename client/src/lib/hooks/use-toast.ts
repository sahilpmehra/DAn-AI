import { writable } from "svelte/store";

export type ToastProps = {
  id?: string;
  title?: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  variant?: "default" | "destructive";
  duration?: number;
};

type ToastState = {
  toasts: ToastProps[];
};

// Create a writable store for the toasts
const toastStore = writable<ToastState>({ toasts: [] });

// Helper to generate unique IDs
const generateId = () => Math.random().toString(36).substr(2, 9);

export function toast(props: ToastProps) {
  const id = props.id || generateId();
  const duration = props.duration || 5000;

  // Add the toast to the store
  toastStore.update((state) => ({
    toasts: [
      ...state.toasts,
      {
        ...props,
        id,
        variant: props.variant || "default",
      },
    ],
  }));

  // Automatically remove the toast after duration
  setTimeout(() => {
    dismissToast(id);
  }, duration);
}

export function dismissToast(id: string) {
  toastStore.update((state) => ({
    toasts: state.toasts.filter((toast) => toast.id !== id),
  }));
}

// Export the store for use in the ToastProvider component
export const toasts = {
  subscribe: toastStore.subscribe,
  dismiss: dismissToast,
};
