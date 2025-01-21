<script lang="ts">
  import { toasts } from '$lib/hooks/use-toast';
  import { fly } from 'svelte/transition';
  import { Button } from '$lib/components/ui/button';

  let mounted = $state(false);
  
  $effect(() => {
    mounted = true;
  });
</script>

<div
  class="fixed top-0 z-[100] flex max-h-screen w-full flex-col-reverse p-4 sm:bottom-0 sm:right-0 sm:top-auto sm:flex-col md:max-w-[420px]"
>
  {#if mounted}
    {#each $toasts.toasts as toast (toast.id)}
      <div
        role="alert"
        transition:fly={{ duration: 150, x: 300 }}
        class="group pointer-events-auto relative flex w-full items-center justify-between space-x-4 overflow-hidden rounded-md border border-slate-200 p-6 pr-8 shadow-lg transition-all data-[swipe=move]:translate-x-[var(--radix-toast-swipe-move-x)] data-[swipe=cancel]:translate-x-0 data-[swipe=end]:translate-x-[var(--radix-toast-swipe-end-x)] data-[state=open]:animate-in data-[state=closed]:animate-out data-[swipe=end]:animate-out data-[state=closed]:fade-out-80 data-[state=open]:slide-in-from-top-full data-[state=open]:sm:slide-in-from-bottom-full data-[state=closed]:slide-out-to-right-full dark:border-slate-800 bg-white dark:bg-slate-950 mt-4"
      >
        <div class="grid gap-1">
          {#if toast.title}
            <div class="text-sm font-semibold">{toast.title}</div>
          {/if}
          {#if toast.description}
            <div class="text-sm opacity-90">{toast.description}</div>
          {/if}
        </div>
        {#if toast.action}
          <Button
            variant="outline"
            size="sm"
            onclick={() => {
              toast.action?.onClick();
              toasts.dismiss(toast.id || '');
            }}
          >
            {toast.action.label}
          </Button>
        {/if}
        <button
          class="absolute right-2 top-2 rounded-md p-1 text-slate-500 opacity-0 transition-opacity hover:text-slate-900 focus:opacity-100 focus:outline-none focus:ring-2 group-hover:opacity-100 dark:text-slate-400 dark:hover:text-slate-50"
          onclick={() => toasts.dismiss(toast.id || '')}
        >
          <span class="sr-only">Close</span>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            class="h-4 w-4"
          >
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>
      </div>
    {/each}
  {/if}
</div>
