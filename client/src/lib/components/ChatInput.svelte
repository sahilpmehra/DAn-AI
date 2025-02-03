<script lang="ts">
    import Paperclip from '$lib/components/ui/icons/Paperclip.svelte';
    import Send from '$lib/components/ui/icons/Send.svelte';
    import { Button } from '$lib/components/ui/button';
    import { Textarea } from '$lib/components/ui/textarea';

    type ChatInputProps = {
        onSend: (message: string) => void;
        class?: string;
    }

    let { onSend, class: className = '' }: ChatInputProps = $props();
    let message = $state('');

    const handleSend = () => {
        if (message.trim() === '') return;
        onSend(message);
        message = '';
    }

    const handleKeyDown = (event: KeyboardEvent) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleSend();
        }
    }
</script>

<div class="bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
    <div class="container flex gap-4 p-4 max-w-3xl">
      <Button variant="secondary" size="icon" class="flex-shrink-0">
        <Paperclip class="w-4 h-4" />
      </Button>
      <Textarea
        bind:value={message}
        onkeydown={handleKeyDown}
        placeholder="Message Dan..."
        class="min-h-0 h-10 resize-none"
      />
      <Button class="flex-shrink-0" onclick={handleSend}>
        <Send class="w-4 h-4" />
      </Button>
    </div>
</div>