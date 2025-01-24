<script lang="ts">
    import { cn } from '$lib/utils';
    import Copy from '$lib/components/ui/icons/Copy.svelte';
    import RotateCcw from '$lib/components/ui/icons/RotateCcw.svelte';
    import Bot from '$lib/components/ui/icons/Bot.svelte';
    import User from '$lib/components/ui/icons/User.svelte';
    import { Button } from '$lib/components/ui/button';
    // import { Toaster } from '$lib/components/ui/sonner';
    import { toast } from '$lib/hooks/use-toast';

    type ChatMessageProps = {
        content: string;
        isAi?: boolean;
        onRegenerate?: () => void;
    }

    let { content, isAi, onRegenerate }: ChatMessageProps = $props();

    const copyToClipboard = () => {
        navigator.clipboard.writeText(content);
        toast({
            title: "Success!",
            description: "Copied to clipboard",
        });
    }
</script>

<div
class={cn(
  "flex gap-4 p-6 message-appear rounded-2xl",
  isAi ? "bg-secondary/50" : "chat-gradient"
)}
>
    <div
        class={cn(
            "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0",
            isAi ? "bg-primary text-primary-foreground" : "bg-background text-foreground"
        )}
    >
        {#if isAi}
            <Bot class="w-4 h-4" />
        {:else}
            <User class="w-4 h-4" />
        {/if}
    </div>
    <div class="flex-1 space-y-4">
        <p class="text-sm leading-relaxed">{content}</p>
        {#if isAi}
            <div class="flex gap-2">
                <Button
                    variant="secondary"
                    size="sm"
                    class="h-8"
                    onclick={copyToClipboard}
                >
                    <Copy class="w-4 h-4 mr-2" />
                    Copy
                </Button>
                {#if onRegenerate}
                    <Button
                        variant="secondary"
                        size="sm"
                        class="h-8"
                        onclick={onRegenerate}
                    >
                        <RotateCcw class="w-4 h-4 mr-2" />
                        Regenerate
                    </Button>
                {/if}
            </div>
        {/if}
    </div>
</div>