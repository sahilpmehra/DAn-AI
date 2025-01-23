<script lang="ts">
    import { Button } from '$lib/components/ui/button';
    import ChatInput from '$lib/components/ChatInput.svelte';
    import MessageSquare from '$lib/components/ui/icons/MessageSquare.svelte';

    type WelcomeScreenProps = {
        onStartChat: () => void;
    }

    let { onStartChat } = $props<WelcomeScreenProps>();

</script>

<div class="flex flex-col min-h-screen">
    <div class="flex-1 flex flex-col items-center justify-center p-4 chat-gradient">
        <h1 class="text-4xl font-bold mb-8 text-foreground">What can I help with?</h1>
        <div class="w-full max-w-3xl mb-8">
            <ChatInput onSend={onStartChat} />
        </div>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl w-full">
            <Button
            variant="secondary"
            class="p-4 h-auto flex flex-col items-center gap-2 hover:bg-secondary/80"
            onclick={onStartChat}
            >
                <MessageSquare class="w-6 h-6" />
                <span>Create image</span>
            </Button>
            <Button
            variant="secondary"
            class="p-4 h-auto flex flex-col items-center gap-2 hover:bg-secondary/80"
            onclick={onStartChat}
            >
                <MessageSquare class="w-6 h-6" />
                <span>Code</span>
            </Button>
            <Button
            variant="secondary"
            class="p-4 h-auto flex flex-col items-center gap-2 hover:bg-secondary/80"
            onclick={onStartChat}
            >
                <MessageSquare class="w-6 h-6" />
                <span>Summarize text</span>
            </Button>
            <Button
            variant="secondary"
            class="p-4 h-auto flex flex-col items-center gap-2 hover:bg-secondary/80"
            onclick={onStartChat}
            >
                <MessageSquare class="w-6 h-6" />
                <span>Analyze data</span>
            </Button>
        </div>
    </div>
</div>