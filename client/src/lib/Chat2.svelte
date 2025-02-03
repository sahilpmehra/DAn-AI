<script lang="ts">
    import ChatInput from '$lib/components/ChatInput.svelte';
    import ChatMessage from '$lib/components/ChatMessage.svelte';
    import ChatInitScreen from '$lib/ChatInitScreen.svelte';
    import { sessionId } from '$lib/stores/stores';

    type Message = {
        id: string;
        content: string;
        isAi: boolean;
    }

    let messages = $state<Message[]>([]);
    let loading = $state(false);
    let chatStarted = $state(false);
    
    const handleSend = async (content: string) => {
        if (!content.trim()) return;

        chatStarted = true;
        const userMessage: Message = {
            id: crypto.randomUUID(),
            content,
            isAi: false,
        };

        messages = [...messages, userMessage];
        loading = true;

        try {
            const response = await fetch('http://localhost:8000/api/v1/analyze/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: content, session_id: $sessionId }),
            });

            if (!response.ok) throw new Error('Failed to get response');

            const data = await response.json();
            
            messages = [...messages, {
                id: crypto.randomUUID(),
                content: data.response,
                isAi: true,
            }];
        } catch (err) {
            messages = [...messages, {
                id: crypto.randomUUID(),
                content: 'Sorry, there was an error processing your request.',
                isAi: true,
            }];
        } finally {
            loading = false;
        }
    };

    const handleRegenerate = () => {
        // TODO: In a real implementation, this would regenerate the last AI response
        console.log("Regenerate response");
        const lastMessage = messages[messages.length - 1];
        if (lastMessage.isAi) {
            handleSend(lastMessage.content);
        }
    }
</script>

{#if !chatStarted}
    <ChatInitScreen handleSend={handleSend} />
{:else}
    <div class="flex flex-col min-h-screen">
        <div class="flex-1 overflow-y-auto">
            <div class="container max-w-3xl py-4 space-y-4">
                {#each messages as message}
                    <ChatMessage
                        content={message.content}
                        isAi={message.isAi}
                    />
                {/each}
            </div>
        </div>
        <ChatInput class="border-t" onSend={handleSend} />
    </div>
{/if}