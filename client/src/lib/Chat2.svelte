<script lang="ts">
    import ChatInput from '$lib/components/ChatInput.svelte';
    import ChatMessage from '$lib/components/ChatMessage.svelte';

    type Message = {
        id: string;
        content: string;
        isAi: boolean;
    }

    let messages = $state<Message[]>([
        {
            id: 1,
            content: "Hi! I'm Dan, your Data Analysis AI assistant. How can I help you today?",
            isAi: true,
        },
    ]);
    
    const handleSend = (content: string) => {
        messages = [...messages, 
            {
                id: crypto.randomUUID(),
                content,
                isAi: false,
            },
            {
                id: crypto.randomUUID(),
                content: "This is a sample response from Dan. In a real implementation, this would be replaced with actual AI responses.",
                isAi: true,
            },
        ];
    };

    const handleRegenerate = () => {
        // In a real implementation, this would regenerate the last AI response
        console.log("Regenerate response");
    }
</script>

<div class="flex flex-col min-h-screen">
    <div class="flex-1 overflow-y-auto">
      <div class="container max-w-3xl py-4 space-y-4">
        {#each messages as message}
            <ChatMessage
                key={message.id}
                content={message.content}
                isAi={message.isAi}
                onRegenerate={message.isAi ? handleRegenerate : undefined}
            />
        {/each}
      </div>
    </div>
    <ChatInput class="border-t" onSend={handleSend} />
</div>