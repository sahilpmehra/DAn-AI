<script lang="ts">
  import { sessionId } from '$lib/stores/stores';

  type Message = {
    id: string;
    content: string;
    role: 'user' | 'assistant';
    timestamp: Date;
  };

  let messages = $state<Message[]>([]);
  let inputMessage = $state('');
  let loading = $state(false);

  async function sendMessage() {
    if (!inputMessage.trim()) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      content: inputMessage,
      role: 'user',
      timestamp: new Date()
    };

    messages = [...messages, userMessage];
    loading = true;
    const currentInput = inputMessage;
    inputMessage = '';

    try {
      const response = await fetch('http://localhost:8000/api/v1/analyze/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: currentInput, session_id: $sessionId }),
      });

      if (!response.ok) throw new Error('Failed to get response');

      const data = await response.json();
      
      messages = [...messages, {
        id: crypto.randomUUID(),
        content: data.response,
        role: 'assistant',
        timestamp: new Date()
      }];
    } catch (err) {
      messages = [...messages, {
        id: crypto.randomUUID(),
        content: 'Sorry, there was an error processing your request.',
        role: 'assistant',
        timestamp: new Date()
      }];
    } finally {
      loading = false;
    }
  }
</script>

<div class="w-full max-w-4xl mx-auto p-4">
  <div class="bg-white rounded-lg shadow-lg">
    <!-- Messages Container -->
    <div class="h-[500px] overflow-y-auto p-4 space-y-4">
      {#each messages as message (message.id)}
        <div class={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
          <div class={`max-w-[70%] rounded-lg p-3 ${
            message.role === 'user' 
              ? 'bg-blue-500 text-white' 
              : 'bg-gray-100 text-gray-800'
          }`}>
            {message.content}
          </div>
        </div>
      {/each}
      
      {#if loading}
        <div class="flex justify-start">
          <div class="bg-gray-100 rounded-lg p-3 text-gray-800">
            Analyzing...
          </div>
        </div>
      {/if}
    </div>

    <!-- Input Area -->
    <div class="border-t p-4">
      <div class="flex space-x-4">
        <input
          type="text"
          bind:value={inputMessage}
          placeholder="Ask about your data..."
          class="flex-1 p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          onkeydown={(e) => e.key === 'Enter' && sendMessage()}
        />
        <button
          onclick={sendMessage}
          disabled={loading}
          class="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-blue-300"
        >
          Send
        </button>
      </div>
    </div>
  </div>
</div>
