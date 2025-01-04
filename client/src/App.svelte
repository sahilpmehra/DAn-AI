<script lang="ts">
  import FileUpload from './lib/FileUpload.svelte';
  import Chat from './lib/Chat.svelte';
  
  let isFileUploaded = $state(false);
</script>

<main class="min-h-screen bg-gray-500 py-8">
  <div class="container mx-auto px-4">
    <h1 class="text-3xl font-bold text-center mb-8">AI Data Analyst</h1>
    
    <FileUpload />
    
    <div class="mt-8">
      <Chat />
    </div>
  </div>
</main> 