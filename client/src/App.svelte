<script lang="ts">
  import ToastProvider from '$lib/components/ui/toast/ToastProvider.svelte';
  import FileUpload from './lib/FileUpload.svelte';
  import Chat from './lib/Chat.svelte';
  import SideBar from './lib/SideBar.svelte';
  // import Visualizations from './lib/Visualizations.svelte';

  // Define current route state
  let currentRoute = $state('upload');
  
  // Navigation handler that can be passed to children
  function navigate(route: string) {
    currentRoute = route;
  }
</script>

<!-- ToastProvider is mounted once at the root -->
<ToastProvider />

<main class="min-h-screen bg-white">
  <div class="flex">
    <SideBar 
      {currentRoute}
      onNavigate={navigate}
    />
    <div class="flex-1 ml-64">
      <h1 class="text-3xl font-bold text-center my-8">AI Data Analyst</h1>
      
      {#if currentRoute === 'upload'}
        <FileUpload {navigate} />
      {:else if currentRoute === 'chat'}
        <Chat />
      {:else if currentRoute === 'visualizations'}
        <div>Key Variables Visualization Component</div>
      {/if} 
    </div>
  </div>
</main> 