<script lang="ts">
  import ToastProvider from '$lib/components/ui/toast/ToastProvider.svelte';
  import FileUpload from '$lib/FileUpload.svelte';
  import DataSummaryPage from '$lib/DataSummaryPage.svelte';
  // import Chat from '$lib/Chat.svelte';
  import Chat2 from '$lib/Chat2.svelte';
  import SideBar from '$lib/SideBar.svelte';
  import Dashboard from '$lib/Dashboard.svelte';
  import ChatInitScreen from '$lib/ChatInitScreen.svelte';
  // import Visualizations from '$lib/Visualizations.svelte';

  // Define current route state
  let currentRoute = $state('upload');

  // Define current chat mode state
  let chatStarted = $state(false);

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
      {#if currentRoute === 'upload'}
        <FileUpload {navigate} />
      {:else if currentRoute === 'chat'}
        {#if chatStarted}
          <Chat2 />
        {:else}
          <ChatInitScreen onStartChat={() => chatStarted = true} />
        {/if}
      {:else if currentRoute === 'visualizations'}
        <Dashboard />
      {:else if currentRoute === 'data-summary'}
        <DataSummaryPage {navigate} />
      {/if} 
    </div>
  </div>
</main> 